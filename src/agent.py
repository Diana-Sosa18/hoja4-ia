"""
Agente de preguntas frecuentes de Parachute S.A.

Usa el SDK de Google Gemini (function calling / tool use) para responder
preguntas del usuario consultando exclusivamente la base de conocimientos
almacenada en PostgreSQL + pgvector. Si la información no está en la base de
datos, el agente debe admitir que no puede responder.

Uso:
    python src/agent.py

Escriba "Bye" o presione Ctrl-C para salir.
"""
import os
import time

# El modelo de embeddings ya se descargó al correr load_data.py; se evita que
# sentence-transformers intente revisar Hugging Face por internet en cada
# arranque (eso puede colgarse varios minutos con conexiones lentas/inestables).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from sentence_transformers import SentenceTransformer

from db import get_connection

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
# OJO: "gemini-flash-latest" puede resolver a un modelo de vista previa con
# cuota gratuita muy baja (se observó un límite de solo 20 solicitudes/día).
# Se fija "gemini-2.5-flash" explícitamente, con cuota gratuita mucho mayor.
# Si tu cuenta se queda sin cuota, puedes sobreescribir con la variable de
# entorno GEMINI_MODEL, por ejemplo a "gemini-2.5-flash-lite".
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TOOL_ITERATIONS = 6
MAX_API_RETRIES = 5

SYSTEM_PROMPT = """Eres el asistente de atención al cliente de Parachute S.A., \
un evento de paracaidismo en Guatemala. Tu única fuente de información es la \
herramienta `buscar_faqs`, que consulta la base de conocimientos oficial de \
preguntas frecuentes de la empresa mediante búsqueda semántica.

Reglas estrictas:
1. Para CUALQUIER pregunta del usuario relacionada con el evento, primero debes \
   llamar a la herramienta `buscar_faqs` para buscar información relevante.
2. Pasa el argumento `query` de `buscar_faqs` lo más parecido posible al texto \
   literal que escribió el usuario. NO lo reformules, resumas, traduzcas ni \
   combines con sinónimos: el motor de búsqueda es un modelo pequeño que \
   funciona mejor con la pregunta original que con una versión "mejorada".
3. La búsqueda es aproximada: puede devolver entradas con un puntaje de \
   "similitud" alto cuyo TEMA en realidad no tiene nada que ver con la \
   pregunta del usuario. IGNORA el puntaje numérico y evalúa tú mismo, \
   comparando el tema de la pregunta del usuario con el campo "pregunta" de \
   cada resultado, cuál(es) tratan genuinamente el mismo tema.
4. Un resultado es válido y debes usarlo aunque su "respuesta" sea breve o \
   genérica (por ejemplo, "consulte a soporte para más detalles"): eso es la \
   información oficial registrada para esa pregunta, no significa que el \
   resultado sea irrelevante. Solo descarta un resultado cuando su "pregunta" \
   trata un TEMA distinto al que se preguntó (p. ej. el usuario pregunta por \
   parqueo y el resultado habla del clima).
5. Responde ÚNICAMENTE con base en los resultados cuyo tema coincida. \
   No inventes ni completes con detalles que no estén en la "respuesta" del \
   resultado, aunque esta se sienta incompleta.
6. Si tras UNA búsqueda ningún resultado trata el mismo tema que la pregunta \
   del usuario, no sigas reintentando con variaciones de la consulta: responde \
   con honestidad que no cuentas con esa información en tu base de \
   conocimientos y que el usuario contacte a soporte@parachutesa.gt. No fuerces \
   una respuesta con datos que no correspondan al tema preguntado.
7. Responde siempre en español, de forma clara y directa.
"""

BUSCAR_FAQS_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="buscar_faqs",
            description=(
                "Busca en la base de conocimientos vectorial de preguntas frecuentes "
                "de Parachute S.A. y devuelve las entradas semánticamente más "
                "relevantes para una consulta dada."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Texto de búsqueda: la pregunta o tema por el que pregunta el usuario.",
                    ),
                    "top_k": types.Schema(
                        type=types.Type.INTEGER,
                        description="Cantidad de resultados a devolver (por defecto 5).",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)

GENERATE_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[BUSCAR_FAQS_TOOL],
)


class FaqSearcher:
    """
    Búsqueda semántica sobre pgvector. No se filtra por un umbral numérico de
    similitud: con un corpus pequeño y un modelo de embeddings liviano en
    español, el puntaje de coseno no separa de forma confiable lo relevante de
    lo irrelevante (una pregunta totalmente ajena puede puntuar similar o más
    alto que un parafraseo válido). Por eso se devuelven siempre los top_k
    resultados con su puntaje, y es el LLM quien decide, leyendo el contenido,
    si de verdad responden la pregunta del usuario.
    """

    def __init__(self):
        print("Cargando modelo de embeddings, un momento...")
        self.model = SentenceTransformer(MODEL_NAME)

    def buscar(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT faq_id, categoria, pregunta, respuesta,
                           embedding <=> %s::vector AS distance
                    FROM faqs
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        return [
            {
                "faq_id": faq_id,
                "categoria": categoria,
                "pregunta": pregunta,
                "respuesta": respuesta,
                "similitud": round(float(1 - distance), 4),
            }
            for faq_id, categoria, pregunta, respuesta, distance in rows
        ]


class QuotaExceededError(Exception):
    """Se agotó la cuota gratuita diaria/por minuto de la API de Gemini."""


def generate_with_retry(client: genai.Client, history: list[types.Content]):
    """
    El nivel gratuito de Gemini a veces responde 503 "high demand" de forma
    transitoria: se reintenta con backoff. Un 429 por cuota agotada no se
    arregla reintentando en segundos, así que se reporta de inmediato.
    """
    for intento in range(1, MAX_API_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=history,
                config=GENERATE_CONFIG,
            )
        except genai_errors.ClientError as e:
            if e.status == "RESOURCE_EXHAUSTED" or getattr(e, "code", None) == 429:
                raise QuotaExceededError(str(e)) from e
            raise
        except genai_errors.ServerError:
            if intento == MAX_API_RETRIES:
                raise
            espera = 2**intento
            print(f"(El servicio de Gemini está saturado, reintentando en {espera}s...)")
            time.sleep(espera)


def run_tool(searcher: FaqSearcher, tool_name: str, tool_args: dict) -> dict:
    if tool_name == "buscar_faqs":
        top_k = tool_args.get("top_k") or 5
        resultados = searcher.buscar(tool_args["query"], top_k=top_k)
        return {
            "resultados": resultados,
            "nota": (
                "El campo 'similitud' es orientativo y no siempre es confiable. "
                "Evalúa el contenido de 'pregunta'/'respuesta' de cada resultado "
                "para decidir si realmente responde lo que preguntó el usuario."
            ),
        }
    raise ValueError(f"Herramienta desconocida: {tool_name}")


def chat_loop():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Falta GEMINI_API_KEY en el entorno (.env).")

    client = genai.Client(api_key=api_key)
    searcher = FaqSearcher()
    history: list[types.Content] = []

    print("=" * 60)
    print("Agente de FAQs - Parachute S.A.")
    print('Escribe tu pregunta. Escribe "Bye" o Ctrl-C para salir.')
    print("=" * 60)

    while True:
        try:
            user_input = input("\nTú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta luego!")
            break

        if not user_input:
            continue
        if user_input.strip().lower() == "bye":
            print("Agente: ¡Hasta luego!")
            break

        history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                response = generate_with_retry(client, history)
                candidate = response.candidates[0]
                history.append(candidate.content)

                function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

                if not function_calls:
                    texto = "".join(p.text for p in candidate.content.parts if p.text)
                    print(f"\nAgente: {texto}")
                    break

                response_parts = []
                for fc in function_calls:
                    resultado = run_tool(searcher, fc.name, dict(fc.args))
                    response_parts.append(
                        types.Part.from_function_response(name=fc.name, response={"result": resultado})
                    )
                history.append(types.Content(role="user", parts=response_parts))
            else:
                print(
                    "\nAgente: No logré encontrar una respuesta confiable en mi base de "
                    "conocimientos. Por favor contacta a soporte@parachutesa.gt."
                )
        except QuotaExceededError:
            print(
                "\nAgente: Se agotó la cuota gratuita de la API de Gemini para este "
                "modelo (el nivel gratuito tiene un límite de solicitudes por día/minuto). "
                "Espera un momento, o cambia GEMINI_MODEL en tu .env a otro modelo "
                "(por ejemplo 'gemini-2.5-flash-lite')."
            )
        except genai_errors.ServerError:
            print(
                "\nAgente: El servicio de Gemini no está disponible en este momento "
                "(alta demanda del nivel gratuito). Intenta de nuevo en un minuto."
            )


if __name__ == "__main__":
    chat_loop()
