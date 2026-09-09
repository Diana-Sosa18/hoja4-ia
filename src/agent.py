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

# El modelo de embeddings ya se descargó al correr load_data.py; se evita que
# sentence-transformers intente revisar Hugging Face por internet en cada
# arranque (eso puede colgarse varios minutos con conexiones lentas/inestables).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from db import get_connection

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """Eres el asistente de atención al cliente de Parachute S.A., \
un evento de paracaidismo en Guatemala. Tu única fuente de información es la \
herramienta `buscar_faqs`, que consulta la base de conocimientos oficial de \
preguntas frecuentes de la empresa mediante búsqueda semántica.

Reglas estrictas:
1. Para CUALQUIER pregunta del usuario relacionada con el evento, primero debes \
   llamar a la herramienta `buscar_faqs` para buscar información relevante.
2. La búsqueda es aproximada: puede devolver entradas con un puntaje de \
   "similitud" alto que en realidad NO responden la pregunta del usuario. \
   IGNORA el puntaje numérico y evalúa tú mismo, leyendo la pregunta y \
   respuesta de cada resultado, si de verdad responde lo que se preguntó.
3. Responde ÚNICAMENTE con base en los resultados que realmente sean relevantes. \
   No inventes ni completes información con conocimiento propio.
4. Si ninguno de los resultados devueltos responde de verdad la pregunta del \
   usuario, responde con honestidad que no cuentas con esa información en tu \
   base de conocimientos y que el usuario contacte a soporte@parachutesa.gt. \
   No fuerces una respuesta con datos que no correspondan al tema preguntado.
5. Responde siempre en español, de forma clara y directa.
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

        while True:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=history,
                config=GENERATE_CONFIG,
            )
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


if __name__ == "__main__":
    chat_loop()
