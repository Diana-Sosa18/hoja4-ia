"""
Agente de preguntas frecuentes de Parachute S.A.

Usa el SDK de Anthropic (tool use) para responder preguntas del usuario
consultando exclusivamente la base de conocimientos almacenada en
PostgreSQL + pgvector. Si la información no está en la base de datos,
el agente debe admitir que no puede responder.

Uso:
    python src/agent.py

Escriba "Bye" o presione Ctrl-C para salir.
"""
import json
import os

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from db import get_connection

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

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

TOOLS = [
    {
        "name": "buscar_faqs",
        "description": (
            "Busca en la base de conocimientos vectorial de preguntas frecuentes "
            "de Parachute S.A. y devuelve las entradas semánticamente más "
            "relevantes para una consulta dada."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Texto de búsqueda: la pregunta o tema por el que pregunta el usuario.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Cantidad de resultados a devolver (por defecto 5).",
                },
            },
            "required": ["query"],
        },
    }
]

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


def run_tool(searcher: FaqSearcher, tool_name: str, tool_input: dict) -> dict:
    if tool_name == "buscar_faqs":
        top_k = tool_input.get("top_k") or 5
        resultados = searcher.buscar(tool_input["query"], top_k=top_k)
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Falta ANTHROPIC_API_KEY en el entorno (.env).")

    client = anthropic.Anthropic(api_key=api_key)
    searcher = FaqSearcher()
    messages: list[dict] = []

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

        messages.append({"role": "user", "content": user_input})

        while True:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                texto = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                print(f"\nAgente: {texto}")
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                resultado = run_tool(searcher, block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(resultado, ensure_ascii=False),
                    }
                )
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    chat_loop()
