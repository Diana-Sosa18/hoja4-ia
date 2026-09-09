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
preguntas frecuentes de la empresa.

Reglas estrictas:
1. Para CUALQUIER pregunta del usuario relacionada con el evento, primero debes \
   llamar a la herramienta `buscar_faqs` para buscar información relevante.
2. Responde ÚNICAMENTE con base en los resultados que te devuelva la herramienta. \
   No inventes ni completes información con conocimiento propio.
3. Si los resultados de la herramienta no contienen información relevante para \
   la pregunta, responde con honestidad que no cuentas con esa información en tu \
   base de conocimientos y que el usuario contacte a soporte@parachutesa.gt.
4. Responde siempre en español, de forma clara y directa.
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
                    "description": "Cantidad de resultados a devolver (por defecto 3).",
                },
            },
            "required": ["query"],
        },
    }
]

SIMILARITY_THRESHOLD = 0.35


class FaqSearcher:
    def __init__(self):
        print("Cargando modelo de embeddings, un momento...")
        self.model = SentenceTransformer(MODEL_NAME)

    def buscar(self, query: str, top_k: int = 3) -> list[dict]:
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

        resultados = []
        for faq_id, categoria, pregunta, respuesta, distance in rows:
            similitud = 1 - distance
            if similitud < SIMILARITY_THRESHOLD:
                continue
            resultados.append(
                {
                    "faq_id": faq_id,
                    "categoria": categoria,
                    "pregunta": pregunta,
                    "respuesta": respuesta,
                    "similitud": round(float(similitud), 4),
                }
            )
        return resultados


def run_tool(searcher: FaqSearcher, tool_name: str, tool_input: dict) -> dict:
    if tool_name == "buscar_faqs":
        top_k = tool_input.get("top_k") or 3
        resultados = searcher.buscar(tool_input["query"], top_k=top_k)
        if not resultados:
            return {"resultados": [], "mensaje": "No se encontró información relevante en la base de conocimientos."}
        return {"resultados": resultados}
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
