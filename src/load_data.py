"""
Script de carga: parsea el corpus de FAQs de Parachute S.A., genera embeddings
con sentence-transformers y los inserta en la tabla `faqs` de PostgreSQL/pgvector.

Uso:
    python src/load_data.py [--corpus data/Corpus_FAQs_Parachute_SA_2026.txt]
"""
import argparse
import json
import re
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from db import connect
from pgvector.psycopg import register_vector

MODEL_NAME = "all-MiniLM-L6-v2"

BLOCK_RE = re.compile(
    r"ID:\s*(?P<id>FAQ-\d+)\s*\n"
    r"CATEG[OÓ]R[IÍ]A:\s*(?P<categoria>.+?)\s*\n"
    r"PREGUNTA:\s*(?P<pregunta>.+?)\s*\n"
    r"RESPUESTA:\s*(?P<respuesta>.+?)\s*\n"
    r"METADATA:\s*(?P<metadata>\{.*?\})",
    re.DOTALL,
)


def parse_corpus(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    faqs = []
    for match in BLOCK_RE.finditer(text):
        faqs.append(
            {
                "faq_id": match.group("id").strip(),
                "categoria": match.group("categoria").strip(),
                "pregunta": match.group("pregunta").strip(),
                "respuesta": match.group("respuesta").strip(),
                "metadata": json.loads(match.group("metadata")),
            }
        )
    return faqs


def ensure_schema(conn):
    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    with conn.cursor() as cur:
        cur.execute(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def load(corpus_path: Path):
    faqs = parse_corpus(corpus_path)
    if not faqs:
        print(f"No se encontraron FAQs en {corpus_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Se parsearon {len(faqs)} FAQs de {corpus_path.name}")

    print(f"Cargando modelo de embeddings '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    textos = [f"Pregunta: {f['pregunta']}\nRespuesta: {f['respuesta']}" for f in faqs]
    print("Generando embeddings...")
    embeddings = model.encode(textos, show_progress_bar=True, normalize_embeddings=True)

    conn = connect(register=False)
    try:
        ensure_schema(conn)
        register_vector(conn)
        with conn.cursor() as cur:
            for faq, embedding in zip(faqs, embeddings):
                cur.execute(
                    """
                    INSERT INTO faqs (faq_id, categoria, pregunta, respuesta, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (faq_id) DO UPDATE SET
                        categoria = EXCLUDED.categoria,
                        pregunta = EXCLUDED.pregunta,
                        respuesta = EXCLUDED.respuesta,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        faq["faq_id"],
                        faq["categoria"],
                        faq["pregunta"],
                        faq["respuesta"],
                        json.dumps(faq["metadata"]),
                        embedding.tolist(),
                    ),
                )
        conn.commit()
        print(f"Se cargaron/actualizaron {len(faqs)} FAQs en la tabla 'faqs'.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "Corpus_FAQs_Parachute_SA_2026.txt",
        help="Ruta al archivo de corpus de FAQs.",
    )
    args = parser.parse_args()
    load(args.corpus)
