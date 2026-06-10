#!/usr/bin/env python3
"""
One-time script: embed customer_history.json and upsert into Pinecone.

Usage:
    uv run python scripts/seed_pinecone.py

Required env vars (in .env or exported):
    PINECONE_API_KEY
    PINECONE_INDEX_NAME   (default: mitsubishi-customers, from config.py)
"""
import json
import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from backend.config import settings
from backend.services.rag import EMBEDDING_MODEL

load_dotenv()

PINECONE_API_KEY    = settings.pinecone_api_key
PINECONE_INDEX_NAME = settings.pinecone_index_name
EMBEDDING_DIM       = 1024
NAMESPACE           = settings.pinecone_namespace or "customers-data"

DATA_PATH = Path(__file__).parent.parent / "backend" / "db" / "customer_history.json"


def embed_batch(pc: Pinecone, texts: list[str]) -> list[list[float]]:
    result = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[t[:8000] for t in texts],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    return [r.values for r in result]


def main() -> None:
    with open(DATA_PATH, encoding="utf-8") as f:
        customers: list[dict] = json.load(f)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = {idx.name: idx for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME in existing:
        current_dim = existing[PINECONE_INDEX_NAME].dimension
        if current_dim != EMBEDDING_DIM:
            print(f"Index '{PINECONE_INDEX_NAME}' has dim={current_dim}, expected {EMBEDDING_DIM}. Recreating…")
            pc.delete_index(PINECONE_INDEX_NAME)
            while PINECONE_INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
                print("  waiting for deletion…")
                time.sleep(2)
            existing = {}

    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIM}, metric=cosine)…")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  waiting for index…")
            time.sleep(2)

    index = pc.Index(PINECONE_INDEX_NAME)

    BATCH = 20
    all_vectors: list[dict] = []

    for start in range(0, len(customers), BATCH):
        batch = customers[start : start + BATCH]
        texts = [c["embedding_text"] for c in batch]
        embeddings = embed_batch(pc, texts)

        for cust, emb in zip(batch, embeddings):
            metadata = {
                "customer_id":       cust["customer_id"],
                "nama":              cust["nama"],
                "usia":              cust["usia"],
                "pekerjaan":         cust["pekerjaan"],
                "kota":              cust["kota"],
                "anggaran_min":      cust["anggaran_juta"]["min"],
                "anggaran_max":      cust["anggaran_juta"]["max"],
                "faktor_utama":      ", ".join(cust["faktor_utama"]),
                "kompetitor":        ", ".join(cust.get("kompetitor_dipertimbangkan", [])),
                "outcome":           cust["outcome"],
                "mobil_dibeli":      (
                    f"{cust['mobil_dibeli']['model']} {cust['mobil_dibeli']['tipe']}"
                    if cust.get("mobil_dibeli") else ""
                ),
                "alasan_tidak_jadi": ", ".join(cust.get("alasan_tidak_jadi") or []),
                "tags":              ", ".join(cust.get("tags", [])),
            }
            all_vectors.append({
                "id":       cust["customer_id"],
                "values":   emb,
                "metadata": metadata,
            })

        end = min(start + BATCH, len(customers))
        print(f"  embedded {end}/{len(customers)}")

    index.upsert(vectors=all_vectors, namespace=NAMESPACE)

    stats = index.describe_index_stats()
    print(f"\nDone. Index '{PINECONE_INDEX_NAME}' now has {stats.total_vector_count} vectors.")


main()
