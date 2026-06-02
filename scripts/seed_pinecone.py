#!/usr/bin/env python3
"""
One-time script: embed customer_history.json and upsert into Pinecone.

Usage:
    uv run python scripts/seed_pinecone.py

Required env vars (in .env or exported):
    OPENAI_API_KEY
    PINECONE_API_KEY
    PINECONE_INDEX_NAME   (default: showroom-ai-customers)
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "showroom-ai-customers")
EMBEDDING_MODEL     = "text-embedding-3-small"
EMBEDDING_DIM       = 1536

DATA_PATH = Path(__file__).parent.parent / "backend" / "db" / "customer_history.json"


async def embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


async def main() -> None:
    with open(DATA_PATH, encoding="utf-8") as f:
        customers: list[dict] = json.load(f)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIM}, metric=cosine)…")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        import time
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  waiting for index…")
            time.sleep(2)

    index = pc.Index(PINECONE_INDEX_NAME)
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Embed in batches of 20
    BATCH = 20
    all_vectors: list[dict] = []

    for start in range(0, len(customers), BATCH):
        batch = customers[start : start + BATCH]
        texts = [c["embedding_text"] for c in batch]
        embeddings = await embed_batch(client, texts)

        for cust, emb in zip(batch, embeddings):
            metadata = {
                "customer_id":    cust["customer_id"],
                "nama":           cust["nama"],
                "usia":           cust["usia"],
                "pekerjaan":      cust["pekerjaan"],
                "kota":           cust["kota"],
                "anggaran_min":   cust["anggaran_juta"]["min"],
                "anggaran_max":   cust["anggaran_juta"]["max"],
                "faktor_utama":   ", ".join(cust["faktor_utama"]),
                "kompetitor":     ", ".join(cust.get("kompetitor_dipertimbangkan", [])),
                "outcome":        cust["outcome"],
                "mobil_dibeli":   (
                    f"{cust['mobil_dibeli']['model']} {cust['mobil_dibeli']['tipe']}"
                    if cust.get("mobil_dibeli") else ""
                ),
                "alasan_tidak_jadi": ", ".join(cust.get("alasan_tidak_jadi") or []),
                "tags":           ", ".join(cust.get("tags", [])),
            }
            all_vectors.append({
                "id":       cust["customer_id"],
                "values":   emb,
                "metadata": metadata,
            })

        end = min(start + BATCH, len(customers))
        print(f"  embedded {end}/{len(customers)}")

    # Upsert in one go (50 records fits easily)
    index.upsert(vectors=all_vectors)

    stats = index.describe_index_stats()
    print(f"\nDone. Index '{PINECONE_INDEX_NAME}' now has {stats.total_vector_count} vectors.")


asyncio.run(main())
