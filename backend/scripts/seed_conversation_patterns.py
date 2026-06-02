"""
Seed sales_history.jsonl → Pinecone namespace: conversation-patterns

Untuk setiap sesi, buat sliding windows dari transcript:
  window[i..i+N]  → vector (sequence yang di-embed)
  next sales turn → effective_next_question (direwrite LLM: max 7 kata, tanpa merek)

Jalankan:
    python -m backend.scripts.seed_conversation_patterns
"""
import json
import uuid
import pathlib

JSONL_PATH  = pathlib.Path(__file__).parent.parent / "db" / "sales_history.jsonl"
WINDOW_SIZE = 4    # utterances per window (~2 full turns)
BATCH_SIZE  = 96   # Pinecone upsert batch limit
LLM_BATCH   = 20   # rewrite per LLM call (batched untuk efisiensi)


def _infer_stage(idx: int, total: int) -> str:
    r = idx / max(total - 1, 1)
    if r < 0.2:  return "early"
    if r < 0.5:  return "rapport"
    if r < 0.8:  return "probing"
    return "closing"


def _fmt(utterances: list[dict]) -> str:
    return "\n".join(
        f"{'Sales' if u['speaker'] == 'sales' else 'Customer'}: {u['text']}"
        for u in utterances
    )


def _rewrite_questions_batch(questions: list[str]) -> list[str]:
    """
    Rewrite daftar pertanyaan via Claude Haiku:
    - Hapus semua nama merek/model (Toyota, Honda, Ertiga, Sigra, dll)
    - Buat pertanyaan generik, natural, max 7 kata
    - Pertahankan topik/dimensi yang digali (keluarga, budget, rutinitas, dll)
    """
    from anthropic import Anthropic
    from backend.config import settings

    client = Anthropic(api_key=settings.claude_api_key)

    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = f"""Kamu adalah asisten yang membantu membersihkan data training untuk AI sales showroom Mitsubishi.

Tugas: Rewrite setiap pertanyaan di bawah menjadi pertanyaan generik untuk sales showroom mobil.
Aturan WAJIB:
- Hapus semua nama merek/model mobil (Toyota, Honda, Suzuki, Daihatsu, Ertiga, Avanza, Jazz, Sigra, dll)
- Maksimal 7 kata
- Natural, tidak formal, to the point
- Pertahankan topik yang digali (jumlah orang, budget, rutinitas, cicilan, dll)
- Jika bukan pertanyaan (kalimat pernyataan), ubah jadi pertanyaan

Pertanyaan:
{numbered}

Jawab HANYA JSON array dengan jumlah elemen sama persis:
["pertanyaan 1", "pertanyaan 2", ...]"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    start, end = raw.find("["), raw.rfind("]") + 1
    result = json.loads(raw[start:end])
    if len(result) != len(questions):
        return questions  # fallback ke original kalau parsing gagal
    return result


def build_records() -> list[dict]:
    with open(JSONL_PATH) as f:
        sessions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(sessions)} sesi dari {JSONL_PATH.name}")

    records = []
    for session in sessions:
        meta       = session.get("metadata", {})
        transcript = meta.get("transcript", [])
        outcome    = meta.get("outcome", "")
        tipe       = meta.get("customer_tipe", "")
        rapport    = meta.get("rapport_style", "")
        key_moments = meta.get("key_moments", [])

        if len(transcript) < WINDOW_SIZE + 1:
            continue

        for i in range(len(transcript) - WINDOW_SIZE):
            window = transcript[i : i + WINDOW_SIZE]
            after  = transcript[i + WINDOW_SIZE :]

            # Pertanyaan/ucapan sales pertama setelah window
            next_sales = next(
                (u["text"] for u in after if u["speaker"] == "sales"), ""
            )
            if not next_sales:
                continue

            # Pakai key_moment terdekat sebagai konteks "kenapa efektif"
            km_idx = min(i // 3, len(key_moments) - 1) if key_moments else -1
            why    = key_moments[km_idx] if km_idx >= 0 else ""

            records.append({
                "sequence":                _fmt(window),
                "stage":                   _infer_stage(i, len(transcript)),
                "effective_next_question": next_sales,
                "why_effective":           why,
                "outcome":                 outcome,
                "customer_tipe":           tipe,
                "rapport_style":           rapport,
            })

    print(f"Generated {len(records)} windows — rewriting questions via LLM...")
    all_questions = [r["effective_next_question"] for r in records]
    rewritten: list[str] = []
    for start in range(0, len(all_questions), LLM_BATCH):
        batch = all_questions[start : start + LLM_BATCH]
        print(f"  Rewriting {start+1}-{start+len(batch)} / {len(all_questions)}...")
        rewritten.extend(_rewrite_questions_batch(batch))

    for rec, q in zip(records, rewritten):
        rec["effective_next_question"] = q

    print(f"Done rewriting. Total records: {len(records)}")
    return records


def _embed_batch(texts: list[str]) -> list[list[float]]:
    from backend.services.rag import EMBEDDING_MODEL, _pinecone_client
    pc = _pinecone_client()
    result = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[t[:8000] for t in texts],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    return [r.values for r in result]


def seed():
    from backend.config import settings
    from backend.services.rag import CONVERSATION_PATTERNS_NS, _pinecone_client

    records = build_records()
    if not records:
        print("Tidak ada data.")
        return

    index         = _pinecone_client().Index(settings.pinecone_index_name)
    total_upserted = 0

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        print(f"Embedding batch {start // BATCH_SIZE + 1}/{-(-len(records) // BATCH_SIZE)} "
              f"({len(batch)} records)...")

        embeddings = _embed_batch([r["sequence"] for r in batch])

        vectors = [
            {
                "id":       f"pattern-{uuid.uuid4().hex[:10]}",
                "values":   emb,
                "metadata": rec,        # sequence ikut disimpan untuk debug
            }
            for rec, emb in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors, namespace=CONVERSATION_PATTERNS_NS)
        total_upserted += len(vectors)
        print(f"  upserted {total_upserted}/{len(records)}")

    print(f"\nDone — {total_upserted} patterns di namespace '{CONVERSATION_PATTERNS_NS}'")


def seed_topic_transitions():
    """
    Baca sales_history.jsonl → extract topic transitions → upsert ke
    namespace "conversation-patterns" dengan record_type="topic_transition".

    Aman dijalankan berulang — id deterministik sehingga Pinecone upsert
    akan overwrite record yang sama.
    """
    import uuid
    from backend.config import settings
    from backend.services.rag import CONVERSATION_PATTERNS_NS, _pinecone_client
    from backend.services.topic_patterns import (
        extract_transitions,
        build_transition_embed_text,
    )

    with open(JSONL_PATH, encoding="utf-8") as f:
        sessions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(sessions)} sesi dari {JSONL_PATH.name}")

    records: list[dict] = []
    for session in sessions:
        sid        = session.get("id", str(uuid.uuid4()))
        meta       = session.get("metadata", {})
        transcript = meta.get("transcript", [])
        if not transcript:
            continue

        transitions = extract_transitions(transcript)
        for i, trans in enumerate(transitions):
            embed_text = build_transition_embed_text(trans, meta)
            records.append({
                "_id":        f"trans_{sid}_{i}",
                "embed_text": embed_text,
                "metadata": {
                    "record_type":               "topic_transition",
                    "from_topic":                trans["from_topic"],
                    "to_topic":                  trans["to_topic"],
                    "trigger_question":          trans.get("trigger_question", ""),
                    "customer_response_preview": trans.get("customer_response_preview", ""),
                    "transition_turn":           trans.get("transition_turn", 0),
                    "naturalness":               trans.get("naturalness", "unknown"),
                    "outcome":                   meta.get("outcome", ""),
                    "customer_tipe":             meta.get("customer_tipe", ""),
                    "session_id":                sid,
                },
            })
        if transitions:
            print(f"  {sid}: {len(transitions)} transisi")

    if not records:
        print("Tidak ada transisi yang bisa di-extract.")
        return

    print(f"\nTotal: {len(records)} transisi — mulai embed & upsert...")

    index          = _pinecone_client().Index(settings.pinecone_index_name)
    total_upserted = 0

    for start in range(0, len(records), BATCH_SIZE):
        batch      = records[start : start + BATCH_SIZE]
        embeddings = _embed_batch([r["embed_text"] for r in batch])

        vectors = [
            {
                "id":       r["_id"],
                "values":   emb,
                "metadata": r["metadata"],
            }
            for r, emb in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors, namespace=CONVERSATION_PATTERNS_NS)
        total_upserted += len(vectors)
        print(f"  upserted {total_upserted}/{len(records)}")

    print(f"\nDone — {total_upserted} topic transitions di namespace '{CONVERSATION_PATTERNS_NS}'")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "transitions":
        seed_topic_transitions()
    else:
        seed()
        seed_topic_transitions()
