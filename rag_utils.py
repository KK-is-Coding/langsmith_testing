"""Helpers to build a FAISS index on Gemini's free embedding tier (about 100 embeddings/minute)."""
import time
from langchain_community.vectorstores import FAISS


def _is_rate_limit(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()


def build_faiss(splits, emb, batch_size: int = 90, pause: float = 62.0, max_attempts: int = 6):
    """Embed `splits` in small batches, pausing between them so the free-tier
    per-minute limit is not exceeded. Retries a batch if a 429 still happens."""
    total = len(splits)
    if total == 0:
        raise ValueError("No chunks to embed.")
    vs = None
    for start in range(0, total, batch_size):
        batch = splits[start:start + batch_size]
        for attempt in range(1, max_attempts + 1):
            try:
                if vs is None:
                    vs = FAISS.from_documents(batch, emb)
                else:
                    vs.add_documents(batch)
                break
            except Exception as e:
                if _is_rate_limit(e) and attempt < max_attempts:
                    print(f"  rate limited, waiting {pause:.0f}s (attempt {attempt}/{max_attempts})...")
                    time.sleep(pause)
                    continue
                raise
        done = min(start + batch_size, total)
        print(f"  embedded {done}/{total} chunks")
        if done < total:
            time.sleep(pause)
    return vs