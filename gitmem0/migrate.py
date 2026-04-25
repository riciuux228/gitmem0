"""Migration utilities for GitMem0.

Re-embeds all memories when the embedding model changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from gitmem0.embeddings import EmbeddingEngine
from gitmem0.store import MemoryStore


def re_embed_all(
    store: MemoryStore,
    embeddings: EmbeddingEngine,
    batch_size: int = 32,
    verbose: bool = False,
) -> dict:
    """Re-embed all memories with the current model.

    Args:
        store: MemoryStore instance
        embeddings: EmbeddingEngine instance
        batch_size: Number of memories to process at once
        verbose: Print progress to stderr

    Returns:
        {total, re_embedded, skipped}
    """
    memories = store.list_memories(limit=999_999)
    total = len(memories)
    re_embedded = 0
    skipped = 0

    # Process in batches for efficiency
    for i in range(0, total, batch_size):
        batch = memories[i : i + batch_size]
        texts = [m.content for m in batch]

        try:
            vectors = embeddings.embed_batch(texts)
        except Exception:
            # Fallback: embed one by one
            vectors = []
            for text in texts:
                try:
                    vectors.append(embeddings.embed(text))
                except Exception:
                    vectors.append(None)

        for mem, vec in zip(batch, vectors):
            if vec is None:
                skipped += 1
                continue
            mem.embedding = vec
            store.update_memory(mem)
            re_embedded += 1

        if verbose:
            pct = min(100, int((i + len(batch)) / total * 100))
            print(f"  [{pct}%] {i + len(batch)}/{total}", file=sys.stderr)

    return {
        "total": total,
        "re_embedded": re_embedded,
        "skipped": skipped,
    }
