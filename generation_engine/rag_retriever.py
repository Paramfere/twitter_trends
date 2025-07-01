"""FAISS-based retrieval of exemplar tweets.

This module builds/updates a single vector index that stores embeddings of
high-engagement tweets collected by the FERE pipeline.  The index lives under
``~/.fere_cache/embeddings/faiss.index`` (configurable via ``FERE_FAISS_PATH``).

Key API
--------
refresh_index_from_session(session_dir: Path) -> None
    Scan the session's ``kaito_tech_summary_*.csv`` files, embed *new* tweet
    texts, and append them to the index.
get_examples(topic: str, k: int = 3) -> list[str]
    Return *k* tweet texts whose embeddings are nearest to the given *topic*
    embedding.

Embeddings
----------
*Primary* – OpenAI ``text-embedding-3-small`` (cheap, high quality).  Requires
``OPENAI_API_KEY``.
*Fallback* – Sentence-Transformers ``all-MiniLM-L6-v2`` (local, no cost).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List

import faiss  # type: ignore
import numpy as np
import pandas as pd

try:
    import openai

    _OPENAI_OK = True if os.getenv("OPENAI_API_KEY") else False
except ImportError:  # pragma: no cover – handled in fallback
    _OPENAI_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _ST_MODEL: SentenceTransformer | None = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:  # pragma: no cover
    _ST_MODEL = None

# ----------------------------------------------------------------------------
# Paths & helpers
# ----------------------------------------------------------------------------
_CACHE_DIR = Path(os.getenv("FERE_CACHE", Path.home() / ".fere_cache"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_INDEX_PATH = Path(os.getenv("FERE_FAISS_PATH", _CACHE_DIR / "embeddings" / "faiss.index"))
_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
_META_PATH = _INDEX_PATH.with_suffix(".jsonl")  # stores tweet text, sha1 hash


def _sha1(text: str) -> str:  # noqa: D401
    """Return SHA-1 hash of *text* (hex)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Embedding back-end
# ----------------------------------------------------------------------------


def _embed(texts: list[str]) -> np.ndarray:  # noqa: D401
    """Return an (N, 1536) float32 numpy array of embeddings."""

    if _OPENAI_OK:
        client = openai.OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)  # type: ignore[attr-defined]
        vectors = np.array([r.embedding for r in resp.data], dtype="float32")
        return vectors

    if _ST_MODEL is None:  # pragma: no cover – fallback unavailable
        raise RuntimeError("No embedding backend available. Install sentence-transformers or set OPENAI_API_KEY")

    vecs = _ST_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


# ----------------------------------------------------------------------------
# Index persistence helpers
# ----------------------------------------------------------------------------


def _load_index() -> tuple[faiss.IndexFlatIP, list[str]]:
    if _INDEX_PATH.exists():
        index = faiss.read_index(str(_INDEX_PATH))
        texts = [_d["text"] for _d in map(json.loads, _META_PATH.read_text().splitlines())]
    else:
        index = faiss.IndexFlatIP(1536)
        texts = []
    return index, texts


def _save_index(index: faiss.IndexFlatIP, texts: list[str]) -> None:
    faiss.write_index(index, str(_INDEX_PATH))
    with _META_PATH.open("w") as fp:
        for t in texts:
            fp.write(json.dumps({"text": t, "hash": _sha1(t)}) + "\n")


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def refresh_index_from_session(session_dir: Path) -> None:  # noqa: D401
    """Embed and add new tweets from *session_dir* into the global index."""
    session_dir = Path(session_dir)
    summary_paths = list(session_dir.glob("raw/kaito_data/kaito_tech_summary_*.csv"))
    if not summary_paths:
        return

    index, texts = _load_index()
    known_hashes = {_sha1(t) for t in texts}

    new_texts: list[str] = []
    for csv_path in summary_paths:
        df = pd.read_csv(csv_path)
        if "tweet_text" not in df.columns:
            continue
        for txt in df["tweet_text"].astype(str).tolist():
            h = _sha1(txt)
            if h not in known_hashes:
                known_hashes.add(h)
                new_texts.append(txt)

    if not new_texts:
        return

    vectors = _embed(new_texts)
    index.add(vectors)
    texts.extend(new_texts)
    _save_index(index, texts)


def get_examples(topic: str, k: int = 3) -> List[str]:  # noqa: D401
    """Return *k* exemplar tweets nearest to *topic* (by embedding cosine sim)."""
    if not _INDEX_PATH.exists():
        return []

    index, texts = _load_index()
    if index.ntotal == 0:
        return []

    query_vec = _embed([topic])
    sims, idxs = index.search(query_vec, k)
    idx_list = idxs[0][:k]
    return [texts[i] for i in idx_list if i < len(texts)] 