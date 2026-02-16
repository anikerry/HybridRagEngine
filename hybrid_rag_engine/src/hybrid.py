import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None

def tokenize(text: str) -> List[str]:
    # simple tokenization; can later switch to better tokenizer
    return [t for t in text.lower().split() if t.strip()]

class BM25Index:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def query(self, q: str, top_k: int) -> List[Tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(q))
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i], float(s)) for i, s in ranked]

def rrf_fuse(
    vec_ranked: List[Tuple[Chunk, float]],
    bm25_ranked: List[Tuple[Chunk, float]],
    k: int = 60
) -> List[Tuple[Chunk, float]]:
    # We only use ranks, not scores. RRF: sum(1/(k + rank))
    score_map: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    def add(rank_list: List[Tuple[Chunk, float]]):
        for r, (ch, _) in enumerate(rank_list, start=1):
            chunk_map[ch.chunk_id] = ch
            score_map[ch.chunk_id] = score_map.get(ch.chunk_id, 0.0) + (1.0 / (k + r))

    add(vec_ranked)
    add(bm25_ranked)

    fused = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], sc) for cid, sc in fused]

def save_bm25(storage_dir: str, chunks: List[Chunk]):
    os.makedirs(storage_dir, exist_ok=True)
    with open(os.path.join(storage_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

def load_bm25(storage_dir: str) -> BM25Index:
    chunks: List[Chunk] = []
    path = os.path.join(storage_dir, "chunks.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return BM25Index(chunks)
