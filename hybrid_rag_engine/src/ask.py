import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

from config import SETTINGS
from hybrid import load_bm25, rrf_fuse, Chunk

load_dotenv()

def format_context(chunks: List[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        cite = f"[{i}] {c.source}" + (f" p.{c.page}" if c.page is not None else "")
        lines.append(f"{cite}\n{c.text}")
    return "\n\n---\n\n".join(lines)

def main():
    question = input("Ask: ").strip()
    if not question:
        return

    # Load BM25
    bm25 = load_bm25("storage")

    # Vector query to Qdrant
    client = QdrantClient(url=SETTINGS.qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name=SETTINGS.collection)

    vq = VectorStoreQuery(query_str=question, similarity_top_k=SETTINGS.top_k_vec)
    vec_res = vector_store.query(vq)

    vec_ranked = []
    for node, score in zip(vec_res.nodes, vec_res.similarities):
        # node.id_ was stored as chunk_id
        ch = Chunk(
            chunk_id=node.node_id,
            text=node.get_content(),
            source=str((node.metadata or {}).get("source", "unknown")),
            page=(node.metadata or {}).get("page", None),
        )
        vec_ranked.append((ch, float(score)))

    bm25_ranked = bm25.query(question, top_k=SETTINGS.top_k_bm25)

    fused = rrf_fuse(vec_ranked, bm25_ranked, k=SETTINGS.rrf_k)[:SETTINGS.final_top_k]
    fused_chunks = [c for c, _ in fused]

    context = format_context(fused_chunks)

    # Citation-enforced response schema
    system = (
        "You are a technical assistant. Use ONLY the provided context.\n"
        "Cite sources using bracket numbers like [1], [2] matching the context chunks.\n"
        "If the answer is not contained in the context, say you don't know.\n"
        "Return JSON with keys: answer, citations.\n"
        "citations must be a list of objects: {chunk: int, source: str, page: int|null}.\n"
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT CHUNKS:\n{context}\n\n"
        "Return strictly valid JSON."
    )

    llm = Ollama(
    model="llama3",
    temperature=0.1,)

    resp = llm.chat([ChatMessage(role="system", content=system),
                     ChatMessage(role="user", content=user)])

    text = resp.message.content.strip()
    try:
        data = json.loads(text)
    except Exception:
        # fallback: print raw if model misbehaves
        print("⚠️ Non-JSON output:\n", text)
        return

    print("\n=== ANSWER ===\n")
    print(data.get("answer", ""))

    print("\n=== CITATIONS ===\n")
    for c in data.get("citations", []):
        print(f"[{c['chunk']}] {c['source']}" + (f" p.{c['page']}" if c.get("page") is not None else ""))

if __name__ == "__main__":
    main()
