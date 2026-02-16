import json
from typing import List
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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

    # 0) Local embed model (must match what you used in ingest)
    print("🔄 Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 1) Load BM25
    print("🔄 Loading BM25 index...")
    bm25 = load_bm25("storage")

    # 2) Vector query to Qdrant (IMPORTANT: use query_embedding, not query_str)
    print("🔄 Connecting to Qdrant...")
    client = QdrantClient(url=SETTINGS.qdrant_url, timeout=SETTINGS.qdrant_timeout)
    vector_store = QdrantVectorStore(client=client, collection_name=SETTINGS.collection)

    print("🔄 Generating query embedding...")
    q_emb = embed_model.get_query_embedding(question)
    if q_emb is None:
        raise RuntimeError("Query embedding is None. Check HuggingFaceEmbedding installation/model.")

    print("🔍 Searching vector database...")
    vq = VectorStoreQuery(query_embedding=q_emb, similarity_top_k=SETTINGS.top_k_vec)
    vec_res = vector_store.query(vq)

    vec_ranked = []
    for node, score in zip(vec_res.nodes, vec_res.similarities):
        ch = Chunk(
            chunk_id=node.node_id,
            text=node.get_content(),
            source=str((node.metadata or {}).get("source", "unknown")),
            page=(node.metadata or {}).get("page", None),
        )
        vec_ranked.append((ch, float(score)))

    # 3) BM25 query
    print("🔍 Performing BM25 search...")
    bm25_ranked = bm25.query(question, top_k=SETTINGS.top_k_bm25)

    # 4) Fuse via RRF
    print("🔄 Fusing search results...")
    fused = rrf_fuse(vec_ranked, bm25_ranked, k=SETTINGS.rrf_k)[: SETTINGS.final_top_k]
    fused_chunks = [c for c, _ in fused]
    context = format_context(fused_chunks)

    # If you haven't installed Ollama yet, uncomment next lines to verify retrieval first:
    # print("\n=== TOP CONTEXT CHUNKS ===\n")
    # for i, c in enumerate(fused_chunks, 1):
    #     print(f"[{i}] {c.source}" + (f" p.{c.page}" if c.page is not None else ""))
    #     print(c.text[:800], "\n")
    # return

    # 5) Citation-enforced response schema
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

    # 6) Local LLM via Ollama
    llm = Ollama(model="llama3", temperature=0.1, request_timeout=SETTINGS.llm_timeout)

    try:
        print("\n🤔 Generating response... (this may take a while)")
        resp = llm.chat(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
        )
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        print("💡 Try: 1) Check if Ollama is running, 2) Reduce query complexity, 3) Check network connection")
        return

    text = resp.message.content.strip()
    try:
        data = json.loads(text)
    except Exception:
        print("⚠️ Non-JSON output:\n", text)
        return

    print("\n=== ANSWER ===\n")
    print(data.get("answer", ""))

    print("\n=== CITATIONS ===\n")
    for c in data.get("citations", []):
        print(
            f"[{c['chunk']}] {c['source']}"
            + (f" p.{c['page']}" if c.get("page") is not None else "")
        )


if __name__ == "__main__":
    main()
