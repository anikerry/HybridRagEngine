import os
import uuid
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from config import SETTINGS
from hybrid import Chunk, save_bm25

load_dotenv()

def main():
    docs_path = os.path.join("data", "docs")
    if not os.path.isdir(docs_path):
        raise RuntimeError(f"Missing folder: {docs_path}")

    # 1) Read PDFs/MD
    reader = SimpleDirectoryReader(docs_path, recursive=True)
    docs: list[Document] = reader.load_data()

    # 2) Chunk
    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)
    nodes = splitter.get_nodes_from_documents(docs)

    # 3) Build chunk metadata for BM25 + citations
    chunks: list[Chunk] = []
    for n in nodes:
        meta = n.metadata or {}
        source = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
        page = meta.get("page_label")
        try:
            page_int = int(page) if page is not None else None
        except:
            page_int = None

        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=n.get_content(),
                source=str(source),
                page=page_int
            )
        )

    # 4) Save BM25 storage
    save_bm25("storage", chunks)

    # 5) Vector index to Qdrant
    client = QdrantClient(url=SETTINGS.qdrant_url)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=SETTINGS.collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # We attach the same metadata so citations work
    # Recreate nodes with metadata aligned to chunks
    # (simple: store source/page/text in Qdrant payload)
    from llama_index.core.schema import TextNode
    vnodes = []
    for c in chunks:
        vnodes.append(
            TextNode(
                text=c.text,
                id_=c.chunk_id,
                metadata={"source": c.source, "page": c.page}
            )
        )

    embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5")

    VectorStoreIndex(
    vnodes,
    storage_context=storage_context,
    embed_model=embed_model)

    print(f"✅ Ingested {len(chunks)} chunks into BM25 + Qdrant collection '{SETTINGS.collection}'")

if __name__ == "__main__":
    main()
