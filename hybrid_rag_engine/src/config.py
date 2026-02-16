from pydantic import BaseModel

class Settings(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    collection: str = "bosch_docs"
    top_k_vec: int = 8
    top_k_bm25: int = 12
    rrf_k: int = 60          # typical 60
    final_top_k: int = 8     # context chunks sent to LLM

SETTINGS = Settings()
