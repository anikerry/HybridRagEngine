from pydantic import BaseModel
from typing import Optional
import os

class Settings(BaseModel):
    # Database settings
    qdrant_url: str = "http://localhost:6333"
    collection: str = "bosch_docs"
    
    # Retrieval settings
    top_k_vec: int = 8
    top_k_bm25: int = 12
    rrf_k: int = 60          # typical 60
    final_top_k: int = 8     # context chunks sent to LLM
    
    # Timeout settings
    qdrant_timeout: float = 30.0      # seconds
    llm_timeout: float = 300.0        # seconds (5 minutes)
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600            # 1 hour
    max_cache_size: int = 1000
    
    # Analytics settings
    enable_analytics: bool = True
    analytics_retention_days: int = 30
    
    # Cross-encoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = True
    
    # Embedding settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"  # or "cuda" for GPU
    
    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    default_llm_provider: str = "ollama"
    default_ollama_model: str = "llama3"
    default_openai_model: str = "gpt-3.5-turbo"
    
    # Security
    api_key: Optional[str] = None
    cors_origins: list = ["http://localhost:3000", "http://localhost:8501"]
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/rag_engine.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Load settings from environment
SETTINGS = Settings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    api_key=os.getenv("API_KEY")
)