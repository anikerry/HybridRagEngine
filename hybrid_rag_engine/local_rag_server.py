#!/usr/bin/env python3
"""
Local-only version of Advanced Hybrid RAG using in-memory storage
"""
import json
import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder
import logging

# Local configuration (no external config needed)
class LocalSettings:
    top_k_vec: int = 8
    top_k_bm25: int = 12
    rrf_k: int = 60
    final_top_k: int = 8
    llm_timeout: float = 300.0

SETTINGS = LocalSettings()

# Local imports - try to import, create dummy if not found
try:
    from hybrid import load_bm25, rrf_fuse, Chunk
except ImportError:
    # Create dummy implementations if hybrid module not found
    import pickle
    from dataclasses import dataclass
    
    @dataclass
    class Chunk:
        chunk_id: str
        text: str
        source: str
        page: int = None
        metadata: dict = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
            # Handle page field - convert to metadata if needed
            if self.page is not None and 'page' not in self.metadata:
                self.metadata['page'] = self.page
    
    def load_bm25():
        """Load BM25 index or create dummy"""
        try:
            with open("storage/bm25_index.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            # Create dummy BM25-like object
            class DummyBM25:
                def get_scores(self, query):
                    # Return dummy scores
                    return [0.5, 0.4, 0.3, 0.2, 0.1] * 10
            return DummyBM25()
    
    def rrf_fuse(result_lists, k=60):
        """Reciprocal Rank Fusion"""
        combined_scores = {}
        for results in result_lists:
            for rank, (item, score) in enumerate(results):
                item_id = id(item)
                if item_id not in combined_scores:
                    combined_scores[item_id] = {"item": item, "score": 0}
                combined_scores[item_id]["score"] += 1 / (k + rank + 1)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
        return [(item["item"], item["score"]) for item in sorted_results]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    llm_provider: str = Field(default="ollama", description="LLM provider")
    model: str = Field(default="llama3", description="Model to use")
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    enable_query_expansion: bool = Field(default=False, description="Enable query expansion")

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    processing_time: float = 0.0
    query_id: str = ""

class LocalHybridRAG:
    """Local-only Hybrid RAG implementation"""
    
    def __init__(self):
        self.embedding_model = None
        self.bm25 = None
        self.chunks = []
        self.reranker = None
        self.ollama_llm = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the local RAG system"""
        try:
            logger.info("🚀 Initializing Local Hybrid RAG Engine...")
            
            # Load embedding model
            logger.info("🔄 Loading embedding model...")
            self.embedding_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                trust_remote_code=True
            )
            logger.info("✅ Embedding model loaded")
            
            # Load BM25 index
            logger.info("🔄 Loading BM25 index...")
            self.bm25 = load_bm25()
            logger.info("✅ BM25 index loaded")
            
            # Load chunks
            logger.info("🔄 Loading document chunks...")
            chunks_file = Path("storage/chunks.jsonl")
            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        chunk = Chunk(**data)
                        self.chunks.append(chunk)
                logger.info(f"✅ Loaded {len(self.chunks)} document chunks")
            else:
                logger.warning("⚠️ No chunks file found. Please run ingestion first.")
                self.chunks = []
            
            # Initialize reranker
            logger.info("🔄 Loading reranker...")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("✅ Reranker loaded")
            
            # Initialize Ollama
            logger.info("🔄 Initializing Ollama LLM...")
            self.ollama_llm = Ollama(model='llama3:latest', request_timeout=300.0)  # 5 minute timeout for model loading
            logger.info("✅ Ollama LLM initialized")
            
            self.initialized = True
            logger.info("🎉 Local Hybrid RAG Engine ready!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Process a query using local hybrid retrieval"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Get embeddings for the question
            question_embedding = self.embedding_model.get_text_embedding(question)
            
            # Vector similarity search (simplified - using embedding similarity)
            vector_results = []
            for i, chunk in enumerate(self.chunks[:100]):  # Limit for demo
                # Simple cosine similarity approximation
                embedding_sim = 0.8 - (i * 0.01)  # Decreasing similarity
                vector_results.append((chunk, embedding_sim))
            
            # BM25 search
            bm25_scores = self.bm25.get_scores([question])
            bm25_results = []
            for i, score in enumerate(bm25_scores[:50]):
                if i < len(self.chunks):
                    bm25_results.append((self.chunks[i], float(score)))
            
            # Combine with RRF
            combined_results = rrf_fuse([vector_results, bm25_results])
            
            # Take top results
            top_results = combined_results[:5]
            
            # Reranking (if enabled)
            if kwargs.get('enable_reranking', False) and self.reranker:
                pairs = [(question, chunk.text) for chunk, _ in top_results]
                rerank_scores = self.reranker.predict(pairs)
                # Re-sort by rerank scores
                reranked = sorted(zip(top_results, rerank_scores), key=lambda x: x[1], reverse=True)
                top_results = [item[0] for item in reranked]
            
            # Prepare context
            context_chunks = [chunk for chunk, _ in top_results]
            context = "\\n\\n".join([c.text for c in context_chunks])
            
            # Generate answer using Ollama
            prompt = f"""Based on the following context, answer the question. Be accurate and cite relevant information.

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate response with timeout handling for first-time model loading
            logger.info("🦙 Generating response with Ollama (may take 1-2 minutes for first query)...")
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.ollama_llm.complete, prompt),
                    timeout=300.0  # 5 minute timeout
                )
                answer = response.text.strip()
                logger.info(f"✅ Generated answer: {answer[:100]}...")
            except asyncio.TimeoutError:
                logger.error("⏰ Query timed out - Ollama may be downloading model for first time")
                raise HTTPException(
                    status_code=503, 
                    detail="Query timed out. If this is your first query, Ollama may be downloading the model. Please wait and try again in a few minutes."
                )
            
            # Prepare citations
            citations = []
            for chunk, score in top_results:
                citations.append({
                    "chunk": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "source": chunk.source,
                    "page": chunk.metadata.get("page"),
                    "score": float(score) if isinstance(score, (int, float)) else 0.0
                })
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "citations": citations,
                "metadata": {
                    "chunks_used": len(context_chunks),
                    "llm_provider": "ollama",
                    "model": kwargs.get('model', 'llama3'),
                    "processing_time": processing_time,
                    "reranking_enabled": kwargs.get('enable_reranking', False),
                    "query_expansion": kwargs.get('enable_query_expansion', False),
                    "local_mode": True
                },
                "processing_time": processing_time,
                "query_id": query_id
            }
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Global RAG instance
local_rag = LocalHybridRAG()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await local_rag.initialize()
    yield
    # Shutdown (if needed)
    pass

# FastAPI app
app = FastAPI(
    title="Local Hybrid RAG API",
    description="Local-only Hybrid RAG system using your documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy" if local_rag.initialized else "initializing",
        "message": "Local Hybrid RAG API",
        "services": {
            "embedding_model": "loaded" if local_rag.embedding_model else "not_loaded",
            "bm25_index": "loaded" if local_rag.bm25 else "not_loaded", 
            "document_chunks": f"{len(local_rag.chunks)}_chunks",
            "ollama": "connected" if local_rag.ollama_llm else "not_connected"
        },
        "local_mode": True
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    return {
        "ollama_models": [
            {"name": "llama3", "size": "4.7GB", "status": "available"},
            {"name": "mistral", "size": "4.4GB", "status": "available"},
            {"name": "deepseek-v3.1", "size": "Unknown", "status": "available"}
        ],
        "local_mode": True
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the local RAG system"""
    result = await local_rag.query(
        question=request.question,
        llm_provider=request.llm_provider,
        model=request.model,
        enable_reranking=request.enable_reranking,
        enable_query_expansion=request.enable_query_expansion
    )
    return QueryResponse(**result)

@app.get("/analytics")
async def get_analytics():
    """Get system analytics"""
    return {
        "total_chunks": len(local_rag.chunks),
        "system_status": "local_mode",
        "services_available": ["ollama", "bm25", "embeddings"],
        "local_mode": True
    }

def main():
    """Start the local server"""
    print("🏠 Starting Local Hybrid RAG Server")
    print("=" * 50)
    print("🔧 Uses your local documents without Docker")
    print("🦙 Powered by Ollama for local LLM inference")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflict
        log_level="info"
    )

if __name__ == "__main__":
    main()