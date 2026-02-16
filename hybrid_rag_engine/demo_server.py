#!/usr/bin/env python3
"""
Demo version of Advanced Hybrid RAG that can work without Qdrant
"""
import json
import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Demo responses for when services are unavailable
DEMO_RESPONSES = [
    {
        "answer": "This is a demo response from the Advanced Hybrid RAG Engine. In a real deployment, this would search through your document collection using hybrid retrieval (combining dense vector search and sparse BM25 ranking) to provide accurate answers based on your data.",
        "citations": [
            {"text": "Demo citation showing source attribution", "score": 0.95, "doc_id": "demo_doc_1"},
            {"text": "Another example citation for demonstration", "score": 0.87, "doc_id": "demo_doc_2"}
        ],
        "metadata": {
            "chunks_used": 5,
            "llm_provider": "demo",
            "processing_time": 1.2,
            "reranking_enabled": True,
            "query_expansion": False
        }
    },
    {
        "answer": "The Advanced RAG system combines multiple retrieval strategies: 1) Dense vector search using BGE embeddings for semantic similarity, 2) Sparse BM25 search for keyword matching, and 3) Reciprocal Rank Fusion (RRF) to merge results. This hybrid approach provides better accuracy than single-method systems.",
        "citations": [
            {"text": "Hybrid retrieval systems show 15-25% improvement in accuracy", "score": 0.92, "doc_id": "research_paper_1"},
            {"text": "RRF fusion technique balances multiple ranking signals", "score": 0.88, "doc_id": "technical_docs"}
        ],
        "metadata": {
            "chunks_used": 3,
            "llm_provider": "demo",
            "processing_time": 0.8,
            "reranking_enabled": False,
            "query_expansion": True
        }
    }
]

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    llm_provider: str = Field(default="ollama", description="LLM provider (ollama/openai)")
    model: str = Field(default="llama3.2", description="Model to use")
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    enable_query_expansion: bool = Field(default=False, description="Enable query expansion")

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    processing_time: float = 0.0
    query_id: str = ""

class HealthResponse(BaseModel):
    status: str
    message: str
    services: Dict[str, str] = {}
    demo_mode: bool = True

# Global state
startup_time = datetime.now()
query_count = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    print("🚀 Starting Demo Hybrid RAG API Server...")
    print("📍 Running in DEMO MODE - no external dependencies required")
    print("🌐 API available at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    
    yield
    
    print("👋 Demo server shutting down")

# Create FastAPI app
app = FastAPI(
    title="Advanced Hybrid RAG API (Demo Mode)",
    description="Production-ready Hybrid RAG system with vector search, BM25, and reranking (Demo Version)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global startup_time, query_count
    
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        message=f"Demo server running for {uptime:.1f}s, {query_count} queries processed",
        services={
            "vector_db": "demo_mode",
            "embedding_model": "demo_mode", 
            "bm25_index": "demo_mode",
            "llm_provider": "demo_mode"
        },
        demo_mode=True
    )

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system (Demo Mode)"""
    global query_count
    
    start_time = time.time()
    query_id = str(uuid.uuid4())
    query_count += 1
    
    # Simulate processing time
    processing_delay = min(2.0, 0.5 + len(request.question) * 0.01)
    await asyncio.sleep(processing_delay)
    
    # Select demo response based on question content
    response_idx = 0 if any(word in request.question.lower() for word in ['how', 'what', 'explain', 'describe']) else 1
    demo_response = DEMO_RESPONSES[response_idx].copy()
    
    # Update metadata with request parameters
    demo_response["metadata"].update({
        "llm_provider": request.llm_provider,
        "model": request.model,
        "reranking_enabled": request.enable_reranking,
        "query_expansion": request.enable_query_expansion,
        "processing_time": time.time() - start_time,
        "demo_mode": True
    })
    
    return QueryResponse(
        answer=demo_response["answer"],
        citations=demo_response["citations"],
        metadata=demo_response["metadata"],
        processing_time=time.time() - start_time,
        query_id=query_id
    )

@app.get("/stream-query")
async def stream_query(
    question: str,
    llm_provider: str = "ollama",
    model: str = "llama3.2",
    enable_reranking: bool = True,
    enable_query_expansion: bool = False
):
    """Streaming query endpoint (Demo Mode)"""
    
    async def generate_response():
        demo_text = "This is a streaming demo response from the Advanced Hybrid RAG Engine. "
        demo_text += "In production, this would stream real-time responses from your LLM while searching through "
        demo_text += "your document collection. The system uses hybrid retrieval combining vector search and BM25 "
        demo_text += "for optimal accuracy and relevance."
        
        # Stream word by word
        words = demo_text.split()
        for i, word in enumerate(words):
            chunk = {
                "type": "token",
                "content": word + " ",
                "metadata": {
                    "tokens_streamed": i + 1,
                    "total_tokens": len(words)
                }
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.1)  # Simulate natural streaming delay
        
        # Final chunk with complete metadata
        final_chunk = {
            "type": "complete",
            "content": "",
            "metadata": {
                "processing_time": len(words) * 0.1,
                "chunks_used": 3,
                "demo_mode": True
            }
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/models")
async def get_models():
    """Get available models (Demo Mode)"""
    return {
        "ollama_models": [
            {"name": "llama3.2", "size": "2.0GB", "status": "available"},
            {"name": "mistral", "size": "4.1GB", "status": "available"},
            {"name": "codellama", "size": "3.8GB", "status": "available"}
        ],
        "openai_models": [
            {"name": "gpt-4o", "context": "128k", "status": "available"},
            {"name": "gpt-4o-mini", "context": "128k", "status": "available"},
            {"name": "gpt-3.5-turbo", "context": "16k", "status": "available"}
        ],
        "demo_mode": True
    }

@app.get("/analytics")
async def get_analytics():
    """Get system analytics (Demo Mode)"""
    global startup_time, query_count
    
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "total_queries": query_count,
        "queries_per_minute": query_count / max(uptime / 60, 1),
        "system_status": "demo_mode",
        "demo_mode": True
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced Hybrid RAG API (Demo Mode)",
        "version": "1.0.0",
        "demo_mode": True,
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "stream": "/stream-query",
            "analytics": "/analytics",
            "docs": "/docs"
        }
    }

def main():
    """Start the demo server"""
    print("🤖 Starting Advanced Hybrid RAG Demo Server")
    print("=" * 50)
    print("📋 This is a demonstration version that works without external dependencies")
    print("🔄 For full functionality, start Qdrant and configure your data")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()