import json
import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from qdrant_client import QdrantClient
try:
    # Try the new import structure first
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.core.vector_stores import VectorStoreQuery
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.ollama import Ollama
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    # Fallback to older structure if needed
    from llama_index_vector_stores_qdrant import QdrantVectorStore
    from llama_index_core.vector_stores import VectorStoreQuery
    from llama_index_core.llms import ChatMessage
    from llama_index_llms_ollama import Ollama
    from llama_index_llms_openai import OpenAI
    from llama_index_embeddings_huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder
import logging
from functools import lru_cache
import hashlib

from config import SETTINGS
from hybrid import load_bm25, rrf_fuse, Chunk

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for responses
response_cache: Dict[str, Dict] = {}
analytics_data: List[Dict] = []

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    use_streaming: bool = Field(default=True, description="Whether to stream the response")
    llm_provider: str = Field(default="ollama", description="LLM provider: ollama or openai")
    model: str = Field(default="llama3", description="Model name")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    enable_query_expansion: bool = Field(default=False, description="Expand query with synonyms")

class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    cached: bool = False

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    version: str = "2.0.0"

class AnalyticsResponse(BaseModel):
    total_queries: int
    avg_response_time: float
    popular_topics: List[str]
    cache_hit_rate: float

# Enhanced RAG Engine Class
class AdvancedHybridRAG:
    def __init__(self):
        self.embed_model = None
        self.bm25 = None
        self.vector_store = None
        self.client = None
        self.cross_encoder = None
        self.ollama_llm = None
        self.openai_llm = None
        
    async def initialize(self):
        """Initialize all components asynchronously"""
        logger.info("Initializing Advanced Hybrid RAG Engine...")
        
        # Load embedding model
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("✅ Embedding model loaded")
        
        # Load BM25
        self.bm25 = load_bm25("storage")
        logger.info("✅ BM25 index loaded")
        
        # Initialize Qdrant
        self.client = QdrantClient(url=SETTINGS.qdrant_url, timeout=SETTINGS.qdrant_timeout)
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=SETTINGS.collection)
        logger.info("✅ Qdrant connection established")
        
        # Initialize cross-encoder for reranking
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✅ Cross-encoder reranker loaded")
        except Exception as e:
            logger.warning(f"Cross-encoder not available: {e}")
            
        # Initialize LLMs
        self.ollama_llm = Ollama(model="llama3", temperature=0.1, request_timeout=SETTINGS.llm_timeout)
        
        if os.getenv("OPENAI_API_KEY"):
            self.openai_llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
            logger.info("✅ OpenAI LLM available")
        else:
            logger.info("⚠️ OpenAI API key not found, using Ollama only")
            
        logger.info("🚀 Advanced Hybrid RAG Engine ready!")
    
    def expand_query(self, query: str) -> str:
        """Simple query expansion with synonyms"""
        expansions = {
            "requirements": "requirements specifications needs criteria",
            "compliance": "compliance regulatory rules regulations standards", 
            "process": "process procedure workflow methodology",
            "documentation": "documentation docs manual guide reference"
        }
        
        expanded = query
        for key, synonyms in expansions.items():
            if key.lower() in query.lower():
                expanded += f" {synonyms}"
        return expanded
    
    def rerank_chunks(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """Rerank chunks using cross-encoder"""
        if not self.cross_encoder or len(chunks) <= 3:
            return chunks
            
        try:
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = self.cross_encoder.predict(pairs)
            
            # Sort chunks by cross-encoder scores
            ranked_pairs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in ranked_pairs]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return chunks
    
    def cache_key(self, query: str, params: Dict) -> str:
        """Generate cache key for query"""
        key_data = f"{query}_{params.get('llm_provider')}_{params.get('model')}_{params.get('enable_reranking')}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """Enhanced query processing with caching and analytics"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        # Check cache
        cache_key = self.cache_key(request.question, request.dict())
        if cache_key in response_cache:
            cached_response = response_cache[cache_key]
            cached_response["cached"] = True
            cached_response["query_id"] = query_id
            return QueryResponse(**cached_response)
        
        try:
            # Query expansion
            query_text = request.question
            if request.enable_query_expansion:
                query_text = self.expand_query(request.question)
                logger.info(f"Expanded query: {query_text}")
            
            # Vector search
            q_emb = self.embed_model.get_query_embedding(request.question)
            if q_emb is None:
                raise ValueError("Failed to generate query embedding")
            
            vq = VectorStoreQuery(query_embedding=q_emb, similarity_top_k=SETTINGS.top_k_vec)
            vec_res = self.vector_store.query(vq)
            
            vec_ranked = []
            for node, score in zip(vec_res.nodes, vec_res.similarities):
                ch = Chunk(
                    chunk_id=node.node_id,
                    text=node.get_content(),
                    source=str((node.metadata or {}).get("source", "unknown")),
                    page=(node.metadata or {}).get("page", None),
                )
                vec_ranked.append((ch, float(score)))
            
            # BM25 search
            bm25_ranked = self.bm25.query(query_text, top_k=SETTINGS.top_k_bm25)
            
            # RRF fusion
            fused = rrf_fuse(vec_ranked, bm25_ranked, k=SETTINGS.rrf_k)[:SETTINGS.final_top_k]
            fused_chunks = [c for c, _ in fused]
            
            # Reranking
            if request.enable_reranking:
                fused_chunks = self.rerank_chunks(request.question, fused_chunks)
            
            # Format context
            context = format_context(fused_chunks)
            
            # Generate response
            llm = self.ollama_llm if request.llm_provider == "ollama" else self.openai_llm
            if not llm:
                raise ValueError(f"LLM provider '{request.llm_provider}' not available")
            
            system_prompt = (
                "You are a technical assistant. Use ONLY the provided context.\n"
                "Cite sources using bracket numbers like [1], [2] matching the context chunks.\n"
                "If the answer is not contained in the context, say you don't know.\n"
                "Return JSON with keys: answer, citations.\n"
                "citations must be a list of objects: {chunk: int, source: str, page: int|null}.\n"
            )
            
            user_prompt = (
                f"QUESTION:\n{request.question}\n\n"
                f"CONTEXT CHUNKS:\n{context}\n\n"
                "Return strictly valid JSON."
            )
            
            resp = llm.chat([
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ])
            
            # Parse response
            try:
                data = json.loads(resp.message.content.strip())
            except json.JSONDecodeError:
                # Fallback for non-JSON responses
                data = {
                    "answer": resp.message.content.strip(),
                    "citations": [{"chunk": i+1, "source": c.source, "page": c.page} for i, c in enumerate(fused_chunks[:3])]
                }
            
            processing_time = time.time() - start_time
            
            # Build response
            response = QueryResponse(
                query_id=query_id,
                question=request.question,
                answer=data.get("answer", ""),
                citations=data.get("citations", []),
                metadata={
                    "llm_provider": request.llm_provider,
                    "model": request.model,
                    "reranking_enabled": request.enable_reranking,
                    "query_expansion_enabled": request.enable_query_expansion,
                    "chunks_used": len(fused_chunks),
                    "vector_results": len(vec_ranked),
                    "bm25_results": len(bm25_ranked)
                },
                processing_time=processing_time
            )
            
            # Cache response
            response_cache[cache_key] = response.dict()
            
            # Analytics
            analytics_data.append({
                "timestamp": datetime.now().isoformat(),
                "query_id": query_id,
                "question": request.question,
                "question_length": len(request.question),
                "processing_time": processing_time,
                "llm_provider": request.llm_provider,
                "cached": False
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def stream_query(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Stream query response for real-time interaction"""
        try:
            # For demo purposes, we'll simulate streaming by yielding the regular response in chunks
            response = await self.query(request)
            
            # Simulate streaming by breaking answer into sentences
            answer = response.answer
            sentences = answer.split('. ')
            
            yield f"data: {{\"type\": \"start\", \"query_id\": \"{response.query_id}\", \"question\": \"{response.question}\"}}\n\n"
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    yield f"data: {{\"type\": \"content\", \"content\": \"{sentence.strip()}\", \"index\": {i}}}\n\n"
                    await asyncio.sleep(0.1)  # Simulate processing delay
            
            yield f"data: {{\"type\": \"citations\", \"citations\": {json.dumps(response.citations)}}}\n\n"
            yield f"data: {{\"type\": \"metadata\", \"metadata\": {json.dumps(response.metadata)}}}\n\n"
            yield f"data: {{\"type\": \"end\", \"processing_time\": {response.processing_time}}}\n\n"
            
        except Exception as e:
            yield f"data: {{\"type\": \"error\", \"error\": \"{str(e)}\"}}\n\n"

def format_context(chunks: List[Chunk]) -> str:
    """Format chunks for LLM context"""
    lines = []
    for i, c in enumerate(chunks, start=1):
        cite = f"[{i}] {c.source}" + (f" p.{c.page}" if c.page is not None else "")
        lines.append(f"{cite}\n{c.text}")
    return "\n\n---\n\n".join(lines)

# Global RAG engine instance
rag_engine = AdvancedHybridRAG()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await rag_engine.initialize()
    yield
    # Cleanup (if needed)
    pass

app = FastAPI(
    title="Advanced Hybrid RAG Engine",
    description="Production-ready Hybrid Retrieval-Augmented Generation system with streaming, analytics, and multi-LLM support",
    version="2.0.0",
    lifespan=lifespan
)

# FastAPI Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {
        "qdrant": "healthy" if rag_engine.client else "unavailable",
        "bm25": "healthy" if rag_engine.bm25 else "unavailable",
        "embeddings": "healthy" if rag_engine.embed_model else "unavailable",
        "ollama": "healthy" if rag_engine.ollama_llm else "unavailable",
        "openai": "healthy" if rag_engine.openai_llm else "unavailable"
    }
    
    return HealthResponse(
        status="healthy",
        services=services
    )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query and return the response"""
    return await rag_engine.query(request)

@app.post("/query/stream")
async def stream_query_endpoint(request: QueryRequest):
    """Stream query response for real-time interaction"""
    return StreamingResponse(
        rag_engine.stream_query(request),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get system analytics and metrics"""
    if not analytics_data:
        return AnalyticsResponse(
            total_queries=0,
            avg_response_time=0.0,
            popular_topics=[],
            cache_hit_rate=0.0
        )
    
    total_queries = len(analytics_data)
    avg_response_time = sum(q["processing_time"] for q in analytics_data) / total_queries
    cached_queries = sum(1 for q in analytics_data if q.get("cached", False))
    cache_hit_rate = cached_queries / total_queries if total_queries > 0 else 0.0
    
    # Simple topic analysis based on question words
    topics = {}
    for q in analytics_data:
        # This is a simplified version - in production you'd use NLP for topic modeling
        question = q.get("question", "")
        words = question.lower().split()
        for word in words:
            if len(word) > 4:  # Filter short words
                topics[word] = topics.get(word, 0) + 1
    
    popular_topics = [topic for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    return AnalyticsResponse(
        total_queries=total_queries,
        avg_response_time=avg_response_time,
        popular_topics=popular_topics,
        cache_hit_rate=cache_hit_rate
    )

@app.delete("/cache")
async def clear_cache():
    """Clear response cache"""
    global response_cache
    response_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/models")
async def list_models():
    """List available models and providers"""
    return {
        "providers": {
            "ollama": {
                "available": rag_engine.ollama_llm is not None,
                "default_model": "llama3",
                "supported_models": ["llama3", "mistral", "phi3", "codellama"]
            },
            "openai": {
                "available": rag_engine.openai_llm is not None,
                "default_model": "gpt-3.5-turbo",
                "supported_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            }
        }
    }

# CLI Interface (backward compatibility)
def cli_main():
    """CLI interface for backward compatibility"""
    question = input("Ask: ").strip()
    if not question:
        return
    
    # Simple CLI query
    import asyncio
    
    async def run_query():
        await rag_engine.initialize()
        
        request = QueryRequest(
            question=question,
            use_streaming=False,
            llm_provider="ollama",
            model="llama3"
        )
        
        response = await rag_engine.query(request)
        
        print("\n=== ANSWER ===\n")
        print(response.answer)
        
        print("\n=== CITATIONS ===\n")
        for c in response.citations:
            print(f"[{c['chunk']}] {c['source']}" + (f" p.{c['page']}" if c.get('page') else ""))
        
        print(f"\n=== METADATA ===\n")
        print(f"Processing time: {response.processing_time:.2f}s")
        print(f"Chunks used: {response.metadata['chunks_used']}")
        print(f"LLM Provider: {response.metadata['llm_provider']}")
    
    asyncio.run(run_query())

def start_server():
    """Start the FastAPI server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        print("🚀 Starting Advanced Hybrid RAG API Server on http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        start_server()
    else:
        cli_main()