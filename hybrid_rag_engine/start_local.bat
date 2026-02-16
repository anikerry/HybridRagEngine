@echo off
echo 🏠 Starting Local Hybrid RAG System (No Docker Required)...
echo.
echo This version uses:
echo - Your local documents (Bosch thesis, Thor PDF)
echo - Ollama for LLM (llama3, mistral)
echo - In-memory vector storage
echo.
echo 🌐 API will be available at: http://localhost:8001
echo 📚 API Documentation: http://localhost:8001/docs
echo.

python local_rag_server.py

pause