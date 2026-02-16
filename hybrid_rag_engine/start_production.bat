@echo off
echo 🐳 Starting Production Hybrid RAG System...
echo.

echo Step 1: Starting Qdrant Database...
docker compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start Qdrant. Make sure Docker Desktop is running.
    pause
    exit /b 1
)

echo.
echo Step 2: Starting Production FastAPI Server...
echo 🌐 API will be available at: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo.
python src/advanced_ask.py --server

pause