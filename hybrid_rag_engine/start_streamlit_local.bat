@echo off
echo 📱 Starting Streamlit Interface for Local RAG...
echo.
echo Make sure the Local RAG Server is running first!
echo Server should be at: http://localhost:8001
echo.
echo 🌐 Streamlit will be available at: http://localhost:8503
echo.

streamlit run src/streamlit_app.py --server.port 8503 -- --api-url http://localhost:8001

pause