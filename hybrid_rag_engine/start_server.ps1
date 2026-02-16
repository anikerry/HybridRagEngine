# Advanced Hybrid RAG API Server Launcher (PowerShell)
Write-Host "🚀 Advanced Hybrid RAG API Server" -ForegroundColor Green
Write-Host "=" * 50

# Change to script directory
Set-Location -Path $PSScriptRoot

# Check Python
try {
    python --version | Out-Null
    Write-Host "✅ Python found" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python and add it to your PATH"
    Read-Host "Press Enter to exit"
    exit
}

# Start the server
Write-Host "Starting API server..." -ForegroundColor Yellow
python start_api_server.py