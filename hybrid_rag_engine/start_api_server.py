#!/usr/bin/env python3
"""
Simple script to start the Advanced Hybrid RAG API Server
"""
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ["fastapi", "uvicorn", "qdrant_client", "llama_index"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("\n🔧 Install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def start_server():
    """Start the FastAPI server"""
    # Change to the correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Add src to Python path
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("🚀 Starting Advanced Hybrid RAG API Server...")
    print("📍 Project directory:", project_root)
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print("🔄 Starting server (press Ctrl+C to stop)...\n")
    
    try:
        # Import and run the server
        from advanced_ask import app
        import uvicorn
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    print("🤖 Advanced Hybrid RAG API Server Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("✅ All dependencies found")
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()