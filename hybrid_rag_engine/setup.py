#!/usr/bin/env python3
"""
Advanced RAG Engine Setup & Deployment Script

Production utility script for setting up, testing, and deploying the Advanced Hybrid RAG Engine.
Demonstrates DevOps practices and production deployment workflows.
"""

import subprocess
import sys
import os
import json
import time
import requests
from pathlib import Path
import argparse

class RAGSetup:
    """Setup and deployment manager for Advanced RAG Engine"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        
    def check_python_version(self):
        """Check Python version compatibility"""
        print("🐍 Checking Python version...")
        if sys.version_info < (3, 10):
            print("❌ Python 3.10+ is required")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("📦 Checking dependencies...")
        
        required_packages = [
            "fastapi", "uvicorn", "streamlit", "qdrant-client",
            "llama-index", "sentence-transformers", "plotly"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print("💡 Run: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies installed")
        return True
    
    def setup_environment(self):
        """Set up the environment configuration"""
        print("⚙️ Setting up environment...")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if not env_file.exists() and env_example.exists():
            print("📝 Creating .env file from template...")
            content = env_example.read_text()
            env_file.write_text(content)
            print("✅ .env file created")
            print("💡 Please edit .env with your API keys and settings")
        elif env_file.exists():
            print("✅ .env file already exists")
        else:
            print("⚠️ No .env.example found, creating basic .env...")
            basic_env = """# Basic configuration
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=bosch_docs
LOG_LEVEL=INFO
ENABLE_CACHING=true
"""
            env_file.write_text(basic_env)
    
    def check_qdrant(self):
        """Check if Qdrant is running"""
        print("🔍 Checking Qdrant status...")
        try:
            response = requests.get("http://localhost:6333", timeout=3)
            if response.status_code == 200:
                print("✅ Qdrant is running")
                return True
        except:
            pass
        
        print("❌ Qdrant not running")
        return False
    
    def start_qdrant(self):
        """Start Qdrant using Docker Compose"""
        print("🚀 Starting Qdrant...")
        try:
            subprocess.run(["docker", "compose", "up", "-d"], 
                         cwd=self.project_root, check=True)
            
            # Wait for Qdrant to be ready
            print("⏳ Waiting for Qdrant to be ready...")
            for i in range(30):
                if self.check_qdrant():
                    break
                time.sleep(1)
            else:
                print("❌ Qdrant failed to start within 30 seconds")
                return False
            
            print("✅ Qdrant started successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to start Qdrant")
            return False
    
    def check_ollama(self):
        """Check if Ollama is installed and models are available"""
        print("🦙 Checking Ollama...")
        
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = result.stdout.strip()
                if "llama3" in models or "mistral" in models:
                    print("✅ Ollama with models detected")
                    return True
                else:
                    print("⚠️ Ollama installed but no models found")
                    print("💡 Run: ollama pull llama3")
                    return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print("❌ Ollama not found")
        print("💡 Install from: https://ollama.com")
        return False
    
    def run_ingestion(self):
        """Run document ingestion"""
        print("📚 Running document ingestion...")
        
        docs_dir = self.project_root / "data" / "docs"
        if not docs_dir.exists() or not list(docs_dir.glob("*.pdf")):
            print("⚠️ No documents found in data/docs/")
            print("💡 Add PDF files to data/docs/ directory")
            return False
        
        try:
            subprocess.run([sys.executable, "src/ingest.py"], 
                         cwd=self.project_root, check=True)
            print("✅ Document ingestion completed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Document ingestion failed")
            return False
    
    def test_system(self):
        """Test the system end-to-end"""
        print("🧪 Running system tests...")
        
        try:
            # Test CLI interface
            print("  Testing CLI interface...")
            result = subprocess.run([
                sys.executable, "-c",
                "import sys; sys.path.append('src'); "
                "from advanced_ask import cli_main; "
                "print('CLI test passed')"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ CLI interface OK")
            else:
                print("  ❌ CLI interface failed")
                return False
            
            # Test evaluation framework
            print("  Testing evaluation framework...")
            result = subprocess.run([
                sys.executable, "-c",
                "import sys; sys.path.append('src'); "
                "from evaluation import RAGEvaluator; "
                "print('Evaluation test passed')"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ Evaluation framework OK")
            else:
                print("  ❌ Evaluation framework failed")
                return False
            
            print("✅ All system tests passed")
            return True
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
            return False
    
    def start_api_server(self, port=8000):
        """Start the FastAPI server"""
        print(f"🚀 Starting API server on port {port}...")
        
        try:
            subprocess.run([
                sys.executable, "src/advanced_ask.py", "--server"
            ], cwd=self.project_root)
        except KeyboardInterrupt:
            print("\n🛑 API server stopped")
    
    def start_streamlit(self, port=8501):
        """Start the Streamlit interface"""
        print(f"🎨 Starting Streamlit interface on port {port}...")
        
        try:
            subprocess.run([
                "streamlit", "run", "src/streamlit_app.py", 
                "--server.port", str(port)
            ], cwd=self.project_root)
        except KeyboardInterrupt:
            print("\n🛑 Streamlit interface stopped")
    
    def run_demo(self):
        """Run the comprehensive demo"""
        print("🎯 Running comprehensive demo...")
        
        try:
            subprocess.run([sys.executable, "run_demo.py"], 
                         cwd=self.project_root)
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped")
    
    def full_setup(self):
        """Run complete setup process"""
        print("🎯 ADVANCED RAG ENGINE - FULL SETUP")
        print("=" * 50)
        
        # Pre-flight checks
        if not self.check_python_version():
            return False
        
        if not self.check_dependencies():
            print("💡 Install dependencies first: pip install -r requirements.txt")
            return False
        
        # Environment setup
        self.setup_environment()
        
        # Start services
        if not self.check_qdrant():
            if not self.start_qdrant():
                return False
        
        # Check Ollama
        ollama_ok = self.check_ollama()
        if not ollama_ok:
            print("⚠️ Ollama not available - some features will be limited")
        
        # Run ingestion
        if not self.run_ingestion():
            print("⚠️ Ingestion failed - using existing data if available")
        
        # Test system
        if not self.test_system():
            print("⚠️ Some tests failed - system may have issues")
        
        print("\n🎉 Setup completed!")
        print("\n🚀 Next steps:")
        print("1. Start API server: python setup.py --api")
        print("2. Start Streamlit UI: python setup.py --streamlit") 
        print("3. Run demo: python setup.py --demo")
        
        return True
    
    def status_check(self):
        """Check overall system status"""
        print("📊 SYSTEM STATUS CHECK")
        print("=" * 30)
        
        checks = [
            ("Python Version", self.check_python_version()),
            ("Dependencies", self.check_dependencies()),
            ("Qdrant", self.check_qdrant()),
            ("Ollama", self.check_ollama()),
        ]
        
        print("\n📋 Status Summary:")
        for name, status in checks:
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {name}")
        
        # Overall health
        all_ok = all(status for _, status in checks)
        print(f"\n🎯 Overall Status: {'✅ HEALTHY' if all_ok else '⚠️ NEEDS ATTENTION'}")
        
        return all_ok

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Advanced RAG Engine Setup & Deployment")
    parser.add_argument("--setup", action="store_true", help="Run full setup")
    parser.add_argument("--status", action="store_true", help="Check system status")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--streamlit", action="store_true", help="Start Streamlit interface")
    parser.add_argument("--demo", action="store_true", help="Run comprehensive demo")
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion")
    parser.add_argument("--test", action="store_true", help="Run system tests")
    
    args = parser.parse_args()
    
    setup_manager = RAGSetup()
    
    if args.setup:
        setup_manager.full_setup()
    elif args.status:
        setup_manager.status_check()
    elif args.api:
        setup_manager.start_api_server()
    elif args.streamlit:
        setup_manager.start_streamlit()
    elif args.demo:
        setup_manager.run_demo()
    elif args.ingest:
        setup_manager.run_ingestion()
    elif args.test:
        setup_manager.test_system()
    else:
        # Default: show status and available commands
        print("🧠 Advanced Hybrid RAG Engine - Setup Manager")
        print("=" * 50)
        
        status = setup_manager.status_check()
        
        print("\n🛠️ Available Commands:")
        print("  --setup      Full system setup")
        print("  --status     Check system status")
        print("  --api        Start API server")
        print("  --streamlit  Start web interface")
        print("  --demo       Run comprehensive demo")
        print("  --ingest     Run document ingestion")
        print("  --test       Run system tests")
        
        if not status:
            print("\n💡 Recommendation: Run --setup to configure the system")

if __name__ == "__main__":
    main()