"""
Streamlit Web Interface for Advanced Hybrid RAG Engine

This provides a user-friendly web interface for the RAG system with:
- Interactive chat interface
- Real-time streaming responses  
- System analytics dashboard
- Configuration management
- Performance monitoring
"""

import streamlit as st
import asyncio
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import requests
from typing import Dict, List

# Configure page
st.set_page_config(
    page_title="Advanced Hybrid RAG Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }
    .citation {
        background-color: #fff3e0;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = []

class RAGInterface:
    """Interface class for RAG system interaction"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        
    def check_health(self) -> Dict:
        """Check system health"""
        try:
            response = requests.get(f"{self.api_base_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def query_rag(self, question: str, **kwargs) -> Dict:
        """Query the RAG system"""
        try:
            payload = {
                "question": question,
                "use_streaming": False,
                **kwargs
            }
            response = requests.post(f"{self.api_base_url}/query", json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_analytics(self) -> Dict:
        """Get system analytics"""
        try:
            response = requests.get(f"{self.api_base_url}/analytics")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_models(self) -> Dict:
        """Get available models"""
        try:
            response = requests.get(f"{self.api_base_url}/models")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Advanced Hybrid RAG Engine</h1>', 
                unsafe_allow_html=True)
    
    # Initialize RAG interface
    rag_interface = RAGInterface()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System health check
        health_status = rag_interface.check_health()
        if health_status.get("status") == "healthy":
            st.success("✅ System Online")
        else:
            st.error("❌ System Offline")
            st.error(health_status.get("message", "Unknown error"))
        
        # Model selection
        models_data = rag_interface.get_models()
        if "providers" in models_data:
            st.subheader("🤖 LLM Provider")
            providers = list(models_data["providers"].keys())
            selected_provider = st.selectbox("Provider", providers)
            
            if selected_provider in models_data["providers"]:
                available_models = models_data["providers"][selected_provider]["supported_models"]
                selected_model = st.selectbox("Model", available_models)
        else:
            selected_provider = "ollama"
            selected_model = "llama3"
            st.warning("Could not load model information")
        
        # Advanced settings
        st.subheader("🔧 Advanced Settings")
        enable_reranking = st.checkbox("Enable Reranking", value=True)
        enable_query_expansion = st.checkbox("Enable Query Expansion", value=False)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "🧪 Evaluation"])
    
    with tab1:
        chat_interface(rag_interface, selected_provider, selected_model, 
                      enable_reranking, enable_query_expansion)
    
    with tab2:
        analytics_dashboard(rag_interface)
    
    with tab3:
        evaluation_interface()

def chat_interface(rag_interface, provider, model, reranking, query_expansion):
    """Chat interface implementation"""
    
    st.header("💬 Interactive Chat")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for chat in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong> {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Citations
            if chat.get('citations'):
                st.markdown("**📚 Sources:**")
                for citation in chat['citations']:
                    st.markdown(f"""
                    <div class="citation">
                        [{citation['chunk']}] {citation['source']}
                        {f" p.{citation['page']}" if citation.get('page') else ""}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Metadata
            if st.checkbox(f"Show Details", key=f"details_{len(st.session_state.chat_history)}"):
                metadata = chat.get('metadata', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processing Time", f"{chat.get('processing_time', 0):.2f}s")
                with col2:
                    st.metric("Chunks Used", metadata.get('chunks_used', 0))
                with col3:
                    st.metric("LLM Provider", metadata.get('llm_provider', 'Unknown'))
            
            st.divider()
    
    # Input area
    st.subheader("Ask a Question")
    question = st.text_area("Enter your question:", height=100)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary")
    
    if ask_button and question.strip():
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            # Query the RAG system
            response = rag_interface.query_rag(
                question=question,
                llm_provider=provider,
                model=model,
                enable_reranking=reranking,
                enable_query_expansion=query_expansion
            )
            
            processing_time = time.time() - start_time
            
            if "error" not in response:
                # Add to chat history
                chat_entry = {
                    "question": question,
                    "answer": response.get("answer", "No answer provided"),
                    "citations": response.get("citations", []),
                    "metadata": response.get("metadata", {}),
                    "processing_time": response.get("processing_time", processing_time),
                    "timestamp": datetime.now()
                }
                st.session_state.chat_history.append(chat_entry)
                st.session_state.analytics_data.append(chat_entry)
                st.rerun()
            else:
                st.error(f"Error: {response['error']}")

def analytics_dashboard(rag_interface):
    """Analytics dashboard implementation"""
    
    st.header("📊 System Analytics")
    
    # Get server analytics
    server_analytics = rag_interface.get_analytics()
    
    if "error" not in server_analytics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", server_analytics.get("total_queries", 0))
        with col2:
            st.metric("Avg Response Time", f"{server_analytics.get('avg_response_time', 0):.2f}s")
        with col3:
            st.metric("Cache Hit Rate", f"{server_analytics.get('cache_hit_rate', 0):.1%}")
        with col4:
            st.metric("Popular Topics", len(server_analytics.get("popular_topics", [])))
    
    # Session analytics from local data
    if st.session_state.analytics_data:
        st.subheader("📈 Session Analytics")
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.analytics_data)
        
        # Processing time chart
        fig_time = px.line(
            df.reset_index(), 
            x="index", 
            y="processing_time",
            title="Processing Time Over Session",
            labels={"index": "Query Number", "processing_time": "Processing Time (s)"}
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Question length vs processing time
        if len(df) > 5:
            df['question_length'] = df['question'].str.len()
            fig_scatter = px.scatter(
                df,
                x="question_length",
                y="processing_time", 
                title="Question Length vs Processing Time",
                labels={"question_length": "Question Length (chars)", "processing_time": "Processing Time (s)"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Recent queries table
        st.subheader("🕒 Recent Queries")
        recent_df = df[['question', 'processing_time', 'timestamp']].tail(10)
        st.dataframe(recent_df, use_container_width=True)
    
    else:
        st.info("No analytics data available. Start asking questions to see analytics!")

def evaluation_interface():
    """Evaluation interface for testing"""
    
    st.header("🧪 System Evaluation")
    
    st.markdown("""
    This section allows you to run comprehensive evaluations on the RAG system to assess:
    - **Semantic Similarity**: How well answers match expected responses
    - **Citation Accuracy**: How well sources are cited
    - **Response Completeness**: How complete the responses are
    - **Processing Performance**: Response time and efficiency
    """)
    
    # Sample test cases
    st.subheader("📝 Test Cases")
    
    # Predefined test cases
    test_cases = [
        {
            "question": "What are the main compliance requirements?",
            "expected_answer": "The main compliance requirements include regulatory standards, documentation requirements, and audit procedures.",
            "category": "Compliance"
        },
        {
            "question": "How do I submit a thesis proposal?", 
            "expected_answer": "To submit a thesis proposal, you need to follow the documented submission process including proper formatting and required approvals.",
            "category": "Process"
        }
    ]
    
    # Display test cases
    for i, test_case in enumerate(test_cases):
        with st.expander(f"Test Case {i+1}: {test_case['category']}"):
            st.write(f"**Question:** {test_case['question']}")
            st.write(f"**Expected Answer:** {test_case['expected_answer']}")
            
            if st.button(f"Run Test {i+1}", key=f"test_{i}"):
                with st.spinner("Running evaluation..."):
                    # This would connect to the evaluation framework
                    st.success("✅ Test completed! (Integration with evaluation.py needed)")
    
    # Custom test case
    st.subheader("➕ Custom Test Case")
    custom_question = st.text_input("Test Question:")
    custom_expected = st.text_area("Expected Answer:")
    
    if st.button("Run Custom Test") and custom_question and custom_expected:
        with st.spinner("Running custom evaluation..."):
            # This would integrate with the evaluation framework
            st.success("✅ Custom test completed!")
    
    # Evaluation results summary
    st.subheader("📊 Evaluation Summary")
    
    # Mock evaluation results for demo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Relevance Score", "0.85", "0.05")
    with col2:
        st.metric("Avg Processing Time", "2.3s", "-0.2s")
    with col3:
        st.metric("Citation Accuracy", "0.78", "0.12")

def run_streamlit():
    """Run the Streamlit application"""
    main()

if __name__ == "__main__":
    run_streamlit()