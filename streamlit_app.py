"""Streamlit interface for LLM Fine-tuning Framework."""
import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "your-secure-api-key")

# Page config
st.set_page_config(
    page_title="LLM Fine-tuning Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .response-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class LLMInterface:
    """Interface for interacting with the LLM API."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def check_health(self) -> Dict:
        """Check API health."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        try:
            response = requests.get(
                f"{self.api_url}/model/info",
                headers=self.headers,
                timeout=5
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def generate_text(
        self,
        instruction: str,
        input_text: str = "",
        **kwargs
    ) -> Dict:
        """Generate text from the model."""
        payload = {
            "instruction": instruction,
            "input_text": input_text,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            return response.json() if response.status_code == 200 else {
                "error": response.text
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize interface
@st.cache_resource
def get_llm_interface():
    return LLMInterface(API_URL, API_KEY)

# Sidebar
def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Status
        llm = get_llm_interface()
        health = llm.check_health()
        
        if health and health.get("status") == "healthy":
            st.success("✅ API Connected")
            model_info = llm.get_model_info()
            if model_info:
                st.info(f"**Model**: {model_info.get('model_name', 'Unknown')}")
                st.info(f"**Device**: {model_info.get('device', 'Unknown')}")
        else:
            st.error("❌ API Disconnected")
        
        st.divider()
        
        # Generation Parameters
        st.subheader("Generation Parameters")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in generation"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=10,
            max_value=1000,
            value=256,
            step=10,
            help="Maximum number of tokens to generate"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Nucleus sampling parameter"
        )
        
        return {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "top_p": top_p
        }

# Main interface
def render_main():
    """Render main interface."""
    st.title("🤖 LLM Fine-tuning Interface")
    st.markdown("Interact with your fine-tuned language model")
    
    # Get generation parameters
    params = render_sidebar()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🧪 Batch Testing", "📚 Examples"])
    
    with tab1:
        render_chat_tab(params)
    
    with tab2:
        render_analytics_tab()
    
    with tab3:
        render_batch_tab(params)
    
    with tab4:
        render_examples_tab()

def render_chat_tab(params: Dict):
    """Render chat interface."""
    st.header("Chat with the Model")
    
    # Input fields
    col1, col2 = st.columns([3, 1])
    
    with col1:
        instruction = st.text_area(
            "Instruction",
            placeholder="Enter your instruction here...",
            height=100
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_button = st.button("🚀 Generate", use_container_width=True)
    
    input_text = st.text_area(
        "Context (Optional)",
        placeholder="Provide additional context if needed...",
        height=100
    )
    
    # Generation
    if generate_button and instruction:
        llm = get_llm_interface()
        
        with st.spinner("Generating response..."):
            start_time = time.time()
            response = llm.generate_text(
                instruction=instruction,
                input_text=input_text,
                **params
            )
            end_time = time.time()
        
        if "error" in response:
            st.error(f"Error: {response['error']}")
        else:
            # Display response
            st.markdown("### 📝 Response")
            st.markdown(f"<div class='response-box'>{response['response']}</div>", 
                       unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Generation Time", f"{response['generation_time']:.2f}s")
            with col2:
                st.metric("Tokens Generated", response['tokens_generated'])
            with col3:
                st.metric("Total Time", f"{end_time - start_time:.2f}s")

def render_analytics_tab():
    """Render analytics dashboard."""
    st.header("📊 Model Analytics")
    
    # Mock data for demonstration
    # In production, fetch from Prometheus/API
    
    # Request metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", "1,234", "+12%")
    with col2:
        st.metric("Avg Response Time", "0.85s", "-5%")
    with col3:
        st.metric("Success Rate", "99.2%", "+0.3%")
    with col4:
        st.metric("Active Users", "42", "+5")
    
    # Charts
    st.subheader("Performance Metrics")
    
    # Response time chart
    time_data = pd.DataFrame({
        'Time': pd.date_range(start='2024-01-01', periods=24, freq='H'),
        'Response Time': [0.5 + i*0.02 + (i%3)*0.1 for i in range(24)]
    })
    
    fig = px.line(time_data, x='Time', y='Response Time', 
                  title='Response Time Over Last 24 Hours')
    st.plotly_chart(fig, use_container_width=True)
    
    # Token distribution
    col1, col2 = st.columns(2)
    
    with col1:
        token_dist = pd.DataFrame({
            'Tokens': ['0-50', '51-100', '101-200', '201-500', '500+'],
            'Count': [300, 450, 320, 150, 80]
        })
        fig = px.pie(token_dist, values='Count', names='Tokens', 
                    title='Token Generation Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model usage by endpoint
        endpoint_data = pd.DataFrame({
            'Endpoint': ['Generate', 'Batch', 'Health', 'Model Info'],
            'Calls': [850, 250, 180, 54]
        })
        fig = px.bar(endpoint_data, x='Endpoint', y='Calls',
                    title='API Endpoint Usage')
        st.plotly_chart(fig, use_container_width=True)

def render_batch_tab(params: Dict):
    """Render batch testing interface."""
    st.header("🧪 Batch Testing")
    st.markdown("Test multiple prompts at once")
    
    # Sample prompts
    default_prompts = [
        {"instruction": "Translate to French", "input": "Hello, how are you?"},
        {"instruction": "Summarize", "input": "Machine learning is a subset of AI..."},
        {"instruction": "Write a haiku about", "input": "artificial intelligence"}
    ]
    
    # JSON input
    batch_input = st.text_area(
        "Batch Input (JSON)",
        value=json.dumps(default_prompts, indent=2),
        height=300
    )
    
    if st.button("🚀 Run Batch Test"):
        try:
            prompts = json.loads(batch_input)
            results = []
            
            llm = get_llm_interface()
            progress_bar = st.progress(0)
            
            for i, prompt in enumerate(prompts):
                with st.spinner(f"Processing {i+1}/{len(prompts)}..."):
                    response = llm.generate_text(
                        instruction=prompt.get("instruction", ""),
                        input_text=prompt.get("input", ""),
                        **params
                    )
                    results.append({
                        "instruction": prompt.get("instruction", ""),
                        "input": prompt.get("input", ""),
                        "response": response.get("response", response.get("error", "Error")),
                        "time": response.get("generation_time", 0)
                    })
                progress_bar.progress((i + 1) / len(prompts))
            
            # Display results
            st.success(f"✅ Completed {len(results)} tests")
            
            # Results table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except json.JSONDecodeError:
            st.error("Invalid JSON format")

def render_examples_tab():
    """Render examples tab."""
    st.header("📚 Example Use Cases")
    
    examples = [
        {
            "title": "🌐 Translation",
            "instruction": "Translate the following text to Spanish",
            "input": "The quick brown fox jumps over the lazy dog",
            "expected": "El rápido zorro marrón salta sobre el perro perezoso"
        },
        {
            "title": "📝 Summarization",
            "instruction": "Summarize this text in one sentence",
            "input": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
            "expected": "AI is machine intelligence that perceives environments and takes actions to achieve goals, contrasting with natural human intelligence."
        },
        {
            "title": "💻 Code Generation",
            "instruction": "Write a Python function to calculate the factorial of a number",
            "input": "",
            "expected": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
        },
        {
            "title": "❓ Question Answering",
            "instruction": "Answer the question based on the context",
            "input": "Context: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Question: Who designed the Eiffel Tower?",
            "expected": "Gustave Eiffel's company designed the Eiffel Tower."
        }
    ]
    
    for example in examples:
        with st.expander(example["title"]):
            st.markdown(f"**Instruction:** {example['instruction']}")
            if example["input"]:
                st.markdown(f"**Input:** {example['input']}")
            st.markdown(f"**Expected Output:** {example['expected']}")
            
            if st.button(f"Try this example", key=example["title"]):
                st.session_state.example_instruction = example["instruction"]
                st.session_state.example_input = example["input"]
                st.info("Example loaded! Go to the Chat tab to run it.")

# Main execution
if __name__ == "__main__":
    render_main()
