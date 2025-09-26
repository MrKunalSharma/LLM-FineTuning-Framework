"""Deploy Streamlit app."""
import subprocess
import os
import sys

def deploy_streamlit():
    """Deploy the Streamlit application."""
    # Set environment variables
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    deploy_streamlit()
