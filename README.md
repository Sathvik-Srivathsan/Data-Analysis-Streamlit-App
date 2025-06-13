# Data-Analysis-Streamlit-App
Data Analyst Agent

A Streamlit app to chat with your documents. Upload a file and ask the AI to analyze, summarize, or create visualizations from it.
Setup

    Install libraries:

    pip install streamlit pandas pypdf python-docx openpyxl requests

    Add API Key:

        Create a folder in your project directory named .streamlit.

        Inside that folder, create a file named secrets.toml.

        Add your Together.ai API key to the file like this:

        TOGETHER_API_KEY = "your_key_here"

Run

    streamlit run app.py
