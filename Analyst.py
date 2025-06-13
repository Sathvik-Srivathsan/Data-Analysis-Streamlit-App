# app.py
import streamlit as st
import pandas as pd
import pypdf
import docx
import os
import replicate
import io
import plotly.express as px
import re

st.set_page_config(page_title="Data Analyst Agent", layout="wide")

try:
    # Set the Replicate API token from Streamlit secrets
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    REPLICATE_API_TOKEN_SET = True
except (FileNotFoundError, KeyError):
    st.error(
        "Replicate API token not found. Please create a .streamlit/secrets.toml file and add your REPLICATE_API_TOKEN.")
    REPLICATE_API_TOKEN_SET = False


# Helper Functions for File Processing
def process_text(file_contents):
    """Processes text from a .txt file."""
    return file_contents.decode("utf-8")


def process_pdf(file):
    #Processes text from a .pdf file
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Add check for None
    return text


def process_docx(file):
    """Processes text from a .docx file."""
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def process_csv(file):
    #Processes data from a .csv file and returns it as a DataFrame
    try:
        file.seek(0)
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None


def process_xlsx(file):
    #Processes data from an .xlsx file and returns it as a DataFrame
    try:
        file.seek(0)
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error processing Excel: {e}")
        return None


# Visualization Function

def execute_viz_code(code, df):
    if code:
        try:
            local_vars = {"st": st, "pd": pd, "px": px, "df": df}
            exec(code, {}, local_vars)
        except Exception as e:
            st.error(f"Error executing visualization code: {e}")
            st.code(code)


# --- NEW: Function to Query the Replicate API with Llama 3 ---

def query_replicate_llama3(context, question, chat_history):
    # Sends query to Replicate API to interact with Llama 3
    if not REPLICATE_API_TOKEN_SET:
        st.error("Cannot call Replicate API because the token is not set.")
        return None

    # Prompt
    system_prompt = """
    You are an expert data analyst working inside a Streamlit app. Your goal is to help a user analyze their data.

    **Instructions:**
    1.  Analyze the provided data context and the user's question from the conversation history.
    2.  If the user asks for a visualization, you MUST generate Python code to create and display a plot within Streamlit.
    3.  The data is available in a pandas DataFrame named `df`.
    4.  Use the `plotly.express` library (imported as `px`) for all visualizations. Do not use matplotlib.
    5.  After creating a figure (e.g., `fig = px.histogram(df, x='some_column')`), you MUST display it using `st.plotly_chart(fig)`.
    6.  Your response for a visualization request MUST ONLY be the Python code block, enclosed in ```python ... ```. Do not add any explanation or other text.
    7.  If the user asks a general question that does not require a plot, provide a clear and concise text-based answer without any code.
    """

    # Format the chat history and the new question into a single prompt string
    prompt_for_model = f"Data Context:\n{context}\n\n"
    for msg in chat_history:
        if msg["role"] == "user":
            prompt_for_model += f"User: {msg['content']}\n"
        else:  # assistant
            prompt_for_model += f"Assistant: {msg['content']}\n"
    prompt_for_model += f"User: {question}"

    try:
        # Model identifier for Llama 3
        model_id = "meta/meta-llama-3-70b-instruct"

        output_stream = replicate.run(
            model_id,
            input={
                "prompt": prompt_for_model,
                "system_prompt": system_prompt,
                "max_new_tokens": 2048,
                "temperature": 0.5,

            }
        )

        response_text = "".join(output_stream)
        return response_text

    except replicate.exceptions.ReplicateError as e:
        st.error(f"Error calling Replicate API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# Streamlit App UI (Logic adapted for Replicate)

st.title("Intelligent Data Analyst Agent (using Llama 3 on Replicate)")
st.markdown("Upload a document (.txt, .pdf, .docx, .csv, .xlsx) and ask questions about it.")

# Session State Initialization
if 'file_context_str' not in st.session_state:
    st.session_state.file_context_str = None
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Sidebar for File Upload
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "docx", "csv", "xlsx"]
    )

    if uploaded_file is not None:
        # Clear state on new file upload
        if st.session_state.get('file_name') != uploaded_file.name:
            st.session_state.messages = []
            st.session_state.file_name = uploaded_file.name
            st.session_state.dataframe = None
            st.session_state.file_context_str = None

            with st.spinner(f"Processing {uploaded_file.name}..."):
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                file_wrapper = io.BytesIO(uploaded_file.getvalue())

                context_str = ""
                df = None

                if file_extension == ".txt":
                    context_str = process_text(file_wrapper.read())
                elif file_extension == ".pdf":
                    context_str = process_pdf(file_wrapper)
                elif file_extension == ".docx":
                    context_str = process_docx(file_wrapper)
                elif file_extension in [".csv", ".xlsx"]:
                    if file_extension == ".csv":
                        df = process_csv(file_wrapper)
                    else:
                        df = process_xlsx(file_wrapper)

                    if df is not None:
                        st.session_state.dataframe = df
                        # For dataframes, context is a summary
                        context_str = f"Dataframe columns: {df.columns.tolist()}\nFirst 5 rows:\n{df.head().to_string()}"
                        st.subheader("Data Preview:")
                        st.dataframe(df.head())
                    else:
                        context_str = "Error processing the file."
                else:
                    context_str = "Unsupported file type."

                st.session_state.file_context_str = context_str
                st.success("File processed successfully!")
                st.info("You can now ask questions about the document in the main chat.")


# Main Chat Interface
if st.session_state.get('file_name'):
    st.subheader(f"Chat about: `{st.session_state.file_name}`")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check for python code blocks to execute for past messages
            code_match = re.search(r"```python\n(.*?)```", message["content"], re.DOTALL)
            if code_match and message["role"] == "assistant":
                code = code_match.group(1).strip()
                st.markdown("Here is the visualization you requested:")
                execute_viz_code(code, st.session_state.dataframe)
            else:
                st.markdown(message["content"])

    # Chat input for new messages
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Llama 3 is thinking..."):
                if st.session_state.file_context_str and REPLICATE_API_TOKEN_SET:
                    # Pass the last 10 messages for history context
                    chat_history = st.session_state.messages[-11:-1]

                    response = query_replicate_llama3(
                        st.session_state.file_context_str,
                        prompt,
                        chat_history
                    )
                    if response:
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        # Check for and execute visualization code from the new response
                        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
                        if code_match:
                            code = code_match.group(1).strip()
                            st.markdown("Here is the visualization you requested:")
                            execute_viz_code(code, st.session_state.dataframe)
                        else:
                            st.markdown(response)
                    else:
                        st.error("Failed to get a response from the agent.")
                elif not REPLICATE_API_TOKEN_SET:
                     st.warning("Please set your Replicate API Token to chat.")
                else:
                    st.warning("Please upload a file first.")
else:
    st.info("Please upload a file using the sidebar to begin.")
