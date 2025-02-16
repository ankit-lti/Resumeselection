import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from docx import Document  # For processing .docx files
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import re
from PyPDF2 import PdfReader

# Set your Hugging Face API token
api_token = "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"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
#API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": f"Bearer {api_token}"}

def extract_text_from_docx(docx_file):
    """Extracts text from a .docx file."""
    doc = Document(BytesIO(docx_file))
    return "\n".join([para.text for para in doc.paragraphs])

def summarize_resume(text, sentence_count=5):
    """Summarizes resume to reduce token count before sending to the model."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

def process_pdf_resume(pdf_file):
    """Processes a PDF resume to extract and summarize its content."""
    resume_text = extract_text_from_pdf(pdf_file)
    if len(resume_text.split()) > 500:
        resume_text = summarize_resume(resume_text, sentence_count=8)
    # Further processing can be done here
    return resume_text

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def analyze_resume(resume_text):
    """Sends resume to Hugging Face API for faster processing."""
    
    # Summarize if resume is too long (more than 500 words)
    if len(resume_text.split()) > 500:
        resume_text = summarize_resume(resume_text, sentence_count=8)

    prompt = f"""
  Only add the generated response in your output. Don't include the prompt and the resume of the candidate in your output.
    You are a brilliant AI assisstant whose work will be to analyse the resume provided and provide the summary of the candidate resume and key skills he has worked with. ALso add only below details as asked 
    Extract the following details from the resume:
    - Candidate Name: Extract the name (assumed to be at the top).
    - Key Skills: List all key skills, and highlight AIML-related skills in **bold**.
    - AIML Experience Level:
      - >10 years → **Expert**
      - 5-10 years → **Intermediate**
      - <5 years → **Novice**
    
    Output Format (ONLY output this format, NO extra text):
    Candidate Name: [Extracted Name]
    Key Skills: [List of Skills, AIML in bold]
    AIML Experience: [Novice/Intermediate/Expert]

    Resume:
    {resume_text}
"""

    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt},timeout=30)
    print(response.status_code, response.text)  # Debug API response
    response_json = response.json()
    if isinstance(response_json, list) and "generated_text" in response_json[0]:
        output_text = response_json[0]["generated_text"]
            
            # Ensure we extract only the second occurrence of "Candidate Name:"
        parts = output_text.split("Candidate Name:")
            
        if len(parts) > 1:
            output_text = "Candidate Name:" + parts[-1]  # Keep only the second occurrence onwards
            print(output_text)
            return output_text.strip()
    
    return f"Error: Unexpected response - {response.json()}"# Streamlit App

def parse_analysis_result(result):
    """Parses the analysis result into a dictionary."""
    data = {}
    # Use regex to extract each field
    name_match = re.search(r"Candidate Name:\s*(.*)", result)
    skills_match = re.search(r"Key Skills:\s*(.*)", result)
    experience_match = re.search(r"AIML Experience:\s*(.*)", result)

    if name_match:
        data["Candidate Name"] = name_match.group(1).strip()
    if skills_match:
        data["Key Skills"] = skills_match.group(1).strip()
    if experience_match:
        data["AIML Experience"] = experience_match.group(1).strip()

    return data
st.markdown(
    """
    <style>
    /* Align table headers to the left */
    th {
        text-align: left !important;
    }
    /* Align table cells to the left */
    td {
        text-align: left !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 AIML Resume Analyzer (Fast API Version)")

uploaded_files = st.file_uploader("📂 Upload Resumes", accept_multiple_files=True, type=["txt", "docx","pdf"])

if uploaded_files:
    data = []
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1]
        
        # Extract text based on file type
        if file_type == "txt":
            resume_text = uploaded_file.read()
        elif file_type == "docx":
            resume_text = extract_text_from_docx(uploaded_file.read())
        elif file_type == "pdf":
            resume_text = process_pdf_resume(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_type}")
            continue

        # Extract candidate name (assuming first non-empty line)
        candidate_name = next((line.strip() for line in resume_text.split("\n") if line.strip()), "Unknown")

        # Analyze the resume
        result = analyze_resume(resume_text)
        print(result)
        # Add data for display
        #data.append({
        #    "Resume File": uploaded_file.name,
        #    "Candidate Name": candidate_name,
        #    "Analysis": result
        #})
    
    # Convert to DataFrame
    #df = pd.DataFrame(data)

    # Display results
    #st.write("### 🔍 Analysis Results")
    #st.write(result)
     # Parse the structured output
        if result.startswith("Candidate Name:"):
            parsed_data = parse_analysis_result(result)
            data.append(parsed_data)
        else:
            st.error("Failed to parse the analysis result.")
    
    # Convert to DataFrame for tabular display
    if data:
        df = pd.DataFrame(data)
        st.write("### 🔍 Analysis Results")
        st.table(df)  # Display as a table
    else:
        st.write("No valid data to display.")
