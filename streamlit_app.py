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
import nltk

# Set your Hugging Face API token
api_token = "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"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
#API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": f"Bearer {api_token}"}

# Download NLTK data
nltk.download('punkt_tab')
nltk.data.path.append('/home/vscode/nltk_data')

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
Only include the generated response in your output. Do not include the prompt or the candidate's resume in your output.

You are a brilliant AI assistant whose task is to analyze the provided resume and generate a summary of the candidate's resume and key skills. Please include only the details requested below:

Analyze the provided resume and extract the candidate name. Ensure that the name is accurately identified and included in your response.
Provide a brief summary of the candidate and their past work experience, specifically highlighting any experience in the Media & Entertainment Industry. Also highlight in bold letters any specific Jobs role done in Media & Entertainment industry. 
Calculate the total number of years of experience the candidate has based on the work history provided in the resume.
Calculate the relevant Media & Entertainment experience, considering any work done at companies within the Media & Entertainment Industry (e.g., Paramount, HBO, Disney, etc.).
Analyze the provided resume and identify any media-related platforms or technologies the candidate has worked with. Specifically, look for platforms similar to VidiSpine, Fabric,Charles Proxy, Apple Configurator,MPX in th resume. List these platforms in your response. Iy they are none then specified as None Specified.
List all the companies where the candidate has worked, in comma-separated values.
Based on the resume analyzer, identify the Industry Domain the candidate has worked in. If it is part of the Media & Entertainment domain, list out the subdomains (such as media supply chain (Content acquisition, Media processing, Quality Control, delivery, etc.), streaming, broadcasting, film production, post-production, animation, gaming, and digital media).
Always start and end the response by adding three backticks like ``` to identify the actual response before the first field.
    Candidate Name: [Provide Candidate Name from the resume]
    Total Experience: [total number of years experience]
    Relevant M&E Experience: [relevant years of experience based on Media and Entertainment industry]
    Companies: [List all companies in comma separated]
    Summary: [Brief summary about the candidate and past experience]
    Domain:[List out the domain and subdomain]
    Platform: [List media Platform if any]
    Resume:
    {resume_text}
"""

    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"temperature": 0.9}}, timeout=30)
    print(response.status_code, response.text)  # Debug API response
    response_json = response.json()
    if isinstance(response_json, list) and "generated_text" in response_json[0]:
        output_text = response_json[0]["generated_text"]
            
            # Ensure we extract only the second occurrence of "Candidate Name:"
        #parts = output_text.split("Candidate Name:")
            
        parts = output_text.split("```")
        if len(parts) > 1:
            output_text = "```".join(parts[1:]).strip()  # Join parts after the first occurrence of ```
            return output_text
    
    return f"Error: Unexpected response - {response.json()}"# Streamlit App

def parse_analysis_result(result):
    """Parses the analysis result into a dictionary."""
    data = {}
    name_matches = re.findall(r"Candidate Name:\s*(.*)", result, re.IGNORECASE)
    #skills_match = re.findall(r"Key Skills:\s*(.*)", result, re.IGNORECASE)
    totalExperience_match = re.findall(r"Total Experience:\s*(.*)", result, re.IGNORECASE)
    relevantExperience_match = re.findall(r"Relevant M&E Experience:\s*(.*)", result, re.IGNORECASE)
    companies_match = re.findall(r"Companies:\s*(.*)", result, re.IGNORECASE)
    summary_match = re.findall(r"Summary:\s*(.*)", result, re.IGNORECASE)
    domain_match = re.findall(r"Domain:\s*(.*)", result, re.IGNORECASE)
    platform_match = re.findall(r"Platform:\s*(.*)", result, re.IGNORECASE)
    print("Result:", result)  # Print the entire result for debugging
    print("Name Matches:", name_matches)  # Print all name matches
    #print("Skills Match:", skills_match)  # Print the skills match result
    print("Total Experience:", totalExperience_match)  # Print the experience match result
    print("Relevant M&E Experience:", relevantExperience_match)  # Print the relevant experience match result
    print("Companies:", companies_match)  # Print the relevant experience match result
    print("Summary Match:", summary_match)  # Print the summary match result
    print("Domain Match:", domain_match)  # Print the domain match result
    print("Platform Match:", platform_match)  # Print the domain match result
    if name_matches:
        data["Candidate Name"] = name_matches[-1].strip()  # Take the last occurrence
    if summary_match:
        data["Summary"] = summary_match[-1].strip()
    #if skills_match:
    #    data["Key Skills"] = skills_match[-1].strip()
    if totalExperience_match:
        data["Total Experience"] = totalExperience_match[-1].strip()
    if relevantExperience_match:
        data["Relevant M&E Experience"] = relevantExperience_match[-1].strip()
    if companies_match:
        data["Companies"] = companies_match[-1].strip()
    if platform_match:
        data["Platform"] = platform_match[-1].strip()
    if domain_match:
        data["Domain"] = domain_match[-1].strip()


    return data

# Function to convert DataFrame to Excel and return as bytes
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

st.set_page_config(layout="wide")  # Set the layout to wide
st.markdown(
    """
    <style>
    th, td {
        word-wrap: break-word;
        white-space: normal;
    }
    th {
        text-align: left !important;
    }
    td {
        text-align: left !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 AIML Resume Analyzer")

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
        #print(result)
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
        #print("Before parsing",result)
        if "candidate name" in result.lower():
            parsed_data = parse_analysis_result(result)
            data.append(parsed_data)
        else:
            st.error("Failed to parse the analysis result.")
    
    # Convert to DataFrame for tabular display
    if data:
        df = pd.DataFrame(data)
        st.write("### 🔍 Analysis Results")
        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
        # Add download button for Excel file
        st.download_button(
        label="Download data as Excel",
        data=to_excel(df),
        file_name='analysis_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        #st.dataframe(df,use_container_width=True)
        #st.dataframe(df, column_config={
        #    "Candidate Name": {"width": 150},
        #    "Key Skills": {"width": 300},
        #    "AIML Experience": {"width": 150},
        #    "Summary": {"width": 600},
        #})
    else:
        st.write("No valid data to display.")   
