import os
import re
import io
import base64
import spacy
import torch
import pdfplumber
import pandas as pd
import streamlit as st
from io import BytesIO
from docx import Document
from openpyxl import Workbook
from sentence_transformers import SentenceTransformer, util

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Force CPU & float32 usage to avoid torch meta tensor error
model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.float()
model.to(torch.device("cpu"))

# Streamlit layout
st.set_page_config(page_title="Automated Resume Analyzer", layout="wide")
st.title("📄 Automated Resume Analyzer & Tracker")

# Job Description input (Always visible)
st.subheader("📋 Job Description")
jd_text = st.text_area("Paste Job Description Here:", height=150, placeholder="E.g. Looking for a Software Engineer with 3+ years of experience in Python and ML...")

# Resume upload (Always visible)
st.subheader("📁 Upload Candidate Resumes")
uploaded_files = st.file_uploader("Upload PDFs or DOCX resumes:", type=["pdf", "docx"], accept_multiple_files=True)

# --- Utility Functions ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""

def extract_email(text):
    match = re.search(r"\b[\w.-]+@[\w.-]+\.\w{2,4}\b", text)
    return match.group(0) if match else ""

def extract_phone(text):
    match = re.search(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", text)
    return match.group(0) if match else ""

def extract_field(field_name, text):
    pattern = rf"{field_name}[:\s]*([^\n,;|]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match and match.lastindex >= 1 else ""

def extract_role_from_jd(jd_text):
    role_keywords = ["Software Engineer", "Data Scientist", "Frontend Developer", "Backend Developer", "Marketing Manager"]
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    scores = [util.cos_sim(jd_embedding, model.encode(role, convert_to_tensor=True)) for role in role_keywords]
    best_match_index = scores.index(max(scores))
    return role_keywords[best_match_index]

def get_excel_download_link(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

def get_download_link(file, filename):
    b64 = base64.b64encode(file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📎 Download Resume</a>'
    return href

# --- Main Processing ---
if uploaded_files and jd_text.strip():
    role_from_jd = extract_role_from_jd(jd_text)
    st.success(f"🧠 Inferred Role from JD: **{role_from_jd}**")
    st.subheader("📌 Extracted Resume Information")

    results = []

    for file in uploaded_files:
        text = extract_text(file)
        name = extract_name(text)
        email = extract_email(text)
        phone = extract_phone(text)
        role = extract_field("Role|Position", text)
        company = extract_field("Company|Currently working", text)

        results.append({
            "Candidate Name": name,
            "Email": email,
            "Phone": phone,
            "Current Company": company,
            "Role (in Resume)": role,
            "Role (from JD)": role_from_jd,
            "Filename": file.name
        })

        with st.expander(f"👤 {name or file.name}"):
            st.markdown(f"**Email:** {email or 'N/A'}")
            st.markdown(f"**Phone:** {phone or 'N/A'}")
            st.markdown(f"**Current Company:** {company or 'N/A'}")
            st.markdown(f"**Role Mentioned:** {role or 'N/A'}")
            st.markdown(get_download_link(file, file.name), unsafe_allow_html=True)

    df = pd.DataFrame(results)

    if st.button("💾 Generate Excel and Enable Resume Downloads"):
        st.success("✅ Tracker Generated Below")
        excel_data = get_excel_download_link(df)
        st.download_button("📊 Download Tracker Excel", data=excel_data, file_name="resume_tracker.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif not jd_text.strip():
    st.info("✏️ Please paste a Job Description to get started.")

elif not uploaded_files:
    st.info("📂 Please upload at least one resume.")
