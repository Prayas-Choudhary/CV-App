import spacy.cli
spacy.cli.download("en_core_web_sm")

import streamlit as st
import os
import re
import pdfplumber
from docx import Document
import pandas as pd
import spacy
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return '\n'.join([p.text for p in doc.paragraphs])

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else ""

def extract_field(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def get_similarity(text1, text2):
    doc1 = nlp(text1.lower())
    doc2 = nlp(text2.lower())
    return round(doc1.similarity(doc2) * 100, 2)

def remove_company_name(jd_text, company_name):
    return jd_text.replace(company_name, "Confidential")

st.title("📋 Automated Hiring Assistant")
st.write("Upload resumes and job description to extract data, compare, and create reports.")

uploaded_jd = st.file_uploader("Upload Job Description (TXT)", type=['txt'])
company_name = st.text_input("Enter Company Name to Redact from JD")
uploaded_resumes = st.file_uploader("Upload CVs (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

if st.button("Process Resumes"):
    if not uploaded_jd or not uploaded_resumes or not company_name:
        st.warning("Please upload JD, at least one CV, and enter company name.")
        st.stop()

    jd_text = uploaded_jd.read().decode('utf-8')
    redacted_jd = remove_company_name(jd_text, company_name)

    os.makedirs("output/emails", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    candidates = []

    for file in uploaded_resumes:
        filename = file.name
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            continue

        email = extract_email(text)
        name = extract_field(r"Name[:\-]?\s*(.*)", text)
        role = extract_field(r"(Role|Position)[:\-]?\s*(.*)", text)
        ctc = extract_field(r"CTC[:\-]?\s*([0-9]+)", text)
        ectc = extract_field(r"Expected CTC[:\-]?\s*([0-9]+)", text)
        company = extract_field(r"Working at[:\-]?\s*(.*)", text)
        notice = extract_field(r"Notice Period[:\-]?\s*(.*)", text)
        exp = extract_field(r"Experience[:\-]?\s*(\d+)", text)
        match = get_similarity(text, jd_text)

        if email:
            draft = f"""Subject: Opportunity Matching Your Profile

Hi {name or "Candidate"},

We found your profile suitable for a role that matches your experience. Please find the job description attached.

If you're interested, kindly respond with your updated resume and availability.

Regards,
Recruitment Team
"""
            with open(f"output/emails/{email}.txt", "w", encoding="utf-8") as f:
                f.write(draft)
            with open(f"output/emails/{email}_JD.txt", "w", encoding="utf-8") as f:
                f.write(redacted_jd)

        candidates.append({
            "Name": name,
            "Email": email,
            "Role": role,
            "CTC": ctc,
            "Expected CTC": ectc,
            "Current Company": company,
            "Notice Period": notice,
            "Experience": exp,
            "Match %": match,
            "Resume File": filename,
            "Status": "Profile Shared"
        })

    df = pd.DataFrame(candidates)
    excel_path = "output/candidates.xlsx"
    df.to_excel(excel_path, index=False)

    wb = load_workbook(excel_path)
    ws = wb.active
    dv = DataValidation(type="list", formula1='"Profile Shared,Shortlisted,Interview L-1,Interview L-2,Selected,Rejected"', allow_blank=True)
    ws.add_data_validation(dv)
    for row in range(2, len(df) + 2):
        dv.add(ws[f"K{row}"])
    wb.save(excel_path)

    st.success("✅ Done! Files saved in 'output' folder.")
    st.download_button("📥 Download Excel", data=open(excel_path, 'rb'), file_name="candidates.xlsx")
