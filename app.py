import streamlit as st
import spacy
import pdfplumber
import docx
import re
import pandas as pd
import datetime
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_contact_info(text):
    phones = re.findall(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b", text)
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return ", ".join(set(phones)), ", ".join(set(emails))

def score_resume(resume_text, jd_text):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    return round(util.cos_sim(emb_resume, emb_jd).item() * 100, 2)

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return output.getvalue()

# Streamlit UI
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("📄 Smart Resume Analyzer")
st.markdown("Upload resumes and analyze them against a job description.")

with st.sidebar:
    st.header("💼 Job Description")
    jd_text = st.text_area("Paste JD here", height=250)
    save_jd = st.checkbox("Save this JD to history")
    analyze = st.button("Start Analysis")

uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if analyze and jd_text and uploaded_files:
    results = []

    for file in uploaded_files:
        name = file.name
        try:
            ext = name.split(".")[-1].lower()
            if ext == "pdf":
                text = extract_text_from_pdf(file)
            elif ext == "docx":
                text = extract_text_from_docx(file)
            else:
                st.error(f"Unsupported file: {name}")
                continue

            text = clean_text(text)
            phone, email = extract_contact_info(text)
            score = score_resume(text, jd_text)

            email_draft = (
                f"Dear Candidate,\n\n"
                f"Thank you for applying. Based on our analysis, your resume matches {score}% of our job requirements.\n"
                f"We will contact you shortly.\n\nBest,\nHR Team"
            )

            results.append({
                "File Name": name,
                "Score (%)": score,
                "Phone": phone,
                "Email": email,
                "Email Draft": email_draft,
                "Preview": text[:500] + "..." if len(text) > 500 else text
            })

        except Exception as e:
            st.error(f"Error processing {name}: {str(e)}")

    if results:
        df = pd.DataFrame(results)

        st.success(f"✅ Processed {len(results)} resumes successfully.")
        st.subheader("📊 Summary Table")
        st.dataframe(df[["File Name", "Score (%)", "Phone", "Email"]].sort_values("Score (%)", ascending=False))

        st.subheader("📬 Email Drafts")
        for row in results:
            with st.expander(f"✉️ {row['File Name']} - {row['Score (%)']}%"):
                st.code(row["Email Draft"], language="text")

        st.subheader("📝 Resume Previews")
        for row in results:
            with st.expander(f"📄 {row['File Name']} Preview"):
                st.write(row["Preview"])

        st.download_button(
            label="📥 Download Excel Report",
            data=convert_df_to_excel(df),
            file_name=f"resume_report_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if save_jd:
        with open("jd_history.txt", "a", encoding="utf-8") as f:
            f.write(f"\n---\n{datetime.datetime.now()}\n{jd_text.strip()}\n")
        st.info("✅ JD saved to history.")

elif analyze:
    st.warning("⚠️ Please upload resumes and enter a JD to start analysis.")

