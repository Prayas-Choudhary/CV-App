import streamlit as st
import spacy
import pdfplumber
import docx
import re
import pandas as pd
import datetime
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Helpers
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_contact_info(text):
    phone_match = re.findall(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b", text)
    email_match = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return (", ".join(set(phone_match)), ", ".join(set(email_match)))

def score_resume(resume_text, jd_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    score = util.cos_sim(resume_embedding, jd_embedding).item()
    return round(score * 100, 2)

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

# Streamlit App UI
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")
st.markdown("Upload resumes and compare them against a Job Description.")

with st.sidebar:
    st.header("💼 Job Description")
    jd_text = st.text_area("Paste the JD here", height=250)
    save_jd = st.checkbox("Save this JD to history")
    analyze = st.button("Start Analysis")

uploaded_files = st.file_uploader("Upload Resume files (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if analyze and jd_text and uploaded_files:
    results = []
    for file in uploaded_files:
        file_name = file.name
        try:
            ext = file.name.split('.')[-1].lower()
            if ext == "pdf":
                text = extract_text_from_pdf(file)
            elif ext == "docx":
                text = extract_text_from_docx(file)
            else:
                st.error(f"{file_name}: Unsupported file format.")
                continue

            text = clean_text(text)
            phone, email = extract_contact_info(text)
            percent = score_resume(text, jd_text)
            email_draft = f"Dear Candidate,\n\nThank you for applying. Based on our analysis, your profile matches approximately {percent}% of our requirements.\nWe will get back to you soon.\n\nRegards,\nHR Team"

            results.append({
                "File Name": file_name,
                "Score (%)": percent,
                "Phone": phone,
                "Email": email,
                "Email Draft": email_draft,
                "Preview": text[:300] + "..." if len(text) > 300 else text
            })

        except Exception as e:
            st.error(f"❌ Error reading {file_name}: {str(e)}")

    if results:
        df = pd.DataFrame(results)
        st.success(f"✅ {len(results)} resume(s) processed successfully!")

        st.subheader("📊 Summary Table")
        st.dataframe(df[["File Name", "Score (%)", "Phone", "Email"]].sort_values("Score (%)", ascending=False), use_container_width=True)

        st.subheader("📬 Email Drafts")
        for row in results:
            with st.expander(f"✉️ Draft for {row['File Name']} ({row['Score (%)']}%)"):
                st.code(row["Email Draft"], language="text")

        st.subheader("🔎 Resume Previews")
        for row in results:
            with st.expander(f"📝 {row['File Name']} Preview"):
                st.write(row["Preview"])

        st.subheader("⬇️ Download All Results")
        st.download_button(
            label="Download Excel",
            data=convert_df_to_excel(pd.DataFrame(results)),
            file_name=f"resume_results_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if save_jd:
        with open("jd_history.txt", "a", encoding="utf-8") as f:
            f.write(f"\n---\n{datetime.datetime.now()}\n{jd_text.strip()}\n")
        st.info("✅ JD saved to history.")
elif analyze:
    st.warning("⚠️ Please upload resumes and provide a JD to analyze.")
