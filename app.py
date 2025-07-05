import streamlit as st
import cohere
import PyPDF2
import os
from dotenv import load_dotenv
load_dotenv()  # loads your .env file

import cohere
co = cohere.Client(os.getenv("CO_API_KEY"))

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

chat_history = []

def ask_cohere_question(text, question):
    full_context = "\n".join(chat_history[-5:])

    prompt = f"""
You're an AI assistant helping with this PDF content.
Use the information below to answer questions.

PDF Text:
{text}

Previous Chat:
{full_context}

User's Question:
{question}

Answer:
"""

    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300
    )

    answer = response.generations[0].text.strip()

    chat_history.append(f"User: {question}")
    chat_history.append(f"Bot: {answer}")

    return answer

st.set_page_config(page_title="Smart PDF Chatbot", page_icon="🤖")
st.title("📚 VynAI – Where PDFs Talk Back 🤖")
st.markdown("Upload one or more PDFs and ask anything about them!")

pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if pdfs:
    full_text = ""
    for pdf in pdfs:
        full_text += extract_text_from_pdf(pdf)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the PDFs...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        reply = ask_cohere_question(full_text, user_input)

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": reply})

