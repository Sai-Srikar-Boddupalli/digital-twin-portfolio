import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# 1. Setup
load_dotenv()
st.set_page_config(page_title="Saisrikar's Digital Resume", page_icon="🤖")

# 2. UI Layout
st.header("🤖 Chat with Saisrikar's Experience (Free & Fast Edition)")

# Sidebar
with st.sidebar:
    st.title("Data Ingestion")
    pdf_docs = st.file_uploader("Upload Resume (PDF)", accept_multiple_files=True)
    
    if st.button("Process Resume"):
        with st.spinner("Processing (Downloading Free Models)..."):
            # A. Extract Text
            raw_text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
            
            # B. Split Text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(raw_text)

            # C. Embed & Save (Using HuggingFace - Runs on your CPU)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            
            st.success("Resume Processed! The AI is ready.")

# 3. Chat Logic
user_question = st.text_input("Ask a question (e.g., 'What ETL tools does Sai know?')")

if user_question:
    # Use the same free embeddings to understand the question
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index"):
        # Load the DB
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # 1. Retrieve Docs
        docs = new_db.similarity_search(user_question)
        
        # 2. Setup the "Fast" LLM (UPDATED MODEL NAME HERE)
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile" 
        )
        
        # 3. The Prompt
        context_text = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
        You are Saisrikar Boddupalli. Answer the question based ONLY on the context provided below.
        If the answer is not in the context, say "I don't have that info in my resume."
        
        Context:
        {context_text}
        
        Question: 
        {user_question}
        """
        
        # 4. Generate
        response = llm.invoke(prompt)
        
        st.write("### Answer:")
        st.write(response.content)
    else:
        st.error("Please upload and process the resume in the sidebar first!")
