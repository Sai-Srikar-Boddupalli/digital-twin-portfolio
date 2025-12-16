import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Saisrikar's AI Portfolio",
    page_icon="🚀",
    layout="wide"
)

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CUSTOM CSS FOR "COOL" LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatInput {
        position: fixed;
        bottom: 30px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE (MEMORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Saisrikar's AI Assistant. Ask me anything about his Data Engineering projects, Python skills, or ETL experience."}
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- AUTO-LOAD RESUME LOGIC ---
@st.cache_resource
def get_vector_store():
    # 1. Check if the index already exists to save time
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # 2. If not, build it from the hardcoded 'resume.pdf'
    if os.path.exists("resume.pdf"):
        pdf_reader = PdfReader("resume.pdf")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    else:
        return None

# Load the DB immediately
vector_store = get_vector_store()

# --- SIDEBAR (PROFILE INFO) ---
with st.sidebar:
    st.title("👨‍💻 Saisrikar Boddupalli")
    st.markdown("### Data Engineer | Python | SQL")
    
    st.markdown("---")
    st.markdown("**📍 Location:** Buffalo, NY")
    st.markdown("**📧 Email:** saisrikarboddupalli@gmail.com")
    st.markdown("[LinkedIn Profile](https://linkedin.com/in/sai-srikar-boddupalli)")
    
    st.markdown("---")
    st.info("This agent is powered by Llama-3 (Groq) and RAG technology.")

# --- MAIN CHAT INTERFACE ---
st.title("🚀 Chat with My Resume")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if prompt := st.chat_input("Ask me about my Smart City project..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate Answer
    if vector_store:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieval
                docs = vector_store.similarity_search(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Generation
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile"
                )
                
                system_prompt = f"""
                You are Saisrikar's professional AI representative. 
                Answer the question based ONLY on the context below. 
                Keep answers concise and professional. Use "I" statements (e.g., "I built...", "I worked on...").
                
                Context: {context}
                
                Question: {prompt}
                """
                
                response = llm.invoke(system_prompt)
                st.markdown(response.content)
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response.content})
    else:
        st.error("Please upload and process the resume in the sidebar first!")