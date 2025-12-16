# 🤖 Saisrikar's AI Digital Twin
### [🔴 Live Demo: Chat with My Resume](https://digital-twin-portfolio.streamlit.app/)

> *"Instead of reading a static PDF, interview my experience directly."*

This is an **Agentic RAG (Retrieval-Augmented Generation)** application that serves as an interactive portfolio. It ingests my professional resume and uses a Vector Database + LLM to answer questions about my work history, technical skills, and projects.

## 🛠️ Tech Stack (Free & Fast)
* **Engine:** Python 3.13 & Streamlit
* **Brain:** Llama-3-70b (via Groq LPU for <1s inference)
* **Memory:** FAISS Vector Store (Local CPU)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Orchestration:** LangChain

## 🚀 Key Features
* **Automatic Ingestion:** No file uploads required; the app auto-loads my resume on startup.
* **Semantic Search:** Finds relevant experience even if keywords don't match exactly.
* **Hallucination Guard:** The AI is instructed to answer *only* based on my factual resume data.

## 📂 Project Structure
```bash
├── app.py              # Main application logic (Streamlit + LangChain)
├── resume.pdf          # The source data (My Resume)
├── requirements.txt    # Dependency list for Cloud Deployment
└── README.md           # Documentation