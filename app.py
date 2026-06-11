# ╔══════════════════════════════════════════════════════════════╗
# ║  Sai Srikar Boddupalli — Digital Twin Portfolio V4          ║
# ║  AI/ML Engineer · Data Systems                              ║
# ║  Theme: Midnight Royale (Navy & Champagne Gold)             ║
# ╚══════════════════════════════════════════════════════════════╝

import streamlit as st
import time, os, json, smtplib, re, datetime, requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# ── 1. INIT & ROUTING (STEALTH MODE) ───────────────────────────
load_dotenv()
st.set_page_config(
    page_title="Sai Srikar | AI/ML Engineer",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

target_company = st.query_params.get("company", "")
company_context = f" I am highly interested in bringing my skills to {target_company.title()}." if target_company else ""
is_founder_mode = st.query_params.get("mode", "") == "founder"

# ── 2. SESSION STATE ───────────────────────────────────────────
for k, v in {
    "messages": [], "feedback": {}, "follow_ups": [],
    "admin_auth": False, "jd_result": None, "visitor_counted": False,
}.items():
    if k not in st.session_state: st.session_state[k] = v

ANALYTICS_FILE = "analytics.log"
CONTACTS_FILE  = "contacts.json"
VISITOR_FILE   = "visitor_count.json"

# ── 3. HELPER FUNCTIONS & ANALYTICS ────────────────────────────
def get_visitor_count():
    try:
        if os.path.exists(VISITOR_FILE):
            with open(VISITOR_FILE) as f: return json.load(f).get("count", 0)
    except: pass
    return 0

def increment_visitor():
    if not st.session_state.visitor_counted:
        count = get_visitor_count() + 1
        with open(VISITOR_FILE, "w") as f: json.dump({"count": count}, f)
        st.session_state.visitor_counted = True
        log_analytics("page_view", f"Company Param: {target_company} | Founder Mode: {is_founder_mode}")

def log_analytics(event, content, feedback=None):
    entry = {"ts": datetime.datetime.now().isoformat(), "event": event, "content": str(content)[:300], "feedback": feedback}
    with open(ANALYTICS_FILE, "a") as f: f.write(json.dumps(entry) + "\n")

def save_contact(name, email, message):
    contacts = []
    if os.path.exists(CONTACTS_FILE):
        try:
            with open(CONTACTS_FILE) as f: contacts = json.load(f)
        except: pass
    contacts.append({"ts": datetime.datetime.now().isoformat(), "name": name, "email": email, "message": message})
    with open(CONTACTS_FILE, "w") as f: json.dump(contacts, f, indent=2)

def send_email(name, sender_email, message):
    try:
        gmail_user, gmail_pass, receiver = os.getenv("GMAIL_SENDER"), os.getenv("GMAIL_APP_PASSWORD"), os.getenv("GMAIL_RECEIVER")
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = f"Portfolio Contact: {name}", gmail_user, receiver
        html = f"""<h2 style="color:#D4AF37">New Portfolio Contact</h2>
        <p><b>Name:</b> {name}</p><p><b>Email:</b> {sender_email}</p><p><b>Message:</b><br>{message}</p>"""
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gmail_user, gmail_pass)
            s.send_message(msg)
        return True
    except Exception as e: return str(e)

# ── 4. KNOWLEDGE BASE PIPELINE ─────────────────────────────────
@st.cache_data(ttl=3600) # Reduced cache time to prevent stale data
def fetch_resume_data():
    return """
    Sai Srikar Boddupalli | New York, NY | (716) 986-4873 | saisrikarboddupalli@gmail.com

    PROFESSIONAL SUMMARY:
    AI & ML Engineer with 5+ years of experience bridging scalable data infrastructure and advanced predictive modeling. Expertise in designing robust machine learning pipelines, integrating LLMs, and training deep learning and gradient algorithms (PyTorch, XGBoost) to drive outcomes in fintech and healthcare. Strong foundation in Big Data (Spark, Kafka) and MLOps (Docker, Kubernetes, AWS, Azure), enabling full lifecycle model development from feature engineering and real time data ingestion to production deployment.

    EXPERIENCE:
    AI / ML Engineer | RS Technologies Inc | Remote | Dec 2025 - Present
    - Architected end-to-end Python-based machine learning pipelines orchestrated via Apache Airflow.
    - Developed automated feature extraction workflows using Pandas and NumPy.
    - Extended RESTful APIs to serve ML model predictions and data to third-party vendors across AWS and Azure.

    Machine Learning Engineer (Contract) | Tekly Studio LLC | Remote | Jan 2025 - Nov 2025
    - Spearheaded the integration of Large Language Models (LLMs) via PyTorch and Hugging Face.
    - Built real-time Python ingestion pipelines using Apache Kafka for cryptocurrency market data.
    - Engineered optimized, multi-tenant SQL Server schemas.
    - Managed end-to-end MLOps deployment using Docker and Kubernetes.

    Machine Learning Engineer | SelectFi | Buffalo, NY | Aug 2024 - Dec 2024
    - Developed and evaluated gradient boosting models (XGBoost, LightGBM) to predict optimal lender-to-borrower rate matching.
    - Engineered comprehensive credit risk feature pipelines.
    - Deployed automated, real-time credit decisioning logic as containerized microservices.

    Graduate Research Assistant (Applied ML) | University at Buffalo | Sep 2023 - Jul 2024
    - Architected analytical models and data pipelines utilizing SciPy and Scikit-learn.
    - Applied advanced statistical analysis and machine learning techniques for research publications.

    Data Engineer (Advanced Analytics) | Cognizant | Hyderabad, India | Jun 2021 - Aug 2023
    - Built the enterprise data foundation using Python, PySpark, and Azure Data Factory (ADF).
    - Optimized complex SQL queries and relational models in Microsoft SQL Server.
    - Enforced strict HIPAA-compliant protocols across the pipeline lifecycle.

    SKILLS:
    - Machine Learning & AI: Deep Learning (PyTorch, TensorFlow), LLMs, RAG, XGBoost, Scikit-learn, NLP
    - Languages & Libraries: Python, SQL, Java, C#, JavaScript, Pandas, NumPy, SciPy
    - Big Data & Engineering: Apache Spark, PySpark, Kafka, Airflow, Azure Data Factory, ETL/ELT
    - MLOps & Cloud: AWS, GCP, Azure, Docker, Kubernetes, CI/CD, REST APIs
    - Databases: Microsoft SQL Server, Vector Databases, Schema Design

    EDUCATION:
    - Master of Professional Studies - Computer Information Systems & Data Sciences, University at Buffalo
    - Bachelor of Engineering - Computer Science, Sri Chandrasekharendra Saraswathi Viswa Maha Vidhyalaya
    """

# ── 5. AI BACKEND ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text = fetch_resume_data()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_text(text)
    return FAISS.from_texts(chunks, embedding=embeddings)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0.2)

vector_store = get_vector_store()
llm = get_llm()
increment_visitor()

# ── 6. ADMIN DASHBOARD ─────────────────────────────────────────
if st.query_params.get("admin", "") == "true":
    st.markdown("## 📊 Admin Analytics")
    if not st.session_state.admin_auth:
        if st.button("Login", type="primary") and st.text_input("Password", type="password") == os.getenv("ADMIN_PASSWORD", ""):
            st.session_state.admin_auth = True
            st.rerun()
        st.stop()
    st.success("✅ Authenticated")
    
    analytics = load_analytics()
    jd_events = len([a for a in analytics if a["event"] == "jd_match"])
    questions = len([a for a in analytics if a["event"] == "question"])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("👁 Visitors", get_visitor_count())
    c2.metric("🎯 JD Engagements", jd_events)
    c3.metric("💬 Chat Queries", questions)

    st.markdown("---")
    st.subheader("📝 Activity Log")
    for a in reversed(analytics[-20:]):
        st.markdown(f"`{a.get('ts','')[:16]}` | **{a['event']}** | {a['content']}")
        
    if st.button("🚪 Logout"): st.session_state.admin_auth = False; st.rerun()
    st.stop()

# ── 7. MAIN CSS — MIDNIGHT ROYALE THEME ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0A1128; 
    --bg-card:   #121B33; 
    --bg-card2:  #1A2442;
    --border:    #2A3655; 
    --accent:    #D4AF37; 
    --accent-hi: #FDE047; 
    --accent-lo: rgba(212, 175, 55, 0.08); 
    --accent-md: rgba(212, 175, 55, 0.20);
    --text-1:    #F8FAFC; 
    --text-2:    #CBD5E1; 
    --text-3:    #94A3B8; 
    --radius:    8px;     
    --font-head: 'Playfair Display', serif;
    --font-body: 'Inter', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

html, body, .stApp { background-color: var(--bg) !important; font-family: var(--font-body); color: var(--text-1); }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
.stMainBlockContainer { max-width: 880px; margin: 0 auto; padding: 2rem 1.5rem 6rem; }
::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: var(--bg); } ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }

.fade-up { opacity:0; transform:translateY(20px); animation: fadeUp 0.65s ease forwards; }
.d1 { animation-delay:0.05s; } .d2 { animation-delay:0.15s; } .d3 { animation-delay:0.25s; }
.d4 { animation-delay:0.35s; } .d5 { animation-delay:0.45s; } .d6 { animation-delay:0.55s; }
@keyframes fadeUp { to { opacity:1; transform:translateY(0); } }

.hero { padding:5rem 0 3.5rem; text-align:center; }
.hero-name { font-family:var(--font-head); font-size:clamp(2.5rem,6vw,4rem); font-weight:600; color:var(--text-1); margin-bottom:0.5rem; letter-spacing: -0.5px;}
.hero-name .g { color:var(--accent); font-style: italic; } 
.hero-role { font-family:var(--mono); font-size:0.85rem; color:var(--accent); letter-spacing:0.2em; text-transform:uppercase; margin-bottom:0.8rem; }
.hero-sub { font-size:1.05rem; color:var(--text-2); margin-bottom:2.5rem; font-weight: 300; }
.hero-links { display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; margin-bottom:1.6rem; }
.hl { display:inline-flex; align-items:center; gap:5px; padding:8px 24px; border:1px solid var(--border); border-radius:4px; color:var(--text-2) !important; text-decoration:none !important; font-size:0.85rem; font-family:var(--mono); background:var(--bg-card); transition:all 0.3s ease; }
.hl:hover { border-color:var(--accent); color:var(--accent-hi) !important; background:var(--accent-lo); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }

.sl { font-family:var(--font-head); font-size:1.8rem; color:var(--text-1); font-weight:500; margin-bottom:1.5rem; margin-top:4.5rem; display:flex; align-items:center; gap:15px; }
.sl::after { content:''; flex:1; height:1px; background:linear-gradient(90deg, var(--accent) 0%, transparent 100%); opacity: 0.3; }

.sg { margin-bottom:1.5rem; } 
.sgt { font-size:0.7rem; color:var(--accent); font-weight:600; margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.15em; font-family:var(--mono); }
.sp { display:flex; flex-wrap:wrap; gap:0.5rem; } 
.pill { background:var(--bg-card); border:1px solid var(--border); color:var(--text-2); font-size:0.8rem; padding:6px 16px; border-radius:4px; font-family:var(--font-body); transition:all 0.2s; cursor:default; }
.pill:hover { border-color:var(--accent); color:var(--accent-hi); background:var(--accent-lo); } 

.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1.2rem; }
.pc { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:1.8rem; position:relative; overflow:hidden; transition:all 0.3s; }
.pc:hover { border-color:var(--accent); transform:translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
.pt { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em; color:var(--accent); font-weight:600; margin-bottom:0.8rem; font-family:var(--mono); } 
.pti { font-family:var(--font-head); font-size:1.3rem; font-weight:500; color:var(--text-1); margin-bottom:0.6rem; } 
.pd { font-size:0.85rem; color:var(--text-2); line-height:1.7; margin-bottom:1rem; opacity:0.9; } 
.pch { display:flex; flex-wrap:wrap; gap:0.4rem; } 
.ch { font-size:0.7rem; padding:3px 10px; border-radius:2px; background:var(--bg-card2); color:var(--text-3); font-family:var(--mono); border:1px solid var(--border); }

.tl { position:relative; padding-left:1.8rem; } .tl::before { content:''; position:absolute; left:0; top:8px; bottom:8px; width:1px; background:var(--border); } .ti { position:relative; padding-bottom:2.5rem; } .ti:last-child { padding-bottom:0; }
.td { position:absolute; left:-1.8rem; top:6px; width:9px; height:9px; border-radius:50%; background:var(--bg); border:2px solid var(--accent); transform:translateX(-4px); } 
.th { display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.2rem; margin-bottom:0.3rem; } 
.tt { font-family:var(--font-head); font-size:1.1rem; font-weight:600; color:var(--text-1); } 
.tda { font-size:0.75rem; color:var(--accent); font-family:var(--mono); } 
.tc { font-size:0.85rem; color:var(--text-2); margin-bottom:0.8rem; font-weight:400; font-family:var(--font-body); } 
.tb { list-style:none; padding:0; margin:0; } 
.tb li { font-size:0.85rem; color:var(--text-3); line-height:1.7; padding-left:1.2rem; position:relative; } 
.tb li::before { content:'—'; position:absolute; left:0; color:var(--accent); font-size:0.8rem; }

.widget-wrap { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:2rem; } 
.widget-h { font-family:var(--font-head); font-size:1.4rem; font-weight:500; color:var(--text-1); margin-bottom:0.5rem; } 
.widget-s { font-size:0.85rem; color:var(--text-2); margin-bottom:1.5rem; }

.score-box { display: flex; align-items: center; justify-content: center; gap: 1rem; padding:1.5rem; background:var(--bg-card2); border:1px solid var(--border); border-radius:4px; margin:1.5rem 0; } 
.score-num { font-family:var(--font-head); font-size:3.5rem; font-weight:600; line-height: 1; } 
.score-lbl { font-size:0.75rem; color:var(--text-3); text-transform:uppercase; letter-spacing:0.15em; font-family:var(--mono); }

[data-testid="stBottom"] { background-color:var(--bg) !important; border:none !important; }
.stChatInput textarea { background-color:var(--bg-card) !important; color:var(--text-1) !important; border:1px solid var(--border) !important; border-radius:4px !important; font-family:var(--font-body) !important; } 
.stChatInput textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 1px var(--accent) !important; } 
.stTextInput input, .stTextArea textarea { background-color:var(--bg-card2) !important; color:var(--text-1) !important; border:1px solid var(--border) !important; border-radius:4px !important; } 
.stTextInput input:focus, .stTextArea textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 1px var(--accent) !important; }
.stButton > button { background:var(--bg-card) !important; border:1px solid var(--border) !important; color:var(--text-2) !important; border-radius:4px !important; font-size:0.85rem !important; font-family:var(--mono) !important; transition:all 0.3s !important; } 
.stButton > button:hover { border-color:var(--accent) !important; color:var(--accent-hi) !important; background:var(--accent-lo) !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ── 8. HERO ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
hero_title = f"Engineering for <span class='g'>{target_company.title()}</span>" if target_company else "Sai Srikar <span class='g'>Boddupalli</span>"
hero_sub = "Ship faster. Scale harder. Own the product." if is_founder_mode else "Architecting production-grade AI systems, ML pipelines, and scalable cloud backends."

st.markdown(f"""
<div class="hero fade-up d1">
    <div class="hero-role">AI / ML Engineer &nbsp;·&nbsp; Data Systems</div>
    <div class="hero-name">{hero_title}</div>
    <div class="hero-sub">{hero_sub}</div>
    <div class="hero-links">
        <a class="hl" href="https://linkedin.com/in/sai-srikar-boddupalli" target="_blank">LinkedIn</a>
        <a class="hl" href="https://github.com/Sai-Srikar-Boddupalli" target="_blank">GitHub</a>
        <a class="hl" href="#connect">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ── 9. SKILLS & PROJECTS 
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="sl fade-up d2">Technical Arsenal</div>', unsafe_allow_html=True)
st.markdown("""
<div class="fade-up d2">
  <div class="sg"><div class="sgt">Languages</div><div class="sp"><span class="pill">Python</span><span class="pill">SQL</span><span class="pill">Java</span><span class="pill">C#</span><span class="pill">JavaScript</span></div></div>
  <div class="sg"><div class="sgt">AI & Machine Learning</div><div class="sp"><span class="pill">PyTorch</span><span class="pill">TensorFlow</span><span class="pill">LLMs & RAG</span><span class="pill">XGBoost</span></div></div>
  <div class="sg"><div class="sgt">Big Data & Cloud</div><div class="sp"><span class="pill">Apache Spark</span><span class="pill">Kafka</span><span class="pill">Azure</span><span class="pill">AWS</span><span class="pill">Docker</span></div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sl fade-up d3">Featured Engineering</div>', unsafe_allow_html=True)
st.markdown("""
<div class="grid2 fade-up d3">
  <div class="pc"><div class="pt">Production System</div><div class="pti">High-Frequency Trading Data</div><div class="pd">Built real-time Python ingestion pipelines using Apache Kafka for cryptocurrency market data, ensuring low-latency forecasting.</div><div class="pch"><span class="ch">Python</span><span class="ch">Kafka</span><span class="ch">SQL</span></div></div>
  <div class="pc"><div class="pt">Architecture</div><div class="pti">Digital Twin RAG Agent</div><div class="pd">Designed a robust Retrieval-Augmented Generation (RAG) LLM application embedded with professional vector data.</div><div class="pch"><span class="ch">FAISS</span><span class="ch">LangChain</span><span class="ch">LLMOps</span></div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sl fade-up d4">Professional History</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tl fade-up d4">
  <div class="ti"><div class="td"></div><div class="th"><div class="tt">AI / ML Engineer</div><div class="tda">Dec 2025 → Present</div></div><div class="tc">RS Technologies Inc</div><ul class="tb"><li>Architected end-to-end Python ML pipelines orchestrated via Apache Airflow.</li><li>Extended RESTful APIs to serve ML model predictions across AWS and Azure.</li></ul></div>
  <div class="ti"><div class="td"></div><div class="th"><div class="tt">Machine Learning Engineer</div><div class="tda">Jan 2025 → Nov 2025</div></div><div class="tc">Tekly Studio LLC</div><ul class="tb"><li>Spearheaded LLM integration via PyTorch and Hugging Face.</li><li>Built real-time Python ingestion pipelines using Apache Kafka.</li></ul></div>
  <div class="ti"><div class="td"></div><div class="th"><div class="tt">Machine Learning Engineer</div><div class="tda">Aug 2024 → Dec 2024</div></div><div class="tc">SelectFi</div><ul class="tb"><li>Developed gradient boosting models (XGBoost, LightGBM) for rate matching.</li><li>Deployed automated, real-time credit decisioning logic as microservices.</li></ul></div>
  <div class="ti"><div class="td"></div><div class="th"><div class="tt">Data Engineer</div><div class="tda">Jun 2021 → Aug 2023</div></div><div class="tc">Cognizant</div><ul class="tb"><li>Built enterprise data foundation using Python, PySpark, and Azure Data Factory.</li><li>Optimized complex SQL queries and relational models in MS SQL Server.</li></ul></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ── 10. JD ANALYSIS TOOL ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="sl fade-up d5">Alignment Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="widget-wrap fade-up d5"><div class="widget-h">Job Description Matcher</div><div class="widget-s">Paste a JD. My agent will evaluate it against my live data and calculate an alignment score.</div></div>', unsafe_allow_html=True)

jd_text = st.text_area("Job Description", placeholder="Paste the target job description...", height=160, label_visibility="collapsed")
if st.button("⚡ Execute Analysis", use_container_width=True):
    if jd_text.strip() and vector_store:
        with st.spinner("Analyzing alignment vectors..."):
            docs = vector_store.similarity_search(jd_text, k=6)
            resume_ctx = "\n\n".join(d.page_content for d in docs)
            match_prompt = f"""Evaluate this candidate against the JD.
CANDIDATE DATA: {resume_ctx}
JOB DESCRIPTION: {jd_text}

Respond EXACTLY in this format, with no conversational filler:
MATCH_SCORE: [Number from 0-100]
SUMMARY: [2-3 sentences evaluating the overall architectural and skill fit]
STRENGTHS:
- [Key strength 1]
- [Key strength 2]
- [Key strength 3]
GAPS:
- [Identified gap or missing requirement, or 'None explicitly identified']
"""
            try:
                res = llm.invoke([HumanMessage(content=match_prompt)])
                st.session_state.jd_result = res.content
                log_analytics("jd_match", jd_text[:120])
            except Exception as e: st.error(f"Analysis failed: {e}")

if st.session_state.jd_result:
    result_text = st.session_state.jd_result
    score = 0
    try:
        score_line = [l for l in result_text.split('\n') if l.startswith('MATCH_SCORE:')][0]
        score = int(re.search(r'\d+', score_line).group())
    except: pass
    
    color = "#D4AF37" if score >= 75 else "#CBD5E1" if score >= 50 else "#94A3B8"
    st.markdown(f'<div class="score-box"><div class="score-num" style="color:{color}">{score}%</div><div><div class="score-lbl">Alignment Score</div></div></div>', unsafe_allow_html=True)
    
    clean_text = result_text.replace(f"MATCH_SCORE: {score}", "").replace(f"MATCH_SCORE:{score}", "").strip()
    st.markdown(f"<div style='background: var(--bg-card2); padding: 1.5rem; border-radius: 4px; border: 1px solid var(--border);'>{clean_text}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ── 11. CHAT & CONNECT ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="sl fade-up d6">Connect With Me</div><div id="connect"></div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.markdown('<div class="widget-wrap fade-up d6" style="height: 100%;"><div class="widget-h">Direct Inquiry</div><div class="widget-s">Initiate a conversation regarding roles or architecture consulting.</div>', unsafe_allow_html=True)
    ct_name = st.text_input("Name", placeholder="Your name")
    ct_email = st.text_input("Email", placeholder="your@email.com")
    ct_msg = st.text_area("Message", placeholder="How can we collaborate?", height=120)
    if st.button("✉ Send Message", use_container_width=True):
        if ct_name and ct_email and ct_msg:
            with st.spinner("Routing message..."):
                save_contact(ct_name, ct_email, ct_msg)
                
                # Capture the actual result from the email function
                email_status = send_email(ct_name, ct_email, ct_msg)
                
                if email_status is True:
                    st.success("Message secured and delivered.")
                else:
                    # Print the exact Google SMTP error to the screen
                    st.error(f"Email routing failed: {email_status}")
        else:
            st.warning("All fields are required.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="widget-wrap fade-up d6" style="height: 100%;"><div class="widget-h">Interactive AI Twin</div><div class="widget-s">Query my knowledge base directly. Grounded on my professional data.</div>', unsafe_allow_html=True)
    
    chat_container = st.container(height=250)
    with chat_container:
        for msg in st.session_state.messages:
            st.markdown(f"**{'You' if msg['role'] == 'user' else 'Sai'}:** {msg['content']}")
            
    prompt = st.chat_input("Ask about my architecture experience...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_prompt = st.session_state.messages[-1]["content"]
        with chat_container:
            with st.spinner("Processing query..."):
                ctx = "\n".join(d.page_content for d in vector_store.similarity_search(last_prompt, k=4)) if vector_store else ""
                
                # STRICT FIRST-PERSON SYSTEM PROMPT
                persona = "You are a highly ambitious, product-focused Founding Engineer. Emphasize speed to market, shipping fast, and absolute ownership." if is_founder_mode else "You are a Senior Data / ML Engineer. Emphasize scalable architecture, robust data pipelines, and enterprise-grade stability."
                
                sys_msg = SystemMessage(content=f"""
                You are Sai Srikar Boddupalli. 
                CRITICAL INSTRUCTION: You must answer strictly in the FIRST PERSON ("I", "my", "me"). 
                Never refer to Sai Srikar in the third person ("He", "Sai", "His"). 
                {company_context}
                {persona}
                Answer confidently and professionally using ONLY this context: {ctx}
                """)
                
                try:
                    res = llm.invoke([sys_msg, HumanMessage(content=last_prompt)])
                    st.session_state.messages.append({"role": "assistant", "content": res.content})
                    st.rerun()
                except Exception as e: st.error(str(e))
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ── 12. FOOTER ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align:center; padding:3rem 0 1rem; margin-top: 4rem; border-top: 1px solid var(--border);">
    <div style="font-family: var(--mono); color: var(--accent); font-size: 0.8rem; margin-bottom: 0.5rem;">
        {get_visitor_count():,} SECURE SESSIONS
    </div>
    <div style="color: var(--text-3); font-size: 0.85rem;">
        Engineered with Python · Streamlit · LangChain
    </div>
</div>
""", unsafe_allow_html=True)