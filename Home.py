import streamlit as st

st.set_page_config(page_title="Early Alzheimer's Detection", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    /* Hide Streamlit Header */
    header { visibility: hidden; }
    
    /* Main background */
    .stApp { 
        background-color: #0a0e1a; 
        color: #f1f5f9; 
        background-image: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0a0e1a 70%);
    }
    
    /* Main Project Title */
    .main-title { 
        font-size: 4.5rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        text-align: center; 
        margin-top: 40px;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    
    .subtitle { 
        text-align: center; 
        color: #94a3b8; 
        font-size: 1.4rem; 
        margin-bottom: 60px; 
        font-weight: 400;
    }
    
    /* Image Card (Purple/Blue Theme) */
    .card-img { 
        background: rgba(26, 34, 53, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 24px; 
        padding: 50px 40px; 
        text-align: center; 
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    .card-img:hover { 
        border-color: #7c3aed; 
        box-shadow: 0 0 50px rgba(124,58,237,0.3), 0 20px 40px rgba(0,0,0,0.6); 
        transform: translateY(-10px); 
    }
    
    /* Free Speech Card (Teal/Cyan Theme) */
    .card-speech { 
        background: rgba(26, 34, 53, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 24px; 
        padding: 50px 40px; 
        text-align: center; 
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    .card-speech:hover { 
        border-color: #06b6d4; 
        box-shadow: 0 0 50px rgba(6, 182, 212, 0.3), 0 20px 40px rgba(0,0,0,0.6); 
        transform: translateY(-10px); 
    }
    
    .card-icon {
        font-size: 5rem;
        margin-bottom: 20px;
        display: inline-block;
        filter: drop-shadow(0 0 20px rgba(167, 139, 250, 0.5));
    }
    
    .card-img h2 { color: #a78bfa; font-size: 2.2rem; margin-bottom: 15px; font-weight: 700; }
    .card-speech h2 { color: #06b6d4; font-size: 2.2rem; margin-bottom: 15px; font-weight: 700; }
    
    .card-img p, .card-speech p { 
        color: #94a3b8; 
        font-size: 1.15rem; 
        line-height: 1.7;
        margin-bottom: 30px;
    }
    
    /* Button Styling - Purple for Image */
    .stButton > button:first-child {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.2rem !important;
        transition: all 0.3s !important;
        width: 100%;
        box-shadow: 0 8px 20px rgba(124,58,237,0.4) !important;
    }
    
    /* Button Styling - Teal for Speech */
    .stButton > button:last-child {
        background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.2rem !important;
        transition: all 0.3s !important;
        width: 100%;
        box-shadow: 0 8px 20px rgba(6,182,212,0.4) !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.03) !important;
    }
    
    /* Team Badge */
    .team-badge {
        text-align: center;
        margin-top: 80px;
        color: #64748b;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 4px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer-text {
        text-align: center; 
        color: #475569; 
        font-size: 0.9rem; 
        margin-top: 30px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='main-title'>Early Alzheimer's Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Cognitive Assessment via Speech Analysis</div>", unsafe_allow_html=True)

# Two-column layout for the options
col1, col2 = st.columns(2, gap="large")

# --- Option 1: Image Test ---
with col1:
    st.markdown("""
    <div class="card-img">
        <div class="card-icon">🏞️</div>
        <h2>Image Description</h2>
        <p>Standard clinical test using the Cookie Theft picture. The patient describes the scene to assess cognitive recall, vocabulary, and speech pauses.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Image Test →", key="btn_image", use_container_width=True):
        st.switch_page("pages/1_Image_Test.py")

# --- Option 2: Free Speech Test ---
with col2:
    st.markdown("""
    <div class="card-speech">
        <div class="card-icon">🎙️</div>
        <h2>Free Speech</h2>
        <p>Open-ended speech analysis. The patient speaks freely about any topic to assess fluency, cognitive load, and natural pauses without visual prompts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Free Speech Test →", key="btn_free", use_container_width=True):
        st.switch_page("pages/2_Free_Speech_Test.py")

# Footer / Team Name
st.markdown("<div class='team-badge'>Powered by Team MediSquad</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    ⚕️ <strong>Medical Disclaimer:</strong> This tool is an AI research prototype and should not be used as a substitute for professional medical diagnosis or clinical assessment.
</div>
""", unsafe_allow_html=True)