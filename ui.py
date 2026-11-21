"""
Streamlit Web UI for Brand Guardian AI (100% Free & Local).

Run locally with:
    uv run streamlit run ui.py
"""

import os
import json
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from backend.src.graph.workflow import app as compliance_graph

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Brand Guardian — Video Compliance Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-pass {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        display: inline-block;
        border-left: 6px solid #31C48D;
    }
    .status-fail {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        display: inline-block;
        border-left: 6px solid #F05252;
    }
    .violation-card {
        background: #FFF5F5;
        border-left: 4px solid #E53E3E;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 6px;
    }
    .badge-critical {
        background-color: #E53E3E;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #DD6B20;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: SYSTEM HEALTH & SETTINGS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("System Status")

    # Check Ollama status
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if r.status_code == 200:
            st.success(f"● Ollama: Connected ({ollama_model})")
        else:
            st.warning(f"▲ Ollama: HTTP {r.status_code}")
    except Exception:
        st.error("✕ Ollama: Offline")

    # Check Vector DB
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "backend/data/chroma_db")
    if os.path.exists(chroma_dir):
        st.success(f"● ChromaDB: Active (`{chroma_dir}`)")
    else:
        st.warning("▲ ChromaDB: Index not found")

    st.info("● Speech-to-Text: `faster-whisper` (Local CPU)")
    st.info("● OCR Engine: `RapidOCR` (Local CPU)")

    st.markdown("---")
    st.subheader("💡 Sample Videos")
    if st.button("Sample 1: Endorsement Ad (Neutrogena)"):
        st.session_state["video_input"] = "https://youtu.be/dT7S75eYhcQ"

    st.markdown("---")
    st.caption("100% Free & Local • LangGraph • Ollama • ChromaDB")

# --- MAIN CONTENT ---
st.markdown('<div class="main-header">🛡️ Brand Guardian AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Video Compliance Auditor against FTC Guidelines & YouTube Advertising Specifications.</div>', unsafe_allow_html=True)

# Input Row
default_url = st.session_state.get("video_input", "https://youtu.be/dT7S75eYhcQ")
col1, col2 = st.columns([4, 1])
with col1:
    video_url = st.text_input("Enter YouTube Video URL or Local Video Path:", value=default_url)
with col2:
    st.write("")
    st.write("")
    run_audit = st.button("🚀 Audit Video", type="primary", use_container_width=True)

# Display Video Preview
if video_url:
    with st.expander("📺 Video Preview", expanded=False):
        try:
            st.video(video_url)
        except Exception:
            st.info("Video preview unavailable for this URL.")

# --- EXECUTION FLOW ---
if run_audit and video_url:
    session_id = str(uuid.uuid4())
    video_id = f"vid_{session_id[:8]}"

    initial_inputs = {
        "video_url": video_url,
        "video_id": video_id,
        "compliance_results": [],
        "errors": []
    }

    with st.status("🔍 Auditing Video Content...", expanded=True) as status_box:
        st.write("📥 Step 1/4: Downloading video with `yt-dlp`...")
        st.write("🎙️ Step 2/4: Transcribing speech with `faster-whisper` & extracting on-screen text with `RapidOCR`...")
        st.write("📚 Step 3/4: Querying regulatory rulebook in local `ChromaDB`...")
        st.write(f"🧠 Step 4/4: Evaluating compliance violations with `Ollama` (`{ollama_model}`)...")

        try:
            final_state = compliance_graph.invoke(initial_inputs)
            status_box.update(label="✅ Audit Complete!", state="complete", expanded=False)
            st.session_state["last_audit"] = final_state
        except Exception as e:
            status_box.update(label="❌ Audit Failed!", state="error", expanded=True)
            st.error(f"Execution Error: {str(e)}")
            st.stop()

# --- DISPLAY AUDIT RESULTS ---
if "last_audit" in st.session_state:
    state = st.session_state["last_audit"]
    audit_status = state.get("final_status", "UNKNOWN")
    violations = state.get("compliance_results", [])
    metadata = state.get("video_metadata", {})
    transcript = state.get("transcript", "")
    ocr_text = state.get("ocr_text", [])
    report = state.get("final_report", "No report available.")

    st.markdown("---")
    st.subheader("📊 Audit Results")

    # Status Banner
    col_status, col_meta1, col_meta2, col_meta3 = st.columns([2, 1, 1, 1])
    with col_status:
        if audit_status == "PASS":
            st.markdown('<div class="status-pass">✅ STATUS: PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-fail">❌ STATUS: FAIL ({len(violations)} Violation{"s" if len(violations) != 1 else ""})</div>', unsafe_allow_html=True)

    with col_meta1:
        st.metric("Video Duration", f"{metadata.get('duration', 0)}s")
    with col_meta2:
        st.metric("Resolution", metadata.get("resolution", "N/A"))
    with col_meta3:
        st.metric("OCR Detections", f"{len(ocr_text)} items")

    # Tabs for detailed review
    tab_violations, tab_report, tab_extracted, tab_json = st.tabs([
        "⚠️ Violations Detected",
        "📝 Executive Summary",
        "📜 Extracted Data (Speech & OCR)",
        "⚙️ Raw State (JSON)"
    ])

    with tab_violations:
        if violations:
            for issue in violations:
                cat = issue.get("category", "General Compliance")
                sev = issue.get("severity", "WARNING").upper()
                desc = issue.get("description", "No description provided.")
                badge_class = "badge-critical" if sev in ["CRITICAL", "HIGH"] else "badge-warning"

                st.markdown(f"""
                <div class="violation-card">
                    <span class="{badge_class}">{sev}</span> <b>{cat}</b><br>
                    <p style="margin-top: 8px; color: #1F2937;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No compliance violations detected! The video meets all evaluated regulatory criteria.")

    with tab_report:
        st.markdown("#### Auditor Report")
        st.write(report)

    with tab_extracted:
        col_tr, col_ocr = st.columns(2)
        with col_tr:
            st.markdown("##### 🎙️ Audio Transcript")
            if transcript:
                st.text_area("Transcript", transcript, height=250, disabled=True)
            else:
                st.info("No speech detected in this video.")
        with col_ocr:
            st.markdown("##### 🔍 On-Screen Text (OCR)")
            if ocr_text:
                st.write(f"Detected **{len(ocr_text)}** unique on-screen text snippets:")
                st.code("\n".join(ocr_text), language="text")
            else:
                st.info("No on-screen text recognized.")

    with tab_json:
        st.json(state)
