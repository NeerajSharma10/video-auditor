"""
LangGraph Nodes for Brand Guardian Compliance QA Pipeline.

Uses 100% Free & Local components:
- Local Video Indexer: Whisper transcription + RapidOCR
- Local Knowledge Base: ChromaDB with local ONNX MiniLM embeddings
- Local LLM: Ollama (gemma3:1b, llama3.2, etc.)
"""

import os
import json
import logging
import re
from typing import Dict, Any, List

from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.graph.state import VideoAuditState, ComplianceIssue
from backend.src.services.video_indexer import LocalVideoIndexerService

logger = logging.getLogger("brand-guardian-nodes")

# --- NODE 1: THE INDEXER (Local Whisper + OCR) ---
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Downloads YouTube video, extracts speech-to-text with faster-whisper,
    and extracts on-screen text with RapidOCR locally.
    """
    video_url = state.get("video_url")
    video_id = state.get("video_id", "vid_local")

    logger.info(f"--- [Node: Indexer] Processing: {video_url} ---")

    temp_filename = f"temp_{video_id}.mp4"

    try:
        service = LocalVideoIndexerService()
        data = service.process_video(video_url, output_path=temp_filename)

        logger.info(f"--- [Node: Indexer] Completed: transcript length={len(data.get('transcript', ''))}, OCR items={len(data.get('ocr_text', []))} ---")
        return data

    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "transcript": "",
            "ocr_text": []
        }

# --- NODE 2: THE COMPLIANCE AUDITOR (Local ChromaDB + Ollama) ---
def audit_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Performs RAG query against local ChromaDB and audits content with local Ollama LLM.
    """
    logger.info("--- [Node: Auditor] Querying Local ChromaDB & Ollama ---")

    transcript = state.get("transcript", "")
    ocr_text = state.get("ocr_text", [])

    if not transcript and not ocr_text:
        logger.warning("Neither transcript nor OCR text available. Skipping audit.")
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (No transcript or OCR extracted)."
        }

    # 1. Initialize Local Chroma Vector Store
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "backend/data/chroma_db")
    if not os.path.isabs(persist_dir):
        persist_dir = os.path.join(project_root, persist_dir)

    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "compliance_rules")

    chroma_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    class LocalChromaEmbeddings:
        def __init__(self, ef):
            self.ef = ef
        def embed_documents(self, texts):
            return self.ef(texts)
        def embed_query(self, text):
            return self.ef([text])[0]

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=LocalChromaEmbeddings(chroma_embedding_fn)
    )

    # 2. Similarity Search for Regulatory Rules
    query_text = f"{transcript[:500]} {' '.join(ocr_text[:10])}".strip()
    if not query_text:
        query_text = "social media influencer disclosures FTC advertising compliance"

    docs = vector_store.similarity_search(query_text, k=3)
    retrieved_rules = "\n\n".join([f"[{d.metadata.get('source', 'Rule')}]: {d.page_content}" for d in docs])

    # 3. Initialize Local Ollama LLM
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:1b")

    logger.info(f"Auditing using Ollama model '{ollama_model}' at {ollama_base_url}...")

    llm = ChatOllama(
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=0.0
    )

    system_prompt = f"""You are a Senior Brand Compliance Auditor.

OFFICIAL REGULATORY RULES:
{retrieved_rules}

INSTRUCTIONS:
1. Analyze the video transcript and on-screen text (OCR).
2. Compare against the official regulatory rules (FTC influencer disclosures, advertising standards, misleading claims).
3. Identify ANY violations or required disclosures missing.
4. Output STRICT JSON only. Format:
{{
  "compliance_results": [
    {{
      "category": "FTC Disclosure",
      "severity": "CRITICAL",
      "description": "Explanation of violation..."
    }}
  ],
  "status": "FAIL",
  "final_report": "Summary explanation of findings."
}}

If no violations are found, return:
{{
  "compliance_results": [],
  "status": "PASS",
  "final_report": "Video complies with all evaluated regulatory standards."
}}
"""

    user_message = f"""
VIDEO METADATA: {state.get('video_metadata', {})}
TRANSCRIPT: {transcript if transcript else 'No speech detected.'}
ON-SCREEN TEXT (OCR): {ocr_text if ocr_text else 'No on-screen text detected.'}
"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])

        content = response.content.strip()

        # Handle Markdown code block formatting if present
        if "```" in content:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        # Parse JSON
        audit_data = json.loads(content)

        results = audit_data.get("compliance_results", [])
        status = audit_data.get("status", "FAIL" if results else "PASS")
        report = audit_data.get("final_report", "Audit completed.")

        return {
            "compliance_results": results,
            "final_status": status,
            "final_report": report
        }

    except Exception as e:
        logger.error(f"Error in Auditor Node: {e}")
        return {
            "errors": [f"Auditor node error: {str(e)}"],
            "final_status": "FAIL",
            "final_report": f"Audit execution encountered an error: {str(e)}"
        }
