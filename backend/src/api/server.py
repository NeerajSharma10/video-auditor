"""
FastAPI Server for Brand Guardian AI (Local & Free).

Endpoints:
- GET  /health: Verify server is running
- POST /audit:  Audit a video URL against local compliance rules
"""

import uuid
import logging
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(override=True)

from backend.src.api.telemetry import setup_telemetry
setup_telemetry()

from backend.src.graph.workflow import app as compliance_graph

logger = logging.getLogger("api-server")

tags_metadata = [
    {
        "name": "Audit",
        "description": "Endpoints for submitting videos to the LangGraph compliance auditing pipeline.",
    },
    {
        "name": "System",
        "description": "System health and operational status.",
    },
]

app = FastAPI(
    title="Brand Guardian AI API (Local & Free)",
    description="""
### 🛡️ Brand Guardian Video Compliance Auditor
Audits social media & video ads against official regulatory rules (FTC & YouTube Ad Guidelines).

**100% Free & Local Pipeline:**
- **Speech-to-Text**: `faster-whisper` (local CPU)
- **On-Screen OCR**: `RapidOCR` + `OpenCV` (local CPU)
- **Vector Knowledge Base**: `ChromaDB` + `ONNX MiniLM`
- **Auditor Reasoning**: `Ollama` (`gemma3:1b` / `llama3.2:3b`)
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
    openapi_url="/openapi.json"
)

class AuditRequest(BaseModel):
    video_url: str = Field(
        ...,
        description="Public YouTube video URL or local video path to audit",
        example="https://youtu.be/dT7S75eYhcQ"
    )

class ComplianceIssue(BaseModel):
    category: str = Field(..., description="Category of violation (e.g., 'FTC Disclosure', 'Misleading Claims')")
    severity: str = Field(..., description="Severity level: 'CRITICAL', 'WARNING', or 'LOW'")
    description: str = Field(..., description="Detailed explanation of the regulatory violation")

class AuditResponse(BaseModel):
    session_id: str = Field(..., description="Unique audit session UUID")
    video_id: str = Field(..., description="Trackable video identifier")
    status: str = Field(..., description="Overall compliance status: 'PASS' or 'FAIL'")
    final_report: str = Field(..., description="Auditor's summary assessment and remediation advice")
    compliance_results: List[ComplianceIssue] = Field(..., description="List of detected compliance violations")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "361ed768-9a1e-4337-837a-71291f8a646e",
                "video_id": "vid_361ed768",
                "status": "FAIL",
                "final_report": "The video contains a critical disclosure violation regarding the commercial endorsement...",
                "compliance_results": [
                    {
                        "category": "FTC Disclosure",
                        "severity": "CRITICAL",
                        "description": "Product branding displayed without prominent material connection disclosures."
                    }
                ]
            }
        }

@app.get("/health", tags=["System"], summary="Health Check")
def health_check():
    """Returns server and engine operational status."""
    return {
        "status": "healthy",
        "service": "Brand Guardian AI",
        "engine": "100% Local (Ollama + ChromaDB + Whisper)",
        "docs": "/docs"
    }

@app.post("/audit", response_model=AuditResponse, tags=["Audit"], summary="Audit Video for Compliance")
async def audit_video(request: AuditRequest):
    """
    Audits a video URL for compliance violations using local AI.
    """
    session_id = str(uuid.uuid4())
    video_id_short = f"vid_{session_id[:8]}"

    logger.info(f"Received Audit Request: {request.video_url} (Session: {session_id})")

    initial_inputs = {
        "video_url": request.video_url,
        "video_id": video_id_short,
        "compliance_results": [],
        "errors": []
    }

    try:
        final_state = compliance_graph.invoke(initial_inputs)

        return AuditResponse(
            session_id=session_id,
            video_id=final_state.get("video_id", video_id_short),
            status=final_state.get("final_status", "UNKNOWN"),
            final_report=final_state.get("final_report", "No report generated."),
            compliance_results=final_state.get("compliance_results", [])
        )

    except Exception as e:
        logger.error(f"Audit Failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow Execution Failed: {str(e)}"
        )
