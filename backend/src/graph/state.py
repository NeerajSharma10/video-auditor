import operator
from typing import Annotated, List, Dict, Optional, Any
from typing_extensions import TypedDict

class ComplianceIssue(TypedDict):
    category: str           # e.g., "FTC_DISCLOSURE", "Misleading Claims"
    description: str        # Detail of the violation
    severity: str           # "CRITICAL" | "WARNING"
    timestamp: Optional[str]# Timestamp of occurrence if detected

class VideoAuditState(TypedDict):
    """
    Defines the data schema for the LangGraph execution context.
    """
    # --- Input Parameters ---
    video_url: str
    video_id: str

    # --- Ingestion & Extraction Data ---
    local_file_path: Optional[str]
    video_metadata: Dict[str, Any]
    transcript: Optional[str]
    ocr_text: List[str]

    # --- Analysis Output ---
    compliance_results: Annotated[List[ComplianceIssue], operator.add]

    # --- Final Deliverables ---
    final_status: str               # "PASS" | "FAIL"
    final_report: str               # Summary of the audit report

    # --- Observability ---
    errors: Annotated[List[str], operator.add]
