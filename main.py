"""
Main Execution Entry Point for Brand Guardian AI (100% Free & Local).

Runs the video compliance audit workflow end-to-end:
1. Initializes video audit session
2. Downloads and transcribes video locally (Whisper + RapidOCR)
3. Queries local ChromaDB for regulatory rules
4. Audits compliance via local Ollama LLM
5. Prints full compliance audit report
"""

import sys
import uuid
import json
import logging
from dotenv import load_dotenv

# Load local environment variables
load_dotenv(override=True)

from backend.src.graph.workflow import app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("brand-guardian-runner")

def run_compliance_audit(video_url: str = "https://youtu.be/dT7S75eYhcQ"):
    """
    Executes a Video Compliance Audit request.
    """
    session_id = str(uuid.uuid4())
    video_id = f"vid_{session_id[:8]}"

    logger.info(f"Starting Local Audit Session: {session_id} for {video_url}")

    initial_inputs = {
        "video_url": video_url,
        "video_id": video_id,
        "compliance_results": [],
        "errors": []
    }

    print("\n" + "=" * 60)
    print(" 1. INITIALIZING LOCAL WORKFLOW ")
    print("=" * 60)
    print(f"Payload: {json.dumps(initial_inputs, indent=2)}")

    try:
        final_state = app.invoke(initial_inputs)

        print("\n" + "=" * 60)
        print(" 2. WORKFLOW EXECUTION COMPLETE ")
        print("=" * 60)

        print("\n=== COMPLIANCE AUDIT REPORT ===")
        print(f"Video ID:    {final_state.get('video_id')}")
        print(f"Status:      {final_state.get('final_status')}")

        metadata = final_state.get('video_metadata', {})
        if metadata:
            print(f"Duration:    {metadata.get('duration')}s")
            print(f"Resolution:  {metadata.get('resolution')}")

        print("\n[ VIOLATIONS DETECTED ]")
        results = final_state.get('compliance_results', [])
        if results:
            for issue in results:
                cat = issue.get('category', 'General')
                sev = issue.get('severity', 'WARNING')
                desc = issue.get('description', '')
                print(f"- [{sev}] {cat}: {desc}")
        else:
            print("No violations found. Video complies with rules.")

        print("\n[ AUDITOR SUMMARY ]")
        print(final_state.get('final_report', 'No report available.'))

        errors = final_state.get('errors', [])
        if errors:
            print("\n[ SYSTEM NOTICES / ERRORS ]")
            for err in errors:
                print(f"- {err}")

        print("\n" + "=" * 60)
        return final_state

    except Exception as e:
        logger.error(f"Workflow Execution Failed: {e}")
        raise e

if __name__ == "__main__":
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/dT7S75eYhcQ"
    run_compliance_audit(test_url)
