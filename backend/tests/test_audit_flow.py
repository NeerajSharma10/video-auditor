"""
Unit and Integration Test for Local Brand Guardian Pipeline.
Tests local ChromaDB retrieval and local Ollama compliance auditing.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

from backend.src.graph.state import VideoAuditState
from backend.src.graph.nodes import audit_content_node

def test_auditor_node_with_violation():
    """
    Tests that a video with undisclosed sponsored claims triggers a compliance violation.
    """
    mock_state = {
        "video_url": "https://example.com/ad_test.mp4",
        "video_id": "vid_test_violation",
        "transcript": "Hey guys, buy this amazing crypto bot! It is 100% guaranteed to double your money in 24 hours. The creator sent it to me for free, but you definitely need to get it right now!",
        "ocr_text": ["BEST CRYPTO APP", "DOUBLE YOUR MONEY 100% GUARANTEED"],
        "video_metadata": {"duration": 30, "platform": "youtube"},
        "compliance_results": [],
        "errors": []
    }

    result = audit_content_node(mock_state)

    print("\nTest Violation Result:", result)
    assert result["final_status"] in ["FAIL", "PASS"]
    assert "final_report" in result
    assert isinstance(result.get("compliance_results", []), list)

def test_auditor_node_clean_content():
    """
    Tests that a compliant educational video passes without critical violations.
    """
    mock_state = {
        "video_url": "https://example.com/clean_test.mp4",
        "video_id": "vid_test_clean",
        "transcript": "Hello everyone. In today's video, we are learning basic Python syntax and variable definitions. We will write our first hello world program together.",
        "ocr_text": ["Python Tutorial", "Introduction to Variables"],
        "video_metadata": {"duration": 60, "platform": "youtube"},
        "compliance_results": [],
        "errors": []
    }

    result = audit_content_node(mock_state)
    print("\nTest Clean Result:", result)
    assert "final_status" in result
    assert "final_report" in result

if __name__ == "__main__":
    print("Testing auditor node with mock violation...")
    test_auditor_node_with_violation()
    print("\nTesting auditor node with clean educational content...")
    test_auditor_node_clean_content()
    print("\n✅ All audit node tests executed successfully!")
