"""
Generates a modern, clean architecture diagram for Brand Guardian.
Highlights the 100% Free & Local tech stack (Whisper, RapidOCR, ChromaDB, Ollama, LangGraph).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Remove axes
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(8, 8.4, "Brand Guardian — 100% Free & Local Architecture", 
            ha='center', va='center', color='#FFFFFF', fontsize=20, weight='bold', fontfamily='sans-serif')
    ax.text(8, 8.0, "Fully Offline Multimodal Video Compliance Auditor with LangGraph, ChromaDB & Ollama", 
            ha='center', va='center', color='#94A3B8', fontsize=12, fontfamily='sans-serif')

    # Helper function for drawing rounded boxes
    def draw_card(x, y, w, h, bg_color, border_color, title, subtitle="", badge=""):
        box = patches.FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.08,rounding_size=0.15",
                                     linewidth=1.8, edgecolor=border_color, facecolor=bg_color)
        ax.add_patch(box)
        if badge:
            # Badge background
            badge_box = patches.FancyBboxPatch((x + 0.15, y + h - 0.35), len(badge)*0.11 + 0.2, 0.22,
                                               boxstyle="round,pad=0.02,rounding_size=0.06",
                                               linewidth=0, facecolor=border_color)
            ax.add_patch(badge_box)
            ax.text(x + 0.25 + len(badge)*0.05, y + h - 0.24, badge,
                    ha='center', va='center', color='#FFFFFF', fontsize=7.5, weight='bold', fontfamily='sans-serif')
            
        ax.text(x + w/2, y + h*0.62, title,
                ha='center', va='center', color='#F8FAFC', fontsize=11, weight='bold', fontfamily='sans-serif')
        if subtitle:
            ax.text(x + w/2, y + h*0.28, subtitle,
                    ha='center', va='center', color='#94A3B8', fontsize=8.5, fontfamily='sans-serif')

    def draw_arrow(x1, y1, x2, y2, color='#38BDF8', text=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", 
                                    color=color, lw=2.2, connectionstyle="arc3,rad=0"))
        if text:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.18
            ax.text(mid_x, mid_y, text, ha='center', va='center', color=color, fontsize=8, weight='bold')

    # --- SECTION 1: INGESTION ---
    # Container box
    c1 = patches.FancyBboxPatch((0.6, 1.8), 2.8, 5.5,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                linewidth=1.2, edgecolor='#334155', facecolor='#111827')
    ax.add_patch(c1)
    ax.text(2.0, 7.0, "1. INGESTION", ha='center', va='center', color='#38BDF8', fontsize=12, weight='bold')

    draw_card(0.85, 5.4, 2.3, 1.2, '#1E293B', '#38BDF8', "YouTube URL", "or Local Video MP4", "INPUT")
    draw_card(0.85, 3.6, 2.3, 1.2, '#1E293B', '#38BDF8', "yt-dlp Downloader", "Local Stream Caching", "CLI TOOL")
    draw_card(0.85, 2.0, 2.3, 1.1, '#1E293B', '#10B981', "Auto Cleanup", "Deletes temp MP4", "STORAGE")

    draw_arrow(2.0, 5.4, 2.0, 4.8)
    draw_arrow(2.0, 3.6, 2.0, 3.1)

    # --- SECTION 2: MULTIMODAL EXTRACTION ---
    c2 = patches.FancyBboxPatch((3.8, 1.8), 3.4, 5.5,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                linewidth=1.2, edgecolor='#334155', facecolor='#111827')
    ax.add_patch(c2)
    ax.text(5.5, 7.0, "2. LOCAL EXTRACTION", ha='center', va='center', color='#A855F7', fontsize=12, weight='bold')

    draw_card(4.05, 5.3, 2.9, 1.3, '#1E293B', '#A855F7', "faster-whisper", "base.en (INT8 on CPU)\nExtracts Speech to Text", "AUDIO STT")
    draw_card(4.05, 3.6, 2.9, 1.3, '#1E293B', '#A855F7', "RapidOCR + OpenCV", "Keyframe Sampling (3s)\nExtracts On-Screen Text", "VISUAL OCR")
    draw_card(4.05, 2.0, 2.9, 1.2, '#1E293B', '#A855F7', "Video Metadata", "Resolution, Duration & FPS", "METADATA")

    draw_arrow(3.15, 4.2, 3.8, 5.9, color='#A855F7', text="Audio")
    draw_arrow(3.15, 4.2, 3.8, 4.25, color='#A855F7', text="Frames")

    # --- SECTION 3: KNOWLEDGE BASE (RAG) ---
    c3 = patches.FancyBboxPatch((7.6, 1.8), 3.4, 5.5,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                linewidth=1.2, edgecolor='#334155', facecolor='#111827')
    ax.add_patch(c3)
    ax.text(9.3, 7.0, "3. LOCAL KNOWLEDGE BASE", ha='center', va='center', color='#F59E0B', fontsize=12, weight='bold')

    draw_card(7.85, 5.3, 2.9, 1.3, '#1E293B', '#F59E0B', "Regulatory PDFs", "FTC Influencer Guidelines\nYouTube Ad Specs", "DOCUMENTS")
    draw_card(7.85, 3.6, 2.9, 1.3, '#1E293B', '#F59E0B', "ONNX MiniLM", "all-MiniLM-L6-v2 (CPU)\nNo API Embedding Model", "EMBEDDINGS")
    draw_card(7.85, 2.0, 2.9, 1.2, '#1E293B', '#F59E0B', "ChromaDB", "Persistent Vector Store\nSimilarity Search (k=3)", "VECTOR DB")

    draw_arrow(9.3, 5.3, 9.3, 4.9)
    draw_arrow(9.3, 3.6, 9.3, 3.2)

    # --- SECTION 4: REASONING & ORCHESTRATION ---
    c4 = patches.FancyBboxPatch((11.4, 1.8), 4.0, 5.5,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                linewidth=1.2, edgecolor='#334155', facecolor='#111827')
    ax.add_patch(c4)
    ax.text(13.4, 7.0, "4. REASONING & OUTPUT", ha='center', va='center', color='#10B981', fontsize=12, weight='bold')

    draw_card(11.65, 5.3, 3.5, 1.3, '#1E293B', '#10B981', "LangGraph Workflow", "StateGraph Orchestration\nSTART -> Indexer -> Auditor", "DAG PIPELINE")
    draw_card(11.65, 3.6, 3.5, 1.3, '#1E293B', '#10B981', "Ollama Local LLM", "gemma3:1b / llama3.2:3b\nZero Cloud / Zero API Cost", "REASONING")
    draw_card(11.65, 2.0, 3.5, 1.2, '#1E293B', '#10B981', "Audit Deliverables", "PASS / FAIL + Severity Breakdown\nExecutive Summary JSON", "REPORT")

    draw_arrow(6.95, 5.9, 11.65, 5.9, color='#10B981', text="Transcript")
    draw_arrow(6.95, 4.2, 11.65, 4.2, color='#10B981', text="OCR Text")
    draw_arrow(10.75, 2.6, 11.65, 4.0, color='#F59E0B', text="Rules RAG")
    draw_arrow(13.4, 5.3, 13.4, 4.9)
    draw_arrow(13.4, 3.6, 13.4, 3.2)

    # --- BOTTOM SECTION: INTERFACES & OBSERVABILITY ---
    draw_card(1.0, 0.45, 3.5, 0.95, '#1E293B', '#0284C7', "Streamlit Web UI", "Visual dashboard with video player & metrics", "PORT 8501")
    draw_card(5.2, 0.45, 3.5, 0.95, '#1E293B', '#059669', "FastAPI + Swagger", "REST API with interactive /docs UI", "PORT 8000")
    draw_card(9.4, 0.45, 3.2, 0.95, '#1E293B', '#D97706', "LangSmith Tracing", "Optional real-time execution observability", "TELEMETRY")
    draw_card(13.0, 0.45, 2.4, 0.95, '#1E293B', '#E11D48', "100% Free & Local", "Runs on standard CPU", "ZERO COST")

    plt.tight_layout()
    output_path = "docs/architecture.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Diagram successfully generated at: {output_path}")

if __name__ == "__main__":
    draw_diagram()
