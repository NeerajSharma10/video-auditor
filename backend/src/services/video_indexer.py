"""
Local Video Indexer Service.

Replaces Azure Video Indexer completely with 100% free, local tools:
1. Video download: yt-dlp
2. Speech-to-text (Transcription): faster-whisper (CPU, int8)
3. On-screen text (OCR): OpenCV keyframe sampling + RapidOCR (ONNX, CPU)
"""

import os
import glob
import logging
import yt_dlp
import cv2

# Fix PyAV >= 14 incompatibility with faster-whisper (removed metadata_errors argument)
import av
_orig_av_open = av.open
def _safe_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)
av.open = _safe_av_open

from faster_whisper import WhisperModel
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger("local-video-indexer")

class LocalVideoIndexerService:
    def __init__(self, whisper_model_size="base.en", ocr_sample_interval_sec=3):
        self.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", whisper_model_size)
        self.whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.ocr_sample_interval_sec = int(os.getenv("OCR_SAMPLE_RATE_SEC", str(ocr_sample_interval_sec)))
        
        # Lazy initialization
        self._whisper_model = None
        self._ocr_engine = None

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            logger.info(f"Initializing faster-whisper ({self.whisper_model_size}) on {self.whisper_device}...")
            self._whisper_model = WhisperModel(
                self.whisper_model_size,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type
            )
        return self._whisper_model

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            logger.info("Initializing RapidOCR engine...")
            self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def download_youtube_video(self, url: str, output_path: str = "temp_audit_video.mp4") -> str:
        """
        Downloads a YouTube video using yt-dlp.
        """
        logger.info(f"Downloading YouTube video: {url}")
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if not os.path.exists(output_path):
                # In case yt-dlp added extension
                matches = glob.glob(f"{os.path.splitext(output_path)[0]}*")
                if matches:
                    output_path = matches[0]
            logger.info(f"Download complete: {output_path}")
            return output_path
        except Exception as e:
            raise RuntimeError(f"YouTube Download Failed: {str(e)}")

    def transcribe_audio(self, video_path: str) -> str:
        """
        Extracts speech-to-text transcript using faster-whisper.
        """
        logger.info(f"Transcribing audio from: {video_path}...")
        segments, info = self.whisper_model.transcribe(
            video_path,
            beam_size=5,
            language="en",
            vad_filter=True
        )
        
        transcript_lines = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript_lines.append(text)
                
        full_transcript = " ".join(transcript_lines)
        logger.info(f"Transcription complete ({len(transcript_lines)} segments, {len(full_transcript)} chars).")
        return full_transcript

    def extract_ocr(self, video_path: str) -> list[str]:
        """
        Samples video frames every `ocr_sample_interval_sec` seconds and extracts
        on-screen text with RapidOCR.
        """
        logger.info(f"Extracting on-screen text (OCR) every {self.ocr_sample_interval_sec}s from: {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video file {video_path} for OCR extraction.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = int(fps * self.ocr_sample_interval_sec)
        
        ocr_lines = []
        seen_lines = set()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                try:
                    result, _ = self.ocr_engine(frame)
                    if result:
                        for box, text, score in result:
                            cleaned = text.strip()
                            # Filter short noise and duplicates
                            if len(cleaned) >= 3 and cleaned.lower() not in seen_lines:
                                seen_lines.add(cleaned.lower())
                                ocr_lines.append(cleaned)
                except Exception as e:
                    logger.debug(f"OCR frame error: {e}")

            frame_idx += 1

        cap.release()
        logger.info(f"OCR Extraction complete: found {len(ocr_lines)} unique text elements.")
        return ocr_lines

    def get_video_metadata(self, video_path: str) -> dict:
        """
        Extracts duration and resolution metadata using OpenCV.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"duration": 0, "platform": "youtube"}
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration_sec = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        return {
            "duration": round(duration_sec, 2),
            "resolution": f"{width}x{height}",
            "platform": "youtube"
        }

    def process_video(self, video_url: str, output_path: str = "temp_audit_video.mp4") -> dict:
        """
        Coordinates full local pipeline: download -> whisper transcription -> OCR -> metadata -> cleanup.
        """
        local_video = None
        try:
            if os.path.exists(video_url):
                # Direct local file passed
                local_video = video_url
            else:
                local_video = self.download_youtube_video(video_url, output_path=output_path)
            
            metadata = self.get_video_metadata(local_video)
            transcript = self.transcribe_audio(local_video)
            ocr_text = self.extract_ocr(local_video)

            return {
                "transcript": transcript,
                "ocr_text": ocr_text,
                "video_metadata": metadata
            }
        finally:
            # Clean up temporary download file if created
            if local_video and local_video != video_url and os.path.exists(local_video):
                try:
                    os.remove(local_video)
                    logger.info(f"Cleaned up temporary video: {local_video}")
                except Exception as e:
                    logger.warning(f"Failed to remove {local_video}: {e}")
