"""
SadTalker REST API
==================
A standalone FastAPI server that wraps SadTalker's GPU inference pipeline.
Accepts image + audio file uploads and returns a generated talking-head MP4 video.

Usage:
    uvicorn sadtalker_api:app --host 0.0.0.0 --port 7861
    
    Or simply run: run_api.bat
"""

import os
import sys
import glob
import uuid
import shutil
import threading
import tempfile

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# FFmpeg PATH injection (same logic as launcher.py)
# ---------------------------------------------------------------------------
def _inject_ffmpeg():
    """Inject the WinGet-installed FFmpeg binary directory into PATH."""
    try:
        winget_path = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft", "WinGet", "Packages"
        )
        ffmpeg_paths = glob.glob(os.path.join(winget_path, "Gyan.FFmpeg*", "*", "bin"))
        for path in ffmpeg_paths:
            if path not in os.environ["PATH"]:
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                print(f"[API] Injected FFmpeg path: {path}")
    except Exception as e:
        print(f"[API] Warning: Could not inject FFmpeg path: {e}")

_inject_ffmpeg()

# ---------------------------------------------------------------------------
# SadTalker model (lazy-loaded on first request)
# ---------------------------------------------------------------------------
from src.gradio_demo import SadTalker

_sad_talker: SadTalker | None = None
_gpu_lock = threading.Lock()


def _get_model() -> SadTalker:
    """Lazy-load the SadTalker model on first use."""
    global _sad_talker
    if _sad_talker is None:
        print("[API] Loading SadTalker model...")
        _sad_talker = SadTalker(
            checkpoint_path="checkpoints",
            config_path="src/config",
            lazy_load=True,
        )
        print(f"[API] Model loaded on device: {_sad_talker.device}")
    return _sad_talker


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app_description = """
### remote machine instructions

This API allows you to submit an image and an audio file to generate a talking-head video.

#### Quickstart with `curl`
```bash
curl -X POST "http://<SERVER_IP>:7861/generate" \\
  -F "image=@your_photo.png" \\
  -F "audio=@your_voice.wav" \\
  -F "use_enhancer=true" \\
  --output final_video.mp4
```

#### Python Example
```python
import requests

url = "http://<SERVER_IP>:7861/generate"
files = {
    "image": open("photo.png", "rb"),
    "audio": open("voice.wav", "rb")
}
data = {"use_enhancer": "true"}

resp = requests.post(url, files=files, data=data)
with open("result.mp4", "wb") as f:
    f.write(resp.content)
```
"""

app = FastAPI(
    title="SadTalker API",
    description=app_description,
    version="1.0.0",
)

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>SadTalker API Wrapper</title>
            <style>
                body { font-family: sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>SadTalker Inference API is Running!</h1>
            <p>Welcome to the SadTalker REST API. You can use this service to generate talking-head videos from any machine on your network.</p>
            <ul>
                <li><strong>Interactive Documentation:</strong> <a href="/docs">/docs (Swagger UI)</a></li>
                <li><strong>System Health / GPU Status:</strong> <a href="/health">/health</a></li>
            </ul>
            <p>Please visit the <a href="/docs">/docs</a> page for full endpoint schemas to see how to use the <code>/generate</code> endpoint.</p>
        </body>
    </html>
    """

@app.get("/health")
def health():
    """Return server and GPU status."""
    gpu_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": gpu_available,
        "cuda_version": torch.version.cuda if gpu_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "gpu_memory_mb": (
            torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if gpu_available else None
        ),
        "model_loaded": _sad_talker is not None,
    }


@app.post("/generate")
def generate(
    image: UploadFile = File(..., description="Source face image (PNG/JPG)"),
    audio: UploadFile = File(..., description="Driving audio (WAV/MP3)"),
    preprocess: str = Form("crop", description="crop | resize | full | extcrop | extfull"),
    still_mode: bool = Form(False, description="Fewer head motion (works best with 'full')"),
    use_enhancer: bool = Form(False, description="Use GFPGAN face enhancer"),
    size: int = Form(256, description="Face model resolution: 256 or 512"),
    pose_style: int = Form(0, description="Pose style index (0-46)"),
    exp_scale: float = Form(1.0, description="Expression scale"),
    batch_size: int = Form(2, description="Batch size for generation"),
):
    """
    Generate a talking-head video.

    Upload a face image and an audio file. The API will return the generated
    MP4 video as a downloadable file.
    """
    # ------------------------------------------------------------------
    # Save uploaded files to a temp directory
    # ------------------------------------------------------------------
    job_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(tempfile.gettempdir(), "sadtalker_api", job_id)
    os.makedirs(work_dir, exist_ok=True)

    image_ext = os.path.splitext(image.filename or "image.png")[1] or ".png"
    audio_ext = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"

    image_path = os.path.join(work_dir, f"input{image_ext}")
    audio_path = os.path.join(work_dir, f"input{audio_ext}")

    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save uploaded files: {e}")

    # ------------------------------------------------------------------
    # Run inference (serialized — one generation at a time)
    # ------------------------------------------------------------------
    acquired = _gpu_lock.acquire(timeout=300)  # Wait up to 5 minutes
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="GPU is busy. Another generation is in progress. Try again later.",
        )

    try:
        model = _get_model()
        result_dir = os.path.join(work_dir, "results")
        os.makedirs(result_dir, exist_ok=True)

        video_path = model.test(
            source_image=image_path,
            driven_audio=audio_path,
            preprocess=preprocess,
            still_mode=still_mode,
            use_enhancer=use_enhancer,
            batch_size=batch_size,
            size=size,
            pose_style=pose_style,
            exp_scale=exp_scale,
            result_dir=result_dir,
        )

        if video_path is None or not os.path.isfile(video_path):
            raise HTTPException(
                status_code=500,
                detail="Generation completed but no video file was produced.",
            )

        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"sadtalker_{job_id}.mp4",
        )

    except HTTPException:
        raise
    except AttributeError as e:
        if "No face" in str(e):
            raise HTTPException(
                status_code=422,
                detail="No face detected in the uploaded image. Please use a clear, front-facing photo.",
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        _gpu_lock.release()


# ---------------------------------------------------------------------------
# Direct launch support: python sadtalker_api.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("[API] Starting SadTalker API on http://0.0.0.0:7861")
    print("[API] Docs available at http://0.0.0.0:7861/docs")
    uvicorn.run(app, host="0.0.0.0", port=7861)
