"""FastAPI Application for Clinical Retinal Screening Web Studio.
Serves interactive REST endpoints, model inference, and static web assets.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.web.inference_engine import InferenceEngine, MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "src" / "web" / "static"
SAMPLES_MANIFEST = STATIC_DIR / "samples" / "samples.json"

app = FastAPI(
    title="Retinal Disease Clinical Screening Studio",
    description="Automated multi-architecture retinal screening system with Grad-CAM visual explainability.",
    version="1.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine lazily
engine = InferenceEngine()

# Mount static files (CSS, JS, sample images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
DOCS_DIR = PROJECT_ROOT / "docs"
app.mount("/figures", StaticFiles(directory=str(DOCS_DIR / "figures")), name="figures")

@app.get("/presentation")
def get_presentation():
    """Serve supervisor executive briefing presentation."""
    presentation_file = DOCS_DIR / "presentation" / "index.html"
    if not presentation_file.exists():
        raise HTTPException(status_code=404, detail="Presentation not found")
    return FileResponse(presentation_file)



@app.get("/")
def get_index():
    """Serve main interactive clinical studio page."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index HTML not found")
    return FileResponse(index_file)


@app.get("/api/health")
def get_health():
    """System telemetry, GPU hardware profile, and memory metrics."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Host)"
    vram_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2) if cuda_avail else 0
    vram_res_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2) if cuda_avail else 0

    return {
        "status": "healthy",
        "device": str(engine.device),
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_allocated_mb": vram_alloc_mb,
        "vram_reserved_mb": vram_res_mb,
        "active_model": engine.active_model_key[1] if engine.active_model_key else "None"
    }


@app.get("/api/models")
def get_models():
    """Retrieve all available benchmarks, models, and published accuracy milestones."""
    return MODEL_REGISTRY


@app.get("/api/samples")
def get_samples():
    """List curated fundus clinical sample cases."""
    if not SAMPLES_MANIFEST.exists():
        return []
    with open(SAMPLES_MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/predict")
async def run_prediction(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    benchmark: str = Form("10class"),
    model_id: Optional[str] = Form(None),
    generate_cam: bool = Form(True)
):
    """Execute clinical classification and Grad-CAM explainability."""
    rgb_image = None

    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file uploaded")
        rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif sample_id:
        if not SAMPLES_MANIFEST.exists():
            raise HTTPException(status_code=404, detail="Samples manifest not found")
        with open(SAMPLES_MANIFEST, "r", encoding="utf-8") as f:
            samples = json.load(f)
        matched = next((s for s in samples if s["id"] == sample_id), None)
        if not matched or not Path(matched["file_path"]).exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
        bgr = cv2.imread(matched["file_path"])
        rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise HTTPException(status_code=400, detail="Either an image file or sample_id must be provided.")

    try:
        results = engine.predict(
            rgb_image=rgb_image,
            benchmark=benchmark,
            model_id=model_id,
            generate_cam=generate_cam
        )
        return JSONResponse(content=results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
