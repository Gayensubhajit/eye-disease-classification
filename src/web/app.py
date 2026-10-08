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
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_name = "Apple Silicon (MPS)"
    else:
        gpu_name = "CPU (Host)"
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


@app.get("/api/analytics")
def get_analytics():
    """Return precomputed clinical error analysis and performance metrics."""
    metrics_file = PROJECT_ROOT / "outputs" / "clinical_error_analysis_data.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Analytics data not found")
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/download/paper-package")
def download_paper_package():
    """Download 1-click Overleaf ready-to-import paper package zip."""
    pkg_file = PROJECT_ROOT / "outputs" / "Overleaf_Paper_Package.zip"
    if not pkg_file.exists():
        raise HTTPException(status_code=404, detail="Paper package not found")
    return FileResponse(pkg_file, filename="Overleaf_Paper_Package.zip", media_type="application/zip")


@app.get("/api/samples")
def get_samples():
    """List curated fundus clinical sample cases."""
    if not SAMPLES_MANIFEST.exists():
        return []
    with open(SAMPLES_MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)


async def _resolve_image(file: Optional[UploadFile], sample_id: Optional[str]) -> np.ndarray:
    """Helper to load and decode RGB image from file upload or sample ID."""
    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file uploaded")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif sample_id:
        if not SAMPLES_MANIFEST.exists():
            raise HTTPException(status_code=404, detail="Samples manifest not found")
        with open(SAMPLES_MANIFEST, "r", encoding="utf-8") as f:
            samples = json.load(f)
        matched = next((s for s in samples if s["id"] == sample_id), None)
        if not matched:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
        p = Path(matched["file_path"])
        if not p.exists():
            p = STATIC_DIR / "samples" / p.name
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Sample image file not found for '{sample_id}'")
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise HTTPException(status_code=400, detail=f"Failed to load sample image for '{sample_id}'")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise HTTPException(status_code=400, detail="Either an image file or sample_id must be provided.")


@app.post("/api/predict")
async def run_prediction(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    benchmark: str = Form("10class"),
    model_id: Optional[str] = Form(None),
    generate_cam: bool = Form(True)
):
    """Execute clinical classification and Grad-CAM explainability for a single eye."""
    rgb_image = await _resolve_image(file, sample_id)

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


@app.post("/api/predict-bilateral")
async def run_predict_bilateral(
    od_file: Optional[UploadFile] = File(None),
    od_sample_id: Optional[str] = Form(None),
    os_file: Optional[UploadFile] = File(None),
    os_sample_id: Optional[str] = Form(None),
    benchmark: str = Form("10class"),
    model_id: Optional[str] = Form(None),
    generate_cam: bool = Form(True)
):
    """Execute bilateral comparative retinal screening (OD vs OS) with Asymmetry Index."""
    try:
        od_rgb = await _resolve_image(od_file, od_sample_id)
    except HTTPException as e:
        raise HTTPException(status_code=400, detail=f"Right Eye (OD) Error: {e.detail}")

    try:
        os_rgb = await _resolve_image(os_file, os_sample_id)
    except HTTPException as e:
        raise HTTPException(status_code=400, detail=f"Left Eye (OS) Error: {e.detail}")

    try:
        res_od = engine.predict(
            rgb_image=od_rgb,
            benchmark=benchmark,
            model_id=model_id,
            generate_cam=generate_cam
        )
        res_os = engine.predict(
            rgb_image=os_rgb,
            benchmark=benchmark,
            model_id=model_id,
            generate_cam=generate_cam
        )

        # Bilateral Asymmetry Index (BAI) computation
        od_probs = {d["class_name"]: d["probability"] for d in res_od["distribution"]}
        os_probs = {d["class_name"]: d["probability"] for d in res_os["distribution"]}
        class_keys = list(od_probs.keys())
        total_variation = 0.5 * sum(abs(od_probs[k] - os_probs.get(k, 0.0)) for k in class_keys)
        asymmetry_pct = round(total_variation * 100, 1)

        od_top = res_od["top_prediction"]["class_name"]
        os_top = res_os["top_prediction"]["class_name"]
        od_short = res_od["top_prediction"]["short_name"]
        os_short = res_os["top_prediction"]["short_name"]
        concordant = (od_top == os_top)

        if concordant:
            if od_top == "Healthy":
                clinical_category = "Bilateral Physiological Normal"
                clinical_summary = "Both eyes (OD & OS) demonstrate normal retinal architecture with physiological optic nerve cupping and no microvascular anomalies."
                recommended_action = "Routine annual preventive screening schedule recommended."
                severity = "low"
                badge_color = "emerald"
            else:
                clinical_category = f"Bilateral Concordant ({od_short})"
                clinical_summary = f"Symmetrical pathology detected across both fundi ({od_short}). Consistent presentation indicates systemic disease requiring concurrent systemic workup and bilateral retinal monitoring."
                recommended_action = f"Immediate comprehensive bilateral management for {od_short}."
                severity = "medium"
                badge_color = "amber"
        else:
            clinical_category = f"Discordant Unilateral Asymmetry (OD: {od_short} vs OS: {os_short})"
            clinical_summary = f"High inter-ocular discrepancy (Asymmetry: {asymmetry_pct}%). Right eye presents with {od_short} while left eye exhibits {os_short}. Asymmetric presentations frequently indicate acute unilateral pathologies (e.g., focal detachment, CSCR, or branch occlusion)."
            recommended_action = "Urgent targeted ophthalmologic examination focusing on the acutely affected eye, supplemented by OCT imaging."
            severity = "high"
            badge_color = "rose"

        return JSONResponse(content={
            "od": res_od,
            "os": res_os,
            "bilateral_analysis": {
                "concordant": concordant,
                "asymmetry_score": round(float(total_variation), 4),
                "asymmetry_percentage": asymmetry_pct,
                "od_top": od_short,
                "os_top": os_short,
                "od_confidence": res_od["top_prediction"]["percentage"],
                "os_confidence": res_os["top_prediction"]["percentage"],
                "clinical_category": clinical_category,
                "clinical_summary": clinical_summary,
                "recommended_action": recommended_action,
                "severity": severity,
                "badge_color": badge_color
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
