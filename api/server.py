"""
Maize disease prediction API with cascaded models.
1. Leaf vs Non-Leaf model (gatekeeper): rejects non-leaf images.
2. Maize disease model: runs only when image is classified as Leaf.
Run from project root: uvicorn api.server:app --reload
"""
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

# Paths relative to project root (run server from project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "export"
LEAF_VS_NONLEAF_MODEL_PATH = EXPORT_DIR / "leaf_vs_nonleaf_model.pt"
LEAF_VS_NONLEAF_CLASS_NAMES_PATH = EXPORT_DIR / "leaf_vs_nonleaf_class_names.json"
MAIZE_MODEL_PATH = EXPORT_DIR / "maize_disease_model.pt"
MAIZE_CLASS_NAMES_PATH = EXPORT_DIR / "maize_class_names.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAF_CONFIDENCE_THRESHOLD = 0.9
LOW_CONFIDENCE_THRESHOLD = 0.85
GREEN_MIN_RATIO = 0.18

def _is_green_dominant(img: Image.Image) -> bool:
    img = img.copy()
    img.thumbnail((80, 80), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    n = len(pixels)
    if n == 0:
        return False
    r = sum(p[0] for p in pixels) / n
    g = sum(p[1] for p in pixels) / n
    b = sum(p[2] for p in pixels) / n
    total = r + g + b
    if total <= 0:
        return False
    return (g / total) >= GREEN_MIN_RATIO

# Same transform as notebooks (resize + ImageNet normalize)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Maize Disease Prediction API (Cascaded)", version="2.0.0")

# Cascaded models: gatekeeper (leaf vs non-leaf) and disease classifier
leaf_model = None
leaf_class_names = None
disease_model = None
disease_class_names = None


def load_models():
    global leaf_model, leaf_class_names, disease_model, disease_class_names

    # 1. Leaf vs Non-Leaf (gatekeeper)
    if not LEAF_VS_NONLEAF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Leaf model not found: {LEAF_VS_NONLEAF_MODEL_PATH}")
    with open(LEAF_VS_NONLEAF_CLASS_NAMES_PATH, "r") as f:
        leaf_class_names = json.load(f)
    n_leaf = len(leaf_class_names)
    leaf_model = models.resnet18(weights=None)
    leaf_model.fc = nn.Linear(leaf_model.fc.in_features, n_leaf)
    leaf_model.load_state_dict(torch.load(LEAF_VS_NONLEAF_MODEL_PATH, map_location=device))
    leaf_model = leaf_model.to(device)
    leaf_model.eval()

    # 2. Maize disease classifier
    if not MAIZE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Disease model not found: {MAIZE_MODEL_PATH}")
    with open(MAIZE_CLASS_NAMES_PATH, "r") as f:
        disease_class_names = json.load(f)
    n_disease = len(disease_class_names)
    disease_model = models.resnet18(weights=None)
    disease_model.fc = nn.Linear(disease_model.fc.in_features, n_disease)
    disease_model.load_state_dict(torch.load(MAIZE_MODEL_PATH, map_location=device))
    disease_model = disease_model.to(device)
    disease_model.eval()


@app.on_event("startup")
def startup():
    load_models()


@app.get("/")
def root():
    """Serve upload page for browser; API info at /info."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "message": "Maize Disease Prediction API (Cascaded)",
        "docs": "/docs",
        "predict": "POST /predict with multipart form 'file' (image)",
    }


@app.get("/info")
def info():
    return {
        "message": "Maize Disease Prediction API (Cascaded)",
        "pipeline": "Leaf vs Non-Leaf → Maize Disease (only if Leaf)",
        "check_leaf": "POST /check-leaf (leaf vs non-leaf only)",
        "predict": "POST /predict (cascade: leaf check then disease)",
        "leaf_model": "export/leaf_vs_nonleaf_model.pt",
        "leaf_classes": leaf_class_names or [],
        "disease_model": "export/maize_disease_model.pt",
        "disease_classes": disease_class_names or [],
        "docs": "/docs",
    }


async def _read_image(file: UploadFile):
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed)}")
    contents = await file.read()
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")


@app.post("/check-leaf")
async def check_leaf(file: UploadFile = File(...)):
    if leaf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    img = await _read_image(file)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        leaf_logits = leaf_model(img_tensor)
        leaf_probs = torch.softmax(leaf_logits, dim=1).cpu().numpy()[0]
    pred_idx = int(leaf_probs.argmax())
    pred_class = leaf_class_names[pred_idx]
    confidence = float(leaf_probs[pred_idx])
    probabilities = {leaf_class_names[i]: float(leaf_probs[i]) for i in range(len(leaf_class_names))}
    return JSONResponse(content={
        "class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    })


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Cascaded prediction: first Leaf vs Non-Leaf; if Leaf, then Maize disease.
    Rejects with 400 if image is classified as non-leaf.
    """
    global leaf_model, leaf_class_names, disease_model, disease_class_names
    if leaf_model is None or disease_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    img = await _read_image(file)
    if not _is_green_dominant(img):
        raise HTTPException(
            status_code=400,
            detail="Image does not look like plant/leaf content. Please upload a clear photo of a maize leaf.",
        )
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Stage 1: Leaf vs Non-Leaf (gatekeeper)
    with torch.no_grad():
        leaf_logits = leaf_model(img_tensor)
        leaf_probs = torch.softmax(leaf_logits, dim=1).cpu().numpy()[0]
    leaf_pred_idx = int(leaf_probs.argmax())
    leaf_pred_class = leaf_class_names[leaf_pred_idx]
    leaf_confidence = float(leaf_probs[leaf_pred_idx])

    if leaf_pred_class != "Leaf":
        raise HTTPException(
            status_code=400,
            detail="Input image is not a plant leaf. Please upload a clear photo of a maize leaf.",
        )
    if leaf_confidence < LEAF_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail="Image could not be confidently recognized as a leaf. Please upload a clear, close-up photo of a maize leaf.",
        )

    # Stage 2: Maize disease (only if Leaf)
    with torch.no_grad():
        disease_logits = disease_model(img_tensor)
        probs = torch.softmax(disease_logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = disease_class_names[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = {disease_class_names[i]: float(probs[i]) for i in range(len(disease_class_names))}

    payload = {
        "status": "success",
        "class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        payload["low_confidence"] = True
        payload["warning"] = (
            "Prediction confidence is low. Please use a clear, close-up photo of a maize leaf."
        )

    return JSONResponse(content=payload)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
