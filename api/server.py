"""
Cascade API: Leaf vs Non-Leaf gatekeeper, then crop-specific disease prediction.
Run from project root: uvicorn api.server:app --reload
"""
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "export"
STATIC_DIR = Path(__file__).resolve().parent / "static"

LEAF_MODEL_PATH = EXPORT_DIR / "leaf_vs_nonleaf_model.pt"
LEAF_CLASS_NAMES_PATH = EXPORT_DIR / "leaf_vs_nonleaf_class_names.json"

# Crop id -> (model state path, class names path)
CROPS = {
    "tomato": (EXPORT_DIR / "tomato_disease_model.pt", EXPORT_DIR / "tomato_class_names.json"),
    "maize": (EXPORT_DIR / "maize_disease_model.pt", EXPORT_DIR / "maize_class_names.json"),
    "potato": (EXPORT_DIR / "potato_disease_model.pt", EXPORT_DIR / "potato_class_names.json"),
}

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAF_CONFIDENCE_THRESHOLD = 0.9

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Crop Disease API (Cascade)", version="2.0.0")

leaf_model = None
leaf_class_names = None
disease_models = {}
disease_class_names = {}


def load_models():
    global leaf_model, leaf_class_names, disease_models, disease_class_names

    if not LEAF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Leaf model not found: {LEAF_MODEL_PATH}")
    with open(LEAF_CLASS_NAMES_PATH, "r") as f:
        leaf_class_names = json.load(f)
    n_leaf = len(leaf_class_names)
    leaf_model = models.resnet18(weights=None)
    leaf_model.fc = nn.Linear(leaf_model.fc.in_features, n_leaf)
    leaf_model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=device))
    leaf_model = leaf_model.to(device)
    leaf_model.eval()

    for crop_id, (model_path, names_path) in CROPS.items():
        if not model_path.exists() or not names_path.exists():
            continue
        with open(names_path, "r") as f:
            names = json.load(f)
        n = len(names)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        disease_models[crop_id] = model
        disease_class_names[crop_id] = names


@app.on_event("startup")
def startup():
    load_models()


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Crop Disease API (Cascade)", "predict": "POST /predict (file + crop)", "docs": "/docs"}


@app.get("/info")
def info():
    return {
        "message": "Crop Disease API (Cascade)",
        "pipeline": "Leaf vs Non-Leaf → Crop disease (if Leaf)",
        "check_leaf": "POST /check-leaf (leaf vs non-leaf only)",
        "predict": "POST /predict (file + crop); crop = tomato | maize | potato",
        "crops": list(disease_models.keys()),
        "leaf_classes": leaf_class_names or [],
        "disease_classes": {c: disease_class_names.get(c, []) for c in disease_models},
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
        raise HTTPException(status_code=503, detail="Model not loaded")
    img = await _read_image(file)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = leaf_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    pred_class = leaf_class_names[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = {leaf_class_names[i]: float(probs[i]) for i in range(len(leaf_class_names))}
    return JSONResponse(content={
        "class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    })


@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: str = Form(...)):
    """
    Cascade: 1) Leaf vs Non-Leaf. 2) If Leaf, run disease model for selected crop.
    Rejects with 400 if not classified as Leaf.
    """
    crop = crop.strip().lower()
    if crop not in disease_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop '{crop}'. Choose from: {list(disease_models.keys())}",
        )

    if leaf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    img = await _read_image(file)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        leaf_logits = leaf_model(img_tensor)
        leaf_probs = torch.softmax(leaf_logits, dim=1).cpu().numpy()[0]
    leaf_pred_idx = int(leaf_probs.argmax())
    leaf_pred_class = leaf_class_names[leaf_pred_idx]
    leaf_confidence = float(leaf_probs[leaf_pred_idx])

    if leaf_pred_class != "Leaf":
        raise HTTPException(
            status_code=400,
            detail="Image is not classified as a plant leaf. Please upload a clear leaf photo.",
        )
    if leaf_confidence < LEAF_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail="Leaf confidence too low. Please upload a clear, close-up leaf photo.",
        )

    disease_model = disease_models[crop]
    class_names = disease_class_names[crop]
    with torch.no_grad():
        logits = disease_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return JSONResponse(content={
        "status": "success",
        "crop": crop,
        "leaf_check": {"class": leaf_pred_class, "confidence": round(leaf_confidence, 4)},
        "class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    })


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
