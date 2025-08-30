from fastapi import FastAPI, UploadFile, File
import tempfile, shutil, os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from src.infer import predict

app = FastAPI(title="Potato Disease Classifier", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("api/static/index.html", "r") as f:
        return f.read()

@app.get("/labels")
def labels():
    import torch
    ckpt = torch.load("leafnet_best.pt", map_location="cpu")
    return {"classes": ckpt["classes"]}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        return predict(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
