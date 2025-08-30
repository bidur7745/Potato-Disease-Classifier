import torch, numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model import LeafNet

def _tfm(size=224):
    return A.Compose([
        A.Resize(size,size),
        A.CenterCrop(size,size),
        A.Normalize(),
        ToTensorV2()
    ])


def predict(image_path: str):
    ckpt = torch.load("leafnet_best.pt", map_location="cpu")
    classes = ckpt["classes"]

    model = LeafNet(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = _tfm()(image=np.array(img))["image"].unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(1).squeeze(0).numpy()

    idx = int(probs.argmax())
    return {
        "label": classes[idx],
        "confidence": float(probs[idx]),
        "probs": {c: float(p) for c, p in zip(classes, probs)}
    }
