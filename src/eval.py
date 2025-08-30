import torch, yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from src.dataset import AlbumentationsFolder, build_transforms
from src.model import LeafNet

def evaluate(cfg_path="configs/base.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ckpt = torch.load("leafnet_best.pt", map_location="cpu")
    classes = ckpt["classes"]

    model = LeafNet(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = AlbumentationsFolder(cfg["data"]["test_dir"], build_transforms(cfg["data"]["img_size"], False))
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(1).tolist())

    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
