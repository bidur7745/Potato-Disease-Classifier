import yaml, torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from sklearn.metrics import f1_score
from src.dataset import make_loaders
from src.model import LeafNet

def run(cfg_path="configs/base.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    torch.manual_seed(cfg["seed"])

    train_loader, val_loader, classes = make_loaders(**cfg["data"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LeafNet(num_classes=len(classes), dropout=cfg["model"]["dropout"], pretrained=cfg["model"]["pretrained"]).to(device)
    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    sch = CosineAnnealingLR(opt, T_max=cfg["optim"]["epochs"])

    best_f1, best_path = 0.0, Path("leafnet_best.pt")

    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                y_true.extend(y.tolist())
                y_pred.extend(logits.argmax(1).cpu().tolist())

        f1 = f1_score(y_true, y_pred, average="macro")
        sch.step()
        print(f"epoch {epoch+1}: macroF1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"state_dict": model.state_dict(), "classes": classes}, best_path)

    print(f"best macroF1={best_f1:.4f}, saved -> {best_path}")

if __name__ == "__main__":
    run()
