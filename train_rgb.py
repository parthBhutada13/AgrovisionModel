import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import AgroVision

# ─────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train AgroVision on EuroSAT dataset")
parser.add_argument(
    "--data",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"),
    help="Path to dataset root (ImageFolder format). Defaults to ./dataset/",
)
parser.add_argument("--epochs",     type=int,   default=30,    help="Number of training epochs")
parser.add_argument("--batch_size", type=int,   default=32,    help="Batch size")
parser.add_argument("--lr",         type=float, default=1e-4,  help="Learning rate")
parser.add_argument("--num_classes",type=int,   default=4,     help="Number of output classes")
parser.add_argument("--in_ch",      type=int,   default=3,     help="Input channels (3 for RGB)")
parser.add_argument("--save_path",  type=str,   default="agrovision_best.pt", help="Output model path")
args = parser.parse_args()

print(f"🚀 Training AgroVision | epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")
print(f"📂 Dataset: {args.data}")

# ─────────────────────────────────────────
# Device
# ─────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device: {DEVICE}")

# ─────────────────────────────────────────
# Transforms & Dataset
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.3444, 0.3803, 0.4078],
                         [0.2026, 0.1367, 0.1155]),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.3444, 0.3803, 0.4078],
                         [0.2026, 0.1367, 0.1155]),
])

full_dataset = datasets.ImageFolder(args.data)
train_size   = int(0.8 * len(full_dataset))
val_size     = len(full_dataset) - train_size

train_data, val_data = random_split(full_dataset, [train_size, val_size])

# Apply separate transforms
train_data.dataset.transform = train_transform
val_data.dataset.transform   = val_transform

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"📊 Train: {len(train_data)} | Val: {len(val_data)} | Classes: {full_dataset.classes}")

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
model = AgroVision(in_ch=args.in_ch, num_classes=args.num_classes).to(DEVICE)
print(f"🧠 AgroVision loaded | params: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────
# Loss, Optimizer, Scheduler
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, args.epochs + 1):
    # --- Train ---
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    train_acc = correct / total

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            val_loss    += loss.item()
            preds        = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step()

    print(
        f"Epoch [{epoch:02d}/{args.epochs}] "
        f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}% | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc*100:.2f}%"
    )

    # --- Save best model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "val_acc":          val_acc,
            "classes":          full_dataset.classes,
        }, args.save_path)
        print(f"  💾 Saved best model → {args.save_path} (val_acc={val_acc*100:.2f}%)")

print(f"\n✅ Training complete. Best val accuracy: {best_val_acc*100:.2f}%")