"""
eval_4class.py
Full evaluation of AgroVision on the 4 agricultural classes:
  AnnualCrop | HerbaceousVegetation | Pasture | PermanentCrop

Outputs (saved as PNG in project directory):
  confusion_matrix_4class.png
  roc_curves_4class.png
  per_class_metrics_4class.png
  tsne_4class.png
  gradcam_4class.png
  attention_map_4class.png
  boundary_detection_4class.png
  spectral_profiles_4class.png
"""

import os, sys, random, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

from model import AgroVision

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
OUT_DIR      = BASE_DIR
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

TARGET_CLASSES = ["AnnualCrop", "HerbaceousVegetation", "Pasture", "PermanentCrop"]
NUM_CLASSES    = len(TARGET_CLASSES)
BATCH_SIZE     = 32

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device : {DEVICE}")
print(f"Classes: {TARGET_CLASSES}")
print(f"Output : {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────
# Dataset — filter to 4 classes
# ─────────────────────────────────────────────────────────────────
MEAN = [0.3444, 0.3803, 0.4078]
STD  = [0.2026, 0.1367, 0.1155]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

full_ds    = datasets.ImageFolder(DATASET_DIR, transform=transform)
cls2idx    = full_ds.class_to_idx
target_idx = [cls2idx[c] for c in TARGET_CLASSES if c in cls2idx]

if len(target_idx) != NUM_CLASSES:
    missing = [c for c in TARGET_CLASSES if c not in cls2idx]
    print(f"[WARN] Missing classes in dataset: {missing}")
    sys.exit(1)

label_remap = {old: new for new, old in enumerate(target_idx)}

class FourClassDataset(Dataset):
    def __init__(self, base_ds, indices, remap):
        self.base  = base_ds
        self.idx   = indices
        self.remap = remap
    def __len__(self):  return len(self.idx)
    def __getitem__(self, i):
        img, lbl = self.base[self.idx[i]]
        return img, self.remap[lbl]

filtered_idx = [i for i, (_, l) in enumerate(full_ds.samples) if l in target_idx]
four_ds      = FourClassDataset(full_ds, filtered_idx, label_remap)

n_val   = int(0.2 * len(four_ds))
n_train = len(four_ds) - n_val
_, val_ds = torch.utils.data.random_split(
    four_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

print(f"Val samples : {len(val_ds)} ({len(val_ds)//NUM_CLASSES} per class approx)")

# ─────────────────────────────────────────────────────────────────
# Load model — try best.pt then checkpoint.pt
# ─────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    m = AgroVision(in_ch=3, num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    sd   = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    md   = m.state_dict()
    fil  = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
    md.update(fil)
    m.load_state_dict(md)
    m.eval()
    return m, len(fil), len(md)

NESTED = os.path.join(BASE_DIR, "main_cv_withoutput")
ckpt_paths = [
    os.path.join(BASE_DIR,  "agrovision_best.pt"),
    os.path.join(BASE_DIR,  "agrovision_ms_checkpoint.pt"),
    os.path.join(NESTED,    "agrovision_ms_best.pt"),
    os.path.join(NESTED,    "agrovision_ms_checkpoint.pt"),
]
model = None
for cp in ckpt_paths:
    if os.path.exists(cp):
        model, n_loaded, n_total = load_model(cp)
        print(f"Loaded checkpoint: {os.path.basename(cp)}  ({n_loaded}/{n_total} layers matched)")
        break

if model is None:
    print("[ERROR] No checkpoint found. Train first with train_rgb.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# Inference pass — collect preds, probs, features
# ─────────────────────────────────────────────────────────────────
all_labels, all_preds, all_probs = [], [], []
feat_buf = []

def feat_hook_fn(m, inp, out):
    # out: (B, num_classes, 256) from cross_attn
    feat_buf.append(out.detach().cpu().float())

hook = model.cross_attn.register_forward_hook(feat_hook_fn)

with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(DEVICE)
        logits, _ = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_labels.append(lbls.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

hook.remove()

all_labels = np.concatenate(all_labels)
all_preds  = np.concatenate(all_preds)
all_probs  = np.concatenate(all_probs)

# Features for t-SNE: mean over class tokens → (N, 256)
all_feats = torch.cat(feat_buf, dim=0).numpy().mean(axis=1) if feat_buf else None

# ─────────────────────────────────────────────────────────────────
# Metrics summary
# ─────────────────────────────────────────────────────────────────
acc = (all_labels == all_preds).mean()
f1w = f1_score(all_labels, all_preds, average="weighted")
f1m = f1_score(all_labels, all_preds, average="macro")
prec_cls = precision_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
rec_cls  = recall_score   (all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
f1_cls   = f1_score       (all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

print("\n" + "="*60)
print(f"  Overall Accuracy : {acc*100:.2f}%")
print(f"  Weighted F1      : {f1w:.4f}")
print(f"  Macro F1         : {f1m:.4f}")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, zero_division=0))

# ─────────────────────────────────────────────────────────────────
# 1. Confusion Matrix (raw + normalised side-by-side)
# ─────────────────────────────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"Confusion Matrix  |  OA={acc*100:.2f}%   Weighted-F1={f1w:.4f}",
             fontsize=14, fontweight="bold", y=1.01)

for ax, data, cmap, fmt, title in [
    (ax1, cm,      "YlOrRd", lambda v: str(int(v)), "Raw Counts"),
    (ax2, cm_norm, "Blues",  lambda v: f"{v*100:.1f}%", "Recall per Class (%)"),
]:
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=data.max(), aspect="auto")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(TARGET_CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(TARGET_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    thresh = data.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, fmt(data[i, j]), ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black",
                    fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
out = os.path.join(OUT_DIR, "confusion_matrix_4class.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("[OK] confusion_matrix_4class.png")

# ─────────────────────────────────────────────────────────────────
# 2. ROC Curves
# ─────────────────────────────────────────────────────────────────
y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

fig, ax = plt.subplots(figsize=(9, 7))
aucs = []
for i, cls in enumerate(TARGET_CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    ax.plot(fpr, tpr, color=PALETTE[i], lw=2.2, label=f"{cls}  (AUC = {roc_auc:.3f})")

mean_auc = np.nanmean(aucs)
ax.plot([0,1],[0,1], "k--", lw=1.2, alpha=0.5, label="Random")
ax.fill_between([0,1],[0,1], alpha=0.05, color="grey")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(f"ROC Curves  |  Mean AUC = {mean_auc:.3f}", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves_4class.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] roc_curves_4class.png")

# ─────────────────────────────────────────────────────────────────
# 3. Per-Class Metrics Bar Chart
# ─────────────────────────────────────────────────────────────────
x, w = np.arange(NUM_CLASSES), 0.26
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w, prec_cls, w, label="Precision", color=PALETTE[0], alpha=0.88)
b2 = ax.bar(x,     rec_cls,  w, label="Recall",    color=PALETTE[1], alpha=0.88)
b3 = ax.bar(x + w, f1_cls,   w, label="F1-Score",  color=PALETTE[2], alpha=0.88)
ax.axhline(f1w, color="grey", ls="--", lw=1.5, label=f"Weighted F1 = {f1w:.3f}")
ax.set_xticks(x); ax.set_xticklabels(TARGET_CLASSES, fontsize=11)
ax.set_ylim(0, 1.18); ax.set_ylabel("Score", fontsize=12)
ax.set_title(f"Per-Class Metrics  |  Accuracy = {acc*100:.2f}%  |  Weighted-F1 = {f1w:.4f}",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9); ax.grid(axis="y", alpha=0.25)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.012,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_metrics_4class.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] per_class_metrics_4class.png")

# ─────────────────────────────────────────────────────────────────
# 4. t-SNE
# ─────────────────────────────────────────────────────────────────
if all_feats is not None and len(all_feats) >= 50:
    print("Computing t-SNE ...")
    n_samp  = min(1200, len(all_feats))
    samp_i  = np.random.choice(len(all_feats), n_samp, replace=False)
    feats_s = all_feats[samp_i]
    labs_s  = all_labels[samp_i]
    tsne    = TSNE(n_components=2, perplexity=40, max_iter=1200,
                   learning_rate="auto", init="pca", random_state=SEED)
    emb     = tsne.fit_transform(feats_s)

    fig, ax = plt.subplots(figsize=(10, 8))
    markers = ["o", "s", "^", "D"]
    for i, cls in enumerate(TARGET_CLASSES):
        mask = labs_s == i
        ax.scatter(emb[mask, 0], emb[mask, 1], s=22, alpha=0.75,
                   marker=markers[i], label=cls, color=PALETTE[i],
                   edgecolors="white", linewidths=0.3)
    ax.set_title("t-SNE Feature Embeddings — AgroVision 4-Class",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1", fontsize=11); ax.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2); ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_4class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] tsne_4class.png")
else:
    print("[WARN] Skipping t-SNE (no features captured)")

# ─────────────────────────────────────────────────────────────────
# 5. Grad-CAM
# ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.acts, self.grads = None, None
        self._h = [
            target_layer.register_forward_hook(self._fwd),
            target_layer.register_full_backward_hook(self._bwd),
        ]
    def _fwd(self, m, inp, out): self.acts = out.detach()
    def _bwd(self, m, gi, go):   self.grads = go[0].detach()
    def generate(self, x, cls_idx):
        model.zero_grad()
        logits, _ = model(x)
        logits[0, cls_idx].backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, (128, 128), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        denom = cam.max() - cam.min()
        return (cam - cam.min()) / denom if denom > 0 else cam
    def remove(self):
        for h in self._h: h.remove()

model.train()   # need grads
gc = GradCAM(model, model.fpn.smooth)

DENORM_MEAN = torch.tensor(MEAN).view(3,1,1)
DENORM_STD  = torch.tensor(STD).view(3,1,1)

# collect 2 samples per class from val_loader
samples = {i: [] for i in range(NUM_CLASSES)}
for imgs, lbls in val_loader:
    for img, lbl in zip(imgs, lbls):
        l = lbl.item()
        if len(samples[l]) < 2:
            samples[l].append(img.clone())
    if all(len(v) >= 2 for v in samples.values()):
        break

ROWS = NUM_CLASSES * 2
fig, axes = plt.subplots(ROWS, 3, figsize=(12, ROWS * 3.5))
fig.suptitle("AgroVision 4-Class — Grad-CAM Explainability\n"
             "False Colour  |  Grad-CAM Heatmap  |  Overlay",
             fontsize=12, fontweight="bold")

row = 0
for ci in range(NUM_CLASSES):
    for img in samples[ci]:
        x_in = img.unsqueeze(0).to(DEVICE)
        cam  = gc.generate(x_in, ci)

        img_d = (img * DENORM_STD + DENORM_MEAN).permute(1,2,0).numpy().clip(0,1)
        heat  = plt.cm.jet(cam)[:,:,:3]
        over  = 0.55 * img_d + 0.45 * heat

        with torch.no_grad():
            logits_d, _ = model(x_in)
        pred_c = logits_d.argmax(dim=1).item()
        ok = "✓" if pred_c == ci else "✗"

        axes[row, 0].imshow(img_d);                axes[row,0].set_title(f"GT: {TARGET_CLASSES[ci]}", fontsize=8)
        axes[row, 1].imshow(cam, cmap="jet");      axes[row,1].set_title("Grad-CAM", fontsize=8)
        axes[row, 2].imshow(over.clip(0,1));       axes[row,2].set_title(f"Pred: {TARGET_CLASSES[pred_c]} {ok}", fontsize=8)
        for a in axes[row]: a.axis("off")
        row += 1

gc.remove()
model.eval()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gradcam_4class.png"), dpi=120, bbox_inches="tight")
plt.close()
print("[OK] gradcam_4class.png")

# ─────────────────────────────────────────────────────────────────
# 6. Boundary Detection — Sobel on sample images
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(12, NUM_CLASSES * 3.2))
fig.suptitle("Field Boundary Detection (Sobel)  |  One sample per class",
             fontsize=12, fontweight="bold")

for ci in range(NUM_CLASSES):
    if samples[ci]:
        img  = samples[ci][0]
        img_d = (img * DENORM_STD + DENORM_MEAN).permute(1,2,0).numpy().clip(0,1)
        gray  = cv2.cvtColor((img_d * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag   = np.sqrt(sx**2 + sy**2)
        mag   = (mag / mag.max() * 255).astype(np.uint8)
        thresh = cv2.threshold(mag, 35, 255, cv2.THRESH_BINARY)[1]
        overlay = img_d.copy()
        overlay[thresh > 0] = [1, 0.2, 0]

        axes[ci, 0].imshow(img_d);               axes[ci,0].set_title(f"{TARGET_CLASSES[ci]}\nOriginal", fontsize=9)
        axes[ci, 1].imshow(mag, cmap="hot");     axes[ci,1].set_title("Sobel Magnitude", fontsize=9)
        axes[ci, 2].imshow(overlay.clip(0,1));   axes[ci,2].set_title("Boundary Overlay", fontsize=9)
        for a in axes[ci]: a.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boundary_detection_4class.png"), dpi=130, bbox_inches="tight")
plt.close()
print("[OK] boundary_detection_4class.png")

# ─────────────────────────────────────────────────────────────────
# 7. Spectral Reflectance Profiles — mean RGB per class
# ─────────────────────────────────────────────────────────────────
class_means = {i: [] for i in range(NUM_CLASSES)}

with torch.no_grad():
    for imgs, lbls in val_loader:
        for img, lbl in zip(imgs, lbls):
            l = lbl.item()
            img_d = (img * DENORM_STD + DENORM_MEAN).numpy().clip(0, 1)  # (3, H, W)
            class_means[l].append(img_d.reshape(3, -1).mean(axis=1))

fig, ax = plt.subplots(figsize=(10, 6))
band_names = ["Band R (Red)", "Band G (Green)", "Band B (Blue)"]
for ci, cls in enumerate(TARGET_CLASSES):
    arr  = np.array(class_means[ci])          # (N, 3)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    x    = np.arange(3)
    ax.plot(x, mean, "o-", color=PALETTE[ci], lw=2.2, label=cls, ms=7)
    ax.fill_between(x, mean - std, mean + std, color=PALETTE[ci], alpha=0.12)

ax.set_xticks(range(3)); ax.set_xticklabels(band_names, fontsize=11)
ax.set_ylabel("Mean Reflectance (normalised)", fontsize=12)
ax.set_title("Spectral Reflectance Profiles — 4 Agricultural Classes",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9); ax.grid(True, alpha=0.3)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spectral_profiles_4class.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] spectral_profiles_4class.png")

# ─────────────────────────────────────────────────────────────────
# 8. Confidence Distribution — violin / box per class
# ─────────────────────────────────────────────────────────────────
conf_by_class = {i: [] for i in range(NUM_CLASSES)}
for prob, lbl in zip(all_probs, all_labels):
    conf_by_class[lbl].append(prob.max())

fig, ax = plt.subplots(figsize=(11, 6))
data_vp  = [conf_by_class[i] for i in range(NUM_CLASSES)]
parts    = ax.violinplot(data_vp, showmedians=True, showmeans=False)
for pc, color in zip(parts["bodies"], PALETTE):
    pc.set_facecolor(color); pc.set_alpha(0.75)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
parts["cbars"].set_color("grey");   parts["cbars"].set_linewidth(1)
parts["cmins"].set_color("grey");   parts["cmaxes"].set_color("grey")

ax.set_xticks(range(1, NUM_CLASSES+1))
ax.set_xticklabels(TARGET_CLASSES, fontsize=10)
ax.set_ylabel("Max Softmax Confidence", fontsize=12)
ax.set_title("Prediction Confidence Distribution per Class",
             fontsize=13, fontweight="bold")
ax.axhline(0.5, color="red", ls="--", lw=1.2, alpha=0.6, label="0.5 threshold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
ax.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confidence_dist_4class.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] confidence_dist_4class.png")

# ─────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────
print(f"""
+--------------------------------------------------+
|            AgroVision 4-Class Results            |
+--------------------------------------------------+
|  Overall Accuracy  : {acc*100:6.2f}%                    |
|  Weighted F1-Score : {f1w:6.4f}                    |
|  Macro F1-Score    : {f1m:6.4f}                    |
|  Mean AUC (ROC)    : {mean_auc:6.3f}                    |
+--------------------------------------------------+""")
for ci, cls in enumerate(TARGET_CLASSES):
    print(f"|  {cls:<24} P={prec_cls[ci]:.2f} R={rec_cls[ci]:.2f} F1={f1_cls[ci]:.2f}  |")
print("+--------------------------------------------------+")
print("\n[OK] All PNGs saved. Ready to push to GitHub.")
