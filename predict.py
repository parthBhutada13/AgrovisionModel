import torch
from torchvision import transforms
from PIL import Image
import os

from model import AgroVision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['AnnualCrop', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NESTED = os.path.join(BASE_DIR, "main_cv_withoutput")
ckpt_paths = [
    os.path.join(BASE_DIR,  "agrovision_best.pt"),
    os.path.join(BASE_DIR,  "agrovision_ms_checkpoint.pt"),
    os.path.join(NESTED,    "agrovision_ms_best.pt"),
    os.path.join(NESTED,    "agrovision_ms_checkpoint.pt"),
]

# ✅ Load model
model = AgroVision(in_ch=3, num_classes=4).to(DEVICE)

checkpoint = None
for cp in ckpt_paths:
    if os.path.exists(cp):
        checkpoint = torch.load(cp, map_location=DEVICE)
        print(f"Loaded checkpoint: {os.path.basename(cp)}")
        break

if checkpoint is None:
    raise FileNotFoundError("❌ No checkpoint found. Please train the model first or place the weights in the project directory.")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# ✅ Safe load (ignore mismatched layers)
model_dict = model.state_dict()
filtered_dict = {}

for k, v in state_dict.items():
    if k in model_dict and model_dict[k].shape == v.shape:
        filtered_dict[k] = v

model_dict.update(filtered_dict)
model.load_state_dict(model_dict)

model.eval()

# ✅ Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.3444, 0.3803, 0.4078],
        [0.2026, 0.1367, 0.1155]
    )
])

# ✅ Prediction function
def predict(image):
    image = Image.open(image).convert("RGB")
    image = image.resize((128, 128))

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return CLASSES[pred], confidence, image