AgroVision — Main Project

Overview
- AgroVision is a PyTorch project for agricultural land classification using Sentinel-2 multispectral / RGB imagery.
- It includes training (`train_rgb.py`), evaluation (`eval_4class.py`), inference (`predict.py`), a Streamlit demo (`app.py`), and model definitions (`model.py`).

Files to include on GitHub
- `app.py` (Streamlit demo)
- `main.ipynb` (final project notebook, renamed)
- `predict.py` (inference helper that loads model)
- `model.py` (model architecture)
- `eval_4class.py` (evaluation scripts and metrics)
- `train_rgb.py` (training script)
- `requirements.txt` (dependencies)
- `README.md` (this file)

Files to exclude (add to `.gitignore`)
- `dataset/` (large image dataset)
- `*.pt` and `*.onnx` model checkpoints
- `__pycache__/`, `.ipynb_checkpoints/`

Quick run (demo)

1. Create virtual environment and install:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the Streamlit demo:

```bash
streamlit run app.py
```

Resume bullet (short)
- Developed "AgroVision": a PyTorch-based land-use classification system with a Streamlit demo, training and evaluation pipelines, explainability (Grad-CAM) and boundary detection.

Resume bullet (expanded)
- Architected and implemented `AgroVision`, a hybrid CNN-Transformer model for multispectral crop classification. Implemented data pipeline, training (MixUp, label smoothing), multi-scale FPN fusion, boundary-detection head, evaluation scripts, and an interactive Streamlit demo; prepared reproducible notebooks and deployment-ready export (ONNX).

 Project report
- `PROJECT_REPORT.md`: Detailed project report and abstract suitable for academic submission and resume references.