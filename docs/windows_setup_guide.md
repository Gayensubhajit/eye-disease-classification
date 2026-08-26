# Windows Setup & Execution Guide

Step-by-step instructions to run the eye disease classification pipeline natively on Windows (PowerShell, Command Prompt, or VS Code).

---

## 1. Navigate to Project Directory

Open **PowerShell** or **Command Prompt** and navigate to your project directory:

```powershell
cd C:\Users\subha\Documents\GitHub\eye-disease-classification
```

---

## 2. Virtual Environment Setup

### On Windows PowerShell:
```powershell
# If PowerShell gives a script execution policy error:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Create virtual environment (uses .venv_win to avoid conflicting with Linux .venv)
python -m venv .venv_win

# Activate virtual environment
.venv_win\Scripts\Activate.ps1
```

### On Windows Command Prompt (CMD):
```cmd
python -m venv .venv_win
.venv_win\Scripts\activate.bat
```

---

## 3. Install Dependencies (with NVIDIA GPU CUDA Support)

```powershell
# 1. Install PyTorch with CUDA 12.1 acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install project requirements
pip install -e .[dev]
```

### Verify GPU Acceleration:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 4. Build 70/15/15 Balanced Dataset Folders

```powershell
python scripts\build_balanced_folders.py
```

---

## 5. Train Models on Windows

### A. High-Resolution ($384\times 384$) EfficientNet-B3 + CLAHE (SOTA 90.17% Acc):
```powershell
python -m src.train --config configs\efficientnet_b3_384_clahe.yaml
```

### B. Microsoft BiomedCLIP + CBAM Dual Attention:
```powershell
python -m src.train --config configs\biomedclip_cbam_fusion.yaml
```

### C. Standard CNN Baseline (EfficientNet-B0):
```powershell
python -m src.train --config configs\config.yaml
```

---

## 6. Evaluation & Multi-Model Ensemble

### Single Checkpoint Evaluation:
```powershell
python -m src.evaluate --checkpoint outputs\efficientnet_b3_384_clahe\best_model.pth --config configs\efficientnet_b3_384_clahe.yaml
```

### Multi-Model Ensemble with 4-View Test-Time Augmentation (TTA):
```powershell
python scripts\evaluate_ensemble.py
```

---

## 7. Generate Publication PDF Report

```powershell
python scripts\generate_research_pdf.py
```
Output PDF: `docs\Literature_Review_and_SOTA_Benchmarks.pdf`

---

## Common Windows Troubleshooting Tips

1. **PowerShell Script Policy:**
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
2. **CUDA Memory Fragmentation (4GB VRAM):**
   `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`
