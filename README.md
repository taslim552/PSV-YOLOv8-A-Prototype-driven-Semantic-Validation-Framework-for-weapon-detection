# Hybrid X-Ray Weapon Detection (YOLO + SSL Prototypes)

This project implements a hybrid concealed-weapon detection pipeline for X-ray baggage images.

The pipeline combines:
- YOLO object detection
- SSL (SimCLR + ViT-Small) feature learning
- Multi-prototype class matching
- Class-adaptive thresholding
- Hybrid score fusion for final decisions

Main workflow is in `main.ipynb`.

## Dataset Reference

- Roboflow SIXray dataset: https://universe.roboflow.com/rabbanitechpro/sixray-hqgro/dataset/1

## Project Structure

- `main.ipynb` - End-to-end pipeline notebook (setup -> training -> evaluation -> export)
- `SIXray.v1i.yolov8/` - YOLO-format dataset used by the notebook
- `SIXray.v1i.coco/` - COCO-format dataset copy (optional for current notebook flow)
- `artifacts/` - saved metadata, SSL weights, prototypes, thresholds, prediction exports
- `yolo_runs/` - YOLO staged training outputs (best/last weights)
- `runs/` - additional run outputs
- `multitask_runs/` - multitask checkpoints (optional for current hybrid flow)
- `yolov8m.pt`, `yolo26n.pt` - base model weights

## Hybrid Pipeline Steps

The notebook is organized as:

1. Project setup and dataset path checks
2. Dataset cleaning and label consistency validation
3. YOLO training and benchmark selection
4. SSL pretraining (SimCLR + ViT-Small)
5. Multi-prototype generation with K-Means
6. Class-adaptive threshold estimation
7. Final hybrid fusion evaluation
8. Overall precision/recall summary
9. Prediction visualization export

## Requirements

## Hardware

- Recommended: NVIDIA GPU with CUDA
- Enough VRAM for YOLO + ViT workflows

## Software

- Windows
- Python 3.11 (project tested with 3.11.5)
- Jupyter kernel linked to project virtual environment

## Python Packages

Install at least:
- torch
- torchvision
- ultralytics
- timm
- pillow
- numpy
- pyyaml
- scikit-learn
- tqdm
- ipykernel

Example install command:

```bash
python -m pip install torch torchvision ultralytics timm pillow numpy pyyaml scikit-learn tqdm ipykernel
```

## Quick Start

1. Open the project folder in VS Code.
2. Open `main.ipynb`.
3. Select the project Python kernel (for example `.venv`).
4. Run cells in order for full training pipeline.

If you only want final evaluation + metrics + prediction export (after artifacts already exist), run:
- Cell 14
- Cell 15
- Cell 16

These cells expect:
- YOLO best weights available
- SSL backbone weights saved
- Prototype file saved
- Adaptive threshold file saved

## Required Artifacts for Final Hybrid Inference

These files are expected in `artifacts/`:
- `best_yolo_hybrid.json`
- `ssl_vit_small_backbone.pth`
- `prototypes_multi_v1.pth`
- `adaptive_thresholds.pth`

Prediction exports are written to:
- `artifacts/hybrid_test_predictions/`

## Notes

- Paths in some cells were made resilient to workspace variants.
- If a cell fails with missing package errors, install packages in the same notebook kernel.
- If a cell fails with missing weights, run earlier training cells or verify artifact paths.

## Typical Output Targets

Recent successful run showed:
- mAP@0.50 around 0.93
- mAP@0.50:0.95 around 0.70+

Exact values can vary by environment, random seed, and checkpoint version.

## License

Add your preferred license here.
