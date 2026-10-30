# Results — YOLO11n road damage detector

Run on 31 Oct 2026 (IST) on CPU (8 cores), from the executed notebook
[`notebooks/01_road_damage_detection_yolo11.ipynb`](../notebooks/01_road_damage_detection_yolo11.ipynb).

- Dataset: 2,009 images (640 × 360), 4,737 objects — Pothole 1,261 · Crack 2,519 · Manhole 957
- Split: 80/20 random (seed 42) → 1,608 train / 401 validation images
- Model: `yolo11n.pt` (COCO-pretrained) fine-tuned for 30 epochs, imgsz 640, batch 16

## Validation metrics (best weights)

| Class | Val instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **All** | 548 | 0.531 | 0.500 | **0.496** | **0.225** |
| Pothole | 160 | 0.574 | 0.283 | 0.383 | 0.148 |
| Crack | 309 | 0.410 | 0.366 | 0.336 | 0.128 |
| Manhole | 79 | 0.609 | 0.851 | 0.770 | 0.398 |

Manholes (regular, round shapes) are learned well. Potholes and cracks are harder: their boundaries are
ambiguous and cracks are thin, so recall is the weak point. More epochs, a larger model (`yolo11s`) or a
higher input resolution would be the next steps.

## Files
- `training/results.csv`, `training/results.png`: per-epoch losses and metrics
- `training/labels.jpg`, `training/train_batch0.jpg`: label statistics and an augmented training batch
- `validation/`: confusion matrices, PR / P / R / F1 curves, a labelled vs predicted validation batch
- `predictions/`: predictions (conf ≥ 0.25) on 8 validation images
- `../models/best.pt` (5.3 MB) and `../models/best.onnx` (11 MB, used by the web demo)
