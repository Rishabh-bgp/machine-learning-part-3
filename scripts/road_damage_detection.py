# Machine Learning Part 3 — Road damage detection with YOLO11
# Copyright (c) 2026 Rishabh Aryan — MIT License
# Detects potholes, cracks and manholes on road images with Ultralytics YOLO11.

# %% [markdown]
# # Road Damage Detection with YOLO11 — Potholes, Cracks & Manholes
# Fine-tuning **YOLO11n** (Ultralytics) on the Kaggle *Road Damage Dataset* by Lorenzo Arcioni.
# Classes: `0 Pothole`, `1 Crack`, `2 Manhole`.

# %% [markdown]
# ## 1. Download the dataset with kagglehub

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes")

print("Path to dataset files:", path)

# %% [markdown]
# ## 2. Inspect the dataset
# `labels/` holds 4-point polygons; `labels-YOLO/` holds the same objects as axis-aligned YOLO boxes
# (`class x_center y_center width height`, normalised). We train on the YOLO boxes.

# %%
import os, glob, random, shutil, collections
from pathlib import Path

DATA = Path(path) / "data"
CLASSES = ["Pothole", "Crack", "Manhole"]
images = sorted(glob.glob(str(DATA / "images" / "*.jpg")))
print("Images:", len(images))
counts = collections.Counter()
for lbl in glob.glob(str(DATA / "labels-YOLO" / "*.txt")):
    for line in open(lbl):
        if line.strip():
            counts[CLASSES[int(line.split()[0])]] += 1
print("Objects per class:", dict(counts))

# %% [markdown]
# ## 3. Build a YOLO train/val split (80/20, seed 42)

# %%
WORK = Path("road_damage_yolo")
random.seed(42)
shuffled = images[:]
random.shuffle(shuffled)
n_val = int(0.2 * len(shuffled))
splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}
for split, files in splits.items():
    (WORK / "images" / split).mkdir(parents=True, exist_ok=True)
    (WORK / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img in files:
        stem = Path(img).stem
        shutil.copy(img, WORK / "images" / split / Path(img).name)
        src = DATA / "labels-YOLO" / f"{stem}.txt"
        dst = WORK / "labels" / split / f"{stem}.txt"
        if src.exists():
            shutil.copy(src, dst)
        else:
            open(dst, "w").close()
    print(split, len(files))

yaml_path = WORK / "data.yaml"
yaml_path.write_text(
    f"path: {WORK.resolve()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: Pothole\n  1: Crack\n  2: Manhole\n"
)
print(yaml_path.read_text())

# %% [markdown]
# ## 4. Fine-tune YOLO11n

# %%
from ultralytics import YOLO

RUNS = Path("runs").resolve()  # absolute, so outputs land next to the notebook
EPOCHS = int(os.environ.get("EPOCHS", 30))
model = YOLO("yolo11n.pt")
model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=640, batch=16, workers=4,
            seed=42, project=str(RUNS), name="road_damage", exist_ok=True, plots=True)

# %% [markdown]
# ## 5. Evaluate the best weights on the validation set

# %%
best = YOLO(RUNS / "road_damage" / "weights" / "best.pt")
metrics = best.val(data=str(yaml_path), imgsz=640, split="val", plots=True,
                   project=str(RUNS), name="road_damage_val", exist_ok=True)
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
for i, name in enumerate(CLASSES):
    p, r, ap50, ap = metrics.box.class_result(i)
    print(f"{name:8s} P={p:.3f} R={r:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")

# %% [markdown]
# ## 6. Predict on validation images

# %%
os.makedirs("predictions", exist_ok=True)
samples = sorted(splits["val"])[:8]
for img in samples:
    res = best.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
    found = collections.Counter(CLASSES[int(c)] for c in res.boxes.cls)
    print(Path(img).name, dict(found))
    res.save(filename=f"predictions/{Path(img).stem}_pred.jpg")

# %% [markdown]
# ## 7. Export to ONNX for in-browser inference

# %%
onnx_path = best.export(format="onnx", imgsz=640, opset=12, simplify=True)
print("ONNX model:", onnx_path)
