"""
Generate Ultralytics ground truth for monocular metric depth estimation.

Runs every ONNX depth model in models/ over every image in data/images/ and
writes summary statistics plus a fixed sample grid to results/results_ultralytics.json.

Full depth maps are far too large to store in JSON, so the comparison contract
is: per-image min/max/mean/median depth in meters, plus depth sampled on a
deterministic GRID_N x GRID_N lattice of relative coordinates. That is enough
to catch scale errors, letterbox/crop misalignment and flipped axes, while
staying tolerant of small fp16-vs-fp32 numeric differences.
"""

import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_PATH, "data", "images")
MODELS_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

# Deterministic sample lattice, expressed in relative image coordinates so the
# C++ side can reproduce the exact same pixels regardless of image size.
GRID_N = 5
IMG_SIZE = 768  # released YOLO26 depth weights are trained at imgsz=768

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def sample_points(width: int, height: int):
    """Deterministic GRID_N x GRID_N pixel lattice, matching the C++ driver."""
    points = []
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            # Cell centres: (i + 0.5) / N  — avoids image borders entirely.
            x = int((gx + 0.5) / GRID_N * width)
            y = int((gy + 0.5) / GRID_N * height)
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            points.append((x, y))
    return points


def summarize(depth: np.ndarray) -> dict:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("Depth map contains no finite values")
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
    }


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"

    if not os.path.isdir(IMAGES_PATH):
        print(f"ERROR: images directory not found: {IMAGES_PATH}")
        return 1

    image_files = [
        f for f in sorted(os.listdir(IMAGES_PATH))
        if f.lower().endswith(VALID_EXT)
    ]
    if not image_files:
        print(f"ERROR: no images found in {IMAGES_PATH}")
        return 1

    model_files = [
        f for f in sorted(os.listdir(MODELS_PATH)) if f.endswith(".onnx")
    ]
    if not model_files:
        print(f"ERROR: no ONNX depth models found in {MODELS_PATH}")
        return 1

    os.makedirs(RESULTS_PATH, exist_ok=True)
    output = {}

    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(MODELS_PATH, model_file)
        print(f"\n======== {model_name} ========")

        model = YOLO(model_path, task="depth")

        model_results = {
            "weights_path": os.path.join("models", model_file),
            "task": "depth",
            "results": [],
        }

        for image_file in tqdm(image_files, desc="Images", unit="image"):
            image_path = os.path.join(IMAGES_PATH, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"WARNING: could not read {image_path}, skipping")
                continue

            height, width = image.shape[:2]

            predictions = model.predict(
                image, imgsz=IMG_SIZE, device=device, verbose=False
            )
            depth = predictions[0].depth.data.cpu().numpy().astype(np.float32)
            depth = np.squeeze(depth)

            # Ultralytics returns the depth map at the original image size; if a
            # future version changes that, resize so sampling stays comparable.
            if depth.shape != (height, width):
                depth = cv2.resize(
                    depth, (width, height), interpolation=cv2.INTER_LINEAR
                )

            entry = {
                "image_path": os.path.join("data", "images", image_file),
                "width": width,
                "height": height,
            }
            entry.update(summarize(depth))
            entry["samples"] = [
                {"x": x, "y": y, "depth": float(depth[y, x])}
                for (x, y) in sample_points(width, height)
            ]
            model_results["results"].append(entry)

        output[model_name] = model_results

    out_file = os.path.join(RESULTS_PATH, "results_ultralytics.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
