import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from scipy.spatial.distance import mahalanobis

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DEFAULT_IMAGE_PATH = "test_crash.jpg"
BASELINE_DIR = "baseline_data"
BASELINE_EMBEDDINGS_FILE = os.path.join(BASELINE_DIR, "baseline_embeddings.npy")

# ──────────────────────────────────────────────
# 1. LOAD MODEL
# ──────────────────────────────────────────────
def load_model():
    print("Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()
    print("Model loaded.")
    return processor, model


# ──────────────────────────────────────────────
# 2. EXTRACT EMBEDDING
# ──────────────────────────────────────────────
def get_embedding(image_path: str, processor, model) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding


# ──────────────────────────────────────────────
# 3. BUILD BASELINE FROM FOLDER
#    Run once on 20–50 real crash images to
#    create the reference distribution.
# ──────────────────────────────────────────────
def build_baseline_from_folder(folder: str, processor, model) -> np.ndarray:
    """Extract embeddings from every image in `folder` and save to .npy."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in '{folder}'. "
            f"Add 20–50 real car crash photos to this folder."
        )

    print(f"Building baseline from {len(image_paths)} images in '{folder}'...")
    embeddings = []
    for i, path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(path)}")
        emb = get_embedding(path, processor, model)
        embeddings.append(emb)

    baseline = np.array(embeddings)

    os.makedirs(BASELINE_DIR, exist_ok=True)
    np.save(BASELINE_EMBEDDINGS_FILE, baseline)
    print(f"Baseline saved to '{BASELINE_EMBEDDINGS_FILE}' "
          f"({baseline.shape[0]} embeddings, dim={baseline.shape[1]})")
    return baseline


# ──────────────────────────────────────────────
# 4. LOAD BASELINE (or fall back to mock)
# ──────────────────────────────────────────────
def load_baseline(query_embedding: np.ndarray) -> np.ndarray:
    """Load saved baseline embeddings, or generate mock data with a warning."""
    if os.path.exists(BASELINE_EMBEDDINGS_FILE):
        baseline = np.load(BASELINE_EMBEDDINGS_FILE)
        print(f"Loaded baseline: {baseline.shape[0]} reference embeddings.")
        return baseline

    print("⚠ WARNING: No baseline file found. Using MOCK baseline.")
    print(f"  To build a real baseline, run:")
    print(f"    python task1_dinov2.py --build-baseline <folder_of_crash_images>")
    print(f"  The mock baseline adds noise to the query itself,")
    print(f"  so fraud detection will NOT work properly.\n")
    noise = np.random.normal(0, 0.05, (50, len(query_embedding)))
    return query_embedding + noise


# ──────────────────────────────────────────────
# 5. MAHALANOBIS DISTANCE
# ──────────────────────────────────────────────
def compute_mahalanobis(query_embedding: np.ndarray,
                        baseline_embeddings: np.ndarray) -> float:
    mean_vec = np.mean(baseline_embeddings, axis=0)
    cov = np.cov(baseline_embeddings.T)
    # Using pseudo-inverse to handle the curse of dimensionality (samples < dims)
    cov_pinv = np.linalg.pinv(cov, rcond=1e-5)
    
    diff = query_embedding - mean_vec
    distance = np.sqrt(diff.T @ cov_pinv @ diff)
    return float(distance)


# ──────────────────────────────────────────────
# 6. NORMALIZE TO 0–1 OUTLIER SCORE
# ──────────────────────────────────────────────
def normalize_score(distance: float) -> float:
    """Map pseudo-inverse Mahalanobis distance to [0, 1].
       Typical inliers have distances ~12-16.
    """
    shifted_dist = max(0.0, distance - 14.0)
    score = 1.0 - np.exp(-shifted_dist / 10.0)
    return round(float(score), 4)


# ──────────────────────────────────────────────
# 7. INTERPRET SCORE
# ──────────────────────────────────────────────
def interpret_score(score: float) -> str:
    if score < 0.3:
        return "NORMAL — visually consistent with real crashes"
    elif score < 0.6:
        return "SUSPICIOUS — moderate visual anomaly detected"
    else:
        return "HIGH RISK — strong visual outlier, possible staging"


# ──────────────────────────────────────────────
# 8. MAIN — produces the dict for downstream
# ──────────────────────────────────────────────
def run_task1(image_path: str) -> dict:
    print(f"\nProcessing: {image_path}")

    processor, model = load_model()
    query_emb = get_embedding(image_path, processor, model)
    baseline = load_baseline(query_emb)
    raw_distance = compute_mahalanobis(query_emb, baseline)
    outlier_score = normalize_score(raw_distance)
    label = interpret_score(outlier_score)

    result = {
        "layer":                2,
        "task":                 "dinov2_mahalanobis",
        "image_path":           image_path,
        "embedding_dim":        len(query_emb),
        "mahalanobis_raw":      round(raw_distance, 4),
        "dinov2_outlier_score": outlier_score,
        "anomaly_label":        label,
        "is_outlier":           outlier_score >= 0.3,
    }

    print("\n── LAYER 2 / TASK 1 OUTPUT ──────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("─────────────────────────────────────────────")
    return result


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINOv2 + Mahalanobis outlier detection for crash images"
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to the crash image to analyse")
    parser.add_argument("--build-baseline", type=str, default=None,
                        metavar="FOLDER",
                        help="Build baseline from a folder of real crash images")
    args = parser.parse_args()

    if args.build_baseline:
        proc, mdl = load_model()
        build_baseline_from_folder(args.build_baseline, proc, mdl)
    else:
        layer2_task1_output = run_task1(args.image)
