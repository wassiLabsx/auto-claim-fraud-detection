import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# ──────────────────────────────────────────────
# CONFIG — same image as Task 1
# ──────────────────────────────────────────────
DEFAULT_IMAGE_PATH = "test_crash.jpg"

# ──────────────────────────────────────────────
# 1. LOAD MODEL
# ──────────────────────────────────────────────
def load_model():
    print("Loading MobileSAM model...")
    model = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
    model.eval()
    print("Model loaded.")
    return model

# ──────────────────────────────────────────────
# 2. LOAD AND PREPARE IMAGE
# ──────────────────────────────────────────────
def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# ──────────────────────────────────────────────
# 3. GENERATE MASKS
# ──────────────────────────────────────────────
def generate_masks(model, image: np.ndarray) -> list:
    print("Generating masks...")
    mask_generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=16,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.70,
    )
    masks = mask_generator.generate(image)
    print(f"Found {len(masks)} segments.")
    return masks

# ──────────────────────────────────────────────
# 4. ESTIMATE DAMAGE SEVERITY
#    Largest foreground mask = car body
#    All smaller masks = damage regions
#    Output: sam_severity 0.0–1.0
# ──────────────────────────────────────────────
def estimate_damage(masks: list, image: np.ndarray) -> dict:
    if len(masks) < 2:
        return {
            "car_area_px": 0,
            "damage_area_px": 0,
            "sam_severity": 0.0,
            "severity_label": "UNKNOWN — not enough segments found",
        }

    total_pixels = image.shape[0] * image.shape[1]

    # Filter out background masks (those covering > 70% of the image)
    foreground_masks = [
        m for m in masks if m["area"] < total_pixels * 0.70
    ]

    if len(foreground_masks) < 1:
        # Fall back to all masks if filtering removed everything
        foreground_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    # Sort foreground masks by area (largest first)
    sorted_masks = sorted(foreground_masks, key=lambda m: m["area"], reverse=True)

    # Largest foreground segment = car body
    car_mask = sorted_masks[0]
    car_area = car_mask["area"]

    # Sum all smaller foreground segments as potential damage
    damage_area = sum(m["area"] for m in sorted_masks[1:])

    # Normalize to 0.0–1.0: damage / (car + damage)
    total = car_area + damage_area
    severity = round(damage_area / total, 4) if total > 0 else 0.0

    return {
        "car_area_px":     car_area,
        "damage_area_px":  damage_area,
        "sam_severity":    severity,
        "severity_label":  interpret_severity(severity),
    }

# ──────────────────────────────────────────────
# 5. INTERPRET SEVERITY (0.0–1.0 scale)
# ──────────────────────────────────────────────
def interpret_severity(score: float) -> str:
    if score < 0.10:
        return "MINOR — small dent or scratch"
    elif score < 0.30:
        return "MODERATE — significant panel damage"
    elif score < 0.60:
        return "SEVERE — major structural damage"
    else:
        return "TOTAL LOSS — extreme damage"

# ──────────────────────────────────────────────
# 6. VISUALIZE — saves a result image
# ──────────────────────────────────────────────
def visualize(image: np.ndarray, masks: list, output_path="task2_result.png"):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    for mask in sorted(masks, key=lambda m: m["area"], reverse=True)[:5]:  # top 5 largest
        m = mask["segmentation"]
        color = np.concatenate([np.random.random(3), [0.4]])
        plt.imshow(np.dstack([m, m, m, m]) * color)
    plt.axis("off")
    plt.title("SAM Segmentation — Top Segments")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {output_path}")

# ──────────────────────────────────────────────
# 7. MAIN — produces the dict for Person 4
# ──────────────────────────────────────────────
def run_task2(image_path: str) -> dict:
    print(f"\nProcessing: {image_path}")

    model   = load_model()
    image   = load_image(image_path)
    masks   = generate_masks(model, image)
    damage  = estimate_damage(masks, image)

    visualize(image, masks)

    result = {
        "layer":            2,
        "task":             "sam_damage_severity",
        "image_path":       image_path,
        "car_area_px":      damage["car_area_px"],
        "damage_area_px":   damage["damage_area_px"],
        "sam_severity":     damage["sam_severity"],
        "severity_label":   damage["severity_label"],
    }

    print("\n── LAYER 2 / TASK 2 OUTPUT ──────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("─────────────────────────────────────────────")
    return result

# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-based car damage severity estimation")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to the crash image")
    args = parser.parse_args()
    layer2_task2_output = run_task2(args.image)
