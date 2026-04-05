import argparse
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DEFAULT_IMAGE_PATH = "test_crash.jpg"
DEFAULT_CLAIM_TEXT = "front bumper damage from rear-end collision"

# ──────────────────────────────────────────────
# 1. LOAD MODEL
# ──────────────────────────────────────────────
def load_model():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print("Model loaded.")
    return processor, model


# ──────────────────────────────────────────────
# 2. COMPUTE TEXT–IMAGE CONSISTENCY
# ──────────────────────────────────────────────
def compute_consistency(image_path: str, claim_text: str,
                        processor, model) -> float:
    """Return cosine similarity between image and text, mapped to [0, 1]."""
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[claim_text],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # CLIP logits_per_image is cosine-similarity × 100
    logit = outputs.logits_per_image.squeeze().item()

    # Map to 0–1.  CLIP cosine similarity typically ranges ~ -1 to +1
    # (logits = similarity * 100), so we normalise:
    similarity = logit / 100.0                      # back to [-1, 1]
    score = float(np.clip((similarity + 1) / 2, 0, 1))   # map to [0, 1]
    return round(score, 4)


# ──────────────────────────────────────────────
# 3. INTERPRET CONSISTENCY
# ──────────────────────────────────────────────
def interpret_consistency(score: float) -> str:
    if score >= 0.80:
        return "HIGH — claim text matches the image well"
    elif score >= 0.60:
        return "MODERATE — partial match, some inconsistency"
    elif score >= 0.40:
        return "LOW — noticeable mismatch between text and image"
    else:
        return "MISMATCH — claim text contradicts the image"


# ──────────────────────────────────────────────
# 4. MAIN — produces the dict for downstream
# ──────────────────────────────────────────────
def run_task3(image_path: str, claim_text: str) -> dict:
    print(f"\nProcessing: {image_path}")
    print(f"Claim text: \"{claim_text}\"")

    processor, model = load_model()
    score = compute_consistency(image_path, claim_text, processor, model)
    label = interpret_consistency(score)

    result = {
        "layer":              2,
        "task":               "clip_text_image_consistency",
        "image_path":         image_path,
        "claim_text":         claim_text,
        "clip_consistency":   score,
        "consistency_label":  label,
    }

    print("\n── LAYER 2 / TASK 3 OUTPUT ──────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("─────────────────────────────────────────────")
    return result


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP text-image consistency check for crash claims"
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to the crash image")
    parser.add_argument("--claim-text", type=str, default=DEFAULT_CLAIM_TEXT,
                        help="The claim description text to compare against the image")
    args = parser.parse_args()

    layer2_task3_output = run_task3(args.image, args.claim_text)
