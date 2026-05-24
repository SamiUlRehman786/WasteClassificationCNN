"""
predict.py
──────────
Run inference on a single image or a folder of images.

Usage
─────
  python predict.py --image path/to/image.jpg
  python predict.py --folder path/to/images/
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

MODEL_PATH  = "./models/waste_cnn_best.h5"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
COLORS      = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"]


def load_model():
    print("🔄 Loading model ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.\n")
    return model


def preprocess(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict_single(model, img_path):
    arr, img = preprocess(img_path)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(probs)
    label = CLASS_NAMES[pred_idx]
    conf  = probs[pred_idx] * 100

    print(f"📷 Image     : {img_path}")
    print(f"🏷️  Predicted : {label.upper()} ({conf:.2f}%)")
    print("   All scores:")
    for cls, p in zip(CLASS_NAMES, probs):
        bar = "█" * int(p * 30)
        print(f"   {cls:<12} {p*100:5.1f}% {bar}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0D1117")

    axes[0].imshow(img)
    axes[0].set_title(f"Prediction: {label.upper()}\nConfidence: {conf:.1f}%",
                      color=COLORS[pred_idx], fontsize=14, fontweight="bold")
    axes[0].axis("off")

    ax = axes[1]
    ax.set_facecolor("#161B22")
    bars = ax.barh(CLASS_NAMES, probs * 100,
                   color=COLORS[:len(CLASS_NAMES)], alpha=0.9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="white")
    ax.set_title("Class Probabilities", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
    for bar, val in zip(bars, probs * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=9)

    fig.suptitle("Waste Classification Inference", fontsize=15,
                 fontweight="bold", color="white")
    plt.tight_layout()
    out = "./results/prediction_result.png"
    os.makedirs("./results", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.show()
    print(f"\n📊 Saved: {out}")
    return label, conf


def predict_folder(model, folder_path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in exts]
    if not imgs:
        print("❌ No images found in folder.")
        return

    results = []
    for fname in imgs:
        path = os.path.join(folder_path, fname)
        arr, _ = preprocess(path)
        probs = model.predict(arr, verbose=0)[0]
        pred_idx = np.argmax(probs)
        results.append({
            "file": fname,
            "class": CLASS_NAMES[pred_idx],
            "confidence": probs[pred_idx] * 100,
        })

    print(f"\n{'FILE':<40} {'CLASS':<15} {'CONFIDENCE':>10}")
    print("-" * 68)
    for r in results:
        print(f"{r['file']:<40} {r['class']:<15} {r['confidence']:>9.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste Classification Inference")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    model = load_model()
    if args.image:
        predict_single(model, args.image)
    elif args.folder:
        predict_folder(model, args.folder)
    else:
        print("❌ Provide --image or --folder argument.")
        parser.print_help()
