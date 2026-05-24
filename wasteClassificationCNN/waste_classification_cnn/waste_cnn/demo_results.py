"""
demo_results.py
───────────────
Generates realistic-looking result graphs using SIMULATED data.
Run this if you don't yet have the dataset downloaded, to see what
the final output looks like — or to generate placeholder graphs for
your GitHub README.

Usage:  python demo_results.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

np.random.seed(42)
os.makedirs("./results", exist_ok=True)

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
COLORS      = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"]
EPOCHS      = 25


# ── Simulate training history ──────────────────────────────────────────
def sim_curve(start, end, epochs, noise=0.015):
    x   = np.linspace(0, 1, epochs)
    val = start + (end - start) * (1 - np.exp(-4 * x))
    return val + np.random.normal(0, noise, epochs)

history = {
    "accuracy":     sim_curve(0.45, 0.93, EPOCHS),
    "val_accuracy": sim_curve(0.42, 0.91, EPOCHS),
    "loss":         sim_curve(1.60, 0.22, EPOCHS)[::-1] + 0.05,
    "val_loss":     sim_curve(1.65, 0.28, EPOCHS)[::-1] + 0.07,
    "precision":    sim_curve(0.40, 0.92, EPOCHS),
    "recall":       sim_curve(0.38, 0.90, EPOCHS),
}


# ── 1. Training history ────────────────────────────────────────────────
def plot_training_history():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0D1117")
    for ax in axes:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for s in ax.spines.values(): s.set_edgecolor("#30363D")

    ep = range(1, EPOCHS + 1)
    axes[0].plot(ep, history["accuracy"],     color="#2ECC71", lw=2, label="Train")
    axes[0].plot(ep, history["val_accuracy"], color="#3498DB", lw=2, ls="--", label="Val")
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1C2128", labelcolor="white"); axes[0].grid(alpha=0.2)

    axes[1].plot(ep, history["loss"],     color="#E74C3C", lw=2, label="Train")
    axes[1].plot(ep, history["val_loss"], color="#F39C12", lw=2, ls="--", label="Val")
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#1C2128", labelcolor="white"); axes[1].grid(alpha=0.2)

    axes[2].plot(ep, history["precision"], color="#9B59B6", lw=2, label="Precision")
    axes[2].plot(ep, history["recall"],    color="#1ABC9C", lw=2, ls="--", label="Recall")
    axes[2].set_title("Precision & Recall", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Score")
    axes[2].legend(facecolor="#1C2128", labelcolor="white"); axes[2].grid(alpha=0.2)

    fig.suptitle("Waste Classification CNN — Training Metrics",
                 fontsize=16, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    plt.savefig("./results/training_history.png", dpi=150,
                bbox_inches="tight", facecolor="#0D1117")
    plt.close(); print("✅ training_history.png")


# ── 2. Confusion matrix ────────────────────────────────────────────────
def plot_confusion_matrix():
    # Simulate a good confusion matrix
    cm = np.array([
        [185,  5,  2,  4,  3,  1],
        [  4,170,  8,  3,  9,  6],
        [  3,  7,178,  4,  5,  3],
        [  2,  3,  4,190,  5,  6],
        [  4,  8,  6,  5,172,  5],
        [  2,  5,  3,  7,  4,179],
    ])
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0D1117")
    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
        ["d", ".2f"],
    ):
        ax.set_facecolor("#161B22")
        sns.heatmap(data, annot=True, fmt=fmt, cmap="YlOrRd",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.5, linecolor="#30363D", ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=12)
        ax.set_xlabel("Predicted", color="white", fontsize=11)
        ax.set_ylabel("Actual",    color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=9)

    fig.suptitle("Waste Classification — Confusion Matrices",
                 fontsize=15, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig("./results/confusion_matrix.png", dpi=150,
                bbox_inches="tight", facecolor="#0D1117")
    plt.close(); print("✅ confusion_matrix.png")


# ── 3. Per-class metrics ───────────────────────────────────────────────
def plot_per_class_metrics():
    prec = [0.92, 0.88, 0.91, 0.93, 0.89, 0.91]
    rec  = [0.93, 0.86, 0.89, 0.94, 0.87, 0.90]
    f1   = [0.925, 0.87, 0.90, 0.935, 0.88, 0.905]

    x = np.arange(len(CLASS_NAMES)); w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")

    b1 = ax.bar(x - w, prec, w, label="Precision", color="#3498DB", alpha=0.9)
    b2 = ax.bar(x,     rec,  w, label="Recall",    color="#2ECC71", alpha=0.9)
    b3 = ax.bar(x + w, f1,   w, label="F1-Score",  color="#E74C3C", alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=10)
    ax.set_yticks(np.arange(0, 1.1, 0.1)); ax.tick_params(colors="white")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Waste Category", color="white", fontsize=12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("Per-Class Precision, Recall & F1-Score",
                 color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1C2128", labelcolor="white", fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", color="white", fontsize=7)

    plt.tight_layout()
    plt.savefig("./results/per_class_metrics.png", dpi=150,
                bbox_inches="tight", facecolor="#0D1117")
    plt.close(); print("✅ per_class_metrics.png")


# ── 4. Class distribution ──────────────────────────────────────────────
def plot_class_distribution():
    counts = [403, 501, 410, 594, 482, 137]  # realistic imbalance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")

    ax = axes[0]; ax.set_facecolor("#161B22")
    bars = ax.bar(CLASS_NAMES, counts, color=COLORS, alpha=0.9, edgecolor="#30363D")
    ax.set_title("Class Distribution (Training Set)", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Waste Category", color="white"); ax.set_ylabel("# Images", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(v), ha="center", color="white", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    ax2 = axes[1]; ax2.set_facecolor("#161B22")
    wedges, texts, autotexts = ax2.pie(
        counts, labels=CLASS_NAMES, colors=COLORS,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="#0D1117", linewidth=1.5),
        textprops=dict(color="white"),
    )
    for at in autotexts: at.set_color("white"); at.set_fontsize(9)
    ax2.set_title("Class Share", color="white", fontsize=13, fontweight="bold")

    fig.suptitle("Dataset Class Distribution", fontsize=15, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig("./results/class_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="#0D1117")
    plt.close(); print("✅ class_distribution.png")


if __name__ == "__main__":
    print("🎨 Generating demo result graphs ...\n")
    plot_training_history()
    plot_confusion_matrix()
    plot_per_class_metrics()
    plot_class_distribution()
    print("\n✅ All graphs saved to ./results/")
    print("   Open them to preview what the real training output looks like.")
