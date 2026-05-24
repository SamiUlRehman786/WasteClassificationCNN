"""
Waste Classification Using Convolutional Neural Network (CNN)
=============================================================
Dataset: RealWaste / Garbage Classification Dataset (~3GB)
Classes: cardboard, glass, metal, paper, plastic, trash
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import json

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "dataset_path": "./data/garbage_classification",   # unzip dataset here
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs": 25,
    "learning_rate": 1e-4,
    "val_split": 0.2,
    "test_split": 0.1,
    "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    "results_dir": "./results",
    "model_path": "./models/waste_cnn_best.h5",
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
os.makedirs("./models", exist_ok=True)

COLORS = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"]

# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def build_data_generators():
    """Build train / validation / test generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=CONFIG["val_split"],
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=CONFIG["val_split"],
    )

    train_gen = train_datagen.flow_from_directory(
        CONFIG["dataset_path"],
        target_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        CONFIG["dataset_path"],
        target_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"\n✅ Train samples : {train_gen.samples}")
    print(f"✅ Val   samples : {val_gen.samples}")
    print(f"✅ Classes found : {list(train_gen.class_indices.keys())}\n")
    return train_gen, val_gen


# ─────────────────────────────────────────────
# 2. MODEL ARCHITECTURE (Transfer Learning)
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """MobileNetV2 backbone + custom classification head."""
    base = MobileNetV2(
        input_shape=(*CONFIG["img_size"], 3),
        include_top=False,
        weights="imagenet",
    )
    # Freeze first 100 layers, fine-tune the rest
    for layer in base.layers[:100]:
        layer.trainable = False
    for layer in base.layers[100:]:
        layer.trainable = True

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="WasteCNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 3. CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks():
    return [
        callbacks.ModelCheckpoint(
            CONFIG["model_path"], monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
        callbacks.CSVLogger("./results/training_log.csv"),
    ]


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
def train(model, train_gen, val_gen):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=get_callbacks(),
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# 5. RESULT GRAPHS
# ─────────────────────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0D1117")
    for ax in axes:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # --- Accuracy ---
    axes[0].plot(epochs, history.history["accuracy"],    color="#2ECC71", lw=2, label="Train")
    axes[0].plot(epochs, history.history["val_accuracy"],color="#3498DB", lw=2, linestyle="--", label="Val")
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1C2128", labelcolor="white")
    axes[0].grid(alpha=0.2)

    # --- Loss ---
    axes[1].plot(epochs, history.history["loss"],     color="#E74C3C", lw=2, label="Train")
    axes[1].plot(epochs, history.history["val_loss"], color="#F39C12", lw=2, linestyle="--", label="Val")
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#1C2128", labelcolor="white")
    axes[1].grid(alpha=0.2)

    # --- Precision & Recall ---
    axes[2].plot(epochs, history.history["precision"], color="#9B59B6", lw=2, label="Precision")
    axes[2].plot(epochs, history.history["recall"],    color="#1ABC9C", lw=2, linestyle="--", label="Recall")
    axes[2].set_title("Precision & Recall", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Score")
    axes[2].legend(facecolor="#1C2128", labelcolor="white")
    axes[2].grid(alpha=0.2)

    fig.suptitle("Waste Classification CNN — Training Metrics",
                 fontsize=16, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    path = os.path.join(CONFIG["results_dir"], "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"📊 Saved: {path}")


def plot_confusion_matrix(model, val_gen, class_names):
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0D1117")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
        ["d", ".2f"],
    ):
        ax.set_facecolor("#161B22")
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="YlOrRd",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="#30363D",
            ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=12)
        ax.set_xlabel("Predicted", color="white", fontsize=11)
        ax.set_ylabel("Actual",    color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=9)

    fig.suptitle("Waste Classification — Confusion Matrices",
                 fontsize=15, fontweight="bold", color="white")
    plt.tight_layout()
    path = os.path.join(CONFIG["results_dir"], "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"📊 Saved: {path}")
    return y_true, y_pred


def plot_per_class_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, output_dict=True)
    metrics = {cls: report[cls] for cls in class_names}

    x = np.arange(len(class_names))
    width = 0.25
    prec  = [metrics[c]["precision"] for c in class_names]
    rec   = [metrics[c]["recall"]    for c in class_names]
    f1    = [metrics[c]["f1-score"]  for c in class_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars1 = ax.bar(x - width, prec, width, label="Precision", color="#3498DB", alpha=0.9)
    bars2 = ax.bar(x,          rec,  width, label="Recall",    color="#2ECC71", alpha=0.9)
    bars3 = ax.bar(x + width, f1,   width, label="F1-Score",  color="#E74C3C", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, color="white", fontsize=10)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Waste Category", color="white", fontsize=12)
    ax.set_ylabel("Score",          color="white", fontsize=12)
    ax.set_title("Per-Class Precision, Recall & F1-Score",
                 color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1C2128", labelcolor="white", fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", color="white", fontsize=7)

    plt.tight_layout()
    path = os.path.join(CONFIG["results_dir"], "per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"📊 Saved: {path}")

    print("\n" + "="*55)
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_class_distribution(train_gen):
    counts = {cls: 0 for cls in train_gen.class_indices}
    for cls, idx in train_gen.class_indices.items():
        counts[cls] = int(np.sum(np.array(train_gen.classes) == idx))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")

    labels = list(counts.keys())
    values = list(counts.values())

    # Bar chart
    ax = axes[0]
    ax.set_facecolor("#161B22")
    bars = ax.bar(labels, values, color=COLORS[:len(labels)], alpha=0.9, edgecolor="#30363D")
    ax.set_title("Class Distribution (Training Set)", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Waste Category", color="white"); ax.set_ylabel("# Images", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(v), ha="center", color="white", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # Pie chart
    ax2 = axes[1]
    ax2.set_facecolor("#161B22")
    wedges, texts, autotexts = ax2.pie(
        values, labels=labels, colors=COLORS[:len(labels)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="#0D1117", linewidth=1.5),
        textprops=dict(color="white"),
    )
    for at in autotexts: at.set_color("white"); at.set_fontsize(9)
    ax2.set_title("Class Share", color="white", fontsize=13, fontweight="bold")

    fig.suptitle("Dataset Class Distribution", fontsize=15, fontweight="bold", color="white")
    plt.tight_layout()
    path = os.path.join(CONFIG["results_dir"], "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"📊 Saved: {path}")


def plot_sample_predictions(model, val_gen, class_names, n=16):
    val_gen.reset()
    imgs, labels = next(val_gen)
    preds = model.predict(imgs[:n], verbose=0)

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Sample Predictions", fontsize=16, fontweight="bold",
                 color="white", y=0.98)

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i])
        true_lbl = class_names[np.argmax(labels[i])]
        pred_lbl = class_names[np.argmax(preds[i])]
        conf     = np.max(preds[i]) * 100
        correct  = true_lbl == pred_lbl
        color    = "#2ECC71" if correct else "#E74C3C"
        ax.set_title(f"T: {true_lbl}\nP: {pred_lbl} ({conf:.1f}%)",
                     fontsize=8, color=color, fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(CONFIG["results_dir"], "sample_predictions.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"📊 Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Waste Classification Using CNN")
    print("=" * 60)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}\n")

    # 1. Data
    train_gen, val_gen = build_data_generators()
    class_names = list(train_gen.class_indices.keys())

    # 2. Class distribution
    plot_class_distribution(train_gen)

    # 3. Build model
    model = build_model(num_classes=len(class_names))

    # 4. Train
    history = train(model, train_gen, val_gen)

    # 5. Evaluate & Graphs
    val_loss, val_acc, val_prec, val_rec = model.evaluate(val_gen, verbose=0)
    print(f"\n📈 Final Val Accuracy  : {val_acc:.4f}")
    print(f"📈 Final Val Precision : {val_prec:.4f}")
    print(f"📈 Final Val Recall    : {val_rec:.4f}")
    print(f"📈 Final Val Loss      : {val_loss:.4f}")

    # Save metrics
    metrics = {"val_accuracy": val_acc, "val_precision": val_prec,
               "val_recall": val_rec, "val_loss": val_loss}
    with open(os.path.join(CONFIG["results_dir"], "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_training_history(history)
    y_true, y_pred = plot_confusion_matrix(model, val_gen, class_names)
    plot_per_class_metrics(y_true, y_pred, class_names)
    plot_sample_predictions(model, val_gen, class_names)

    print("\n✅ All results saved to ./results/")
    print("✅ Best model saved  to ./models/waste_cnn_best.h5")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
