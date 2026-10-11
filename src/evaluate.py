"""
Evaluation metrics, plots, and t-SNE visualisation for the GIN model.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Data

from src.model import EmbeddingExtractor, GINClassifier


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def plot_eda(df, save_path: str = "outputs/eda.png"):
    """Class distribution bar chart + feature correlation heatmap."""
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Bitcoin Heist Dataset — EDA", fontsize=14, fontweight="bold")

    label_counts = df["label"].value_counts()
    colors = ["#e74c3c" if l != "white" else "#2ecc71" for l in label_counts.index]
    axes[0].bar(label_counts.index, label_counts.values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    axes[1].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"EDA plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, epochs: int, save_path: str = "outputs/training_curves.png"):
    """Loss and accuracy curves across training epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GIN Training History", fontsize=14, fontweight="bold")

    ep = range(1, epochs + 1)
    axes[0].plot(ep, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(ep, history["val_loss"], label="Val Loss", color="coral", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="Train Acc", color="steelblue")
    axes[1].plot(ep, history["val_acc"], label="Val Acc", color="coral", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_test(
    model: GINClassifier,
    data: Data,
    save_path: str = "outputs/evaluation.png",
) -> dict:
    """
    Compute classification report, ROC-AUC, confusion matrix, and ROC curve.

    Returns a dict of scalar metrics.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1]

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = logits[data.test_mask].argmax(1).cpu().numpy()
    y_score = probs[data.test_mask].cpu().numpy()

    print("=" * 55)
    print("         TEST SET PERFORMANCE")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=["Benign", "Ransomware"]))

    auc = roc_auc_score(y_true, y_score)
    print(f"ROC-AUC : {auc:.4f}")

    # Plots
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Ransomware"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix (Test Set)")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Evaluation plots saved → {save_path}")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": auc,
    }


# ---------------------------------------------------------------------------
# t-SNE embedding visualisation
# ---------------------------------------------------------------------------

def plot_tsne(
    model: GINClassifier,
    data: Data,
    device: torch.device,
    sample_n: int = 3000,
    save_path: str = "outputs/tsne.png",
):
    """t-SNE projection of GIN node embeddings on test-set nodes."""
    extractor = EmbeddingExtractor(model).to(device)
    extractor.eval()

    with torch.no_grad():
        embs = extractor(data.x, data.edge_index).cpu().numpy()

    test_idx = data.test_mask.cpu().numpy().nonzero()[0]
    sample_n = min(sample_n, len(test_idx))
    sample_idx = np.random.choice(test_idx, sample_n, replace=False)

    print(f"Running t-SNE on {sample_n} test nodes …")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42, n_jobs=-1)
    emb2d = tsne.fit_transform(embs[sample_idx])
    labels_sample = data.y.cpu().numpy()[sample_idx]

    colors_map = {0: "#2ecc71", 1: "#e74c3c"}
    plt.figure(figsize=(9, 7))
    for cls, name in [(0, "Benign"), (1, "Ransomware")]:
        mask = labels_sample == cls
        plt.scatter(emb2d[mask, 0], emb2d[mask, 1],
                    c=colors_map[cls], label=name, s=10, alpha=0.6, linewidths=0)
    plt.legend(markerscale=3)
    plt.title("t-SNE of GIN Node Embeddings (Test Set)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"t-SNE plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(metrics: dict, data: Data, total_params: int):
    print("=" * 40)
    print("   FINAL SUMMARY — GIN on Bitcoin Heist")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("=" * 40)
    print(f"  Graph nodes : {data.num_nodes:,}")
    print(f"  Graph edges : {data.num_edges:,}")
    print(f"  GIN layers  : 3")
    print(f"  Hidden dim  : 128")
    print(f"  Parameters  : {total_params:,}")
    print("=" * 40)
