"""
predict.py — Load a trained GIN checkpoint and run inference on the dataset.

Usage:
    python predict.py --data_path data/BitcoinHeistData.csv --checkpoint outputs/gin_model.pt
    python predict.py --data_path data/BitcoinHeistData.csv --checkpoint outputs/gin_model.pt --save_csv outputs/predictions.csv
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F

from src.data import load_dataset, preprocess, build_graph
from src.model import GINClassifier
from src.evaluate import evaluate_test, plot_tsne, print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained GIN checkpoint")
    parser.add_argument("--data_path", type=str, default="data/BitcoinHeistData.csv",
                        help="Path to BitcoinHeistData.csv")
    parser.add_argument("--checkpoint", type=str, default="outputs/gin_model.pt",
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--max_samples", type=int, default=50_000,
                        help="Must match the value used during training")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="Must match the value used during training")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Optional: save predictions to this CSV path")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save evaluation plots")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load checkpoint ─────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    print(f"Model config: {config}")

    # ── 2. Rebuild the same graph ──────────────────────────────────
    df = load_dataset(args.data_path)
    X_scaled, y_binary = preprocess(df, max_samples=args.max_samples, random_state=args.seed)
    data = build_graph(X_scaled, y_binary, k_neighbors=args.k_neighbors, random_state=args.seed)
    data = data.to(device)

    # ── 3. Load model weights ──────────────────────────────────────
    model = GINClassifier(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✅ Model loaded successfully")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── 4. Run inference ───────────────────────────────────────────
    print("\nRunning inference on test set …")
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    # ── 5. Evaluate on test set ────────────────────────────────────
    metrics = evaluate_test(model, data,
                            save_path=os.path.join(args.output_dir, "evaluation.png"))
    plot_tsne(model, data, device,
              save_path=os.path.join(args.output_dir, "tsne.png"))
    print_summary(metrics, data, total_params)

    # ── 6. Optionally save predictions to CSV ─────────────────────
    if args.save_csv:
        test_mask = data.test_mask.cpu().numpy()
        results_df = pd.DataFrame({
            "node_index": test_mask.nonzero()[0],
            "true_label": data.y[data.test_mask].cpu().numpy(),
            "predicted_label": preds[test_mask],
            "ransomware_probability": probs[test_mask],
        })
        results_df["true_label"] = results_df["true_label"].map({0: "benign", 1: "ransomware"})
        results_df["predicted_label"] = results_df["predicted_label"].map({0: "benign", 1: "ransomware"})
        results_df.to_csv(args.save_csv, index=False)
        print(f"\n✅ Predictions saved → {args.save_csv}")
        print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
