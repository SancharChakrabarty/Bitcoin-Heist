"""
main.py — End-to-end pipeline for Bitcoin Heist Ransomware Detection using GIN.

Usage:
    python main.py --data_path data/BitcoinHeistData.csv
    python main.py --data_path data/BitcoinHeistData.csv --epochs 150 --hidden_dim 256
"""

import argparse
import os
import shutil

import torch

from src.data import load_dataset, preprocess, build_graph
from src.model import GINClassifier
from src.train import train
from src.evaluate import (
    plot_eda,
    plot_training_curves,
    evaluate_test,
    plot_tsne,
    print_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Bitcoin Heist Ransomware Detection via GIN")
    parser.add_argument("--data_path", type=str, default="data/BitcoinHeistData.csv",
                        help="Path to BitcoinHeistData.csv")
    parser.add_argument("--max_samples", type=int, default=50_000,
                        help="Max rows to keep (stratified subsample)")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="k for k-NN graph construction")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension for GIN layers")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GIN layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 weight decay")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save plots and model checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load & EDA ──────────────────────────────────────────────
    df = load_dataset(args.data_path)
    plot_eda(df, save_path=os.path.join(args.output_dir, "eda.png"))

    # ── 2. Preprocess & build graph ────────────────────────────────
    X_scaled, y_binary = preprocess(df, max_samples=args.max_samples, random_state=args.seed)
    data = build_graph(X_scaled, y_binary, k_neighbors=args.k_neighbors, random_state=args.seed)
    data = data.to(device)

    # ── 3. Model ───────────────────────────────────────────────────
    model = GINClassifier(
        in_dim=data.num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Trainable parameters: {total_params:,}")

    # ── 4. Train ───────────────────────────────────────────────────
    history, best_state = train(
        model, data, device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    plot_training_curves(history, args.epochs,
                         save_path=os.path.join(args.output_dir, "training_curves.png"))

    # ── 5. Evaluate ────────────────────────────────────────────────
    metrics = evaluate_test(model, data,
                            save_path=os.path.join(args.output_dir, "evaluation.png"))

    # ── 6. t-SNE ───────────────────────────────────────────────────
    plot_tsne(model, data, device,
              save_path=os.path.join(args.output_dir, "tsne.png"))

    # ── 7. Save model ──────────────────────────────────────────────
    checkpoint_path = os.path.join(args.output_dir, "gin_model.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "in_dim": data.num_node_features,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_classes": 2,
                "dropout": args.dropout,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(f"✅ Model checkpoint saved → {checkpoint_path}")

    # ── 8. Summary ─────────────────────────────────────────────────
    print_summary(metrics, data, total_params)


if __name__ == "__main__":
    main()
