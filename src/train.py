"""
Training and evaluation logic for the GIN model.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from tqdm import trange


def get_class_weights(data: Data, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle class imbalance."""
    train_labels = data.y[data.train_mask].cpu().numpy()
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Class weights → benign: 1.00, ransomware: {pos_weight:.2f}")
    return torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)


def evaluate(model, data: Data, mask: torch.Tensor, criterion) -> tuple[float, float]:
    """Run inference on a subset of nodes and return (loss, accuracy)."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[mask], data.y[mask]).item()
        preds = logits[mask].argmax(dim=1)
        acc = (preds == data.y[mask]).float().mean().item()
    return loss, acc


def train(
    model,
    data: Data,
    device: torch.device,
    epochs: int = 100,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
) -> tuple[dict, dict]:
    """
    Train the GIN model with cosine annealing LR and class-weighted loss.

    Args:
        model: GINClassifier instance (already moved to device).
        data: PyG Data object (already moved to device).
        device: torch.device.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        weight_decay: L2 regularisation coefficient.

    Returns:
        history (dict): Train/val loss and accuracy per epoch.
        best_state (dict): State dict of the best model (by val accuracy).
    """
    class_weights = get_class_weights(data, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    pbar = trange(epochs, desc="Training")
    for epoch in pbar:
        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        train_loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc = (
            (logits[data.train_mask].argmax(1) == data.y[data.train_mask])
            .float()
            .mean()
            .item()
        )

        # --- Validation step ---
        val_loss, val_acc = evaluate(model, data, data.val_mask, criterion)

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        pbar.set_postfix(
            {
                "t_loss": f"{train_loss.item():.4f}",
                "v_loss": f"{val_loss:.4f}",
                "v_acc": f"{val_acc:.4f}",
            }
        )

    model.load_state_dict(best_state)
    print(f"\n✅ Training complete | Best Val Acc: {best_val_acc:.4f}")
    return history, best_state
