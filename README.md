# Bitcoin Heist Ransomware Detection using GIN

A **Graph Isomorphism Network (GIN)** that classifies Bitcoin addresses as ransomware or benign using the [BitcoinHeist Ransomware Address Dataset](https://archive.ics.uci.edu/dataset/526/bitcoinheist+ransomware+address+dataset).

**Author:** Sanchar Chakrabarty

---

## Project Structure

```
bitcoin-heist-gin/
├── main.py              # End-to-end pipeline entry point
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md        # Instructions for downloading the dataset
├── outputs/             # Auto-generated plots and model checkpoint
│   └── README.md
└── src/
    ├── __init__.py
    ├── data.py          # Dataset loading, preprocessing, graph construction
    ├── model.py         # GINLayer, GINClassifier, EmbeddingExtractor
    ├── train.py         # Training loop with cosine LR + class-weighted loss
    └── evaluate.py      # Metrics, plots (EDA, curves, confusion matrix, t-SNE)
```

---

## Pipeline Overview

1. **Load & EDA** — load `BitcoinHeistData.csv`, plot class distribution and feature correlations
2. **Preprocessing** — binary label encoding, stratified subsampling, feature scaling
3. **Graph Construction** — k-NN similarity graph (k=5) over node features
4. **GIN Model** — 3-layer GIN with Jumping Knowledge (JK) concatenation
5. **Training** — cosine annealing LR, class-weighted cross-entropy, best-val-acc checkpoint
6. **Evaluation** — classification report, ROC-AUC, confusion matrix, ROC curve
7. **t-SNE** — 2D projection of learned node embeddings

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **PyG note:** `torch_scatter` and `torch_sparse` must match your CUDA/CPU version.  
> See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### 2. Download the dataset

Place `BitcoinHeistData.csv` inside the `data/` folder (see `data/README.md`).

### 3. Run

```bash
python main.py --data_path data/BitcoinHeistData.csv
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `data/BitcoinHeistData.csv` | Path to dataset CSV |
| `--max_samples` | `50000` | Max nodes (stratified subsample) |
| `--k_neighbors` | `5` | k for k-NN graph |
| `--hidden_dim` | `128` | GIN hidden dimension |
| `--num_layers` | `3` | Number of GIN layers |
| `--dropout` | `0.3` | Dropout probability |
| `--epochs` | `100` | Training epochs |
| `--lr` | `0.003` | Initial learning rate |
| `--output_dir` | `outputs` | Where to save plots and checkpoint |

---

## Model Architecture

```
Input features (8-dim)
       │
  ┌────▼────┐
  │ GINLayer│  ×3 layers  (hidden_dim = 128)
  └────┬────┘
       │  JK Concatenation  →  384-dim
  ┌────▼────────────────┐
  │ MLP: 384 → 128 → 2 │
  └────────────────────-┘
       │
  Logits (Benign / Ransomware)
```

- **Jumping Knowledge (JK)** — outputs from all GIN layers are concatenated before the final MLP, preserving multi-hop structural information.
- **Class-weighted loss** — compensates for the heavily imbalanced dataset (~97% benign addresses).
- **Cosine annealing LR** — smoothly decays the learning rate over training.

---

## Results

Plots are saved to `outputs/` after each run:

| Plot | Content |
|---|---|
| `eda.png` | Class distribution + feature correlation heatmap |
| `training_curves.png` | Loss and accuracy over epochs |
| `evaluation.png` | Confusion matrix + ROC curve |
| `tsne.png` | t-SNE of GIN node embeddings |

---

## Dataset

- **Name:** BitcoinHeist Ransomware Address Dataset  
- **Source:** UCI ML Repository / Kaggle  
- **Features:** `year`, `day`, `length`, `weight`, `count`, `looped`, `neighbors`, `income`  
- **Labels:** 29 ransomware families + `white` (benign)  
- **Task:** Binary classification — ransomware (1) vs. benign (0)

---

## License

MIT
