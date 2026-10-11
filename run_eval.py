import torch
from src.data import load_dataset, preprocess, build_graph
from src.model import GINClassifier
from src.evaluate import evaluate_test, plot_tsne

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = load_dataset("data/BitcoinHeistData.csv")
X_scaled, y_binary = preprocess(df, max_samples=50000, random_state=42)
data = build_graph(X_scaled, y_binary, k_neighbors=5, random_state=42).to(device)

# Load trained model
checkpoint = torch.load("outputs/gin_model.pt", map_location=device)
cfg = checkpoint["model_config"]
model = GINClassifier(**cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluate
evaluate_test(model, data, save_path="outputs/evaluation.png")
plot_tsne(model, data, device, save_path="outputs/tsne.png")