# src/models/kan.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# KAN Layer (RBF-based)
# =========================================================
class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 16):
        super().__init__()

        centers = torch.linspace(-3, 3, num_basis)
        self.register_buffer("centers", centers)

        self.log_gamma = nn.Parameter(torch.zeros(1))
        self.coeffs = nn.Parameter(torch.randn(in_dim, out_dim, num_basis) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        gamma = torch.exp(self.log_gamma)

        basis = torch.exp(-gamma * (x_expanded - self.centers) ** 2)
        out = torch.einsum("bin,ion->bo", basis, self.coeffs)

        return out + self.bias


# =========================================================
# KAN Classifier
# =========================================================
class KANClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_basis=16, dropout=0.2):
        super().__init__()

        self.model = nn.Sequential(
            KANLayer(input_dim, hidden_dim, num_basis),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            KANLayer(hidden_dim, hidden_dim // 2, num_basis),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


# =========================================================
# Data utils
# =========================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_labels(y):
    """
    Standard:
    1 = Fake
    0 = True / Real
    """

    y = np.asarray(y)

    # FIX definitivo para numpy.bool_
    if y.dtype == bool or str(y.dtype) == "bool":
        y = np.array([0 if bool(v) else 1 for v in y])

    elif y.dtype.kind in {"U", "S", "O"}:
        y = np.array([
            1 if str(v).strip().lower() in ["fake", "false", "falsa", "0"]
            else 0
            for v in y
        ])

    else:
        y = y.astype(np.float32)

    return y.astype(np.float32)


def extract_xy(obj, feature_key, label_key):
    if isinstance(obj, pd.DataFrame):

        if feature_key and feature_key in obj.columns:
            X = np.vstack(obj[feature_key].values)
        else:
            numeric_cols = obj.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != label_key]
            X = obj[numeric_cols].values

        y = obj[label_key].values

    elif isinstance(obj, dict):
        X = obj[feature_key] if feature_key else obj["X"]
        y = obj[label_key]

    else:
        raise ValueError("Unsupported format")

    X = np.asarray(X, dtype=np.float32)
    y = normalize_labels(y)

    return X, y


def make_loader(X, y, batch_size, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=shuffle
    )


# =========================================================
# Evaluation
# =========================================================
def evaluate(model, loader, device):
    model.eval()

    all_probs = []
    all_y = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_y.extend(y.numpy())

    all_probs = np.array(all_probs)
    all_y = np.array(all_y)

    preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_y, preds),
        "precision": precision_score(all_y, preds, zero_division=0),
        "recall": recall_score(all_y, preds, zero_division=0),
        "f1": f1_score(all_y, preds, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(all_y, all_probs)
    except:
        metrics["auc"] = float("nan")

    return metrics


# =========================================================
# Training
# =========================================================
def run_training(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_obj = load_pickle(args.train_pkl)
    val_obj = load_pickle(args.val_pkl)
    test_obj = load_pickle(args.test_pkl)

    X_train, y_train = extract_xy(train_obj, args.feature_key, args.label_key)
    X_val, y_val = extract_xy(val_obj, args.feature_key, args.label_key)
    X_test, y_test = extract_xy(test_obj, args.feature_key, args.label_key)

    print("Train labels:", np.unique(y_train, return_counts=True))
    print("Val labels:", np.unique(y_val, return_counts=True))
    print("Test labels:", np.unique(y_test, return_counts=True))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = make_loader(X_train, y_train, args.batch_size, True)
    val_loader = make_loader(X_val, y_val, args.batch_size)
    test_loader = make_loader(X_test, y_test, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KANClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_basis=args.num_basis,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1
    patience_counter = 0
    best_path = output_dir / "best_kan_model.pt"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1:03d} | Loss {total_loss:.4f} | Val F1 {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0

            torch.save(model.state_dict(), best_path)

        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))

    print("\nFinal Results")
    print("Train:", evaluate(model, train_loader, device))
    print("Val:", evaluate(model, val_loader, device))
    print("Test:", evaluate(model, test_loader, device))


# =========================================================
# Wrapper para main.py
# =========================================================
def train_kan_from_pkls(**kwargs):
    class Args:
        pass

    args = Args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    run_training(args)


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--val_pkl", required=True)
    parser.add_argument("--test_pkl", required=True)

    parser.add_argument("--feature_key", default=None)
    parser.add_argument("--label_key", default="label")

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_basis", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--output_dir", default="data/kan_outputs/merged")

    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()