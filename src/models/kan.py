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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# KAN Layer
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
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_basis: int = 16,
        dropout: float = 0.2,
    ):
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
# Data utilities
# =========================================================
def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_labels(y):
    """
    Standard:
    1 = Fake
    0 = True / Real
    """

    y = np.asarray(y)

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


def extract_xy(obj, feature_key: str | None, label_key: str):
    if isinstance(obj, pd.DataFrame):
        if label_key not in obj.columns:
            raise ValueError(f"Label column '{label_key}' not found.")

        if feature_key and feature_key in obj.columns:
            X = np.vstack(obj[feature_key].values)
        else:
            numeric_cols = obj.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != label_key]
            X = obj[numeric_cols].values

        y = obj[label_key].values

    elif isinstance(obj, dict):
        if label_key not in obj:
            raise ValueError(f"Label key '{label_key}' not found.")

        if feature_key:
            X = obj[feature_key]
        else:
            for key in ["merged", "merged_latent", "vae_merged", "features", "X", "x"]:
                if key in obj:
                    X = obj[key]
                    break
            else:
                raise ValueError("No feature key found. Use --feature_key.")

        y = obj[label_key]

    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        X, y = obj

    else:
        raise ValueError(f"Unsupported PKL format: {type(obj)}")

    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = normalize_labels(y)

    return X, y


def make_loader(X, y, batch_size: int, shuffle: bool = False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# =========================================================
# Prediction utilities
# =========================================================
def predict_proba(model, loader, device):
    """
    Returns:
    - y_true
    - y_prob, probability of class 1 = Fake
    """

    model.eval()

    all_y = []
    all_probs = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            all_probs.extend(probs)
            all_y.extend(y.numpy())

    return np.array(all_y), np.array(all_probs)


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

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")

    print("Train labels:", np.unique(y_train, return_counts=True))
    print("Val labels:  ", np.unique(y_val, return_counts=True))
    print("Test labels: ", np.unique(y_test, return_counts=True))

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = KANClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_basis=args.num_basis,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    bad_epochs = 0
    best_path = output_dir / "best_kan_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "num_basis": args.num_basis,
                    "dropout": args.dropout,
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "label_standard": "1=Fake, 0=True/Real",
                },
                best_path,
            )

        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    y_train_true, y_train_prob = predict_proba(model, train_loader, device)
    y_val_true, y_val_prob = predict_proba(model, val_loader, device)
    y_test_true, y_test_prob = predict_proba(model, test_loader, device)

    predictions = {
        "train": {
            "y_true": y_train_true,
            "y_prob": y_train_prob,
        },
        "val": {
            "y_true": y_val_true,
            "y_prob": y_val_prob,
        },
        "test": {
            "y_true": y_test_true,
            "y_prob": y_test_prob,
        },
    }

    with open(output_dir / "predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    print(f"\nBest model saved at: {best_path}")
    print(f"Predictions saved at: {output_dir / 'predictions.pkl'}")

    return {
        "model": model,
        "device": device,
        "best_model_path": str(best_path),
        "predictions_path": str(output_dir / "predictions.pkl"),
        "predictions": predictions,
        "epochs_run": epoch,
        "best_val_loss": best_val_loss,
    }


# =========================================================
# Wrapper for main.py
# =========================================================
def train_kan_from_pkls(
    train_pkl,
    val_pkl,
    test_pkl,
    feature_key=None,
    label_key="label",
    hidden_dim=64,
    num_basis=16,
    dropout=0.2,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    patience=15,
    output_dir="data/07_kan_runs/merged",
):
    class Args:
        pass

    args = Args()

    args.train_pkl = train_pkl
    args.val_pkl = val_pkl
    args.test_pkl = test_pkl

    args.feature_key = feature_key
    args.label_key = label_key

    args.hidden_dim = hidden_dim
    args.num_basis = num_basis
    args.dropout = dropout

    args.epochs = epochs
    args.batch_size = batch_size
    args.lr = lr
    args.weight_decay = weight_decay
    args.patience = patience

    args.output_dir = output_dir

    return run_training(args)


# =========================================================
# Load trained KAN for inference
# =========================================================
def load_trained_kan(checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = KANClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_basis=checkpoint["num_basis"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]

    return model, scaler, device


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train KAN classifier and export probabilities"
    )

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

    parser.add_argument("--output_dir", default="data/07_kan_runs/merged")

    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()