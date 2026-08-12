# src/models/train_vae_from_pkl.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# =====================================================
# Sampling Layer
# =====================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# =====================================================
# VAE Model
# =====================================================
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

    def compute_vae_loss(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(data - reconstruction), axis=1)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )

        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.compute_vae_loss(data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        total_loss, reconstruction_loss, kl_loss = self.compute_vae_loss(data)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# =====================================================
# Data helpers
# =====================================================
def load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def is_vector_column(series: pd.Series) -> bool:
    if len(series) == 0:
        return False

    clean = series.dropna()

    if len(clean) == 0:
        return False

    sample = clean.iloc[0]
    return isinstance(sample, (list, tuple, np.ndarray))


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "label",
        "Label",
        "category",
        "Category",
        "target",
        "Target",
        "y",
        "Y",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def extract_features(
    obj: Any,
    feature_column: Optional[str | list[str]] = None,
    label_column: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    y = None

    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            obj = obj.reshape(-1, 1)
        return obj.astype(np.float32), None

    if isinstance(obj, dict):
        if "X" in obj:
            X = np.asarray(obj["X"], dtype=np.float32)
            y = obj.get("y", None)
            return X, y

        if "features" in obj:
            X = np.asarray(obj["features"], dtype=np.float32)
            y = obj.get("labels", None)
            return X, y

        raise ValueError(
            f"Dict no soportado. Keys disponibles: {list(obj.keys())}. "
            "Usa keys 'X'/'y' o 'features'/'labels'."
        )

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        if label_column is None:
            label_column = detect_label_column(df)

        if label_column and label_column in df.columns:
            y = df[label_column].values
            df = df.drop(columns=[label_column])

        # Caso explícito: una columna o varias columnas
        if feature_column:
            if isinstance(feature_column, str):
                feature_column = [feature_column]

            arrays = []

            for col_name in feature_column:
                if col_name not in df.columns:
                    raise ValueError(
                        f"La columna '{col_name}' no existe. "
                        f"Columnas disponibles: {list(df.columns)}"
                    )

                col = df[col_name]

                if is_vector_column(col):
                    arr = np.vstack(col.values).astype(np.float32)
                else:
                    arr = col.values.reshape(-1, 1).astype(np.float32)

                arrays.append(arr)

            X = np.concatenate(arrays, axis=1).astype(np.float32)
            return X, y

        # Caso automático: una sola columna vectorial
        vector_cols = [c for c in df.columns if is_vector_column(df[c])]

        if len(vector_cols) == 1:
            print(f"[INFO] Detectada columna vectorial: {vector_cols[0]}")
            X = np.vstack(df[vector_cols[0]].values).astype(np.float32)
            return X, y

        if len(vector_cols) > 1:
            raise ValueError(
                "Se detectaron varias columnas vectoriales. "
                f"Indica una con --feature_column. Columnas: {vector_cols}"
            )

        # Fallback: columnas numéricas
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] == 0:
            raise ValueError(
                "No se encontraron features válidas. "
                "Usa --feature_column con la columna de embeddings."
            )

        print("[WARNING] Usando columnas numéricas:", numeric_df.columns.tolist())
        return numeric_df.values.astype(np.float32), y

    raise TypeError(f"Formato no soportado: {type(obj)}")


# =====================================================
# Model builder
# =====================================================
def build_vae(
    input_dim: int,
    latent_dim: int,
    hidden_dims: list[int],
    dropout: float,
    beta: float,
) -> Tuple[VAE, keras.Model, keras.Model]:

    encoder_inputs = keras.Input(shape=(input_dim,), name="input_features")
    x = encoder_inputs

    for h in hidden_dims:
        x = layers.Dense(h, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z_mean, z_log_var, z],
        name="encoder",
    )

    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = latent_inputs

    for h in reversed(hidden_dims):
        x = layers.Dense(h, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    decoder_outputs = layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction",
    )(x)

    decoder = keras.Model(
        latent_inputs,
        decoder_outputs,
        name="decoder",
    )

    vae = VAE(encoder, decoder, beta=beta, name="vae")

    return vae, encoder, decoder


# =====================================================
# Latent saving
# =====================================================
def save_latents(
    encoder: keras.Model,
    X: np.ndarray,
    data_dir: Path,
    split_name: str,
    y: Optional[np.ndarray] = None,
    feature_name: str = "feature",
):
    z_mean, z_log_var, z = encoder.predict(X)

    np.save(data_dir / f"z_mean_{split_name}.npy", z_mean)
    np.save(data_dir / f"z_log_var_{split_name}.npy", z_log_var)
    np.save(data_dir / f"z_{split_name}.npy", z)

    df = pd.DataFrame(
        z_mean,
        columns=[f"{feature_name}_latent_{i}" for i in range(z_mean.shape[1])],
    )

    if y is not None:
        df["label"] = y
        np.save(data_dir / f"y_{split_name}.npy", y)

    df.to_pickle(data_dir / f"{split_name}.pkl")


# =====================================================
# Public function used by main.py
# =====================================================
def train_vae_from_paths(
    train_pkl,
    val_pkl=None,
    test_pkl=None,
    feature_column=None,
    label_column=None,
    latent_dim=32,
    hidden_dims=None,
    dropout=0.1,
    beta=1.0,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    output_data_dir="data/05_vae_latents",
    output_model_dir="models/vae",
    feature_name="feature",
):
    if hidden_dims is None:
        hidden_dims = [512, 256]

    train_pkl = Path(train_pkl)
    val_pkl = Path(val_pkl) if val_pkl else None
    test_pkl = Path(test_pkl) if test_pkl else None
    output_data_dir = Path(output_data_dir)
    output_model_dir = Path(output_model_dir)

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = extract_features(
        load_pkl(train_pkl),
        feature_column=feature_column,
        label_column=label_column,
    )

    X_val, y_val = None, None
    if val_pkl and val_pkl.exists():
        X_val, y_val = extract_features(
            load_pkl(val_pkl),
            feature_column=feature_column,
            label_column=label_column,
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)

    if X_val is not None:
        X_val = scaler.transform(X_val).astype(np.float32)

    input_dim = X_train.shape[1]

    print(f"Input dim: {input_dim}")
    print(f"Latent dim: {latent_dim}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Data output dir: {output_data_dir}")
    print(f"Model output dir: {output_model_dir}")

    if latent_dim >= input_dim:
        print(
            f"[WARNING] latent_dim={latent_dim} >= input_dim={input_dim}. "
            "Para reducción real, usa latent_dim menor que input_dim."
        )

    vae, encoder, decoder = build_vae(
        input_dim=input_dim,
        latent_dim=int(latent_dim),
        hidden_dims=hidden_dims,
        dropout=float(dropout),
        beta=float(beta),
    )

    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate))
    )

    _ = vae(tf.zeros((1, input_dim), dtype=tf.float32))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_model_dir / "vae_best.weights.h5"),
            monitor="val_loss" if X_val is not None else "loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    vae.fit(
        X_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_data=X_val if X_val is not None else None,
        callbacks=callbacks,
    )

    # Models -> models/
    encoder.save(output_model_dir / "encoder.keras")
    decoder.save(output_model_dir / "decoder.keras")
    vae.save_weights(output_model_dir / "vae_final.weights.h5")
    joblib.dump(scaler, output_model_dir / "scaler.joblib")

    # Latent data -> data/
    save_latents(
        encoder=encoder,
        X=X_train,
        data_dir=output_data_dir,
        split_name="train",
        y=y_train,
        feature_name=feature_name,
    )

    if X_val is not None:
        save_latents(
            encoder=encoder,
            X=X_val,
            data_dir=output_data_dir,
            split_name="val",
            y=y_val,
            feature_name=feature_name,
        )

    if test_pkl and test_pkl.exists():
        X_test, y_test = extract_features(
            load_pkl(test_pkl),
            feature_column=feature_column,
            label_column=label_column,
        )

        X_test = scaler.transform(X_test).astype(np.float32)

        save_latents(
            encoder=encoder,
            X=X_test,
            data_dir=output_data_dir,
            split_name="test",
            y=y_test,
            feature_name=feature_name,
        )

    print("VAE training completed.")
    print(f"Models saved in: {output_model_dir}")
    print(f"Latent data saved in: {output_data_dir}")


# =====================================================
# CLI support
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Train VAE from PKL features")

    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--val_pkl", default=None)
    parser.add_argument("--test_pkl", default=None)

    parser.add_argument("--feature_column", nargs="+", default=None)
    parser.add_argument("--label_column", default=None)
    parser.add_argument("--feature_name", default="feature")

    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--output_data_dir", default="data/05_vae_latents")
    parser.add_argument("--output_model_dir", default="models/vae")

    args = parser.parse_args()

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]

    train_vae_from_paths(
        train_pkl=args.train_pkl,
        val_pkl=args.val_pkl,
        test_pkl=args.test_pkl,
        feature_column=args.feature_column,
        label_column=args.label_column,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_data_dir=args.output_data_dir,
        output_model_dir=args.output_model_dir,
        feature_name=args.feature_name,
    )


if __name__ == "__main__":
    main()