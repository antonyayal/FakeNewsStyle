from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import emoji

from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet


# Puedes ajustar/expandir esta lista según tu corpus (headlines/noticias).
DEFAULT_INTENSIFIERS = {
    "terrible",
    "increíble",
    "impactante",
    "alarmante",
    "horrible",
    "devastador",
    "escandaloso",
    "urgente",
    "brutal",
    "extremo",
    "indignante",
    "horrendo",
    "gravísimo",
    "espantoso",
    "inaudito",
    "bomba",
    "shock",
    "pánico",
    "catástrofe",
    "tragedia",
}


@dataclass
class EmotionExtractorConfig:
    lang: str = "es"
    use_preprocess_tweet: bool = False
    intensifiers: Optional[set] = None

    # Normalización de señales:
    # - "chars": ratios normalizados por longitud en caracteres
    # - "tokens": ratios normalizados por cantidad de tokens
    normalize_signals_by: str = "chars"  # "chars" | "tokens"

    # Si True, agrega señales adicionales (útiles para estilo emocional)
    extra_signals: bool = True

    # Si True, devuelve NaN-safe: reemplaza NaNs/inf por 0
    safe_numeric: bool = True


class EmotionExtractor:
    """
    Emotion Feature Extractor (Spanish) using pysentimiento.

    Output vector:
      [emotion_probs..., sentiment_probs..., signals...]

    - emotion_probs: probabilidades por clase emocional (según el modelo de pysentimiento)
    - sentiment_probs: probabilidades por clase POS/NEG/NEU (según el modelo de pysentimiento)
    - signals (base):
        - sig_exclam_ratio
        - sig_question_ratio
        - sig_uppercase_ratio
        - sig_emoji_ratio
        - sig_intensifier_ratio
      + signals (extra, si config.extra_signals=True):
        - sig_len_chars
        - sig_len_tokens
        - sig_punct_ratio
        - sig_digit_ratio
        - sig_repeat_exclam_ratio
        - sig_repeat_question_ratio
        - sig_elipsis_ratio
        - sig_quote_ratio

    Nota: Este extractor NO carga PKL. Solo procesa texto.
    Puede guardar su salida a PKL mediante save_features_pkl / extract_and_save_pkl.
    """

    def __init__(self, config: EmotionExtractorConfig = EmotionExtractorConfig()):
        self.config = config
        self.intensifiers = config.intensifiers or DEFAULT_INTENSIFIERS

        # Analyzers (cargan modelos transformer por debajo)
        self.emotion_analyzer = create_analyzer(task="emotion", lang=config.lang)
        self.sentiment_analyzer = create_analyzer(task="sentiment", lang=config.lang)

        # Cache de nombres (se llena en el primer predict real)
        self._emo_labels: Optional[List[str]] = None
        self._sent_labels: Optional[List[str]] = None

        # Cache de feature names final (cuando ya tenemos labels)
        self._feature_names: Optional[List[str]] = None

    # -------------------------
    # Logging helpers (igual paradigma que preprocess: escribir en logs/)
    # -------------------------

    @staticmethod
    def _ensure_dir(p):
        from pathlib import Path

        pp = Path(p)
        pp.mkdir(parents=True, exist_ok=True)
        return pp

    @staticmethod
    def _append_log_line(log_file, msg: str) -> None:
        from datetime import datetime
        from pathlib import Path

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    @staticmethod
    def _format_kv(d: Dict[str, Any]) -> str:
        return " | ".join([f"{k}={v}" for k, v in d.items()])

    # -------------------------
    # Public API
    # -------------------------

    def extract(self, text: str) -> np.ndarray:
        """
        Extrae el vector de features para 1 texto.
        """
        text_n = self._normalize_input(text)

        emo_vec, emo_labels = self._emotion_probs(text_n)
        sent_vec, sent_labels = self._sentiment_probs(text_n)
        signals = self._signals(text_n)

        self._set_labels_if_needed(emo_labels, sent_labels)

        vec = np.concatenate([emo_vec, sent_vec, signals]).astype(np.float32)
        return self._safe(vec)

    def extract_with_names(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Devuelve (vector, feature_names) para 1 texto.
        """
        vec = self.extract(text)
        names = self.feature_names()
        return vec, names

    def extract_dict(self, text: str) -> Dict[str, float]:
        """
        Devuelve un dict {feature_name: value} para 1 texto.
        """
        vec, names = self.extract_with_names(text)
        return {k: float(v) for k, v in zip(names, vec)}

    def extract_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extrae features para una lista de textos.
        Regresa matriz shape: (N, D).
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        texts_n = [self._normalize_input(t) for t in texts]

        emo_preds = self._predict_many(self.emotion_analyzer, texts_n, batch_size=batch_size)
        sent_preds = self._predict_many(self.sentiment_analyzer, texts_n, batch_size=batch_size)

        emo_labels = self._sorted_proba_keys(emo_preds[0].probas) if emo_preds else []
        sent_labels = self._sorted_proba_keys(sent_preds[0].probas) if sent_preds else []
        self._set_labels_if_needed(emo_labels, sent_labels)

        emo_mat = np.vstack([self._probas_to_vec(p.probas, emo_labels) for p in emo_preds]).astype(np.float32)
        sent_mat = np.vstack([self._probas_to_vec(p.probas, sent_labels) for p in sent_preds]).astype(np.float32)
        sig_mat = np.vstack([self._signals(t) for t in texts_n]).astype(np.float32)

        out = np.hstack([emo_mat, sent_mat, sig_mat]).astype(np.float32)
        return self._safe(out)

    def feature_names(self) -> List[str]:
        """
        Devuelve los nombres de features en el mismo orden del vector.
        """
        if self._feature_names is not None:
            return self._feature_names

        if self._emo_labels is None or self._sent_labels is None:
            _ = self.extract("test")

        names: List[str] = []
        names += [f"emo_{k}" for k in (self._emo_labels or [])]
        names += [f"sent_{k}" for k in (self._sent_labels or [])]
        names += self._signal_feature_names()

        self._feature_names = names
        return names

    def meta(self) -> Dict[str, Any]:
        """
        Metadata útil para logging/experimentos.
        """
        return {
            "module": "EmotionExtractor",
            "lang": self.config.lang,
            "use_preprocess_tweet": self.config.use_preprocess_tweet,
            "normalize_signals_by": self.config.normalize_signals_by,
            "extra_signals": self.config.extra_signals,
            "num_intensifiers": len(self.intensifiers),
            "emo_labels": self._emo_labels,
            "sent_labels": self._sent_labels,
            "feature_dim": len(self.feature_names()) if (self._emo_labels and self._sent_labels) else None,
        }

    # -------------------------
    # Persistence (PKL output) + LOG
    # -------------------------

    def save_features_pkl(
        self,
        texts: List[str],
        output_path,
        ids: Optional[List[Any]] = None,
        batch_size: int = 32,
        metadata: Optional[Dict[str, Any]] = None,
        # ✅ NUEVO: log (mismo patrón que preprocess)
        log_dir=None,
        log_name: str = "emotion_extractor.log",
    ) -> None:
        """
        Extrae features emocionales y los guarda en un archivo PKL.

        El PKL contiene:
        {
          "ids": Optional[List[Any]] (si se provee),
          "X": np.ndarray (N, D),
          "feature_names": List[str],
          "num_samples": int,
          "feature_dim": int,
          "meta": Dict[str, Any]
        }

        Logging:
        - Si log_dir se especifica, escribe un .log con:
          start/end, output_path, #samples, dim, batch_size, flags de config y errores.
        """
        import pickle
        from pathlib import Path

        t0 = time.time()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log_file = None
        if log_dir is not None:
            log_root = self._ensure_dir(log_dir)
            log_file = str(Path(log_root) / log_name)

            self._append_log_line(log_file, "EmotionExtractor: START save_features_pkl")
            self._append_log_line(
                log_file,
                self._format_kv(
                    {
                        "output_path": str(output_path),
                        "num_texts": len(texts),
                        "has_ids": ids is not None,
                        "batch_size": batch_size,
                        "lang": self.config.lang,
                        "use_preprocess_tweet": self.config.use_preprocess_tweet,
                        "normalize_signals_by": self.config.normalize_signals_by,
                        "extra_signals": self.config.extra_signals,
                        "num_intensifiers": len(self.intensifiers),
                    }
                ),
            )
            if metadata:
                self._append_log_line(log_file, f"metadata_keys={list(metadata.keys())}")

        if ids is not None and len(ids) != len(texts):
            msg = f"ids length ({len(ids)}) must match texts length ({len(texts)})."
            if log_file:
                self._append_log_line(log_file, f"ERROR: {msg}")
                self._append_log_line(log_file, "EmotionExtractor: ABORT")
            raise ValueError(msg)

        try:
            X = self.extract_batch(texts, batch_size=batch_size)
            feature_names = self.feature_names()

            payload: Dict[str, Any] = {
                "X": X,
                "feature_names": feature_names,
                "num_samples": int(X.shape[0]) if X.ndim == 2 else int(len(texts)),
                "feature_dim": int(X.shape[1]) if X.ndim == 2 else int(X.shape[0]),
                "meta": {**self.meta(), **(metadata or {})},
            }
            if ids is not None:
                payload["ids"] = list(ids)

            with open(output_path, "wb") as f:
                pickle.dump(payload, f)

            if log_file:
                dt = time.time() - t0
                self._append_log_line(
                    log_file,
                    self._format_kv(
                        {
                            "saved": output_path.name,
                            "num_samples": payload["num_samples"],
                            "feature_dim": payload["feature_dim"],
                            "elapsed_sec": f"{dt:.3f}",
                        }
                    ),
                )
                self._append_log_line(log_file, "EmotionExtractor: END save_features_pkl")

        except Exception as e:
            if log_file:
                dt = time.time() - t0
                self._append_log_line(log_file, f"ERROR: {type(e).__name__}: {e}")
                self._append_log_line(log_file, f"elapsed_sec={dt:.3f}")
                self._append_log_line(log_file, "EmotionExtractor: END (FAILED)")
            raise

    def extract_and_save_pkl(
        self,
        texts: List[str],
        output_path,
        ids: Optional[List[Any]] = None,
        batch_size: int = 32,
        metadata: Optional[Dict[str, Any]] = None,
        # ✅ PASA LOGS
        log_dir=None,
        log_name: str = "emotion_extractor.log",
    ) -> None:
        """
        Wrapper de alto nivel: extrae y guarda features emocionales en un PKL.
        """
        self.save_features_pkl(
            texts=texts,
            output_path=output_path,
            ids=ids,
            batch_size=batch_size,
            metadata=metadata,
            log_dir=log_dir,
            log_name=log_name,
        )

    # -------------------------
    # Internals
    # -------------------------

    def _set_labels_if_needed(self, emo_labels: List[str], sent_labels: List[str]) -> None:
        changed = False
        if self._emo_labels is None:
            self._emo_labels = list(emo_labels)
            changed = True
        if self._sent_labels is None:
            self._sent_labels = list(sent_labels)
            changed = True
        if changed:
            self._feature_names = None

    def _normalize_input(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if self.config.use_preprocess_tweet:
            text = preprocess_tweet(text, lang=self.config.lang)
        return text

    def _emotion_probs(self, text: str) -> Tuple[np.ndarray, List[str]]:
        pred = self.emotion_analyzer.predict(text or "")
        labels = self._sorted_proba_keys(pred.probas)
        vec = self._probas_to_vec(pred.probas, labels)
        return vec, labels

    def _sentiment_probs(self, text: str) -> Tuple[np.ndarray, List[str]]:
        pred = self.sentiment_analyzer.predict(text or "")
        labels = self._sorted_proba_keys(pred.probas)
        vec = self._probas_to_vec(pred.probas, labels)
        return vec, labels

    @staticmethod
    def _sorted_proba_keys(probas: Dict[str, float]) -> List[str]:
        return sorted(probas.keys())

    @staticmethod
    def _probas_to_vec(probas: Dict[str, float], labels: List[str]) -> np.ndarray:
        return np.array([float(probas.get(k, 0.0)) for k in labels], dtype=np.float32)

    def _predict_many(self, analyzer, texts: List[str], batch_size: int = 32):
        """
        Intenta hacer predicción en batch si el analyzer lo soporta.
        Si no, cae a loop.
        """
        if hasattr(analyzer, "predict"):
            try:
                out = analyzer.predict(texts)  # type: ignore[misc]
                if isinstance(out, list):
                    return out
            except Exception:
                pass

        preds = []
        for i in range(0, len(texts), max(batch_size, 1)):
            chunk = texts[i : i + batch_size]
            for t in chunk:
                preds.append(analyzer.predict(t))
        return preds

    def _signals(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+", (text or "").lower())
        num_tokens = max(len(tokens), 1)
        num_chars = max(len(text), 1)

        denom = num_chars if self.config.normalize_signals_by == "chars" else num_tokens

        exclam = (text or "").count("!")
        question = (text or "").count("?")

        exclam_ratio = exclam / max(denom, 1)
        question_ratio = question / max(denom, 1)

        uppercase_ratio = self._uppercase_ratio(text or "")
        emoji_ratio = self._emoji_ratio(text or "")
        intensifier_ratio = self._intensifier_ratio(tokens)

        base = np.array(
            [exclam_ratio, question_ratio, uppercase_ratio, emoji_ratio, intensifier_ratio],
            dtype=np.float32,
        )

        if not self.config.extra_signals:
            return base

        punct_ratio = self._punct_ratio(text or "")
        digit_ratio = self._digit_ratio(text or "")

        repeat_exclam_ratio = self._repeat_punct_ratio(text or "", r"!{2,}", denom)
        repeat_question_ratio = self._repeat_punct_ratio(text or "", r"\?{2,}", denom)

        elipsis_ratio = self._repeat_punct_ratio(text or "", r"\.{3,}", denom)
        quote_ratio = self._quote_ratio(text or "")

        extra = np.array(
            [
                float(num_chars),
                float(num_tokens),
                punct_ratio,
                digit_ratio,
                repeat_exclam_ratio,
                repeat_question_ratio,
                elipsis_ratio,
                quote_ratio,
            ],
            dtype=np.float32,
        )

        return np.concatenate([base, extra]).astype(np.float32)

    def _signal_feature_names(self) -> List[str]:
        names = [
            "sig_exclam_ratio",
            "sig_question_ratio",
            "sig_uppercase_ratio",
            "sig_emoji_ratio",
            "sig_intensifier_ratio",
        ]
        if not self.config.extra_signals:
            return names
        names += [
            "sig_len_chars",
            "sig_len_tokens",
            "sig_punct_ratio",
            "sig_digit_ratio",
            "sig_repeat_exclam_ratio",
            "sig_repeat_question_ratio",
            "sig_elipsis_ratio",
            "sig_quote_ratio",
        ]
        return names

    @staticmethod
    def _uppercase_ratio(text: str) -> float:
        uppercase = sum(1 for c in text if c.isupper())
        letters = sum(1 for c in text if c.isalpha())
        return float(uppercase) / max(letters, 1)

    @staticmethod
    def _emoji_ratio(text: str) -> float:
        emojis = [c for c in text if c in emoji.EMOJI_DATA]
        return float(len(emojis)) / max(len(text.split()), 1)

    def _intensifier_ratio(self, tokens_lower: List[str]) -> float:
        matches = sum(1 for t in tokens_lower if t in self.intensifiers)
        return float(matches) / max(len(tokens_lower), 1)

    @staticmethod
    def _punct_ratio(text: str) -> float:
        if not text:
            return 0.0
        punct = sum(1 for c in text if (not c.isalnum() and not c.isspace()))
        return float(punct) / max(len(text), 1)

    @staticmethod
    def _digit_ratio(text: str) -> float:
        if not text:
            return 0.0
        digits = sum(1 for c in text if c.isdigit())
        return float(digits) / max(len(text), 1)

    @staticmethod
    def _repeat_punct_ratio(text: str, pattern: str, denom: int) -> float:
        if not text:
            return 0.0
        repeats = len(re.findall(pattern, text))
        return float(repeats) / max(denom, 1)

    @staticmethod
    def _quote_ratio(text: str) -> float:
        if not text:
            return 0.0
        quotes = text.count('"') + text.count("“") + text.count("”") + text.count("'")
        return float(quotes) / max(len(text), 1)

    def _safe(self, x: np.ndarray) -> np.ndarray:
        if not self.config.safe_numeric:
            return x
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
