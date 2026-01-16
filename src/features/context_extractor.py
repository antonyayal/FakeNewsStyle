# src/features/context_extractor.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import math
import pickle
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


@dataclass
class ContextExtractorConfig:
    """
    Context features for FakeNewsCorpusSpanish.

    Input columns expected (configurable):
    - topic_column: e.g. "Topic"
    - source_column: e.g. "Source"
    - link_column: e.g. "Link"
    - author_column: optional (si existe)
    - date_column: optional (si existe; para calcular edad)

    Embeddings:
    - Se usa hash-embedding determinístico (feature hashing) para producir vectores
      de dimensión fija SIN entrenamiento.
    """
    topic_column: str = "Topic"
    source_column: str = "Source"
    link_column: str = "Link"
    id_column: str = "Id"  # útil para ids si quieres guardarlos desde DF

    author_column: Optional[str] = None         # e.g. "Author"
    date_column: Optional[str] = None           # e.g. "Date" / "Published" / "created_at"

    # Hash-embedding dimensions
    source_dim: int = 32
    domain_dim: int = 32
    topic_dim: int = 16
    author_dim: int = 16

    # Hashing behavior
    n_hashes: int = 2          # 2–4 típico
    signed: bool = True        # signed hashing reduce sesgo por colisiones
    l2_normalize: bool = True  # normaliza cada sub-embedding

    # Age calculation
    reference_datetime_utc: Optional[datetime] = None  # si None, usa now UTC
    age_cap_days: Optional[int] = 3650                 # cap 10 años
    missing_age_value: float = -1.0                    # sin fecha / parse falla

    # Numeric safety
    safe_numeric: bool = True


class ContextExtractor:
    """
    Context Feature Extractor.

    Características extraídas:
    - Dominio / Fuente:
        - Source (medio) -> hash-embedding
        - Domain del URL (extraído de Link) -> hash-embedding
    - Autor:
        - si hay columna de autor -> hash-embedding
        - si no, heurística conservadora desde URL (/author/<name>/ o ?author=)
    - Categoría temática (Topic) -> hash-embedding
    - Tiempo:
        - Edad de la noticia en días (si hay date_column), respecto a reference_datetime_utc

    Output vector (orden fijo):
      [source_emb..., domain_emb..., topic_emb..., author_emb..., age_days, flags...]

    Flags:
      ctx_has_link, ctx_has_domain, ctx_has_source, ctx_has_topic, ctx_has_author, ctx_has_date
    """

    def __init__(self, config: ContextExtractorConfig = ContextExtractorConfig()):
        self.config = config
        self._feature_names: Optional[List[str]] = None

    # -------------------------
    # Logging helpers (mismo paradigma que preprocess: logs/...)
    # -------------------------

    @staticmethod
    def _ensure_dir(p):
        pp = Path(p)
        pp.mkdir(parents=True, exist_ok=True)
        return pp

    @staticmethod
    def _append_log_line(log_file: Path, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    @staticmethod
    def _format_kv(d: Dict[str, Any]) -> str:
        return " | ".join([f"{k}={v}" for k, v in d.items()])

    # -------------------------
    # Public API
    # -------------------------

    def extract_from_row(self, row: Dict[str, Any]) -> np.ndarray:
        feats = self.extract_dict_from_row(row)
        vec = self._dict_to_vector(feats)
        return self._safe(vec)

    def extract_batch_from_rows(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)

        names = self.feature_names()
        X = np.zeros((len(rows), len(names)), dtype=np.float32)
        for i, row in enumerate(rows):
            d = self.extract_dict_from_row(row)
            X[i, :] = np.array([d.get(k, 0.0) for k in names], dtype=np.float32)
        return self._safe(X)

    def extract_dict_from_row(self, row: Dict[str, Any]) -> Dict[str, float]:
        c = self.config

        topic = self._get_str(row, c.topic_column)
        source = self._get_str(row, c.source_column)
        link = self._get_str(row, c.link_column)

        domain = self._domain_from_url(link) if link else ""
        author = self._get_author(row, link)

        # Hash embeddings
        src_vec = self._hash_embed(source, c.source_dim, field="source")
        dom_vec = self._hash_embed(domain, c.domain_dim, field="domain")
        top_vec = self._hash_embed(topic, c.topic_dim, field="topic")
        aut_vec = self._hash_embed(author, c.author_dim, field="author")

        # Age days
        age_days = self._age_in_days(row)

        # Build dict with stable keys
        feats: Dict[str, float] = {}
        feats.update({f"ctx_source_emb_{i}": float(src_vec[i]) for i in range(c.source_dim)})
        feats.update({f"ctx_domain_emb_{i}": float(dom_vec[i]) for i in range(c.domain_dim)})
        feats.update({f"ctx_topic_emb_{i}": float(top_vec[i]) for i in range(c.topic_dim)})
        feats.update({f"ctx_author_emb_{i}": float(aut_vec[i]) for i in range(c.author_dim)})

        feats["ctx_age_days"] = float(age_days)

        feats["ctx_has_link"] = 1.0 if bool(link) else 0.0
        feats["ctx_has_domain"] = 1.0 if bool(domain) else 0.0
        feats["ctx_has_source"] = 1.0 if bool(source) else 0.0
        feats["ctx_has_topic"] = 1.0 if bool(topic) else 0.0
        feats["ctx_has_author"] = 1.0 if bool(author) else 0.0
        feats["ctx_has_date"] = 1.0 if (c.date_column and self._get_str(row, c.date_column)) else 0.0

        return self._safe_dict(feats)

    def feature_names(self) -> List[str]:
        if self._feature_names is not None:
            return self._feature_names

        c = self.config
        names: List[str] = []
        names += [f"ctx_source_emb_{i}" for i in range(c.source_dim)]
        names += [f"ctx_domain_emb_{i}" for i in range(c.domain_dim)]
        names += [f"ctx_topic_emb_{i}" for i in range(c.topic_dim)]
        names += [f"ctx_author_emb_{i}" for i in range(c.author_dim)]
        names += ["ctx_age_days"]
        names += [
            "ctx_has_link",
            "ctx_has_domain",
            "ctx_has_source",
            "ctx_has_topic",
            "ctx_has_author",
            "ctx_has_date",
        ]
        self._feature_names = names
        return names

    def meta(self) -> Dict[str, Any]:
        return {
            "module": "ContextExtractor",
            "topic_column": self.config.topic_column,
            "source_column": self.config.source_column,
            "link_column": self.config.link_column,
            "id_column": self.config.id_column,
            "author_column": self.config.author_column,
            "date_column": self.config.date_column,
            "source_dim": self.config.source_dim,
            "domain_dim": self.config.domain_dim,
            "topic_dim": self.config.topic_dim,
            "author_dim": self.config.author_dim,
            "n_hashes": self.config.n_hashes,
            "signed": self.config.signed,
            "l2_normalize": self.config.l2_normalize,
            "reference_datetime_utc": (
                self.config.reference_datetime_utc.isoformat() if self.config.reference_datetime_utc else None
            ),
            "feature_dim": len(self.feature_names()),
        }

    # -------------------------
    # Persistence (PKL) + LOG
    # -------------------------

    def save_features_pkl(
        self,
        rows: List[Dict[str, Any]],
        output_path: str | Path,
        ids: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # ✅ NUEVO: logs (paradigma preprocess)
        log_dir: Optional[str | Path] = None,
        log_name: str = "context_extractor.log",
    ) -> None:
        """
        Guarda:
        {
          "ids": optional,
          "X": (N,D),
          "feature_names": [...],
          "num_samples": N,
          "feature_dim": D,
          "meta": {...}
        }

        Logging:
        - Si log_dir se especifica, escribe un .log con:
          start/end, output_path, #rows, dims, config y errores.
        """
        t0 = time.time()

        if ids is not None and len(ids) != len(rows):
            msg = f"ids length ({len(ids)}) must match rows length ({len(rows)})."
            # log si aplica
            if log_dir is not None:
                log_file = Path(self._ensure_dir(log_dir)) / log_name
                self._append_log_line(log_file, f"ERROR: {msg}")
            raise ValueError(msg)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        log_file: Optional[Path] = None
        if log_dir is not None:
            log_file = Path(self._ensure_dir(log_dir)) / log_name
            self._append_log_line(log_file, "ContextExtractor: START save_features_pkl")
            self._append_log_line(
                log_file,
                self._format_kv(
                    {
                        "output_path": str(out),
                        "num_rows": len(rows),
                        "has_ids": ids is not None,
                        "topic_column": self.config.topic_column,
                        "source_column": self.config.source_column,
                        "link_column": self.config.link_column,
                        "author_column": self.config.author_column,
                        "date_column": self.config.date_column,
                        "dims": f"src={self.config.source_dim},dom={self.config.domain_dim},top={self.config.topic_dim},aut={self.config.author_dim}",
                        "n_hashes": self.config.n_hashes,
                        "signed": self.config.signed,
                        "l2_normalize": self.config.l2_normalize,
                        "missing_age_value": self.config.missing_age_value,
                        "age_cap_days": self.config.age_cap_days,
                    }
                ),
            )
            if metadata:
                self._append_log_line(log_file, f"metadata_keys={list(metadata.keys())}")

        try:
            X = self.extract_batch_from_rows(rows)

            payload: Dict[str, Any] = {
                "X": X,
                "feature_names": self.feature_names(),
                "num_samples": int(X.shape[0]),
                "feature_dim": int(X.shape[1]) if X.ndim == 2 else int(X.shape[0]),
                "meta": {**self.meta(), **(metadata or {})},
            }
            if ids is not None:
                payload["ids"] = list(ids)

            with open(out, "wb") as f:
                pickle.dump(payload, f)

            if log_file:
                dt = time.time() - t0
                self._append_log_line(
                    log_file,
                    self._format_kv(
                        {
                            "saved": out.name,
                            "num_samples": payload["num_samples"],
                            "feature_dim": payload["feature_dim"],
                            "elapsed_sec": f"{dt:.3f}",
                        }
                    ),
                )
                self._append_log_line(log_file, "ContextExtractor: END save_features_pkl")

        except Exception as e:
            if log_file:
                dt = time.time() - t0
                self._append_log_line(log_file, f"ERROR: {type(e).__name__}: {e}")
                self._append_log_line(log_file, f"elapsed_sec={dt:.3f}")
                self._append_log_line(log_file, "ContextExtractor: END (FAILED)")
            raise

    def extract_and_save_from_dataframe(
        self,
        df,
        output_path: str | Path,
        ids_column: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # ✅ NUEVO: logs
        log_dir: Optional[str | Path] = None,
        log_name: str = "context_extractor.log",
    ) -> None:
        """
        Convenience: recibe un pandas.DataFrame, extrae rows e ids (por Id),
        y guarda un PKL estándar.
        """
        rows = df.to_dict(orient="records")
        ids = None

        id_col = ids_column or self.config.id_column
        if id_col and hasattr(df, "columns") and id_col in list(df.columns):
            ids = df[id_col].tolist()

        self.save_features_pkl(
            rows=rows,
            ids=ids,
            output_path=output_path,
            metadata=metadata,
            log_dir=log_dir,
            log_name=log_name,
        )

    # -------------------------
    # Internals
    # -------------------------

    @staticmethod
    def _get_str(row: Dict[str, Any], key: str) -> str:
        v = row.get(key, "")
        if v is None:
            return ""
        return str(v).strip()

    @staticmethod
    def _domain_from_url(url: str) -> str:
        try:
            u = url.strip()
            if not u:
                return ""
            parsed = urlparse(u)
            netloc = parsed.netloc or ""
            if not netloc and parsed.path and ("." in parsed.path.split("/")[0]):
                netloc = parsed.path.split("/")[0]
            netloc = netloc.lower().strip()
            netloc = re.sub(r"^www\.", "", netloc)
            return netloc
        except Exception:
            return ""

    def _get_author(self, row: Dict[str, Any], link: str) -> str:
        if self.config.author_column:
            a = self._get_str(row, self.config.author_column)
            if a:
                return a

        if not link:
            return ""
        try:
            parsed = urlparse(link)
            path = (parsed.path or "").strip("/")

            m = re.search(r"(?:^|/)(?:author|autor)/([^/]+)/?", path, flags=re.IGNORECASE)
            if m:
                return self._clean_author_token(m.group(1))

            q = parsed.query or ""
            m2 = re.search(r"(?:^|&)(?:author|autor)=([^&]+)", q, flags=re.IGNORECASE)
            if m2:
                return self._clean_author_token(m2.group(1))
        except Exception:
            pass

        return ""

    @staticmethod
    def _clean_author_token(token: str) -> str:
        t = token.replace("-", " ").replace("_", " ").strip()
        t = re.sub(r"\s+", " ", t)
        if len(t) < 2:
            return ""
        return t

    def _age_in_days(self, row: Dict[str, Any]) -> float:
        dc = self.config.date_column
        if not dc:
            return float(self.config.missing_age_value)

        raw = self._get_str(row, dc)
        if not raw:
            return float(self.config.missing_age_value)

        dt = self._parse_datetime(raw)
        if dt is None:
            return float(self.config.missing_age_value)

        ref = self.config.reference_datetime_utc or datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        delta = ref - dt
        age = float(delta.total_seconds() / 86400.0)

        if self.config.age_cap_days is not None:
            cap = float(self.config.age_cap_days)
            age = max(min(age, cap), -cap)

        return age

    @staticmethod
    def _parse_datetime(s: str) -> Optional[datetime]:
        t = s.strip()

        if re.fullmatch(r"\d{10,13}", t):
            try:
                v = int(t)
                if len(t) == 13:
                    v = v // 1000
                return datetime.fromtimestamp(v, tz=timezone.utc)
            except Exception:
                return None

        try:
            if t.endswith("Z"):
                t2 = t[:-1] + "+00:00"
                return datetime.fromisoformat(t2)
            return datetime.fromisoformat(t)
        except Exception:
            pass

        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(t, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue

        return None

    # -------------------------
    # Hash embedding
    # -------------------------

    def _hash_embed(self, value: str, dim: int, field: str) -> np.ndarray:
        if dim <= 0:
            return np.zeros((0,), dtype=np.float32)

        v = (value or "").strip().lower()
        if not v:
            return np.zeros((dim,), dtype=np.float32)

        out = np.zeros((dim,), dtype=np.float32)
        for j in range(max(self.config.n_hashes, 1)):
            idx, sign = self._hash_to_index_and_sign(f"{field}|{j}|{v}", dim)
            out[idx] += sign

        if self.config.l2_normalize:
            norm = float(np.linalg.norm(out))
            if norm > 0:
                out = out / norm

        return out.astype(np.float32)

    def _hash_to_index_and_sign(self, s: str, dim: int) -> Tuple[int, float]:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim

        if not self.config.signed:
            return idx, 1.0

        sign_bit = int(h[8:9], 16) % 2
        sign = -1.0 if sign_bit == 1 else 1.0
        return idx, sign

    # -------------------------
    # Safety
    # -------------------------

    @staticmethod
    def _dict_to_vector(feat_dict: Dict[str, float]) -> np.ndarray:
        keys = list(feat_dict.keys())
        vals = [feat_dict[k] for k in keys]
        return np.array(vals, dtype=np.float32)

    def _safe(self, x: np.ndarray) -> np.ndarray:
        if not self.config.safe_numeric:
            return x
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _safe_dict(self, d: Dict[str, float]) -> Dict[str, float]:
        if not self.config.safe_numeric:
            return d
        out: Dict[str, float] = {}
        for k, v in d.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                out[k] = 0.0
            else:
                out[k] = float(v)
        return out


"""
Herramientas utilizadas:
- python stdlib: urllib.parse (parseo de URL), datetime (cálculo de edad), hashlib (hash embedding), re (heurísticas)
- numpy: construcción de vectores y normalización
- feature hashing (hash embeddings) determinístico para categorías (Source/Domain/Topic/Author)
"""
