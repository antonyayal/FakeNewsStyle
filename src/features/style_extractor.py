# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional deps (graceful fallback)
try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

try:
    import textstat  # type: ignore
except Exception:  # pragma: no cover
    textstat = None  # type: ignore

try:
    # wordfreq es útil para "vocabulario" y OOV (calidad ortográfica aproximada)
    from wordfreq import zipf_frequency  # type: ignore
except Exception:  # pragma: no cover
    zipf_frequency = None  # type: ignore


# -------------------------
# Defaults
# -------------------------

DEFAULT_STYLE_INTENSIFIERS = {
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

DEFAULT_SPANISH_STOPWORDS = {
    # lista corta (puedes ampliarla)
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
    "las", "por", "un", "para", "con", "no", "una", "su", "al",
    "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este",
    "sí", "porque", "esta", "entre", "cuando", "muy", "sin",
    "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les",
    "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e",
    "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho",
    "quienes", "nada", "muchos", "cual", "poco", "ella", "estar",
    "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú",
    "te", "ti", "tu", "tus",
}


@dataclass
class StyleExtractorConfig:
    # spaCy model for Spanish (recommended)
    spacy_model: str = "es_core_news_sm"

    # Language hint for wordfreq zipf_frequency
    lang: str = "es"

    # Readability features
    compute_readability: bool = True

    # Formality score (Heylighen & Dewaele)
    compute_formality: bool = True

    # Spelling / OOV approximation
    compute_oov: bool = True
    # threshold for zipf_frequency(word) below which we consider "rare"/OOV-ish
    oov_zipf_threshold: float = 1.5

    # Diversity metrics
    compute_diversity: bool = True

    # Extra stylometric signals (punctuation, casing, etc.)
    extra_signals: bool = True

    # Stopwords for function-word / stopword ratios
    stopwords: Optional[set] = None

    # Sensational cues (optional)
    intensifiers: Optional[set] = None

    # Make numeric safe
    safe_numeric: bool = True


class StyleExtractor:
    """
    Style Feature Extractor (Spanish).

    Extrae features de estilo que suelen diferenciar fake vs real:
    - Legibilidad (Flesch–Szigriszt / IFSZ aproximado)
    - Formalidad (F-score de Heylighen & Dewaele)
    - Complejidad sintáctica / estructura
    - Léxico y diversidad (TTR + métricas robustas)
    - Distribución POS (estilo gramatical)
    - Calidad ortográfica aproximada (OOV por frecuencia léxica)
    - Señales de estilo (punct, quotes, uppercase, alargamientos, etc.)

    Diseño:
    - Input: texto (str)
    - Output: vector np.ndarray + feature_names() estables
    - Soporta export a PKL con ids para preservar alineación de filas.
    """

    def __init__(self, config: StyleExtractorConfig = StyleExtractorConfig()):
        self.config = config
        self.stopwords = config.stopwords or DEFAULT_SPANISH_STOPWORDS
        self.intensifiers = config.intensifiers or DEFAULT_STYLE_INTENSIFIERS

        # spaCy pipeline
        self._nlp = None
        if spacy is not None:
            try:
                self._nlp = spacy.load(config.spacy_model, disable=["ner"])
                # aseguramos sents
                if not self._nlp.has_pipe("sentencizer"):
                    # Si el modelo ya tiene parser, no hace falta; pero si no, sentencizer ayuda.
                    # No lo forzamos si ya hay parser:
                    pass
            except Exception:
                self._nlp = None  # fallback sin spaCy

        # cache
        self._feature_names: Optional[List[str]] = None

    # -------------------------
    # Public API
    # -------------------------

    def extract(self, text: str) -> np.ndarray:
        text_n = self._normalize(text)
        feats = self.extract_dict(text_n)
        vec = self._dict_to_vector(feats)
        return self._safe(vec)

    def extract_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # Si hay spaCy, procesamos con pipe (más rápido)
        if self._nlp is not None:
            docs = list(self._nlp.pipe((self._normalize(t) for t in texts), batch_size=batch_size))
            feat_dicts = [self._extract_style_features(self._normalize(texts[i]), docs[i]) for i in range(len(texts))]
        else:
            feat_dicts = [self._extract_style_features(self._normalize(t), None) for t in texts]

        # feature_names estables por orden de dict
        names = self.feature_names()
        X = np.vstack(
            [np.array([d.get(k, 0.0) for k in names], dtype=np.float32) for d in feat_dicts]
        ).astype(np.float32)
        return self._safe(X)

    def extract_with_names(self, text: str) -> Tuple[np.ndarray, List[str]]:
        vec = self.extract(text)
        return vec, self.feature_names()

    def extract_dict(self, text: str) -> Dict[str, float]:
        if self._nlp is not None:
            doc = self._nlp(text)
        else:
            doc = None
        return self._extract_style_features(text, doc)

    def feature_names(self) -> List[str]:
        if self._feature_names is not None:
            return self._feature_names

        # Generamos nombres desde un dict “dummy”
        dummy = self._extract_style_features("test", self._nlp("test") if self._nlp is not None else None)
        self._feature_names = list(dummy.keys())
        return self._feature_names

    def meta(self) -> Dict[str, Any]:
        return {
            "module": "StyleExtractor",
            "spacy_available": self._nlp is not None,
            "spacy_model": self.config.spacy_model,
            "textstat_available": textstat is not None,
            "wordfreq_available": zipf_frequency is not None,
            "compute_readability": self.config.compute_readability,
            "compute_formality": self.config.compute_formality,
            "compute_oov": self.config.compute_oov,
            "oov_zipf_threshold": self.config.oov_zipf_threshold,
            "compute_diversity": self.config.compute_diversity,
            "extra_signals": self.config.extra_signals,
            "feature_dim": len(self.feature_names()),
        }

    # -------------------------
    # Logging helpers (paradigma: escribir en logs/ como preprocess)
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
        parts = []
        for k, v in d.items():
            parts.append(f"{k}={v}")
        return " | ".join(parts)

    # -------------------------
    # Persistence (PKL) + LOG
    # -------------------------

    def save_features_pkl(
        self,
        texts: List[str],
        output_path,
        ids: Optional[List[Any]] = None,
        batch_size: int = 64,
        metadata: Optional[Dict[str, Any]] = None,
        # ✅ NUEVO: log (misma idea que preprocess: se pasa log_dir)
        log_dir=None,
        log_name: str = "style_extractor.log",
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
        - Si log_dir se especifica, se escribe un .log con:
          start/end, input/output, #samples, dim, batch_size, flags de config, errores.
        """
        import pickle
        from pathlib import Path

        t0 = time.time()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        log_file = None
        if log_dir is not None:
            log_root = self._ensure_dir(log_dir)
            log_file = str(Path(log_root) / log_name)
            self._append_log_line(log_file, "StyleExtractor: START save_features_pkl")
            self._append_log_line(
                log_file,
                self._format_kv(
                    {
                        "output_path": str(out),
                        "num_texts": len(texts),
                        "has_ids": ids is not None,
                        "batch_size": batch_size,
                        "spacy_available": (self._nlp is not None),
                        "spacy_model": self.config.spacy_model,
                        "compute_readability": self.config.compute_readability,
                        "compute_formality": self.config.compute_formality,
                        "compute_oov": self.config.compute_oov,
                        "compute_diversity": self.config.compute_diversity,
                        "extra_signals": self.config.extra_signals,
                        "oov_zipf_threshold": self.config.oov_zipf_threshold,
                    }
                ),
            )
            if metadata:
                self._append_log_line(log_file, f"metadata_keys={list(metadata.keys())}")

        # Validaciones
        if ids is not None and len(ids) != len(texts):
            msg = f"ids length ({len(ids)}) must match texts length ({len(texts)})."
            if log_file:
                self._append_log_line(log_file, f"ERROR: {msg}")
                self._append_log_line(log_file, "StyleExtractor: ABORT")
            raise ValueError(msg)

        try:
            X = self.extract_batch(texts, batch_size=batch_size)

            payload: Dict[str, Any] = {
                "X": X,
                "feature_names": self.feature_names(),
                "num_samples": int(X.shape[0]) if X.ndim == 2 else int(len(texts)),
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
                self._append_log_line(log_file, "StyleExtractor: END save_features_pkl")

        except Exception as e:
            if log_file:
                dt = time.time() - t0
                self._append_log_line(log_file, f"ERROR: {type(e).__name__}: {e}")
                self._append_log_line(log_file, f"elapsed_sec={dt:.3f}")
                self._append_log_line(log_file, "StyleExtractor: END (FAILED)")
            raise

    # -------------------------
    # Core extraction
    # -------------------------

    def _extract_style_features(self, text_raw: str, doc_raw) -> Dict[str, float]:
        # Tokenización “mínima” si no hay spaCy
        if doc_raw is None:
            tokens = re.findall(r"\w+|\S", text_raw, flags=re.UNICODE)
            words = [t for t in tokens if re.match(r"^\w+$", t, flags=re.UNICODE)]
            sents = self._split_sentences_simple(text_raw)
            # Sin POS/deps, devolvemos lo mejor posible con heurísticas
            return self._extract_without_spacy(text_raw, words, sents)

        tokens = [t for t in doc_raw if not t.is_space]
        words_alpha = [t for t in tokens if t.is_alpha]
        N = len(tokens)
        N_alpha = len(words_alpha)

        # Sentences
        try:
            sents = list(doc_raw.sents)
            num_sents = len(sents) or 1
        except Exception:
            sents = self._split_sentences_simple(text_raw)
            num_sents = len(sents) or 1

        # ------------
        # 1) Readability: Flesch–Szigriszt (IFSZ) aproximado
        # ------------
        ifsz = 0.0
        if self.config.compute_readability and textstat is not None:
            try:
                # OJO: textstat es principalmente inglés; esto es aproximado pero útil como proxy.
                wc = max(int(textstat.lexicon_count(text_raw, removepunct=True)), 1)
                sc = max(int(textstat.sentence_count(text_raw)), 1)
                syll = max(int(textstat.syllable_count(text_raw)), 1)

                syll_per_word = float(syll) / wc
                words_per_sent = float(wc) / sc

                # Fórmula usada en tu borrador
                ifsz = 206.835 - 62.3 * syll_per_word - words_per_sent
            except Exception:
                ifsz = 0.0

        # ------------
        # 2) Formality: F-score (Heylighen & Dewaele) basado en POS
        # ------------
        formality_f = 0.0
        if self.config.compute_formality:
            pos_counts = self._pos_counter(tokens)
            total_pos = sum(pos_counts.values()) or 1

            noun = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
            adj = pos_counts.get("ADJ", 0)
            prep = pos_counts.get("ADP", 0)
            det = pos_counts.get("DET", 0)

            pron = pos_counts.get("PRON", 0)
            verb = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
            adv = pos_counts.get("ADV", 0)
            interj = pos_counts.get("INTJ", 0)

            num_formal = (noun + adj + prep + det) - (pron + verb + adv + interj)
            formality_f = (float(num_formal) / total_pos) * 100.0

        # ------------
        # 3) Complejidad y estructura
        # ------------
        len_sent = float(N) / float(num_sents)

        # subordinadas (SCONJ) por oración
        num_sconj = sum(1 for t in tokens if t.pos_ == "SCONJ")
        sconj_per_sent = float(num_sconj) / float(num_sents)

        # profundidad sintáctica promedio (proxy de complejidad)
        avg_dep_depth = self._avg_dependency_depth(doc_raw)

        # clauses proxy: conteo de verbos (VERB/AUX) por oración
        num_verbs = sum(1 for t in tokens if t.pos_ in ("VERB", "AUX"))
        verbs_per_sent = float(num_verbs) / float(num_sents)

        # ------------
        # 4) Léxico y diversidad
        # ------------
        words_lower = [t.text.lower() for t in words_alpha]
        V = len(set(words_lower))
        ttr = float(V) / float(max(N_alpha, 1))
        redundancy = 1.0 - ttr

        # Métricas más robustas (menos sensibles a longitud)
        herdans_c = 0.0
        rttr = 0.0
        if self.config.compute_diversity:
            herdans_c = self._herdans_c(V, N_alpha)
            rttr = self._root_ttr(V, N_alpha)

        # ------------
        # 5) Distribución POS (estilo gramatical)
        # ------------
        pos_counts = self._pos_counter(tokens)
        total_pos = sum(pos_counts.values()) or 1

        pos_noun_ratio = float(pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)) / total_pos
        pos_verb_ratio = float(pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)) / total_pos
        pos_adj_ratio = float(pos_counts.get("ADJ", 0)) / total_pos
        pos_adv_ratio = float(pos_counts.get("ADV", 0)) / total_pos
        pos_pron_ratio = float(pos_counts.get("PRON", 0)) / total_pos
        pos_det_ratio = float(pos_counts.get("DET", 0)) / total_pos
        pos_adp_ratio = float(pos_counts.get("ADP", 0)) / total_pos

        # ------------
        # 6) Calidad ortográfica aproximada (OOV por frecuencia)
        # ------------
        error_rate = 0.0
        if self.config.compute_oov and zipf_frequency is not None:
            # palabras “muy raras” como proxy de OOV / errores
            rare = 0
            for w in words_lower:
                if not w.isalpha():
                    continue
                if zipf_frequency(w, self.config.lang) < self.config.oov_zipf_threshold:
                    rare += 1
            error_rate = float(rare) / float(max(len(words_lower), 1))

        # ------------
        # 7) Señales extra de estilo (muy útiles en fake news)
        # ------------
        extra = self._extra_style_signals(text_raw, tokens, words_lower, num_sents) if self.config.extra_signals else {}

        feats: Dict[str, float] = {
            # Legibilidad
            "ifsz": float(ifsz),

            # Formalidad
            "formality_f": float(formality_f),

            # Complejidad / estructura
            "len_sent": float(len_sent),
            "sconj_per_sent": float(sconj_per_sent),
            "avg_dep_depth": float(avg_dep_depth),
            "verbs_per_sent": float(verbs_per_sent),

            # Léxico y diversidad
            "ttr": float(ttr),
            "redundancy": float(redundancy),
            "herdans_c": float(herdans_c),
            "root_ttr": float(rttr),

            # Distribución POS
            "pos_noun_ratio": float(pos_noun_ratio),
            "pos_verb_ratio": float(pos_verb_ratio),
            "pos_adj_ratio": float(pos_adj_ratio),
            "pos_adv_ratio": float(pos_adv_ratio),
            "pos_pron_ratio": float(pos_pron_ratio),
            "pos_det_ratio": float(pos_det_ratio),
            "pos_adp_ratio": float(pos_adp_ratio),

            # Ortografía aproximada
            "error_rate": float(error_rate),
        }

        feats.update(extra)
        return self._safe_dict(feats)

    # -------------------------
    # Fallback sin spaCy (heurístico)
    # -------------------------

    def _extract_without_spacy(self, text_raw: str, words: List[str], sents: List[str]) -> Dict[str, float]:
        N_alpha = len(words)
        num_sents = len(sents) or 1
        words_lower = [w.lower() for w in words]
        V = len(set(words_lower))

        # Readability (aprox)
        ifsz = 0.0
        if self.config.compute_readability and textstat is not None:
            try:
                wc = max(int(textstat.lexicon_count(text_raw, removepunct=True)), 1)
                sc = max(int(textstat.sentence_count(text_raw)), 1)
                syll = max(int(textstat.syllable_count(text_raw)), 1)
                syll_per_word = float(syll) / wc
                words_per_sent = float(wc) / sc
                ifsz = 206.835 - 62.3 * syll_per_word - words_per_sent
            except Exception:
                ifsz = 0.0

        # Formality no disponible sin POS
        formality_f = 0.0

        len_sent = float(max(N_alpha, 0)) / float(num_sents)
        # subordinación proxy: conteo de "que", "porque", "aunque", etc.
        sconj_proxy = sum(1 for w in words_lower if w in {"que", "porque", "aunque", "si", "cuando", "mientras", "donde"})
        sconj_per_sent = float(sconj_proxy) / float(num_sents)

        ttr = float(V) / float(max(N_alpha, 1))
        redundancy = 1.0 - ttr
        herdans_c = self._herdans_c(V, N_alpha) if self.config.compute_diversity else 0.0
        rttr = self._root_ttr(V, N_alpha) if self.config.compute_diversity else 0.0

        error_rate = 0.0
        if self.config.compute_oov and zipf_frequency is not None:
            rare = 0
            for w in words_lower:
                if w.isalpha() and zipf_frequency(w, self.config.lang) < self.config.oov_zipf_threshold:
                    rare += 1
            error_rate = float(rare) / float(max(len(words_lower), 1))

        # Extra signals (sin tokens spaCy, pasamos tokens como lista “fake”)
        extra = self._extra_style_signals(text_raw, None, words_lower, num_sents) if self.config.extra_signals else {}

        feats: Dict[str, float] = {
            "ifsz": float(ifsz),
            "formality_f": float(formality_f),
            "len_sent": float(len_sent),
            "sconj_per_sent": float(sconj_per_sent),
            "avg_dep_depth": 0.0,
            "verbs_per_sent": 0.0,
            "ttr": float(ttr),
            "redundancy": float(redundancy),
            "herdans_c": float(herdans_c),
            "root_ttr": float(rttr),
            "pos_noun_ratio": 0.0,
            "pos_verb_ratio": 0.0,
            "pos_adj_ratio": 0.0,
            "pos_adv_ratio": 0.0,
            "pos_pron_ratio": 0.0,
            "pos_det_ratio": 0.0,
            "pos_adp_ratio": 0.0,
            "error_rate": float(error_rate),
        }
        feats.update(extra)
        return self._safe_dict(feats)

    # -------------------------
    # Extra stylometric signals (focus thesis)
    # -------------------------

    def _extra_style_signals(self, text: str, tokens, words_lower: List[str], num_sents: int) -> Dict[str, float]:
        # Longitudes
        num_chars = max(len(text), 1)
        num_words = max(len(words_lower), 1)

        # Puntuación y signos
        num_excl = text.count("!")
        num_q = text.count("?")
        num_ellipsis = len(re.findall(r"\.{3,}", text))
        num_quotes = text.count('"') + text.count("“") + text.count("”") + text.count("'")

        punct = sum(1 for c in text if (not c.isalnum() and not c.isspace()))
        punct_ratio = float(punct) / num_chars

        # Mayúsculas
        letters = sum(1 for c in text if c.isalpha())
        upper = sum(1 for c in text if c.isupper())
        uppercase_ratio = float(upper) / float(max(letters, 1))

        # Palabras largas / promedio
        avg_word_len = float(sum(len(w) for w in words_lower)) / num_words
        long_word_ratio = float(sum(1 for w in words_lower if len(w) >= 8)) / num_words

        # Stopwords / function words ratio (proxy de estilo más “natural”)
        stop_ratio = float(sum(1 for w in words_lower if w in self.stopwords)) / num_words

        # Dígitos y símbolos (clickbait suele usar números/porcentajes)
        digit_chars = sum(1 for c in text if c.isdigit())
        digit_ratio = float(digit_chars) / num_chars
        percent_ratio = float(text.count("%")) / num_chars

        # Repeticiones (alargamientos tipo "increíííble", "!!!!")
        repeated_char_ratio = self._repeated_char_ratio(text)

        # Intensificadores (lexical cues)
        intensifier_ratio = float(sum(1 for w in words_lower if w in self.intensifiers)) / num_words

        # Pasiva / impersonal (heurística en español):
        # - "se" pasiva/impersonal: frecuencia de "se" por oración
        se_count = sum(1 for w in words_lower if w == "se")
        se_per_sent = float(se_count) / float(max(num_sents, 1))

        # Modalidad / hedging (señales de falta de certeza)
        hedge_words = {"podría", "podrian", "posible", "posiblemente", "al parecer", "según", "presuntamente", "dicen"}
        hedge_count = 0
        for w in words_lower:
            if w in hedge_words:
                hedge_count += 1
        hedge_ratio = float(hedge_count) / num_words

        # Named-entity proxy (sin NER): proporción de tokens Capitalizados “internos”
        proper_like_ratio = 0.0
        if tokens is not None:
            propn = sum(1 for t in tokens if getattr(t, "pos_", "") == "PROPN")
            total = len(tokens) or 1
            proper_like_ratio = float(propn) / total

        # Burstiness: varianza de longitud de oración
        sent_lens = self._sentence_lengths_simple(text)
        burstiness = float(np.std(sent_lens)) if sent_lens else 0.0

        return {
            # punctuation / emphasis
            "sig_punct_ratio": punct_ratio,
            "sig_excl_per_sent": float(num_excl) / float(max(num_sents, 1)),
            "sig_q_per_sent": float(num_q) / float(max(num_sents, 1)),
            "sig_ellipsis_per_sent": float(num_ellipsis) / float(max(num_sents, 1)),
            "sig_quotes_ratio": float(num_quotes) / num_chars,

            # casing / formatting
            "sig_uppercase_ratio": uppercase_ratio,
            "sig_repeated_char_ratio": repeated_char_ratio,

            # lexical surface
            "sig_avg_word_len": avg_word_len,
            "sig_long_word_ratio": long_word_ratio,
            "sig_stopword_ratio": stop_ratio,
            "sig_digit_ratio": digit_ratio,
            "sig_percent_ratio": percent_ratio,
            "sig_intensifier_ratio": intensifier_ratio,

            # discourse/stance proxies
            "sig_se_per_sent": se_per_sent,
            "sig_hedge_ratio": hedge_ratio,
            "sig_proper_like_ratio": proper_like_ratio,
            "sig_burstiness": burstiness,
        }

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _split_sentences_simple(text: str) -> List[str]:
        sents = re.split(r"[.!?]+", text)
        return [s.strip() for s in sents if s.strip()]

    @staticmethod
    def _sentence_lengths_simple(text: str) -> List[int]:
        sents = re.split(r"[.!?]+", text)
        out = []
        for s in sents:
            s = s.strip()
            if not s:
                continue
            words = re.findall(r"\w+", s, flags=re.UNICODE)
            out.append(len(words))
        return out

    @staticmethod
    def _pos_counter(tokens) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in tokens:
            pos = getattr(t, "pos_", None) or "X"
            counts[pos] = counts.get(pos, 0) + 1
        return counts

    @staticmethod
    def _avg_dependency_depth(doc) -> float:
        """
        Proxy de complejidad: profundidad promedio en el árbol de dependencias.
        Requiere spaCy con parser.
        """
        try:
            depths = []
            for token in doc:
                if token.is_space:
                    continue
                d = 0
                cur = token
                while cur.head is not cur and d < 200:
                    d += 1
                    cur = cur.head
                depths.append(d)
            return float(np.mean(depths)) if depths else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _herdans_c(V: int, N: int) -> float:
        N = max(N, 1)
        V = max(V, 1)
        try:
            return float(math.log(V) / max(math.log(N), 1e-8))
        except Exception:
            return 0.0

    @staticmethod
    def _root_ttr(V: int, N: int) -> float:
        N = max(N, 1)
        V = max(V, 0)
        return float(V) / float(math.sqrt(N))

    @staticmethod
    def _repeated_char_ratio(text: str) -> float:
        if not text:
            return 0.0
        repeats = len(re.findall(r"(.)\1\1+", text))
        return float(repeats) / float(max(len(text), 1))

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
Herramientas utilizadas (dependencias / técnicas):
- spaCy: tokenización, segmentación en oraciones, POS tagging y (si el modelo lo incluye) dependencias.
- textstat: conteos y aproximación de sílabas/oraciones/palabras para legibilidad (IFSZ aproximado).
- wordfreq (zipf_frequency): proxy de "OOV"/rare words para calidad ortográfica aproximada.
- regex (re) + heurísticas: señales de estilo (puntuación, repetición, mayúsculas, etc.).
- numpy: construcción de vectores y estadísticas simples (media/std).
"""
