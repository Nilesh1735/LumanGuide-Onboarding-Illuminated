"""
Token-free intent classifier for query routing.

This module replaces the LLM-based query classification step with a small,
deterministic, locally trained model. The goal is to remove an LLM round
trip (and the associated latency and token cost) from the routing decision
while preserving the same three-class output contract the rest of the
pipeline expects: "index", "general" or "search".

Pipeline:
    1. ``bert-base-uncased`` is loaded via Hugging Face ``transformers`` and
       executed on the TensorFlow backend.
    2. The ``[CLS]`` token embedding is extracted for a curated set of
       synthetic training queries spanning the three intent classes.
    3. A scikit-learn ``RandomForestClassifier`` is trained on those
       embeddings and serialised to disk alongside the label encoder.
    4. ``classify_intent(query)`` embeds an incoming query with the same
       BERT/TensorFlow stack and returns the predicted intent string.

The model is trained lazily on first use (or eagerly via ``train()``) and
cached under ``src/ml_pipeline/.cache`` so subsequent process starts incur
no retraining cost. The module is self-contained: importing it never
requires TensorFlow or the model weights to be present; the heavyweight
imports happen only when ``classify_intent`` is first called.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_INTENTS: Tuple[str, ...] = ("index", "general", "search")
BERT_MODEL_NAME = "bert-base-uncased"
MAX_TOKEN_LENGTH = 32
EMBEDDING_DIM = 768  # hidden size of bert-base-uncased

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = _CACHE_DIR / "intent_random_forest.pkl"
LABEL_ENCODER_PATH = _CACHE_DIR / "intent_label_encoder.pkl"

# Default fallback when the classifier cannot produce a prediction. This is
# the safest route because it defers to the general-knowledge LLM path.
DEFAULT_INTENT = "general"

# Curated synthetic training corpus. Each tuple is (query, intent). The set
# is intentionally compact but representative; the Random Forest generalises
# to paraphrases via the BERT embedding space. Extend this list to improve
# coverage for domain-specific phrasings.
_TRAINING_DATA: List[Tuple[str, str]] = [
    # --- index: answerable from the uploaded / indexed documents ---------
    ("What does the document say about the deployment process?", "index"),
    ("Summarize the uploaded runbook.", "index"),
    ("Find the section on database configuration in the docs.", "index"),
    ("Which file describes the API authentication flow?", "index"),
    ("What is covered in the engineering onboarding guide?", "index"),
    ("Show me the policy for access control in the knowledge base.", "index"),
    ("Retrieve the definition of a service level objective.", "index"),
    ("What does the README state about environment variables?", "index"),
    ("Quote the troubleshooting steps from the manual.", "index"),
    ("List the prerequisites mentioned in the uploaded PDF.", "index"),
    ("Explain the architecture diagram in the document.", "index"),
    ("Where in the docs is the backup procedure defined?", "index"),

    # --- general: answerable from model / general knowledge -------------
    ("What is machine learning?", "general"),
    ("Explain the difference between TCP and UDP.", "general"),
    ("Who wrote the novel Nineteen Eighty-Four?", "general"),
    ("Define recursion in programming.", "general"),
    ("What is the capital of France?", "general"),
    ("How does a hash table work?", "general"),
    ("What is the boiling point of water in Celsius?", "general"),
    ("Explain the concept of object-oriented programming.", "general"),
    ("What is the Pythagorean theorem?", "general"),
    ("Describe the theory of relativity in simple terms.", "general"),
    ("What is a design pattern in software engineering?", "general"),
    ("Who founded the Python programming language?", "general"),

    # --- search: requires real-time / live web information --------------
    ("What is the latest news on the stock market today?", "search"),
    ("Current weather in Mumbai right now.", "search"),
    ("Latest version of the Python programming language.", "search"),
    ("What is the live score of the cricket match?", "search"),
    ("Today's exchange rate between USD and INR.", "search"),
    ("Breaking news about the election results.", "search"),
    ("What is trending on technology news sites now?", "search"),
    ("Recent changes to the GitHub pricing model.", "search"),
    ("Live traffic conditions on the highway.", "search"),
    ("Newest release of the transformers library.", "search"),
    ("What are the top headlines this morning?", "search"),
    ("Current price of Bitcoin.", "search"),
]


# ---------------------------------------------------------------------------
# Lazy backend singletons
# ---------------------------------------------------------------------------

# Heavy imports are deferred so that merely importing this module (for
# example, by a documentation generator) does not pull in TensorFlow.
_tf_pipeline = None
_rf_classifier = None
_label_encoder = None
_initialisation_lock_nonce = 0


def _load_embedding_pipeline():
    """Load the BERT model on the TensorFlow backend, returning a callable.

    The callable accepts a string and returns its ``[CLS]`` embedding as a
    1-D float32 numpy array of length ``EMBEDDING_DIM``.

    Returns:
        A function ``embed(text: str) -> np.ndarray``.
    """
    global _tf_pipeline

    if _tf_pipeline is not None:
        return _tf_pipeline

    logger.info("Loading BERT embedding pipeline (backend=TensorFlow).")

    # These imports are intentionally local: TensorFlow and transformers are
    # heavyweight and only needed when classification is actually used.
    import numpy as np
    import tensorflow as tf  # noqa: F401  (registers TF as the HF backend)
    from transformers import TFAutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    encoder = TFAutoModel.from_pretrained(BERT_MODEL_NAME)

    def embed(text: str):
        """Return the BERT ``[CLS]`` embedding for ``text``.

        Args:
            text: The raw input string to embed.

        Returns:
            A 1-D ``np.ndarray`` of shape ``(EMBEDDING_DIM,)`` and dtype
            ``float32``.
        """
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="tf",
        )
        outputs = encoder(encoded, training=False)
        # last_hidden_state shape: (1, seq_len, hidden_size). The first
        # token is always [CLS]; its hidden state is the pooled sentence
        # representation used for classification.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return np.asarray(cls_embedding[0], dtype="float32")

    _tf_pipeline = embed
    return embed


def _load_classifier():
    """Load (or train on demand) the Random Forest and label encoder.

    Returns:
        A tuple ``(classifier, label_encoder)`` where ``classifier`` is a
        fitted scikit-learn estimator exposing ``predict`` and
        ``predict_proba``, and ``label_encoder`` maps integer class ids to
        intent strings via ``inverse_transform``.
    """
    global _rf_classifier, _label_encoder

    if _rf_classifier is not None and _label_encoder is not None:
        return _rf_classifier, _label_encoder

    classifier, label_encoder = _try_load_from_cache()
    if classifier is not None and label_encoder is not None:
        _rf_classifier = classifier
        _label_encoder = label_encoder
        logger.info("Intent classifier loaded from cache: %s", MODEL_PATH)
        return _rf_classifier, _label_encoder

    # No cache present: train eagerly and persist for future starts.
    logger.info("No cached intent model found; training from synthetic corpus.")
    return train()


def _try_load_from_cache():
    """Attempt to load the serialised model and encoder from disk.

    Returns:
        A tuple ``(classifier, label_encoder)``. Both elements are ``None``
        if either artifact is missing or cannot be deserialised.
    """
    try:
        from sklearn.preprocessing import LabelEncoder  # noqa: F401

        if MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            with MODEL_PATH.open("rb") as fh:
                classifier = pickle.load(fh)
            with LABEL_ENCODER_PATH.open("rb") as fh:
                label_encoder = pickle.load(fh)
            return classifier, label_encoder
    except Exception as exc:  # noqa: BLE001 - cache is advisory, not required
        logger.warning("Could not load cached intent model: %s", exc)
    return None, None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    training_data: List[Tuple[str, str]] = None,
    persist: bool = True,
):
    """Train the Random Forest intent classifier.

    Args:
        training_data: Optional list of ``(query, intent)`` tuples. When
            omitted the module-level synthetic corpus is used.
        persist: When ``True`` (default) the fitted classifier and label
            encoder are written to ``MODEL_PATH`` and ``LABEL_ENCODER_PATH``.

    Returns:
        A tuple ``(classifier, label_encoder)``.

    Raises:
        ValueError: If the supplied training data is empty.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    if training_data is None:
        training_data = _TRAINING_DATA
    if not training_data:
        raise ValueError("Cannot train intent classifier on an empty corpus.")

    embed = _load_embedding_pipeline()

    # Validate labels against the supported contract before fitting.
    invalid = {intent for _, intent in training_data if intent not in VALID_INTENTS}
    if invalid:
        raise ValueError(
            f"Training data contains unsupported intents: {sorted(invalid)}. "
            f"Valid intents: {list(VALID_INTENTS)}."
        )

    logger.info(
        "Extracting BERT embeddings for %d training queries.", len(training_data)
    )
    features = np.vstack([embed(text) for text, _ in training_data])
    raw_labels = [intent for _, intent in training_data]

    label_encoder = LabelEncoder()
    label_encoder.fit(list(VALID_INTENTS))
    encoded_labels = label_encoder.transform(raw_labels)

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    classifier.fit(features, encoded_labels)

    train_accuracy = classifier.score(features, encoded_labels)
    logger.info(
        "Intent classifier trained: n_samples=%d n_classes=%d train_accuracy=%.4f",
        features.shape[0],
        len(label_encoder.classes_),
        train_accuracy,
    )

    if persist:
        try:
            with MODEL_PATH.open("wb") as fh:
                pickle.dump(classifier, fh)
            with LABEL_ENCODER_PATH.open("wb") as fh:
                pickle.dump(label_encoder, fh)
            logger.info("Intent classifier persisted to %s", MODEL_PATH)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("Failed to persist intent classifier: %s", exc)

    global _rf_classifier, _label_encoder
    _rf_classifier = classifier
    _label_encoder = label_encoder
    return classifier, label_encoder


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> str:
    """Predict the routing intent for a user query.

    The function is resilient by design: any failure in embedding or
    prediction falls back to ``DEFAULT_INTENT`` so the request pipeline is
    never broken by the classifier. Callers that require the raw
    probability distribution should use ``classify_intent_with_confidence``.

    Args:
        query: The raw user query string.

    Returns:
        One of ``"index"``, ``"general"`` or ``"search"``.
    """
    confidence, intent = classify_intent_with_confidence(query)
    return intent


def classify_intent_with_confidence(query: str) -> Tuple[float, str]:
    """Predict the intent and return its confidence score.

    Args:
        query: The raw user query string.

    Returns:
        A tuple ``(confidence, intent)`` where ``confidence`` is the
        predicted class probability in ``[0.0, 1.0]`` and ``intent`` is the
        predicted label string. On any error the tuple is
        ``(0.0, DEFAULT_INTENT)``.
    """
    if query is None or not query.strip():
        return 0.0, DEFAULT_INTENT

    try:
        import numpy as np

        embed = _load_embedding_pipeline()
        classifier, label_encoder = _load_classifier()

        embedding = embed(query).reshape(1, -1)
        probabilities = classifier.predict_proba(embedding)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        intent = str(label_encoder.inverse_transform([best_index])[0])

        if intent not in VALID_INTENTS:
            logger.warning(
                "Predicted intent %r is outside the valid set; using default.",
                intent,
            )
            return 0.0, DEFAULT_INTENT

        logger.debug(
            "Intent classified: query=%r intent=%s confidence=%.4f",
            query[:80],
            intent,
            confidence,
        )
        return confidence, intent
    except Exception as exc:  # noqa: BLE001 - classifier must never break routing
        logger.exception(
            "Intent classification failed; falling back to %s. Error: %s",
            DEFAULT_INTENT,
            exc,
        )
        return 0.0, DEFAULT_INTENT


# Allow ``python -m src.ml_pipeline.intent_classifier`` to (re)train the
# model eagerly, which is useful in CI or provisioning scripts.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train()
    print(f"Training complete. Artifacts written under: {_CACHE_DIR}")
    for sample_query in (
        "What does the onboarding doc say about VPN setup?",
        "What is the difference between RAM and ROM?",
        "What is the latest version of Node.js right now?",
    ):
        confidence, intent = classify_intent_with_confidence(sample_query)
        print(f"  [{intent:>7}] (p={confidence:.3f})  {sample_query}")


__all__ = [
    "VALID_INTENTS",
    "classify_intent",
    "classify_intent_with_confidence",
    "train",
]
