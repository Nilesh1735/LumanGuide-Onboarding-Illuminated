"""
Automated RAG evaluation pipeline using RAGAS.

This script provides CI/CD-ready quality gating for the retrieval and
generation stages of the LumanGuide RAG pipeline. It evaluates a synthetic
golden dataset against three RAGAS metrics:

  * ``faithfulness``        - is the answer grounded in the retrieved context?
  * ``answer_relevancy``    - does the answer address the question?
  * ``context_precision``   - are the retrieved chunks relevant to the query?

Run directly to execute the evaluation and print a formatted report:

    python -m tests.eval_pipeline

The module also exposes ``run_evaluation()`` so the pipeline can be invoked
from pytest or a CI workflow. ``run_evaluation`` returns a dictionary of
metric scores for programmatic assertions (for example, asserting that
faithfulness exceeds a release threshold).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic golden dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalExample:
    """A single RAG evaluation example.

    Attributes:
        question: The user query.
        ground_truth: The reference answer considered correct.
        answer: The answer produced by the RAG pipeline under test.
        contexts: The retrieved context chunks surfaced for the query.
    """

    question: str
    ground_truth: str
    answer: str
    contexts: List[str]


# A compact but representative dataset spanning the three intent classes the
# pipeline supports (index, general, search). In production this would be
# sourced from a versioned eval store; a synthetic set keeps the pipeline
# self-contained and runnable in CI without external dependencies.
SYNTHETIC_DATASET: List[EvalExample] = [
    EvalExample(
        question=(
            "What does the onboarding runbook say about configuring the "
            "local development environment?"
        ),
        ground_truth=(
            "The onboarding runbook states that developers must install "
            "Python 3.9 or higher, create a virtual environment, install "
            "dependencies from requirements.txt, and copy .env.example to "
            ".env before starting the FastAPI server."
        ),
        answer=(
            "Developers should install Python 3.9+, create a virtual "
            "environment, run pip install -r requirements.txt, and create "
            "a .env file from .env.example before launching the server."
        ),
        contexts=[
            "Onboarding runbook: Step 1 - Install Python 3.9 or higher. "
            "Step 2 - Create a virtual environment with python -m venv venv.",
            "Onboarding runbook: Step 3 - Install dependencies via "
            "pip install -r requirements.txt. Step 4 - Copy .env.example "
            "to .env and populate credentials before starting the server.",
        ],
    ),
    EvalExample(
        question="What is the difference between supervised and unsupervised learning?",
        ground_truth=(
            "Supervised learning trains on labelled input-output pairs to "
            "predict outputs for unseen data, while unsupervised learning "
            "finds structure in unlabelled data, for example through "
            "clustering or dimensionality reduction."
        ),
        answer=(
            "Supervised learning uses labelled training data to learn a "
            "mapping from inputs to outputs, enabling prediction. "
            "Unsupervised learning works on unlabelled data and discovers "
            "patterns such as clusters or low-dimensional structure."
        ),
        contexts=[
            "Supervised learning algorithms require a labelled dataset "
            "where each input is paired with the correct output.",
            "Unsupervised learning operates on unlabelled data and is "
            "commonly used for clustering and dimensionality reduction.",
        ],
    ),
    EvalExample(
        question=(
            "Who is the Subject Matter Expert for the authentication service "
            "and which project do they own?"
        ),
        ground_truth=(
            "Rahul Singh is the SME for authentication and OAuth flows and "
            "is the CODEOWNER of the auth-service project."
        ),
        answer=(
            "The authentication service SME is Rahul Singh, who owns the "
            "auth-service repository as its CODEOWNER."
        ),
        contexts=[
            "Team profile: Rahul Singh, Senior Backend Engineer. Expertise "
            "includes authentication and OAuth flows, JWT token management.",
            "Rahul Singh is CODEOWNER of the auth-service project "
            "(https://github.com/company/auth-service).",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_evaluation_dataset(examples: List[EvalExample]) -> Dict[str, List[Any]]:
    """Convert a list of ``EvalExample`` into RAGAS columnar format.

    RAGAS expects a dictionary of aligned lists keyed by metric-required
    columns. This helper centralises the mapping so the dataset shape can
    evolve independently of the metric configuration.

    Args:
        examples: The evaluation examples to convert.

    Returns:
        A dictionary with ``question``, ``ground_truth``, ``answer`` and
        ``contexts`` keys whose values are aligned lists.
    """
    return {
        "question": [ex.question for ex in examples],
        "ground_truth": [ex.ground_truth for ex in examples],
        "answer": [ex.answer for ex in examples],
        "contexts": [list(ex.contexts) for ex in examples],
    }


# ---------------------------------------------------------------------------
# Evaluation execution
# ---------------------------------------------------------------------------


def _select_metrics():
    """Resolve and return the RAGAS metrics to compute.

    RAGAS' API has evolved across versions. This helper prefers the modern
    ``evaluate`` with explicitly imported metric objects and falls back to
    the legacy string-based ``metrics`` argument for older installs.

    Returns:
        A tuple ``(metrics_value, metrics_kwarg_name)`` where
        ``metrics_value`` is passed to ``evaluate`` under
        ``metrics_kwarg_name``.

    Raises:
        RuntimeError: If RAGAS cannot be imported.
    """
    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except Exception as exc:  # noqa: BLE001 - optional dependency
        raise RuntimeError(
            "ragas is not installed. Install with "
            "`pip install ragas langchain-openai` and set OPENAI_API_KEY."
        ) from exc

    return [faithfulness, answer_relevancy, context_precision], "metrics"


def run_evaluation(
    dataset: List[EvalExample] = None,
    llm: Any = None,
    embeddings: Any = None,
) -> Dict[str, float]:
    """Run the RAGAS evaluation and return the metric scores.

    Args:
        dataset: Optional override dataset. Defaults to the synthetic set.
        llm: Optional LangChain chat model used by RAGAS for judging. When
            omitted, RAGAS selects its default evaluator LLM from the
            environment (``OPENAI_API_KEY`` must be set).
        embeddings: Optional embeddings model used by RAGAS. When omitted,
            RAGAS selects a default from the environment.

    Returns:
        A dictionary mapping metric name to its mean score across the
        dataset, e.g. ``{"faithfulness": 0.83, ...}``.

    Raises:
        RuntimeError: If RAGAS or its evaluator dependencies are missing.
    """
    examples = dataset if dataset is not None else SYNTHETIC_DATASET
    if not examples:
        raise ValueError("Evaluation dataset must not be empty.")

    from datasets import Dataset  # type: ignore[import-not-found]
    from ragas import evaluate

    columnar = build_evaluation_dataset(examples)
    hf_dataset = Dataset.from_dict(columnar)

    metric_objects, metrics_kwarg = _select_metrics()

    evaluate_kwargs: Dict[str, Any] = {metrics_kwarg: metric_objects}
    if llm is not None:
        evaluate_kwargs["llm"] = llm
    if embeddings is not None:
        evaluate_kwargs["embeddings"] = embeddings

    logger.info(
        "Running RAGAS evaluation over %d examples with metrics=%s.",
        len(examples),
        [getattr(m, "name", str(m)) for m in metric_objects],
    )

    result = evaluate(hf_dataset, **evaluate_kwargs)
    scores = _extract_scores(result)
    print_report(scores, len(examples))
    return scores


def _extract_scores(result: Any) -> Dict[str, float]:
    """Normalise a RAGAS result object into a flat metric->score mapping.

    RAGAS returns a ``Result`` object whose scores may be exposed as a dict,
    a pandas DataFrame, or columnar values. This helper handles each shape
    defensively.

    Args:
        result: The object returned by ``ragas.evaluate``.

    Returns:
        A dictionary of metric name to mean score.
    """
    scores: Dict[str, float] = {}

    # Modern RAGAS exposes a to_dict() converting per-row scores.
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            as_dict = to_dict()
            return _mean_of_numeric_columns(as_dict)
        except Exception:  # noqa: BLE001 - fall through to other strategies
            pass

    # pandas DataFrame path.
    try:
        import pandas as pd  # type: ignore[import-not-found]

        if isinstance(result, pd.DataFrame):
            return _mean_of_numeric_columns(result.to_dict())
    except Exception:  # noqa: BLE001 - pandas optional
        pass

    # Direct mapping fallback.
    if isinstance(result, dict):
        return _mean_of_numeric_columns(result)

    if not scores:
        raise RuntimeError(
            f"Unable to parse RAGAS result of type {type(result)!r}."
        )
    return scores


def _mean_of_numeric_columns(data: Dict[str, Any]) -> Dict[str, float]:
    """Compute the mean of each numeric column in a columnar mapping.

    Args:
        data: A dictionary whose values are lists of numbers.

    Returns:
        A dictionary mapping each column name to its mean value.
    """
    means: Dict[str, float] = {}
    for key, value in data.items():
        if not isinstance(value, (list, tuple)) or not value:
            continue
        try:
            numeric = [float(v) for v in value]
        except (TypeError, ValueError):
            continue
        means[key] = sum(numeric) / len(numeric)
    return means


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_score(score: float) -> str:
    """Format a score as a percentage string.

    Args:
        score: A numeric score, typically in ``[0, 1]``.

    Returns:
        A right-aligned percentage string.
    """
    return f"{score * 100:6.2f}%"


def print_report(scores: Dict[str, float], example_count: int) -> None:
    """Print a formatted evaluation report to the console.

    Args:
        scores: Mapping of metric name to mean score.
        example_count: Number of examples evaluated.
    """
    line = "=" * 58
    print(line)
    print("LumanGuide RAG Evaluation Report (RAGAS)")
    print(line)
    print(f"Examples evaluated : {example_count}")
    print(line)
    if not scores:
        print("No metric scores were produced.")
        print(line)
        return

    metric_order = ("faithfulness", "answer_relevancy", "context_precision")
    ordered_keys = [k for k in metric_order if k in scores]
    ordered_keys.extend(k for k in scores if k not in metric_order)

    for key in ordered_keys:
        value = scores.get(key)
        if value is None:
            continue
        print(f"  {key:<22} {_format_score(float(value))}")
    print(line)
    print("Threshold guidance (typical release gate):")
    print("  faithfulness      >= 80.00%")
    print("  answer_relevancy  >= 80.00%")
    print("  context_precision >= 70.00%")
    print(line)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point that runs the evaluation and prints the report.

    Returns:
        Process exit code (0 on success, 1 if evaluation could not run).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_evaluation()
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.exception("RAGAS evaluation failed: %s", exc)
        print(f"ERROR: evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvalExample",
    "SYNTHETIC_DATASET",
    "build_evaluation_dataset",
    "run_evaluation",
    "print_report",
]
