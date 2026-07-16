"""
Multimodal PDF ingestion for architecture diagrams and charts.

Standard text-extraction RAG cannot interpret visual content such as
architecture diagrams, flowcharts and charts embedded in PDFs. This module
bridges that gap by rasterising each PDF page to an image and asking a
vision-capable LLM (``gpt-4o``) to transcribe the visual content into
descriptive Markdown. The resulting text is suitable for ingestion into the
existing FAISS vector store alongside extracted text.

Public entry point:

    ``process_pdf_with_vision(file_path: str) -> str``

The function is resilient: page-level failures are logged and skipped so a
single corrupt page never aborts an entire ingestion run. When the optional
dependencies (``pdf2image``, ``Pillow``) or the OpenAI key are missing, the
function raises a clear, actionable ``RuntimeError`` instead of failing
mid-flight.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_DPI = 200  # balance between OCR fidelity and token cost
DEFAULT_IMAGE_FORMAT = "PNG"
SUPPORTED_PDF_SUFFIX = ".pdf"

# The vision prompt is intentionally prescriptive: it instructs the model to
# emit structured Markdown, to call out diagrams/charts/tables explicitly,
# and to preserve any visible labels, arrows and flow direction so the
# downstream retriever can match on domain vocabulary.
_VISION_SYSTEM_PROMPT = (
    "You are a meticulous technical document analyst. You receive an image "
    "of a single page from an engineering document and must convert its "
    "visual content into well-structured Markdown text. "
    "Rules: "
    "(1) Identify architecture diagrams, flowcharts, sequence diagrams and "
    "charts and describe them under a dedicated Markdown heading, e.g. "
    "'## Architecture Diagram'. "
    "(2) Preserve every visible label, component name, arrow direction and "
    "data-flow relationship in your description. "
    "(3) Transcribe any tables as GitHub-flavoured Markdown tables. "
    "(4) Capture caption text verbatim. "
    "(5) Output only Markdown; do not add commentary or code fences around "
    "the whole response. "
    "If the page contains no visual content, output a short note: "
    "'No diagrammatic content on this page.'"
)


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _validate_pdf_path(file_path: str) -> Path:
    """Validate that ``file_path`` points to a readable PDF.

    Args:
        file_path: Path supplied by the caller.

    Returns:
        A ``Path`` object for the resolved file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not ``.pdf``.
    """
    if not file_path:
        raise ValueError("file_path must be provided.")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != SUPPORTED_PDF_SUFFIX:
        raise ValueError(
            f"Unsupported file type: {path.suffix!r}. Expected a .pdf file."
        )
    return path


def _require_openai_key() -> str:
    """Return the configured OpenAI API key.

    Returns:
        The API key read from the environment.

    Raises:
        RuntimeError: If no OpenAI API key is configured.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Multimodal ingestion requires "
            "a vision-capable OpenAI model."
        )
    return api_key


# ---------------------------------------------------------------------------
# PDF -> images
# ---------------------------------------------------------------------------

def _render_pages_to_images(
    pdf_path: Path,
    dpi: int,
    image_format: str,
) -> List[Any]:
    """Rasterise every page of ``pdf_path`` to a PIL image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution at which to rasterise.
        image_format: Pillow image format label used for in-memory encoding.

    Returns:
        A list of PIL ``Image`` objects, one per page.

    Raises:
        RuntimeError: If ``pdf2image`` or its poppler dependency is missing.
    """
    try:
        from pdf2image import convert_from_path
    except Exception as exc:  # noqa: BLE001 - optional dependency
        raise RuntimeError(
            "pdf2image is not installed. Install with `pip install pdf2image` "
            "and ensure the poppler system binary is available."
        ) from exc

    try:
        images = convert_from_path(str(pdf_path), dpi=dpi, fmt=image_format.lower())
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise RuntimeError(
            f"Failed to convert PDF to images: {exc}. Verify that poppler is "
            "installed and the PDF is not corrupted or encrypted."
        ) from exc

    if not images:
        raise RuntimeError(f"PDF rendered zero pages: {pdf_path}")
    logger.info(
        "Rendered %d page(s) from %s at %d DPI.",
        len(images),
        pdf_path.name,
        dpi,
    )
    return images


def _encode_image(image: Any, image_format: str) -> str:
    """Encode a PIL image as a base64 data URL.

    Args:
        image: A PIL ``Image`` instance.
        image_format: Pillow format label (e.g. ``"PNG"``).

    Returns:
        A base64-encoded ``data:`` URL suitable for the OpenAI vision API.
    """
    import io

    buffer = io.BytesIO()
    # PIL expects a canonical format key (e.g. "PNG"); the format is written
    # into the buffer and read back as the MIME subtype.
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = image_format.lower()
    return f"data:image/{mime};base64,{encoded}"


# ---------------------------------------------------------------------------
# Vision analysis
# ---------------------------------------------------------------------------

def _build_vision_llm(api_key: str, model: str, temperature: float):
    """Construct the vision-capable LangChain chat model.

    Args:
        api_key: OpenAI API key.
        model: Model identifier (defaults to ``gpt-4o``).
        temperature: Sampling temperature for the transcription.

    Returns:
        A configured ``ChatOpenAI`` instance.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=int(os.getenv("MULTIMODAL_MAX_TOKENS", "1500")),
        request_timeout=float(os.getenv("MULTIMODAL_REQUEST_TIMEOUT", "60")),
    )


def _build_vision_messages(image_data_url: str):
    """Assemble the multimodal prompt messages for one page image.

    Args:
        image_data_url: Base64 ``data:`` URL of the page image.

    Returns:
        A list of LangChain message objects combining the system prompt and
        the page image.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(content=_VISION_SYSTEM_PROMPT)
    human = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analyse this page image. Transcribe any architecture "
                    "diagrams, charts, tables and captions into Markdown."
                ),
            },
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    )
    return [system, human]


def _analyze_image(llm: Any, image: Any, image_format: str, page_number: int) -> str:
    """Send a single page image to the vision model and return Markdown.

    Args:
        llm: The configured ``ChatOpenAI`` vision model.
        image: A PIL ``Image`` of the page.
        image_format: Pillow format label.
        page_number: 1-based page index used for logging and headings.

    Returns:
        The Markdown transcription for the page, or an empty string if the
        model returned nothing usable.
    """
    image_data_url = _encode_image(image, image_format)
    messages = _build_vision_messages(image_data_url)
    response = llm.invoke(messages)
    content = getattr(response, "content", "") or ""
    if isinstance(content, list):
        # Some LangChain versions return a list of content blocks.
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = content.strip()
    if not text:
        logger.warning("Vision model returned empty content for page %d.", page_number)
        return ""
    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_pdf_with_vision(
    file_path: str,
    model: str = DEFAULT_VISION_MODEL,
    dpi: int = DEFAULT_DPI,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    temperature: float = 0.1,
    max_pages: Optional[int] = None,
) -> str:
    """Convert a PDF into descriptive Markdown using a vision LLM.

    Each page is rasterised, encoded and sent to the vision model. Page
    outputs are concatenated with explicit page headings so the provenance
    of each section is preserved for downstream retrieval. Page-level
    failures are logged and skipped, ensuring ingestion completes for the
    rest of the document.

    Args:
        file_path: Absolute or relative path to the PDF file.
        model: OpenAI vision model identifier. Defaults to ``gpt-4o``.
        dpi: Rasterisation resolution. Higher values improve OCR fidelity at
            the cost of larger images and token usage.
        image_format: Pillow image format label for in-memory encoding.
        temperature: Sampling temperature for the transcription model.
        max_pages: Optional cap on the number of pages to process, applied
            after rasterisation. Useful for cost control on large PDFs.

    Returns:
        A single Markdown string combining the transcription of every page.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the path is not a PDF.
        RuntimeError: If required dependencies or credentials are missing.
    """
    pdf_path = _validate_pdf_path(file_path)
    api_key = _require_openai_key()

    images = _render_pages_to_images(pdf_path, dpi, image_format)
    if max_pages is not None:
        images = images[: max(0, int(max_pages))]

    llm = _build_vision_llm(api_key, model, temperature)

    sections: List[str] = []
    for index, image in enumerate(images, start=1):
        try:
            markdown = _analyze_image(llm, image, image_format, index)
        except Exception as exc:  # noqa: BLE001 - skip page, keep going
            logger.exception(
                "Vision analysis failed for page %d of %s; skipping.",
                index,
                pdf_path.name,
            )
            markdown = f"<!-- Page {index} skipped due to error: {exc} -->"

        if not markdown:
            markdown = f"<!-- Page {index} produced no content. -->"
        sections.append(f"<!-- Page {index} of {pdf_path.name} -->\n## Page {index}\n\n{markdown}")

    combined = "\n\n".join(sections).strip()
    logger.info(
        "Multimodal ingestion complete for %s: %d page(s), %d characters.",
        pdf_path.name,
        len(images),
        len(combined),
    )
    return combined


__all__ = ["process_pdf_with_vision"]
