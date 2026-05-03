"""
Text processing module: PDF -> Markdown conversion.

Uses PyMuPDF / pymupdf4llm to extract clean Markdown from PDF documents.
Images are stripped and encoding issues are sanitized so downstream chunking
sees uniform text. Already-converted files are skipped unless overwrite is
requested.

Replace this module to plug in a different document loader (Unstructured,
Docling, HTML/DOCX-specific parsers, OCR pipelines, etc.). The contract
required by ``indexing.py`` is simply: produce ``.md`` files inside
``MARKDOWN_DIR``.

Documentation:
    - https://pymupdf.readthedocs.io/
    - https://github.com/pymupdf/PyMuPDF4LLM
"""

import logging
from pathlib import Path
from typing import List

import pymupdf
import pymupdf4llm

from config import DOCS_DIR, MARKDOWN_DIR

logger = logging.getLogger(__name__)


def pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    """Convert a single PDF file to a Markdown file inside ``output_dir``.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory where the Markdown file will be written.

    Returns:
        Path to the generated Markdown file.
    """
    doc = pymupdf.open(str(pdf_path))
    md = pymupdf4llm.to_markdown(
        doc,
        header=False,
        footer=False,
        page_separators=True,
        ignore_images=True,
        write_images=False,
        image_path=None,
    )
    # Sanitize encoding to avoid surrogate or invalid character failures.
    md_cleaned = md.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")

    output_path = (output_dir / Path(doc.name).stem).with_suffix(".md")
    output_path.write_bytes(md_cleaned.encode("utf-8"))
    return output_path


def pdfs_to_markdowns(
    pdf_dir: Path = DOCS_DIR,
    output_dir: Path = MARKDOWN_DIR,
    overwrite: bool = False,
) -> List[Path]:
    """Convert every PDF in ``pdf_dir`` to a Markdown file in ``output_dir``.

    Args:
        pdf_dir: Directory containing source PDFs.
        output_dir: Directory where Markdown files will be written.
        overwrite: If True, regenerate files that already exist.

    Returns:
        List of Markdown file paths produced or already present.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s/", pdf_dir)
        return produced

    for pdf_path in pdf_files:
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            logger.info("Converting %s -> %s", pdf_path.name, md_path.name)
            md_path = pdf_to_markdown(pdf_path, output_dir)
        else:
            logger.info("Skipping existing Markdown: %s", md_path.name)
        produced.append(md_path)

    return produced
