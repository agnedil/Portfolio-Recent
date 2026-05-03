"""
Chunking module: parent-child hierarchical splitting.

Implements the parent-child chunking strategy used for retrieval:

    1. Split Markdown by headers (H1 / H2 / H3) using ``MarkdownHeaderTextSplitter``.
    2. Merge chunks below ``MIN_PARENT_SIZE`` characters with their neighbors.
    3. Split chunks above ``MAX_PARENT_SIZE`` characters with a recursive splitter.
    4. Glue any tail chunks that are still too small back onto siblings.
    5. Build ``CHILD_CHUNK_SIZE``-character child chunks per parent for search.

Parent chunks become a wide-context store for the agent to consult on demand;
child chunks are embedded into the vector store for fast semantic + lexical
search.

Replace this module to plug in a different chunking strategy (semantic
chunking, late chunking, fixed windows, etc.). The contract required by
``indexing.py`` is the ``build_parent_child_chunks`` function returning
``(parent_pairs, children)``.

Documentation:
    - https://docs.langchain.com/oss/python/integrations/splitters
    - https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    MAX_PARENT_SIZE,
    MIN_PARENT_SIZE,
)

# Header levels used to seed the parent split.
HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]


def _merge_metadata(target: Document, incoming: dict) -> None:
    """Merge ``incoming`` metadata into ``target.metadata`` in-place.

    Same-key collisions are joined with " -> " to preserve the section path.
    """
    for k, v in incoming.items():
        if k in target.metadata:
            target.metadata[k] = f"{target.metadata[k]} -> {v}"
        else:
            target.metadata[k] = v


def merge_small_parents(chunks: List[Document], min_size: int) -> List[Document]:
    """Merge consecutive parent chunks until each meets a minimum size."""
    if not chunks:
        return []

    merged: List[Document] = []
    current: Optional[Document] = None

    for chunk in chunks:
        # Start a new accumulator or extend the current one.
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            _merge_metadata(current, chunk.metadata)

        # Flush once the accumulator reaches the minimum size.
        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None

    # Tail handling: append leftovers to the last merged chunk if possible.
    if current is not None:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            _merge_metadata(merged[-1], current.metadata)
        else:
            merged.append(current)

    return merged


def split_large_parents(
    chunks: List[Document], max_size: int, chunk_overlap: int
) -> List[Document]:
    """Recursively split parent chunks that exceed ``max_size``."""
    split_chunks: List[Document] = []

    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
            continue
        large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size, chunk_overlap=chunk_overlap
        )
        split_chunks.extend(large_splitter.split_documents([chunk]))

    return split_chunks


def clean_small_chunks(chunks: List[Document], min_size: int) -> List[Document]:
    """Glue any small leftovers from splitting back into neighboring chunks."""
    cleaned: List[Document] = []

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) >= min_size:
            cleaned.append(chunk)
            continue

        # Below-min chunk: prefer attaching to the previous accepted chunk.
        if cleaned:
            cleaned[-1].page_content += "\n\n" + chunk.page_content
            _merge_metadata(cleaned[-1], chunk.metadata)
        elif i < len(chunks) - 1:
            # No previous chunk yet: push small leading chunk into the next one.
            chunks[i + 1].page_content = (
                chunk.page_content + "\n\n" + chunks[i + 1].page_content
            )
            for k, v in chunk.metadata.items():
                if k in chunks[i + 1].metadata:
                    chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                else:
                    chunks[i + 1].metadata[k] = v
        else:
            # Single tiny chunk in the entire document: keep it.
            cleaned.append(chunk)

    return cleaned


def build_parent_child_chunks(
    md_text: str,
    source_id: str,
) -> Tuple[List[Tuple[str, Document]], List[Document]]:
    """Build parent and child chunks for a single Markdown document.

    Args:
        md_text: Raw Markdown content of the document.
        source_id: Identifier for the source (typically the filename stem).

    Returns:
        ``(parent_pairs, children)`` where ``parent_pairs`` is a list of
        ``(parent_id, Document)`` and ``children`` is the flat list of small
        searchable child Documents (each carrying ``parent_id`` and ``source``
        metadata so the agent can navigate from child to parent).
    """
    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    # Pipeline: header split -> merge small -> split large -> clean tail.
    parent_chunks = parent_splitter.split_text(md_text)
    merged = merge_small_parents(parent_chunks, MIN_PARENT_SIZE)
    split = split_large_parents(merged, MAX_PARENT_SIZE, CHILD_CHUNK_OVERLAP)
    cleaned_parents = clean_small_chunks(split, MIN_PARENT_SIZE)

    parent_pairs: List[Tuple[str, Document]] = []
    children: List[Document] = []

    for i, p_chunk in enumerate(cleaned_parents):
        parent_id = f"{source_id}_parent_{i}"
        # Stamp parent and source metadata before splitting so children inherit it.
        p_chunk.metadata.update({"source": f"{source_id}.pdf", "parent_id": parent_id})
        parent_pairs.append((parent_id, p_chunk))
        children.extend(child_splitter.split_documents([p_chunk]))

    return parent_pairs, children
