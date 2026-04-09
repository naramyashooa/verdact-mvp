#ingestion/chunker.py

import fitz #pymupdf
import tiktoken 
from dataclasses import dataclass, field
from datetime import datetime
import re

HEADER_PATTERN = re.compile(
    r'(confidential|trust services|last revised|soc 2 type|page \d+|not for external)',
    re.IGNORECASE
)
ENCODER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 512 # 256
OVERLAP_TOKENS = 80 # 30, 50 

@dataclass
class Chunk: 
    text: str
    metadata: dict = field(default_factory=dict)

def split_sentences(text: str) -> list[str]:
    """
    Naïve sentence splitter that preserves the delimiter.
    For production, replace with spacy / nltk sent_tokenize.
    """

    import re
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

def split_with_overlap(full_text: str, section_title: str, base_meta: dict) -> list["Chunk"]:
    """
    Splits `full_section_text` into token-bounded chunks with sentence-aware
    overlap, storing the complete section as `parent_text` on every child.
    """
    sentences = split_sentences(full_text)
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
    
        # If adding this sentence would exceed the max, flush the current chunk and start a new one.
        if current_tokens + sentence_tokens > MAX_TOKENS and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **base_meta,
                    'section_title': section_title,
                    'parent_text': full_text,
                    'chunk_type': 'split',
                }
            ))

            # Building overlap: walk backwards through current_sentences keeping
            # Only those that fit within OVERLAP_TOKENS — full sentences only.
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                t = count_tokens(s)
                if overlap_tokens + t > OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += t

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # flash any remaining sentences
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **base_meta,
                "section_title": section_title,
                "parent_text": full_text, # true parent
                "chunk_type": "split"
            }
        )) 
    return chunks
def chunk_pdf(filename: str, ingestion_version: str = "1.0") -> list[Chunk]:
    doc = fitz.open(filename)
    chunks: list[Chunk] = []
    filename = filename.split("/")[-1]
    timestamp = datetime.now().isoformat()

    current_heading = "unknown_section"
    section_text_parts: list[str] = []
    section_start_page = 1  # NEW: track where section begins

    def flush_section(
        heading: str,
        parts: list[str],
        start_page: int,  # CHANGED: now uses section start page
        chunks: list[Chunk],
        filename: str,
        timestamp: str,
        ingestion_version: str,
    ) -> None:
        """Emit chunks for the accumulated section text."""

        if not parts:
            return

        full_text = " ".join(parts).strip()
        if not full_text:
            return

        base_meta = {
            "filename": filename,
            "page_number": start_page,  # FIXED: correct page reference
            "timestamp": timestamp,
            "ingestion_version": ingestion_version,
            "section_title": heading,
        }

        if count_tokens(full_text) <= MAX_TOKENS:
            chunks.append(Chunk(
                text=full_text,
                metadata={**base_meta, "parent_text": full_text, "chunk_type": "paragraph"},
            ))
        else:
            chunks.extend(split_with_overlap(full_text, heading, base_meta))

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:
                if block["type"] == 1:
                    flush_section(
                        current_heading, section_text_parts, section_start_page,
                        chunks, filename, timestamp, ingestion_version,
                    )
                    section_text_parts = []

                    chunks.append(Chunk(
                        text=f"[Image on page {page_num} under section '{current_heading}']",
                        metadata={
                            "filename": filename,
                            "page_number": page_num,
                            "timestamp": timestamp,
                            "ingestion_version": ingestion_version,
                            "section_title": current_heading,
                            "chunk_type": "image_placeholder",
                            "parent_text": None,
                        },
                    ))
                continue

            all_span_texts = [
                span["text"].strip()
                for line in block["lines"]
                for span in line["spans"]
                if span["text"].strip()
            ]

            line_count = len(block["lines"])
            multi_span_lines = sum(1 for line in block["lines"] if len(line["spans"]) > 2)
            is_table_like = line_count >= 3 and multi_span_lines >= line_count * 0.6

            if is_table_like:
                flush_section(
                    current_heading, section_text_parts, section_start_page,
                    chunks, filename, timestamp, ingestion_version,
                )
                section_text_parts = []

                table_text = " | ".join(all_span_texts)
                table_base_meta = {
                    "filename": filename,
                    "page_number": page_num,
                    "timestamp": timestamp,
                    "ingestion_version": ingestion_version,
                    "chunk_type": "table",
                }

                if count_tokens(table_text) <= MAX_TOKENS:
                    chunks.append(Chunk(
                        text=table_text,
                        metadata={
                            **table_base_meta,
                            "section_title": current_heading,
                            "parent_text": table_text,
                        },
                    ))
                else:
                    chunks.extend(
                        split_with_overlap(table_text, current_heading, table_base_meta)
                    )
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    if count_tokens(text) < 12 and HEADER_PATTERN.search(text):
                        continue

                    is_heading = span["size"] > 13 or bool(span["flags"] & 2**4)

                    if is_heading:
                        # flush current section before switching
                        flush_section(
                            current_heading, section_text_parts, section_start_page,
                            chunks, filename, timestamp, ingestion_version,
                        )
                        section_text_parts = []

                        current_heading = text
                        section_start_page = page_num  # NEW: reset start page
                    else:
                        section_text_parts.append(text)

    # flush once at end of document
    flush_section(
        current_heading, section_text_parts, section_start_page,
        chunks, filename, timestamp, ingestion_version,
    )

    return chunks