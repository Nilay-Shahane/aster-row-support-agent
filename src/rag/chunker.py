"""
chunker.py

Takes ParsedDocument objects (metadata + raw markdown body) from parser.py
and splits the body into chunks at '##' (H2) boundaries. Each chunk carries:
  - a heading breadcrumb (H1 > H2)
  - the raw markdown text of that section (shown to user/LLM as the source)
  - a cleaned, markdown-stripped version of the text (used for embedding)
  - key facts pulled from **bold** spans, for cheap deterministic grounding checks
  - the full front-matter metadata inherited from the parent document

This module does not know how to read files — that's parser.py's job. It
only knows how to turn (metadata, body) into a list of chunk dicts.
"""

import re
from dataclasses import dataclass
from rag.parser import ParsedDocument
from rag.loader import load_directory
from rag.parser import parse_document


@dataclass
class Chunk:
    chunk_id: str          # e.g. "RET-2026-01::standard-return-window"
    source_file: str
    heading_path: str       # e.g. "Standard return window"
    raw_text: str            # original markdown, shown to the user/LLM
    embedding_text: str      # markdown stripped, used to generate the embedding
    key_facts: list[str]     # bolded spans extracted from raw_text
    metadata: dict            # copied from the parent document's front matter


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def strip_markdown(text: str) -> str:
    """
    Removes markdown syntax characters so the embedding model sees clean
    prose instead of '**30 calendar days**' or '## Heading'. Embedding
    quality is generally a bit better on plain text. The original raw_text
    is kept separately for anything user-facing or cited.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                     # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)                          # italic
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)        # headings
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def extract_key_facts(raw_text: str) -> list[str]:
    """
    Pulls out every **bolded** span as a standalone fact string. Document
    authors used bold to mark the specific numbers/deadlines that matter
    ('30 calendar days of delivery', '$6.95 return shipping fee'). Keeping
    these separately lets an eval suite check whether a retrieved chunk
    literally contains a given number, without depending on the LLM to get
    the paraphrase right.
    """
    return BOLD_RE.findall(raw_text)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """
    Walks the body line by line. Tracks the current H1 (document title
    inside the body) and starts a new chunk every time it sees an H2. Lines
    under H3+ headings stay inside the current H2 chunk — sections in this
    corpus are short enough that splitting further would produce chunks too
    small to carry useful meaning on their own.
    """
    lines = doc.body.splitlines()

    h1_title = None
    chunks: list[Chunk] = []

    current_h2_heading = None
    current_h2_lines: list[str] = []

    def flush_current_section():
        """Turns the buffered H2 section into a Chunk, if there is one."""
        if current_h2_heading is None:
            return  # nothing buffered yet (e.g. stray text before the first ##)

        raw_section_text = "\n".join(current_h2_lines).strip()
        if not raw_section_text:
            return  # an H2 heading with no body text under it — nothing to index

        heading_path = current_h2_heading

        chunk = Chunk(
            chunk_id=f"{doc.metadata['document_id']}::{slugify(current_h2_heading)}",
            source_file=doc.source_file,
            heading_path=heading_path,
            raw_text=raw_section_text,
            embedding_text=strip_markdown(raw_section_text),
            key_facts=extract_key_facts(raw_section_text),
            metadata=doc.metadata,  # shared reference is fine; never mutated per-chunk
        )
        chunks.append(chunk)

    for line in lines:
        match = HEADING_RE.match(line)

        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            if level == 1:
                h1_title = heading_text
                continue  # H1 is the document title, not a chunk boundary

            if level == 2:
                # New H2 starts: close out whatever section was being built,
                # then start a fresh one.
                flush_current_section()
                current_h2_heading = heading_text
                current_h2_lines = []
                continue

            # H3+ headings: keep as content inside the current H2 chunk so the
            # heading text is preserved as context rather than discarded.
            current_h2_lines.append(line)
            continue

        # Regular body content for whichever section we're currently inside.
        current_h2_lines.append(line)

    flush_current_section()  # don't forget the last section in the file

    return chunks


def chunk_all(documents: list[ParsedDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


def to_embedding_records(chunks: list[Chunk]) -> list[dict]:
    """
    Final shape handed to the embedding step. Keeping this as its own
    function means chunking.py doesn't need to know anything about which
    embedding model or vector store is used downstream.
    """
    records = []
    for c in chunks:
        records.append({
            "id": c.chunk_id,
            "text": c.embedding_text,        # what actually gets embedded
            "raw_text": c.raw_text,           # shown to user/LLM as cited source
            "source_file": c.source_file,
            "heading_path": c.heading_path,
            "key_facts": c.key_facts,
            "document_id": c.metadata.get("document_id"),
            "status": c.metadata.get("status"),
            "policy_authority": c.metadata.get("policy_authority"),
            "effective_date": c.metadata.get("effective_date"),
            "supersedes": c.metadata.get("supersedes"),
            "audience": c.metadata.get("audience"),
            "authoritative": (
                c.metadata.get("status") == "active"
                and c.metadata.get("policy_authority") == "official"
            ),
        })
    return records


if __name__ == "__main__":
    from loader import load_directory
    from parser import parse_document

    KNOWLEDGE_BASE_PATH = "../../ai-agent-intern-test/knowledge-base"

    parsed_documents = []

    for filename, raw_text in load_directory(KNOWLEDGE_BASE_PATH):
        try:
            parsed_documents.append(parse_document(filename, raw_text))
        except ValueError as e:
            print(f"[chunking] WARNING: {e}")

    chunks = chunk_all(parsed_documents)
    records = to_embedding_records(chunks)

    print(f"""
        Documents parsed : {len(parsed_documents)}
        Chunks generated : {len(chunks)}
        Records created  : {len(records)}
    """)

    for record in records:
        print("""
    ===============================
    RECORD
    ===============================

    ID:
    {}

    Text:
    {}

    Raw Text:
    {}

    Source File:
    {}

    Heading:
    {}

    Key Facts:
    {}

    Document ID:
    {}

    Status:
    {}

    Policy Authority:
    {}

    Effective Date:
    {}

    Supersedes:
    {}

    Audience:
    {}

    Authoritative:
    {}

    """.format(
            record["id"],
            record["text"],
            record["raw_text"],
            record["source_file"],
            record["heading_path"],
            record["key_facts"],
            record["document_id"],
            record["status"],
            record["policy_authority"],
            record["effective_date"],
            record["supersedes"],
            record["audience"],
            record["authoritative"]
        ))