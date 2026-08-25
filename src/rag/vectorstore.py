"""
vectorstore.py

Thin wrapper around a local, persistent Chroma collection. This is the
only file that knows Chroma's specific API — chunking.py and retriever.py
work with plain dicts in and dicts out, so swapping the vector store later
(FAISS, pgvector, a hosted store) means changing this file only.

Chroma metadata values must be str, int, float, or bool — no lists or
dicts. Fields like `key_facts` (a list) are flattened to a delimited
string on write and split back into a list on read.
"""

import os
import chromadb
from rag.embedder import Embedder

COLLECTION_NAME = "aster_and_row_kb"
KEY_FACTS_DELIMITER = "||"

# --- PATH FIX ---
# Calculate the absolute path of the directory containing this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Force the chroma_store to always live right next to this file
DEFAULT_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_store")


def _flatten_metadata(record: dict) -> dict:
    """
    Chroma metadata must be flat scalars. Converts list fields to a
    delimited string. raw_text is stored in metadata (not just as the
    Chroma "document") so a single read gives back everything needed to
    cite the chunk verbatim.
    """
    return {
        "source_file": record["source_file"],
        "heading_path": record["heading_path"],
        "document_id": record.get("document_id") or "",
        "status": record.get("status") or "",
        "policy_authority": record.get("policy_authority") or "",
        "effective_date": str(record.get("effective_date") or ""),
        "supersedes": record.get("supersedes") or "",
        "audience": record.get("audience") or "",
        "authoritative": bool(record.get("authoritative", False)),
        "key_facts": KEY_FACTS_DELIMITER.join(record.get("key_facts") or []),
        "raw_text": record["raw_text"],
    }


def _unflatten_metadata(metadata: dict) -> dict:
    """Reverses _flatten_metadata for read paths."""
    out = dict(metadata)
    out["key_facts"] = (
        metadata["key_facts"].split(KEY_FACTS_DELIMITER)
        if metadata.get("key_facts")
        else []
    )
    return out


class VectorStore:
    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        embedder: Embedder | None = None,
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity — standard for OpenAI embeddings
        )
        self.embedder = embedder or Embedder()

    def reset(self) -> None:
        """Wipes the collection. Useful for a clean re-ingest during development."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_records(self, records: list[dict]) -> None:
        """
        Takes the output of chunking.to_embedding_records() and writes it
        into Chroma. Embeds `text` (the markdown-stripped version) in one
        batched call, then upserts by chunk id — upsert means re-running
        ingest after a document edit overwrites the old chunk instead of
        duplicating it.
        """
        if not records:
            return

        ids = [r["id"] for r in records]
        texts_to_embed = [r["text"] for r in records]
        vectors = self.embedder.embed_texts(texts_to_embed)
        metadatas = [_flatten_metadata(r) for r in records]

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts_to_embed,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        k: int = 8,
        authoritative_only: bool = True,
    ) -> list[dict]:
        """
        Embeds the query and returns the top-k chunks. By default only
        searches chunks flagged authoritative=True at ingest time (active +
        official docs), so superseded and internal-only content never
        competes with current policy in a normal user-facing query.

        Set authoritative_only=False only for the specific paths that
        deliberately need visibility into non-authoritative content (e.g.
        detecting a prompt-injection attempt that references the migration
        scratchpad) — that should be an explicit, deliberate call, not the
        default retrieval path.
        """
        query_vector = self.embedder.embed_query(query_text)
        where_clause = {"authoritative": True} if authoritative_only else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where_clause,
        )

        return self._format_results(results)

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        """
        Chroma returns parallel lists (ids[0], documents[0], metadatas[0],
        distances[0]) for a single query. Zip them back into one dict per
        chunk, and convert Chroma's cosine *distance* into a similarity
        score (1 - distance) since "higher is more relevant" is easier to
        reason about in logs and downstream ranking.
        """
        if not results["ids"] or not results["ids"][0]:
            return []

        formatted = []
        for i in range(len(results["ids"][0])):
            metadata = _unflatten_metadata(results["metadatas"][0][i])
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                **metadata,
            })
        return formatted


if __name__ == "__main__":

    from rag.loader import load_directory
    from rag.parser import parse_document
    from rag.chunker import chunk_all, to_embedding_records

    # --------------------------------------------------
    # Resolve knowledge base path
    # backend/
    # ├── src/
    # │   └── rag/
    # │       └── vectorstore.py
    # └── ai-agent-intern-test/
    #     └── knowledge-base/
    # --------------------------------------------------

    KNOWLEDGE_BASE = os.path.abspath(
        os.path.join(
            BASE_DIR,
            "../../ai-agent-intern-test/knowledge-base"
        )
    )

    print("=" * 80)
    print("KNOWLEDGE BASE PATH")
    print(KNOWLEDGE_BASE)
    print("=" * 80)

    if not os.path.exists(KNOWLEDGE_BASE):
        raise FileNotFoundError(
            f"Knowledge base not found at: {KNOWLEDGE_BASE}"
        )

    # 1. Load files
    print("\nLoading documents...")
    raw_files = load_directory(KNOWLEDGE_BASE)

    print(f"Loaded {len(raw_files)} files")


    # 2. Parse documents individually
    docs = []

    for filename, raw_text in raw_files:
        try:
            docs.append(
                parse_document(
                    filename,
                    raw_text
                )
            )

        except ValueError as e:
            print(
                f"Skipping {filename}: {e}"
            )

    print(f"Parsed {len(docs)} documents")


    # 3. Chunk documents
    print("\nChunking documents...")

    chunks = chunk_all(docs)

    print(f"Created {len(chunks)} chunks")


    # 4. Convert chunks into embedding records
    records = to_embedding_records(chunks)

    print(f"Created {len(records)} embedding records")


    # 5. Initialize vector store
    store = VectorStore()


    # Uncomment when you want fresh ingestion
    # store.reset()
    # store.add_records(records)


    # 6. Retrieval test

    case = {
        "id": "standard-return-window",
        "query": "How long does a regular customer have to return an unused backpack?"
    }


    print("\n")
    print("=" * 80)
    print(f"TEST CASE: {case['id']}")
    print(f"QUERY: {case['query']}")
    print("=" * 80)


    hits = store.query(
        case["query"]
    )


    for h in hits[:5]:

        print("=" * 70)

        print(f"ID: {h['id']}")
        print(f"Source: {h['source_file']}")
        print(f"Heading: {h['heading_path']}")
        print(f"Score: {h['score']:.4f}")
        print(f"Authority: {h.get('policy_authority', 'N/A')}")
        print(f"Status: {h.get('status', 'N/A')}")

        print("\nText:")
        print(h["text"])

        print("=" * 70)