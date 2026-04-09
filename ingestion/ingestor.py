#ingestion/ingestor.py - hybrid version

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SparseVectorParams, Modifier, SparseVector, FieldCondition, MatchValue, Filter
)
from fastembed import SparseTextEmbedding
from ingestion.chunker import chunk_pdf
from ingestion.embedder import embed, VECTOR_SIZE
import uuid
import os
import json

COLLECTION = os.getenv('COLLECTION_NAME', 'verdact_policies')
client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))
bm25_model = SparseTextEmbedding("Qdrant/bm25")

UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

def setup_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(modifier=Modifier.IDF)
            }
        )

# Explicit allowlist of metadata keys that are safe to store in Qdrant.
# Every value must be JSON-serializable (str, int, float, bool, None).
_PAYLOAD_KEYS = {
    "filename",
    "page_number",
    "timestamp",
    "ingestion_version",
    "section_title",
    "chunk_type",
}

def _delete_existing(filename: str) -> int:
    result = client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename),
                )
            ]
        ),
    )
    deleted = getattr(result, "deleted", None)
    return deleted if deleted is not None else 0

def _build_payload(chunk) -> dict:

    meta = chunk.metadata

    # Collect whitelisted scalar fields
    payload: dict = {
        key: meta[key]
        for key in _PAYLOAD_KEYS
        if key in meta
    }

    # Store the child chunk text (the actual retrieved snippet)
    payload["text"] = chunk.text

    # Store the parent section text under a single canonical key.
    # Coerce to str or None — never let a non-primitive through.
    parent = meta.get("parent_text")
    payload["parent_context"] = str(parent) if parent is not None else None

    # batch_id mirrors ingestion_version for phantom-file filtering
    payload["batch_id"] = meta.get("ingestion_version", "")

    return payload

 
def _validate_chunks(chunks: list) -> None:
    for i, chunk in enumerate(chunks):
        try:
            json.dumps(chunk.metadata)
        except TypeError as e:
            raise RuntimeError(
                f"Chunk {i} has non-serializable metadata: {e}\n"
                f"metadata={chunk.metadata}"
            )

def _upsert_in_batches(points: list) -> None:
    total = len(points)
    for start in range(0, total, UPSERT_BATCH_SIZE):
        batch = points[start: start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION, points=batch)
        end = min(start + UPSERT_BATCH_SIZE, total)
        print(f"  Upserted chunks {start + 1}–{end} of {total}")   


def ingest_document(filepath: str, ingestion_version: str = "1.0"):
    setup_collection()

    filename = os.path.basename(filepath)

    deleted = _delete_existing(filename)
    if deleted:
        print(f"Deleted {deleted} existing chunks for '{filename}' before re-ingest.")
 
    chunks = chunk_pdf(filepath, ingestion_version)

    if not chunks:
        print(f"No chunks extracted from {filepath} — skipping upsert.")
        return
    
    _validate_chunks(chunks)

    texts = [c.text for c in chunks]
    dense_vectors = embed(texts)
    sparse_vectors = list(bm25_model.embed(texts))
    points = [

        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vec,
                "bm25": SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
            },
            payload=_build_payload(chunk),
        )
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors)
    ]
 
    _upsert_in_batches(points)
    print(f"Ingested {len(points)} chunks from {filepath} under version '{ingestion_version}'")
 