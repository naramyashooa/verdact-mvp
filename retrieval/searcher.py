#retrieval/searcher.py

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from fastembed import SparseTextEmbedding
from ingestion.embedder import embed, VECTOR_SIZE
import os

client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))
COLLECTION = os.getenv('COLLECTION_NAME', 'verdact_policies')
bm25_model = SparseTextEmbedding("Qdrant/bm25")

RRF_SCORE_THRESHOLD = float(os.getenv("RRF_SCORE_THRESHOLD", "0.02"))

def hybrid_search(query: str, top_k: int = 5, ingestion_version: str = "1.0") -> list[dict]:
    dense_vec = embed([query])[0]
    sparse_vec = list(bm25_model.embed([query]))[0]

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch= [ # Prefetch lets you run multiple sub-queries (one dense and one sparse) in one go, and then combine results using a FusionQuery. Each sub-query is limited to 20 (top_k*2) results
            Prefetch(query=dense_vec, using="dense", limit=top_k*2),
            Prefetch(query= {"indices": sparse_vec.indices.tolist(), "values": sparse_vec.values.tolist()},
                using="bm25", 
                limit=top_k*2 ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),  # merge via Reciprocal Rank Fusion
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                        key="ingestion_version",

                        match=models.MatchValue(value=ingestion_version), # matchvalue receives the correct type for qdrant's filter
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )
    
    # Filter out results that fall below the RRF score threshold.
    # Returning an empty list is far better than returning low-relevance chunks
    # that cause the LLM to fabricate an answer — callers should handle [] by
    # telling the user "no relevant policy found" rather than proceeding.

    filtered = [
        {"score": r.score, **r.payload}
        for r in results.points
        if r.score >= RRF_SCORE_THRESHOLD
    ]
 
    return filtered