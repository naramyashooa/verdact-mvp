#ingestion/embedder.py

from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer('all-mpnet-base-v2')
VECTOR_SIZE = 768 # all-mpnet-base-v2 output dimensions –– can change to 384 is performance issue persists (acceptable quality swap)

def embed(text: list[str]) -> list[list[float]]:
    vectors = MODEL.encode(
        text,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True # for costine similarity
    ) 
    return vectors.tolist()
