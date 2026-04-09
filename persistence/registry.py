import json
import os

REGISTRY_PATH = os.getenv("REGISTRY_PATH", "data/ingestion_registry.json")

def _load() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH) as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)

def _save(registry: dict) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def save_version(filename: str, version: str) -> None:
    """Record filename → ingestion_version after a successful ingest."""
    registry = _load()
    registry[filename] = version
    _save(registry)
    
def get_version(filename: str) -> str | None:
    """Return the ingestion_version for a previously ingested file, or None."""
    return _load().get(filename)

def all_documents() -> dict:
    """Return the full registry — {filename: ingestion_version}."""
    return _load()

def remove_document(filename: str) -> None:
    """Remove a document from the registry (e.g. after deletion)."""
    registry = _load()
    registry.pop(filename, None)
    _save(registry)