# generation/generator.py

import httpx
import logging
import json 
import os, re
from json_repair import json_repair

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
MODEL = os.getenv("VERDACT_MODEL", "llama3.2:3b-instruct-q4_K_M")

OLLAMA_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=float(os.getenv("OLLAMA_READ_TIMEOUT", "120")),
    write=30.0,
    pool=10.0,
)

SYSTEM_PROMPT = """You are Verdact, a compliance policy intelligence assistant.
Your job is to analyze retrieved policy chunks and produce a structured evidence report.
 
Rules:
- Every claim must be directly supported by the provided context.
- Never invent, infer, or paraphrase beyond what the source text says.
- Always cite the exact filename, section title, and page number for each claim.
- If the context does not contain enough information, set summary to explain that
  and return an empty evidence array. Do not fabricate claims.
- Your entire response must begin with { and end with }. Do not write anything before or after the JSON object.
 
Output ONLY a single JSON object matching this exact schema — no preamble, no explanation, no markdown:
{
  "summary": "string",
  "evidence": [
    {
      "claim": "string",
      "citation": {
        "filename": "string",
        "section_title": "string",
        "page_number": 0,
        "excerpt": "string"
      }
    }
  ],
  "gaps": ["string"]
}"""

def _extract_json(raw: str) -> str:
    clean = raw.strip()

    clean = re.sub(r'^```(?:json)?\s*', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE)
    clean = clean.strip()

    # try the whole string first
    try:
        json.loads(clean)
        return clean
    except json.JSONDecodeError:
        pass
    
    start = clean.find("{")
    end = clean.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = clean[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
        
    return clean

async def generate_report_async(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        # Short-circuit: no chunks means the search returned nothing above the
        # score threshold. Return a meaningful response immediately without
        # calling the LLM — avoids hallucination on empty context entirely.
        return {
            "summary": "No relevant policy sections were found for this query. "
                       "The document may not cover this topic, or the ingestion "
                       "version filter may not match the uploaded document.",
            "evidence": [],
            "gaps": ["No matching policy content retrieved for this query."],
        }
 
    context = "\n\n".join([
        f"[Source: {c.get('filename')} | Section: {c.get('section_title')} | Page: {c.get('page_number')}]\n"
        f"{c.get('parent_context') or c.get('text', '')}"
        for c in chunks
    ])

    try:
 
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "stream": False,
                    # num_ctx controls the context window sent to Ollama.
                    # phi3:mini defaults to 2048 tokens which is easily exceeded
                    # by 5 full-section chunks. 4096 is the safe minimum; 8192
                    # if your machine has the RAM.
                    "options": {"num_ctx": 8192},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"},
                    ],
                },
            )
            response.raise_for_status()

    except httpx.ConnectError:
        logger.error("Could not connect to Ollama at %s", OLLAMA_URL)
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_URL}. "
            "Ensure Ollama is running: `ollama serve`"
        )
    except httpx.ReadTimeout:
        logger.error("Ollama timed out after %.0fs for model %s", OLLAMA_TIMEOUT.read, MODEL)
        raise RuntimeError(
            f"Ollama timed out after {OLLAMA_TIMEOUT.read:.0f}s. "
            "The model may be too large for available RAM, or still loading. "
            "Try setting OLLAMA_READ_TIMEOUT=180 in your .env if needed."
        )
    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP error: %s", e)
        raise RuntimeError(
            f"Ollama returned an error: {e.response.status_code} — {e.response.text}"
        )
    
    raw = response.json()["message"]["content"]
 
    # Log the raw response so you can see exactly what the model returned
    # when debugging parse failures. Remove or set to DEBUG in production.
    logger.debug("Raw model response:\n%s", raw)
 
    clean = _extract_json(raw)
 
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed, attempting repair. error=%s", e)
        try:
            repaired = json_repair(clean)
            return json.loads(repaired)
        except Exception:
            logger.error("JSON repair failed. raw=\n%s", raw)
            return {
                "summary": f"Model output could not be parsed. Parse error: {e}",
                "evidence": [],
                "gaps": [],
            }

 