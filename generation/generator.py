# generation/generator.py

import httpx
import logging
import json 
import os, re

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
MODEL = os.getenv("VERDACT_MODEL", "phi3:mini")

SYSTEM_PROMPT = """You are Verdact, a compliance policy intelligence assistant.
Your job is to analyze retrieved policy chunks and produce a structured evidence report.
 
Rules:
- Every claim must be directly supported by the provided context.
- Never invent, infer, or paraphrase beyond what the source text says.
- Always cite the exact filename, section title, and page number for each claim.
- If the context does not contain enough information, set summary to explain that
  and return an empty evidence array. Do not fabricate claims.
 
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

def _extract_json(raw: str) -> dict:
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
    end = clean.find("}")

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
        f"POLICIES: {c.get('parent_context', c.get('text', ''))}"
        for c in chunks
    ])
 
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "stream": False,
                # num_ctx controls the context window sent to Ollama.
                # phi3:mini defaults to 2048 tokens which is easily exceeded
                # by 5 full-section chunks. 4096 is the safe minimum; 8192
                # if your machine has the RAM.
                "options": {"num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"},
                ],
            },
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
 
    # Log the raw response so you can see exactly what the model returned
    # when debugging parse failures. Remove or set to DEBUG in production.
    logger.debug("Raw model response:\n%s", raw)
 
    clean = _extract_json(raw)
 
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed after extraction. raw=\n%s\nerror=%s", raw, e)
        return {
            "summary": (
                "The model did not return valid JSON. "
                "This usually means the model is too small to reliably follow "
                "strict output format instructions. Consider switching to a "
                f"larger model by setting VERDACT_MODEL= in your .env. "
                f"Raw output has been logged. Parse error: {e}"
            ),
            "raw": raw,
            "evidence": [],
            "gaps": [],
        }
 