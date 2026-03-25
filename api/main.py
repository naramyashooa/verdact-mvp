# api/main.py

# FIX 1 — Remove duplicate import. The original file had two consecutive
# `from fastapi import ...` lines. Line 2 re-imported without HTTPException,
# silently shadowing line 1's import. Collapsed into one clean import.
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from retrieval.searcher import hybrid_search
from ingestion.ingestor import ingest_document
from generation.generator import generate_report_async
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import asyncio
import shutil
import os
import io

load_dotenv()
app = FastAPI(title="Verdact API", version="1.0")


# ── Request bodies for POST endpoints ────────────────────────────────────────

class InvestigateRequest(BaseModel):
    query: str
    ingestion_version: str = "1.0"

class ExportRequest(BaseModel):
    query: str
    ingestion_version: str = "1.0"
    # FIX 4 — Accept a pre-built report so /export doesn't re-run the LLM.
    # The old /investigate/export re-ran hybrid_search + generate_report_async
    # independently of /investigate, which (a) doubled cost, (b) introduced
    # non-determinism — the exported PDF could differ from what the user saw
    # in the UI because the LLM generated a new response. The client now sends
    # the report it already has; the API only handles PDF rendering.
    report: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _escape_xml(text: str) -> str:
    """
    FIX 5 — Escape XML special characters before passing text to reportlab.
    ReportLab's Paragraph renderer is an XML parser internally. Any unescaped
    `&`, `<`, or `>` in claim/summary/gap text raises an XML parse error and
    crashes doc.build(). Policy documents frequently contain these characters
    (e.g. "CC6.1 & CC6.2", "< 24 hours").
    """
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _build_pdf(query: str, report: dict) -> io.BytesIO:
    """Render a report dict to a PDF BytesIO buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Verdact Evidence Report", styles["Title"]))
    content.append(Paragraph(_escape_xml(f"Query: {query}"), styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Summary", styles["Heading2"]))
    content.append(Paragraph(_escape_xml(report.get("summary", "")), styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Evidence", styles["Heading2"]))
    for item in report.get("evidence", []):
        cit = item.get("citation", {})
        content.append(Paragraph(
            _escape_xml(f"Claim: {item.get('claim', '')}"), styles["Normal"]
        ))
        content.append(Paragraph(
            _escape_xml(
                f"Source: {cit.get('filename')} | "
                f"{cit.get('section_title')} | "
                f"p.{cit.get('page_number')}"
            ),
            styles["Italic"],
        ))
        content.append(Spacer(1, 8))

    if report.get("gaps"):
        content.append(Paragraph("Gaps Identified", styles["Heading2"]))
        for gap in report.get("gaps", []):
            content.append(Paragraph(_escape_xml(f"• {gap}"), styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "product": "Verdact"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    ingestion_version: str = Form("1.0"),
):
    filename = file.filename
    if not filename or filename == "file":
        raise HTTPException(status_code=400, detail="Could not determine filename.")

    filename = os.path.basename(filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported, got: {filename!r}",
        )

    path = f"data/documents/{filename}"
    os.makedirs("data/documents", exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # FIX 3 — Run blocking ingest in a thread pool so it doesn't block the
    # event loop. ingest_document calls fitz (PDF parse), SentenceTransformer
    # (CPU-bound encoding), and Qdrant upsert (network I/O) — all synchronous.
    # Calling it directly inside async def stalls every other request for the
    # full duration of ingestion. asyncio.to_thread() offloads it to a worker
    # thread, keeping the event loop free.
    await asyncio.to_thread(ingest_document, path, ingestion_version)

    return {"status": "ingested", "filename": filename, "version": ingestion_version}


@app.get("/search")
def search(query: str, top_k: int = 10):
    chunks = hybrid_search(query, top_k=top_k)
    return {"query": query, "results": chunks}


# FIX 2 — /investigate changed from GET to POST.
# GET is semantically wrong for an operation that triggers LLM generation
# (a side-effectful, non-idempotent operation). More practically, long
# free-text queries in a GET query string hit browser/proxy URL length limits
# (~2000 chars for IE, ~8000 for most others). POST with a JSON body has no
# such limit and is the correct HTTP verb for "process this input and return
# a result" without storing state.
@app.post("/investigate")
async def investigate(body: InvestigateRequest):
    chunks = hybrid_search(
        body.query, top_k=5, ingestion_version=body.ingestion_version
    )
    report = await generate_report_async(body.query, chunks)
    return report


# FIX 2 + FIX 4 — /investigate/export changed to POST and accepts the report.
# POST: same URL-length and semantics reasoning as /investigate above.
# Pre-built report: the client sends the report JSON it received from
# /investigate. The server only renders it to PDF — no second LLM call,
# no risk of the exported PDF differing from what the user reviewed.
@app.post("/investigate/export")
async def investigate_export(body: ExportRequest):
    # FIX 3 — _build_pdf is CPU-bound (reportlab); run in thread pool.
    buffer = await asyncio.to_thread(_build_pdf, body.query, body.report)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=verdact_evidence_report.pdf"
        },
    )