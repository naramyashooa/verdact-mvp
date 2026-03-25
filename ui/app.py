# ui/app.py

# FIX 6 — Move uuid import to module top level.
# The original imported uuid inside the button callback block. While Python
# allows this, importing inside a hot path (re-evaluated on every Streamlit
# rerun) is wasteful. Module-level imports are cached after the first load.
import uuid
import json
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Verdact", layout="wide", page_icon="🔍")
st.title("Verdact")
st.caption("The evidence engine that proves your company actually follows its security policies")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload a Policy Document")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded and st.button("Ingest PDF", type="primary"):
        session_id = f"v_{uuid.uuid4().hex[:6]}"

        # FIX 7 — Remove erroneous extra indent inside with-block.
        # The original had an extra level of indentation on the requests.post
        # call that didn't match the with: block — cosmetic but signals
        # copy-paste drift and makes the scope misleading.
        with st.spinner("Ingesting document..."):
            res = requests.post(
                f"{API_URL}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                data={"ingestion_version": session_id},
            )

        if res.status_code == 200:
            st.success(f"Ingested: {res.json()['filename']}, version: {session_id}")
            # Store both version AND filename so the investigation panel can
            # reference them without re-deriving from the response.
            st.session_state["current_version"] = session_id
        else:
            st.error(f"Ingestion failed ({res.status_code}): {res.text}")

with col2:
    st.subheader("Run an Investigation")
    query = st.text_input(
        "", placeholder="e.g. 'Show evidence that MFA is required for privileged accounts'"
    )

    # FIX 8 + FIX 9 — Decouple PDF export from the Investigate button click.
    #
    # Original problem:
    #   The "Generate PDF Report" button was nested inside
    #   `if st.button("Investigate")`. In Streamlit, every interaction
    #   triggers a full script rerun. When the user clicks "Generate PDF",
    #   the "Investigate" button is no longer active — its `if` block
    #   evaluates to False — so the inner button and the `report`/`version`
    #   variables it depends on are never reached. The PDF button appeared
    #   to click but nothing happened.
    #
    # Fix: store the report and version in st.session_state after a
    # successful investigation. The PDF export button lives outside the
    # Investigate block and reads from session_state, so it works correctly
    # on its own rerun cycle.

    if st.button("Investigate", type="primary", disabled=not query):
        # FIX 9 — Read version from session_state here (inside the click
        # handler) so it's available at the moment of the API call.
        version = st.session_state.get("current_version", "1.0")

        with st.spinner("Running investigation..."):
            # FIX 2 (frontend side) — Changed from GET to POST to match the
            # updated API. Query and version are sent as JSON body, not URL
            # params, eliminating the query-string length limit.
            res = requests.post(
                f"{API_URL}/investigate",
                json={"query": query, "ingestion_version": version},
            )

        if res.status_code == 200:
            report = res.json()
            # Persist report and query so the PDF button can use them on its
            # own rerun without re-running the investigation.
            st.session_state["last_report"] = report
            st.session_state["last_query"] = query
            st.session_state["last_version"] = version
        else:
            st.error(f"Investigation failed ({res.status_code}): {res.text}")

    # Render results if a report exists in session (persists across reruns)
    report = st.session_state.get("last_report")
    if report:
        st.subheader("Summary")
        st.info(report.get("summary", ""))

        st.subheader("Evidence")
        for item in report.get("evidence", []):
            cit = item.get("citation", {})
            label = (
                f"📄 {cit.get('filename', '')} — "
                f"{cit.get('section_title', '')} "
                f"(p.{cit.get('page_number', '?')})"
            )
            with st.expander(label):
                st.write(f"**Claim:** {item.get('claim', '')}")
                st.caption(f"Excerpt: {cit.get('excerpt', '')}")

        if report.get("gaps"):
            st.subheader("⚠️ Gaps Identified")
            for gap in report.get("gaps", []):
                st.warning(gap)

        st.download_button(
            label="⬇ Export Evidence Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="verdact_evidence_report.json",
            mime="application/json",
        )

        # FIX 8 — PDF button is now a standalone widget outside the
        # Investigate block. It reads query/version/report from session_state
        # and sends the already-generated report to the export endpoint,
        # which only renders PDF — no second LLM call (aligns with FIX 4
        # in main.py).
        if st.button("Generate PDF Report"):
            saved_query = st.session_state.get("last_query", "")
            saved_version = st.session_state.get("last_version", "1.0")

            with st.spinner("Building PDF..."):
                # FIX 2 (frontend side) — POST with JSON body.
                # FIX 4 (frontend side) — Send the cached report so the
                # server doesn't re-run the LLM.
                pdf_res = requests.post(
                    f"{API_URL}/investigate/export",
                    json={
                        "query": saved_query,
                        "ingestion_version": saved_version,
                        "report": report,
                    },
                )

            # FIX 10 — Check PDF response status before offering download.
            # The original passed pdf_res.content to st.download_button
            # unconditionally. If the export endpoint returned an error,
            # the user would download a corrupt file with no indication of
            # failure. Now we surface the error message explicitly.
            if pdf_res.status_code == 200:
                st.download_button(
                    label="⬇ Download Evidence Report (PDF)",
                    data=pdf_res.content,
                    file_name="verdact_evidence_report.pdf",
                    mime="application/pdf",
                )
            else:
                st.error(f"PDF generation failed ({pdf_res.status_code}): {pdf_res.text}")