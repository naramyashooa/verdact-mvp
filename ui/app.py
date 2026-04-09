# ui/app.py

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

    # ── Document selector ────────────────────────────────────────────────────
    # Load all previously ingested documents from the registry on the API.
    # This means returning users can select a document and investigate without
    # re-uploading — the version is fetched from the persistent registry and
    # stored in session_state, exactly as if they had just ingested it.

    try: 
        docs_res = requests.get(f"{API_URL}/documents", timeout=3)
        known_docs = docs_res.json().get("documents", {}) if docs_res.status_code == 200 else {}
    except requests.exceptions.ConnectionError:
        known_docs = {}

    if known_docs:
        st.markdown("**Previously Ingested Document**")
        selected = st.selectbox(
            "Select a document to investigate",
            options=list(known_docs.keys()),
            index=0,
        )
        if st.button("Load document", key="load_existing"):
            st.session_state["current_version"] = known_docs[selected]
            st.session_state["current_filename"] = selected

            # Clear state report when switching reports
            st.session_state.pop("last_report", None)
            st.session_state.pop("last_query", None)
            st.session_state.pop("last_version", None)
            st.success(f"Loaded: {selected}")
        
    st.markdown("**Or Upload a New Document**")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded and st.button("Ingest PDF", type="primary"):
        session_id = f"v_{uuid.uuid4().hex[:6]}"

        with st.spinner("Ingesting document..."):
            res = requests.post(
                f"{API_URL}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                data={"ingestion_version": session_id},
            )

        if res.status_code == 200:
            data = res.json()

            returned_version = data["version"]

            if data.get("status") == "already_ingested":
                st.info(
                    f"'{uploaded.name}' was already ingested. "
                    f"Using existing version. "
                    f"Check 'Force re-ingest' to update it."
                )

            else:
                st.success(f"Ingested: {data['filename']}")
 
            st.session_state["current_version"] = returned_version
            st.session_state["current_filename"] = data["filename"]

            # Clear any stale report from a previous session so the UI
            # doesn't show old results for the newly ingested document.
            st.session_state.pop("last_report", None)
            st.session_state.pop("last_query", None)
            st.session_state.pop("last_version", None)
 
            # Force a rerun so the "Active document" indicator and document
            # selector both refresh immediately with the new document.
            st.rerun()
        else:
            st.error(f"Ingestion failed ({res.status_code}): {res.text}")

    # Show which document is currently active
    if "current_filename" in st.session_state:
        st.info(f"Active document: **{st.session_state['current_filename']}**")

with col2:
    st.subheader("Run an Investigation")
    query = st.text_input(
        "", placeholder="e.g. 'Show evidence that MFA is required for privileged accounts'"
    )

    if st.button("Investigate", type="primary", disabled=not query):
        # Read version from session_state here (inside the click handler) so it's available at the moment of the API call.
        version = st.session_state.get("current_version")

        if not version:
            st.warning(
                "No document selected. Please upload a new document or "
                "select a previously ingested one from the left panel."
            )
        else:
            with st.spinner("Running investigation..."):
                res = requests.post(
                    f"{API_URL}/investigate",
                    json={"query": query, "ingestion_version": version},
                    timeout=120,
                )

            if res.status_code == 200:
                report = res.json()
                # Persist report and query so the PDF button can use them on its
                # own rerun without re-running the investigation.
                st.session_state["last_report"] = report
                st.session_state["last_query"] = query
                st.session_state["last_version"] = version
            elif res.status_code == 503:
                try:
                    detail = res.json().get("detail", res.text)
                except Exception:
                    detail = res.text
                st.error(f"Investigation failed: {detail}")
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

        if st.button("Generate PDF Report"):
            saved_query = st.session_state.get("last_query", "")
            saved_version = st.session_state.get("last_version", "1.0")

            with st.spinner("Building PDF..."):
                pdf_res = requests.post(
                    f"{API_URL}/investigate/export",
                    json={
                        "query": saved_query,
                        "ingestion_version": saved_version,
                        "report": report,
                    },
                )

            # Check PDF response status before offering download.
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