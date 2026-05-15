# frontend/app.py

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ResearchMind", page_icon="🧠")
st.title("🧠 ResearchMind")
st.caption("Ask questions about any webpage or PDF.")

st.subheader("Add a Source")
url_input = st.text_input("Paste a URL to ingest")
if st.button("Ingest URL"):
    if url_input.strip():
        with st.spinner("Ingesting..."):
            res = requests.post(f"{API_URL}/ingest/url", json={"url": url_input})
        if res.status_code == 200:
            st.success(f"Done! {res.json()['chunks_added']} chunks added.")
        else:
            st.error(res.json().get("detail", "Something went wrong."))
    else:
        st.warning("Please enter a URL.")

uploaded = st.file_uploader("Or upload a PDF", type=["pdf"])
if st.button("Ingest PDF"):
    if uploaded:
        with st.spinner("Ingesting..."):
            res = requests.post(
                f"{API_URL}/ingest/pdf",
                files={"file": (uploaded.name, uploaded, "application/pdf")},
            )
        if res.status_code == 200:
            st.success(f"Done! {res.json()['chunks_added']} chunks added.")
        else:
            st.error(res.json().get("detail", "Something went wrong."))
    else:
        st.warning("Please upload a PDF.")

st.divider()

st.subheader("Ask a Question")
query = st.text_input("Your question")
if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_URL}/query/", json={"query": query})
        if res.status_code == 200:
            data = res.json()
            st.markdown(f"**Answer:** {data['answer']}")
            st.caption(f"Query type: `{data['query_type']}`")
            st.markdown("**Sources:** " + ", ".join(data["sources"]))
        else:
            st.error(res.json().get("detail", "Something went wrong."))
    else:
        st.warning("Please enter a question.")