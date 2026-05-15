import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ResearchMind", page_icon="🧠")
st.title("🧠 ResearchMind")
st.caption("Your personal research assistant. Ask questions across multiple documents.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

session_id = st.session_state.session_id
st.sidebar.caption(f"Session: `{session_id}`")

with st.sidebar:
    st.header("📥 Add Documents")

    url_input = st.text_input("Paste a URL")
    if st.button("Ingest URL"):
        if url_input.strip():
            with st.spinner("Ingesting..."):
                res = requests.post(f"{API_URL}/ingest/url", json={
                    "url": url_input,
                    "session_id": session_id,
                })
            if res.status_code == 200:
                st.success(f"{res.json()['chunks_added']} chunks added.")
            else:
                st.error(res.json().get("detail", "Error."))
        else:
            st.warning("Enter a URL first.")

    uploaded = st.file_uploader("Or upload a PDF", type=["pdf"])
    if st.button("Ingest PDF"):
        if uploaded:
            with st.spinner("Ingesting..."):
                res = requests.post(
                    f"{API_URL}/ingest/pdf",
                    files={"file": (uploaded.name, uploaded, "application/pdf")},
                    params={"session_id": session_id},
                )
            if res.status_code == 200:
                st.success(f"{res.json()['chunks_added']} chunks added.")
            else:
                st.error(res.json().get("detail", "Error."))
        else:
            st.warning("Upload a PDF first.")

    st.divider()
    st.header("📂 My Documents")
    if st.button("Refresh Sources"):
        res = requests.get(f"{API_URL}/ingest/sources", params={"session_id": session_id})
        if res.status_code == 200:
            sources = res.json().get("sources", [])
            if sources:
                for src in sources:
                    col1, col2 = st.columns([3, 1])
                    col1.write(src)
                    if col2.button("🗑", key=src):
                        requests.delete(f"{API_URL}/ingest/source", json={
                            "source": src,
                            "session_id": session_id,
                        })
                        st.rerun()
            else:
                st.info("No documents yet.")

st.subheader("💬 Ask a Question")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            col1, col2 = st.columns([1, 3])
            col1.markdown(f"**Confidence:** `{msg.get('confidence', 'N/A')}`")
            col2.markdown(f"**Sources:** {', '.join(msg.get('sources', []))}")
            st.caption(f"Query type: `{msg.get('query_type', '')}`")

query = st.chat_input("Ask something about your documents...")
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_URL}/query/", json={
                "query": query,
                "session_id": session_id,
            })
        if res.status_code == 200:
            data = res.json()
            st.markdown(data["answer"])
            col1, col2 = st.columns([1, 3])
            col1.markdown(f"**Confidence:** `{data['confidence']}`")
            col2.markdown(f"**Sources:** {', '.join(data['sources'])}")
            st.caption(f"Query type: `{data['query_type']}`")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": data["answer"],
                "confidence": data["confidence"],
                "sources": data["sources"],
                "query_type": data["query_type"],
            })
        else:
            err = res.json().get("detail", "Something went wrong.")
            st.error(err)