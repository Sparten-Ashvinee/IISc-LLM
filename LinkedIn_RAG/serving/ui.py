"""Streamlit UI for TalentRAG."""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="TalentRAG", page_icon="🔍", layout="wide")
st.title("TalentRAG — LinkedIn Job Search")
st.caption("Semantic search over 50K+ job postings with grounded answers")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results to show", 1, 10, 5)
    enable_guard = st.checkbox("Enable hallucination guard", value=True)

# Query input
query = st.text_input("Ask a question about jobs...",
                       placeholder="e.g., Find senior ML engineer roles in San Francisco requiring PyTorch")

if query:
    with st.spinner("Searching..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={
                "query": query,
                "top_k": top_k,
                "enable_guard": enable_guard,
            }, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start the server first: `python -m serving.api`")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # Answer
    st.subheader("Answer")
    st.markdown(data["answer"])

    # Faithfulness
    col1, col2, col3 = st.columns(3)
    col1.metric("Faithful", "Yes" if data["is_faithful"] else "No")
    col2.metric("Faithfulness Score", f"{data['faithfulness_score']:.2f}")
    col3.metric("Latency", f"{data['latency_ms']:.0f} ms")

    # Sources
    st.subheader("Sources")
    for i, src in enumerate(data["sources"], 1):
        with st.expander(f"Source {i}: {src['title']} at {src['company']}"):
            st.write(f"**Location:** {src['location']}")
            st.write(f"**Relevance Score:** {src['score']:.4f}")


# Footer
st.divider()
st.caption("Built from scratch — no LangChain, no LlamaIndex")
