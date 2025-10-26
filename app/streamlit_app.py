import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from cache_manager import SemanticCache
from llm_client import query_llm, clear_memory
from config import settings

if "cache" not in st.session_state:
    st.session_state.cache = SemanticCache(
        embedding_model=settings.EMBEDDING_MODEL,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        ttl=settings.CACHE_TTL
    )

st.set_page_config(page_title="LLM Semantic Cache with Memory", layout="wide")
st.title("🧠 LLM Response Caching System + Memory")

user_query = st.text_area("Ask something:", placeholder="Type your question here...")

col1, col2 = st.columns([1, 1])

with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Memory")

if clear:
    clear_memory()
    st.success("🧹 Conversation memory cleared!")

if send and user_query:
    cache = st.session_state.cache

    with st.spinner("Checking cache..."):
        cached_response, sim = cache.get(user_query)

    if cached_response:
        st.success(f"✅ Cache hit (similarity: {sim:.2f})")
        response = cached_response
    else:
        st.info("🚀 Cache miss — querying LLM via ChatGroq...")
        response = query_llm(user_query)
        cache.set(user_query, response)

    st.markdown("### 🤖 Response")
    st.write(response)

    with st.expander("📊 Cache Details"):
        st.json(cache.stats())

    if st.checkbox("Show embedding for query"):
        st.write(cache.get_embedding(user_query)[:10])

else:
    st.caption("Ask a question to see semantic caching and memory in action.")
