import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from cache_manager import SemanticCache
from llm_client import LLMClient
from config import settings

# ---------------------------------------------------------------------
# Initialize state
# ---------------------------------------------------------------------
if "llm_client" not in st.session_state:
    st.session_state.llm_client = LLMClient()

if "cache" not in st.session_state:
    st.session_state.cache = SemanticCache(
        embedding_model=settings.EMBEDDING_MODEL,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        ttl=settings.CACHE_TTL
    )

# Keep persistent data for last query
for key in ["last_query", "last_response", "last_latency", "last_embedding", "cache_hit"]:
    if key not in st.session_state:
        st.session_state[key] = None

llm_client = st.session_state.llm_client
cache = st.session_state.cache

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="LLM Semantic Cache with Memory", layout="wide")
st.title("🧠 LLM Semantic Cache + Conversational Memory")

user_query = st.text_area("Ask something:", placeholder="Type your question here...")

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Memory")

if clear:
    llm_client.clear()
    st.session_state.last_response = None
    st.success("🧹 Conversation memory cleared!")

# ---------------------------------------------------------------------
# Handle LLM Query
# ---------------------------------------------------------------------
if send and user_query.strip():
    with st.spinner("🔍 Checking semantic cache..."):
        cached_response, sim = cache.get(user_query)

    if cached_response:
        st.success(f"✅ Cache hit! (similarity: {sim:.2f})")
        response = cached_response
        latency_ms = 0
        st.session_state.cache_hit = True
    else:
        st.info("🚀 Cache miss — querying ChatGroq via LangChain...")
        start_time = time.time()
        response = llm_client.query(user_query)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        cache.set(user_query, response)
        st.session_state.cache_hit = False

    st.session_state.last_query = user_query
    st.session_state.last_response = response
    st.session_state.last_latency = latency_ms
    st.session_state.last_embedding = cache.get_embedding(user_query)

if st.session_state.last_response:
    st.markdown("### 🤖 Response")
    st.write(st.session_state.last_response)
    st.caption(f"⏱️ Latency: {st.session_state.last_latency} ms")

    with st.expander("📊 Cache Details"):
        st.json(cache.stats())

    show_embed = st.checkbox("🔍 Show embedding for this query", value=False)
    if show_embed and st.session_state.last_embedding is not None:
        st.write(st.session_state.last_embedding[:10])

else:
    st.caption("Ask a question to see semantic caching and conversational memory in action.")
