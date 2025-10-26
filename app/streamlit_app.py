import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import time
from cache_manager import SemanticCache
from llm_client import query_llm
from config import settings

# Initialize
if "cache" not in st.session_state:
    st.session_state.cache = SemanticCache(settings.EMBEDDING_MODEL)

st.title("🧠 LLM Semantic Cache (ChatGroq + LangChain)")
st.markdown("Cache responses by meaning, not exact text — powered by embeddings.")

system_prompt = st.text_area("System Prompt:", "You are a helpful AI assistant.")
user_prompt = st.text_area("User Prompt:", "")

if st.button("Run Query"):
    start = time.time()
    cache = st.session_state.cache
    cached_response = cache.get(user_prompt)

    if cached_response:
        response = cached_response
        latency = time.time() - start
        st.success(f"✅ Cache hit! (took {latency:.2f}s)")
    else:
        response = query_llm(system_prompt, user_prompt)
        cache.set(user_prompt, response)
        latency = time.time() - start
        st.info(f"🆕 Fetched from LLM (took {latency:.2f}s)")

    st.markdown("### Response")
    st.write(response)

    st.markdown("---")
    stats = cache.stats()
    st.metric("Cache Hit Rate (%)", f"{stats['hit_rate']:.1f}")
    st.metric("Total Hits", stats["hits"])
    st.metric("Total Misses", stats["misses"])