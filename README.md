# 🧠 LLM Response Caching System (Streamlit + ChatGroq + LangChain)

### Features
- Semantic caching using Sentence Transformers
- ChatGroq backend via LangChain
- Cache invalidation (TTL)
- Hit-rate metrics + latency tracking
- Fully Streamlit Cloud deployable

### Run Locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_api_key_here
streamlit run app/streamlit_app.py
