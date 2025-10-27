import time
from embeddings import EmbeddingEngine

class SemanticCache:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", similarity_threshold=0.85, ttl=3600):
        self.model = EmbeddingEngine(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.cache = {}  # {query: {"response": str, "embedding": np.ndarray, "timestamp": float}}

    def _cleanup(self):
        """Remove expired cache entries."""
        now = time.time()
        expired_keys = [k for k, v in self.cache.items() if now - v["timestamp"] > self.ttl]
        for k in expired_keys:
            del self.cache[k]

    def get(self, query: str):
        """Retrieve a cached response if similar enough, else return None."""
        self._cleanup()
        query_emb = self.model.embed(query)

        for entry in self.cache.values():
            sim = self.model.cosine_similarity(query_emb, entry["embedding"])
            if sim >= self.similarity_threshold:
                return entry["response"], sim
        return None, 0.0

    def set(self, query: str, response: str):
        """Cache a query-response pair with embedding and timestamp."""
        self.cache[query] = {
            "response": response,
            "embedding": self.model.embed(query),
            "timestamp": time.time()
        }

    def stats(self):
        return {
            "entries": len(self.cache)
        }
