import time
from embeddings import EmbeddingEngine

class SemanticCache:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", similarity_threshold=0.85, ttl=3600):
        self.model = EmbeddingEngine(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.cache = {}  # {query: {"response": str, "embedding": np.ndarray, "timestamp": float}}

    # -----------------------------------------------------------------
    # Public method to get embedding for app display
    # -----------------------------------------------------------------
    def get_embedding(self, query: str):
        """Return the embedding of a query without affecting cache."""
        return self.model.embed(query)

    # -----------------------------------------------------------------
    # Internal cleanup
    # -----------------------------------------------------------------
    def _cleanup(self):
        """Remove expired cache entries based on TTL."""
        now = time.time()
        expired_keys = [k for k, v in self.cache.items() if now - v["timestamp"] > self.ttl]
        for k in expired_keys:
            del self.cache[k]

    # -----------------------------------------------------------------
    # Cache operations
    # -----------------------------------------------------------------
    def get(self, query: str):
        """Retrieve a cached response if similarity threshold is met."""
        self._cleanup()
        query_emb = self.model.embed(query)

        for entry in self.cache.values():
            sim = self.model.cosine_similarity(query_emb, entry["embedding"])
            if sim >= self.similarity_threshold:
                return entry["response"], sim
        return None, 0.0

    def set(self, query: str, response: str):
        """Store a query-response pair in cache with embedding and timestamp."""
        self.cache[query] = {
            "response": response,
            "embedding": self.model.embed(query),
            "timestamp": time.time()
        }

    def stats(self):
        """Return simple cache statistics."""
        return {
            "entries": len(self.cache)
        }
