import time
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", similarity_threshold=0.85, ttl=3600):
        self.model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.cache = {}

    def get_embedding(self, text):
        return self.model.encode(text)

    def _similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def get(self, query):
        now = time.time()
        query_emb = self.get_embedding(query)

        # Clean expired entries
        for key in list(self.cache.keys()):
            if now - self.cache[key]["timestamp"] > self.ttl:
                del self.cache[key]

        for key, entry in self.cache.items():
            sim = self._similarity(query_emb, entry["embedding"])
            if sim >= self.similarity_threshold:
                return entry["response"], sim
        return None, 0.0

    def set(self, query, response):
        self.cache[query] = {
            "response": response,
            "embedding": self.get_embedding(query),
            "timestamp": time.time()
        }

    def stats(self):
        return {
            "entries": len(self.cache),
        }
