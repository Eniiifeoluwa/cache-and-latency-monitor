import threading
import time
from .embeddings import EmbeddingEngine

class SemanticCache:
    def __init__(self, embedding_model: str, similarity_threshold=0.90, ttl=3600):
        self.cache = {}
        self.lock = threading.Lock()
        self.embedder = EmbeddingEngine(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_emb = self.embedder.embed(query)
        with self.lock:
            now = time.time()
            for key, item in list(self.cache.items()):
                if now - item['time'] > self.ttl:
                    del self.cache[key]
                    continue

                sim = self.embedder.cosine_similarity(query_emb, item['embedding'])
                if sim > self.similarity_threshold:
                    self.hits += 1
                    return item['response']
        self.misses += 1
        return None

    def set(self, query, response):
        with self.lock:
            self.cache[query] = {
                'embedding': self.embedder.embed(query),
                'response': response,
                'time': time.time()
            }

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}
