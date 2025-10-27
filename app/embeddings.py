from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Force CPU usage to avoid meta tensor errors
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized embedding vector for a single text."""
        vec = self.model.encode([text])[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Return normalized embeddings for a list of texts."""
        vecs = self.model.encode(texts)
        return [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in vecs]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors (assumes normalized)."""
        return float(np.dot(a, b))
