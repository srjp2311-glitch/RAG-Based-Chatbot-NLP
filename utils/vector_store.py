import os
import faiss
import json
import numpy as np
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []
        self.metadata = []
        
    def add(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict]):
        if len(embeddings) != len(texts) or len(embeddings) != len(metadata):
            logger.error("Mismatch in lengths of embeddings, texts, and metadata")
            return
            
        logger.info(f"Adding {len(embeddings)} items to FAISS index.")
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
    def search(self, query_embedding: np.ndarray, k: int = 20, state_filter: str = None) -> List[Dict]:
        """Searches for top-k nearest embeddings."""
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k * 3) # retrieve more to allow filtering
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
                
            meta = self.metadata[idx]
            
            # Simple state filtering
            if state_filter and state_filter.lower() != "all" and meta.get("state", "").lower() != state_filter.lower():
                continue
                
            results.append({
                "text": self.texts[idx],
                "metadata": meta,
                "distance": float(dist)
            })
            
            if len(results) >= k:
                break
                
        return results
        
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
            
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "data.json"), "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metadata": self.metadata}, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Vector store saved to {path}")
        
    def load(self, path: str):
        index_path = os.path.join(path, "index.faiss")
        data_path = os.path.join(path, "data.json")
        
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            logger.warning(f"Index or data not found at {path}")
            return False
            
        self.index = faiss.read_index(index_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            
        logger.info(f"Vector store loaded from {path}. Contains {self.index.ntotal} vectors.")
        return True
