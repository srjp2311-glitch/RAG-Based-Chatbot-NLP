import numpy as np
from utils.retriever import Retriever
from utils.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_retrieval(query: str, expected_keywords: list = None):
    logger.info("Initializing Evaluation...")
    retriever = Retriever()
    vector_store = VectorStore(embedding_dim=retriever.get_embedding_dimension())
    
    if not vector_store.load("embeddings/"):
        logger.error("Could not load vector store. Ensure ingestion has run.")
        return
        
    query_emb = retriever.encode([query])[0]
    results = vector_store.search(query_emb, k=5)
    
    logger.info(f"Query: {query}")
    logger.info("Top 5 Results:")
    for i, res in enumerate(results):
        logger.info(f"{i+1}. Distance: {res['distance']:.4f} | Source: {res['metadata']['source']}")
        logger.info(f"Text Snippet: {res['text'][:150]}...\n")
        
    if expected_keywords:
        found_keywords = []
        for res in results:
            for kw in expected_keywords:
                if kw.lower() in res['text'].lower() and kw not in found_keywords:
                    found_keywords.append(kw)
                    
        accuracy = len(found_keywords) / len(expected_keywords)
        logger.info(f"Keyword Accuracy (Top 5): {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_query = "What is the penalty for illegal deforestation?"
    evaluate_retrieval(test_query, expected_keywords=["penalty", "punishment", "imprisonment"])
