import os
import warnings
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional

# Suppress all HuggingFace and warning messages for a clean output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from utils.logger import get_logger
from utils.data_pipeline import process_pdfs
from utils.retriever import Retriever
from utils.vector_store import VectorStore
from utils.ranker import Ranker
from utils.qa import QuestionAnswering

logger = get_logger(__name__)

app = FastAPI(title="Legal-Aware Deforestation RAG Chatbot API")

INDEX_PATH = "embeddings/"
UPLOAD_DIR = "data/raw/"

# Global references for models (lazy loaded)
_retriever = None
_vector_store = None
_ranker = None
_qa_model = None

def get_models():
    """Initializes and returns the ML models if not already loaded."""
    global _retriever, _vector_store, _ranker, _qa_model
    
    if _retriever is None:
        logger.info("Initializing models...")
        _retriever = Retriever()
        _vector_store = VectorStore(embedding_dim=_retriever.get_embedding_dimension())
        
        if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")) and os.path.exists(os.path.join(INDEX_PATH, "data.json")):
            _vector_store.load(INDEX_PATH)
            
        _ranker = Ranker()
        _qa_model = QuestionAnswering()
        logger.info("Models initialized successfully.")
        
    return _retriever, _vector_store, _ranker, _qa_model

class QueryRequest(BaseModel):
    query: str
    state: Optional[str] = "All"

@app.on_event("startup")
def startup_event():
    # Pre-load models on startup
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    

@app.post("/ingest")
async def ingest_custom_pdf(file: UploadFile = File(...)):
    """Uploads a PDF file, processes it, and adds it to the FAISS index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    retriever, vector_store, _, _ = get_models()
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Process the file
    processed_data = process_pdfs(file_path)
    if not processed_data:
        raise HTTPException(status_code=400, detail="Failed to process the uploaded PDF.")
        
    texts = [item["text"] for item in processed_data]
    metadata = [item["metadata"] for item in processed_data]
    
    # Generate embeddings and add to index
    embeddings = retriever.encode(texts)
    vector_store.add(embeddings, texts, metadata)
    vector_store.save(INDEX_PATH)
    
    return {
        "message": f"Successfully ingested {file.filename}",
        "chunks_processed": len(texts)
    }

@app.post("/ask")
def ask_question(request: QueryRequest):
    """Answers a question based on indexed legal documents."""
    retriever, vector_store, ranker, qa_model = get_models()
    
    if len(vector_store.texts) == 0:
        raise HTTPException(status_code=400, detail="No data indexed. Please /ingest a PDF first.")
        
    logger.info(f"Received query: '{request.query}' with state filter: '{request.state}'")
    
    # 1. Retrieve
    query_emb = retriever.encode([request.query])[0]
    retrieved_docs = vector_store.search(query_emb, k=20, state_filter=request.state)
    
    if not retrieved_docs:
        return {
            "answer": "No relevant documents found for the given criteria.",
            "confidence": 0.0,
            "sources": []
        }
    
    # 2. Re-rank
    top_docs = ranker.rank(request.query, retrieved_docs, top_k=5)
    
    # 3. QA
    qa_result = qa_model.answer(request.query, top_docs)
    
    return {
        "question": request.query,
        "answer": qa_result["answer"],
        "confidence": qa_result["confidence"],
        "sources": qa_result["sources"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
