# Legal-Aware Deforestation RAG Chatbot for North India

This is an end-to-end Retrieval-Augmented Generation (RAG) system using BERT-based models that answers user queries about deforestation using legal documents, policies, and reports from North Indian states.

## Architecture

- **Data Pipeline**: PyMuPDF for loading PDFs, custom cleaning, and chunking.
- **Embedding**: `sentence-transformers/all-mpnet-base-v2`
- **Vector Store**: FAISS
- **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **QA Model**: `deepset/roberta-base-squad2`
- **CLI Access**: `argparse` based interface

## Folder Structure

```
├── api/                  # (Placeholder for complex routes)
├── data/
│   └── raw/              # Store your PDF documents here
├── embeddings/           # FAISS index will be saved here automatically
├── models/               # (Placeholder for local model weights if needed)
├── utils/
│   ├── data_pipeline.py  # PDF processing
│   ├── vector_store.py   # FAISS logic
│   ├── retriever.py      # Embeddings
│   ├── ranker.py         # Cross-encoder re-ranking
│   ├── qa.py             # Extractive QA
│   └── logger.py         # Centralized logging
├── main.py               # Application Entry Point & CLI
├── evaluate.py           # Simple evaluation script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Setup Instructions

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Documents**:
   Place your legal/policy PDFs inside `data/raw/`.

4. **Run Data Ingestion**:
   You must ingest data to create the FAISS index before querying.
   ```bash
   python main.py ingest
   ```

5. **Ask Questions**:
   You can query the RAG chatbot directly from the terminal.
   ```bash
   python main.py ask "What is the penalty for violating the Forest Act?"
   ```

## Evaluation

Run `evaluate.py` to test embedding search logic and evaluate keyword accuracy.
```bash
python evaluate.py
```
