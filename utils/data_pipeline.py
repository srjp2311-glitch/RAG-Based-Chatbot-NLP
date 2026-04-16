import os
import re
import fitz  # PyMuPDF
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    """Removes extra whitespaces and noise from text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Splits text into chunks of specified word count with overlap."""
    words = text.split()
    chunks = []
    
    if len(words) == 0:
        return chunks
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    return chunks

def extract_metadata_from_filename(filename: str) -> Dict:
    """Heuristic to extract state or info from filename."""
    states = ['Delhi', 'Haryana', 'Punjab', 'Uttar Pradesh']
    found_state = 'Unknown'
    
    for state in states:
        if state.lower() in filename.lower():
            found_state = state
            break
            
    return {
        "source": filename,
        "state": found_state,
        "year": "Unknown",  # Could be parsed via regex if needed
        "document_type": "PDF"
    }

def process_pdfs(path: str) -> List[Dict]:
    """Loads PDFs from directory or a single file, cleans, chunks, and attaches metadata."""
    logger.info(f"Processing path: {path}")
    processed_data = []
    
    if not os.path.exists(path):
        logger.warning(f"Path {path} does not exist.")
        return processed_data
        
    filepaths = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith(".pdf"):
                filepaths.append(os.path.join(path, filename))
    elif os.path.isfile(path) and path.lower().endswith(".pdf"):
        filepaths.append(path)
    else:
        logger.warning(f"Path {path} is neither a directory nor a PDF file.")
        return processed_data
        
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        logger.info(f"Reading {filepath}")
        try:
            doc = fitz.open(filepath)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + " "
                
            doc.close()
            
            cleaned_text = clean_text(full_text)
            chunks = chunk_text(cleaned_text, chunk_size=300, overlap=50)
            metadata = extract_metadata_from_filename(filename)
            
            for i, chunk in enumerate(chunks):
                meta = metadata.copy()
                meta["chunk_id"] = i
                processed_data.append({
                    "text": chunk,
                    "metadata": meta
                })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
                
    logger.info(f"Total chunks created: {len(processed_data)}")
    return processed_data

if __name__ == "__main__":
    # Test locally
    data = process_pdfs("../data/raw")
    if data:
        print(f"Sample: {data[0]}")
