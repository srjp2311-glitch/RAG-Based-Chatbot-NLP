import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, List
from utils.logger import get_logger
import os

# Suppress HF warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logger = get_logger(__name__)

class QuestionAnswering:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        logger.info(f"Loading QA model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def answer(self, question: str, contexts: List[Dict]) -> Dict:
        """Finds the best answer for the question by evaluating contexts individually."""
        if not contexts:
            return {
                "answer": "No relevant context found.",
                "confidence": 0.0,
                "sources": []
            }
            
        logger.info("Extracting answer from retrieved contexts...")
        
        best_answer = "Could not extract a specific answer. Please rephrase your query."
        best_confidence = 0.0
        best_source = None
        
        try:
            for c in contexts:
                inputs = self.tokenizer(question, c["text"], return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                
                start_prob = torch.max(torch.softmax(outputs.start_logits, dim=-1)).item()
                end_prob = torch.max(torch.softmax(outputs.end_logits, dim=-1)).item()
                confidence = (start_prob + end_prob) / 2
                
                # Check if this is a valid answer sequence
                if answer_end > answer_start:
                    answer_text = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
                    ).replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    
                    if len(answer_text) > 1 and confidence > best_confidence:
                        best_confidence = confidence
                        
                        # Expand to the full sentence surrounding the answer
                        chunk_text = c["text"]
                        try:
                            # Try to find the exact substring match in the raw text
                            start_idx = chunk_text.lower().find(answer_text.lower())
                            if start_idx != -1:
                                # Find nearest boundary before the answer
                                prev_bound = max(chunk_text.rfind('.', 0, start_idx), 
                                                 chunk_text.rfind('\n', 0, start_idx))
                                prev_bound = 0 if prev_bound == -1 else prev_bound + 1
                                
                                # Find nearest boundary after the answer
                                next_bound = chunk_text.find('.', start_idx + len(answer_text))
                                next_bound = len(chunk_text) if next_bound == -1 else next_bound + 1
                                
                                best_answer = chunk_text[prev_bound:next_bound].strip()
                            else:
                                best_answer = answer_text
                        except:
                            best_answer = answer_text
                            
                        best_source = c
            
            # Format the sources list based on what we found
            sources = []
            if best_source:
                meta = best_source["metadata"].copy()
                meta["snippet"] = best_source["text"][:150] + "..."
                sources.append(meta)
                
            return {
                "answer": best_answer,
                "confidence": round(best_confidence, 4),
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error during QA extraction: {e}")
            return {
                "answer": "An error occurred during answer generation.",
                "confidence": 0.0,
                "sources": []
            }
