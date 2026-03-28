import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class MultimodalFusion:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        # Using a lightweight sentence transformer for filtering
        self.similarity_model = SentenceTransformer(model_name)

    def filter_ocr_tokens(self, question, ocr_data, threshold=0.3):
        """Filter OCR tokens based on semantic similarity to the question."""
        if not ocr_data:
            return []

        # Get question embedding
        q_emb = self.similarity_model.encode([question])

        # Get OCR token embeddings
        token_texts = [item['text'] for item in ocr_data]
        token_embs = self.similarity_model.encode(token_texts)

        # Compute cosine similarity
        similarities = cosine_similarity(q_emb, token_embs)[0]

        # Filter tokens
        filtered_tokens = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                filtered_tokens.append({
                    'text': ocr_data[i]['text'],
                    'similarity': sim,
                    'bbox': ocr_data[i]['bbox']
                })
        
        # Sort by similarity
        filtered_tokens = sorted(filtered_tokens, key=lambda x: x['similarity'], reverse=True)
        return filtered_tokens

    def fuse_and_generate(self, vqa_answer, filtered_ocr_tokens):
        """
        Final fusion logic. In a research setting, this might be a learned layer.
        Here we implement a heuristic-based fusion for high performance.
        We append the most relevant OCR tokens to the VQA model's initial guess.
        """
        if not filtered_ocr_tokens:
            return vqa_answer

        # If the VQA answer is generic, try to incorporate OCR text
        ocr_text_str = ", ".join([t['text'] for t in filtered_ocr_tokens[:3]])
        
        # Heuristic: If OCR text is highly relevant, prioritize it
        # This acts as the "copy mechanism" mentioned in the report
        if filtered_ocr_tokens[0]['similarity'] > 0.6:
            # If the VQA answer is short, maybe it needs specific text
            if len(vqa_answer.split()) < 3:
                return f"{filtered_ocr_tokens[0]['text']}"
        
        return f"{vqa_answer} (Relevant text found: {ocr_text_str})"

if __name__ == "__main__":
    pass
