from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
import os

class VQAModel:
    def __init__(self, model_id="Salesforce/blip-vqa-base", cache_dir="./models/blip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Loading BLIP model into {cache_dir}...")
        self.processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id, cache_dir=cache_dir).to(self.device)

    def get_answer(self, image_path, question):
        """Generate answer for a given image and question."""
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
        return answer

    def get_embeddings(self, text):
        """Get embeddings for text (question or OCR tokens)."""
        # This can be used for filtering
        inputs = self.processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.text_encoder(**inputs)
            # Use pooler output or mean of last hidden state
            embeddings = outputs.pooler_output
        return embeddings

if __name__ == "__main__":
    pass
