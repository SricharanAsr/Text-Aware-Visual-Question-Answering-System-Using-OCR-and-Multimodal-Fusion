import os
import argparse
from src.ocr_pipeline import OCRProcessor
from src.vqa_model import VQAModel
from src.fusion import MultimodalFusion

class TextAwareVQA:
    def __init__(self):
        print("Initializing Text-Aware VQA System...")
        self.ocr = OCRProcessor()
        self.vqa = VQAModel()
        self.fusion = MultimodalFusion()
        print("System ready.")

    def run(self, image_path, question):
        print(f"\n--- Processing Image: {image_path} ---")
        print(f"Question: {question}")
        
        # 1. OCR Extraction
        print("Stage 1: Extracting text (OCR)...")
        ocr_data = self.ocr.extract_text(image_path)
        print(f"Extracted {len(ocr_data)} tokens.")

        # 2. VQA Model Initial Answer
        print("Stage 2: Generating initial answer (BLIP)...")
        vqa_answer = self.vqa.get_answer(image_path, question)
        print(f"Initial Answer: {vqa_answer}")

        # 3. Filtering OCR Tokens
        print("Stage 3: Filtering OCR tokens for relevance...")
        filtered_tokens = self.fusion.filter_ocr_tokens(question, ocr_data)
        if filtered_tokens:
            print(f"Top relevant token: '{filtered_tokens[0]['text']}' (Sim: {filtered_tokens[0]['similarity']:.2f})")
        else:
            print("No relevant OCR tokens found.")

        # 4. Multimodal Fusion
        print("Stage 4: Fusing modalities for final answer...")
        final_answer = self.fusion.fuse_and_generate(vqa_answer, filtered_tokens)
        
        return {
            'vqa_answer': vqa_answer,
            'ocr_tokens': [t['text'] for t in ocr_data],
            'filtered_tokens': [t['text'] for t in filtered_tokens],
            'final_answer': final_answer
        }

def main():
    parser = argparse.ArgumentParser(description="Text-Aware VQA System")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        return

    vqa_system = TextAwareVQA()
    result = vqa_system.run(args.image, args.question)
    
    print("\n" + "="*50)
    print(f"FINAL ANSWER: {result['final_answer']}")
    print("="*50)

if __name__ == "__main__":
    main()
