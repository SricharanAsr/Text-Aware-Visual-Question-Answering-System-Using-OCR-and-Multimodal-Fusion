# Phase 1 Summary: Text-Aware Visual Question Answering System

## Problem Formulation & Objective
The primary objective of this research project is to develop a lightweight Visual Question Answering (VQA) system that is "text-aware." While traditional VQA models excel at identifying objects and scenes, they often fail to extract or reason over textual information embedded within images (e.g., signboards, product labels). This project aims to bridge this gap by integrating an OCR pipeline with a vision-language model (BLIP).

## Research Gaps Identified
1.  **Lack of Text Awareness**: Most baseline models (e.g., Antol et al. 2015) ignore image-embedded text.
2.  **Noisy OCR Propagation**: Existing text-aware models (LoRRA, M4C) propagate OCR errors directly into answer prediction without cleaning.
3.  **Irrelevant Token Noise**: Using all OCR tokens in fusion introduces significant noise from semantically irrelevant text.
4.  **Resource Constraints**: State-of-the-art models (BLIP-2, LLaVA) require billions of parameters and high-end GPU clusters, making them unsuitable for edge deployment.

## Methodology & Phase 1 Highlights
- **Model Selection**: Selected **BLIP (Bootstrapping Language-Image Pre-training)** for its strong zero-shot performance and efficiency compared to LLM-based models.
- **OCR Integration**: Selected **EasyOCR** for its multi-script support and robustness.
- **Innovation**: Proposed a **Question-Guided Filtering** mechanism to select only relevant OCR tokens based on semantic similarity to the question.
- **Optimization**: Implemented a two-stage cleaning pipeline (image preprocessing and text post-processing).

## Current Progress (End of Phase 1)
- [x] Comprehensive literature review of 10+ papers.
- [x] Problem formulation and gap analysis complete.
- [x] System architecture design finalized.
- [x] Dataset selection (TextVQA, OCR-VQA) verified.
- [x] Initial GitHub repository structure established.

*Prepared for Panel Review-2 | March 28, 2026*
