# src/agents/document_classifier_agent.py
import os
import logging
import openai
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DocumentTypeClassifierAgent:
    """Agent responsible for inferring document type using an LLM."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the agent with an OpenAI client.
        Args:
            model_name: The OpenAI model to use for classification.
        """
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set for DocumentTypeClassifierAgent.")
            raise ValueError("OPENAI_API_KEY not set.")
        self.client = openai.AsyncOpenAI()
        self.model = model_name

    async def infer_document_type(self, filename: str, text_sample: str) -> str:
        """Infers document type using an LLM based on filename and text sample."""
        prompt = f"""Analyze the following document filename and a sample of its text content to determine its primary type.
Focus on common business document types like 'invoice', 'purchase_order', 'goods_receipt_note', 'credit_note', 'debit_note', 'contract', 'policy_document', 'financial_statement', 'report', 'datasheet', 'specification', 'form', 'letter', 'memo', 'resume', 'presentation_slides', 'technical_manual', 'legal_document', 'regulatory_filing', 'other'.

Filename: {filename}

Text Sample (first 500 characters):
{text_sample[:500]}

Based on the filename and text sample, what is the most likely document type? Respond with only the document type string (e.g., 'invoice', 'purchase_order', 'unknown').
If the type is unclear or not a common business document, respond with 'unknown'.
Document Type:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=50
            )
            doc_type = response.choices[0].message.content.strip().lower().replace(" ", "_")
            logger.info(f"LLM inferred document type for {filename} as: {doc_type} using {self.model}")
            return doc_type if doc_type else "unknown_llm_error"
        except Exception as e:
            logger.error(f"Error during LLM document type inference for {filename}: {e}", exc_info=True)
            return "unknown_llm_error"
