# src/agents/structured_data_extractor_agent.py
import os
import json
import logging
import openai
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StructuredDataExtractorAgent:
    """Agent responsible for extracting structured data from document content using an LLM."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the agent with an OpenAI client.
        Args:
            model_name: The OpenAI model to use for data extraction.
        """
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set for StructuredDataExtractorAgent.")
            raise ValueError("OPENAI_API_KEY not set.")
        self.client = openai.AsyncOpenAI()
        self.model = model_name

    async def extract_structured_data(
        self,
        doc_type: str,
        text_content: str,
        combined_tables_html: str,
        filename: str # For logging/context
    ) -> Dict[str, Any]:
        """Extracts structured data from document content using an LLM."""

        system_prompt = f"""You are an advanced AI data extraction assistant. Your task is to analyze the provided document content and its inferred type ('{doc_type}'), then extract all meaningful information into a structured JSON format.
- Identify key entities, attributes, and relationships relevant to a document of type '{doc_type}'.
- For tabular data, represent it as a list of objects, where each object corresponds to a row, using clear header-derived keys.
- Ensure all extracted values are accurately represented (e.g., numbers as numbers, dates in YYYY-MM-DD if possible, otherwise as found).
- The JSON structure should be intuitive and based on the document's content and type. Do not use a predefined schema unless the document type strongly implies one (e.g., common fields for an 'invoice' like 'invoice_number', 'total_amount', 'line_items').
- If the document type is 'unknown' or 'other', be more general in your extraction, focusing on clear headings, sections, and data points.
- Always return a single, valid JSON object. If no meaningful data can be extracted, return an empty JSON object {{}}.
- For the document type '{doc_type}', prioritize extracting information that is typically important for such documents.
"""
        
        user_prompt = f"""Document Type: {doc_type}
Filename: {filename}

Text Content:
{text_content}

Tables (HTML format):
{combined_tables_html}

Extract all relevant information from the above content into a structured JSON format, keeping in mind the document type is '{doc_type}'."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}  # Request JSON output
            )
            
            extracted_json_str = response.choices[0].message.content
            if not extracted_json_str:
                logger.warning(f"LLM returned empty response for {filename} ({doc_type}).")
                return {{}}
            
            extracted_data = json.loads(extracted_json_str)
            logger.info(f"Successfully extracted structured data for {filename} ({doc_type}) using {self.model}.")
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for {filename} ({doc_type}): {e}. Response: {extracted_json_str}", exc_info=True)
            return {"error": "Failed to parse LLM JSON response", "raw_response": extracted_json_str}
        except Exception as e:
            logger.error(f"Error during LLM data extraction for {filename} ({doc_type}): {e}", exc_info=True)
            return {"error": f"LLM API call failed: {str(e)}"}
