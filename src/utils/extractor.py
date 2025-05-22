import json
import logging
from typing import Dict, Any, List

from openai import OpenAI 
from src.utils.config import OPENAI_API_KEY 

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Class for extracting structured data from document content using LLMs dynamically."""
    
    def __init__(self):
        """Initialize the document extractor with OpenAI client."""
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in config or environment.")
            raise ValueError("OPENAI_API_KEY not configured.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-mini" 
    
    def _extract_with_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Helper function to make LLM calls and parse JSON response."""
        try:
            logger.debug(f"System Prompt for LLM: {system_prompt}")
            logger.debug(f"User Prompt for LLM (first 500 chars): {user_prompt[:500]}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.2,
            )
            raw_response = response.choices[0].message.content
            logger.debug(f"Raw LLM response: {raw_response}")
            
            if raw_response:
                extracted_json = json.loads(raw_response)
                return extracted_json
            else:
                logger.warning("LLM returned an empty response.")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM response as JSON: {e}. Response: {raw_response}")
            return {"error": "Failed to decode LLM response as JSON", "raw_response": raw_response}
        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}", exc_info=True)
            return {"error": f"LLM extraction failed: {str(e)}"}

    def extract_data(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically extracts structured data from a document using an LLM.
        The LLM determines the structure of the output JSON based on the document type and content.
        
        Args:
            document: Document dictionary from DocumentLoader, containing:
                      'filename': Name of the document file.
                      'doc_type': Inferred document type (e.g., 'invoice', 'purchase_order', 'unknown').
                      'text': Full text content of the document.
                      'tables_html': List of HTML strings, each representing a table.
            
        Returns:
            Dictionary with extracted structured data, or an error structure if extraction fails.
        """
        filename = document.get('filename', 'N/A')
        doc_type = document.get('doc_type', 'unknown')
        text_content = document.get('text', '')
        tables_html_list = document.get('tables_html', [])

        if not text_content and not tables_html_list:
            logger.warning(f"No text content or tables found for document {filename}. Skipping LLM extraction.")
            return {
                'filename': filename,
                'doc_type': doc_type,
                'extracted_data': {},
                'status': 'No content to process'
            }

        # Combine HTML tables into a single string for the prompt
        combined_tables_html = "\n\n".join(tables_html_list) if tables_html_list else "No tables extracted."

        system_prompt = f"""You are an advanced AI data extraction assistant. Your task is to analyze the provided document content and its inferred type ('{doc_type}'), then extract all meaningful information into a structured JSON format.
- Identify key entities, attributes, and relationships relevant to a document of type '{doc_type}'.
- For tabular data, represent it as a list of objects, where each object corresponds to a row, using clear header-derived keys.
- Ensure all extracted values are accurately represented (e.g., numbers as numbers, dates in YYYY-MM-DD if possible, otherwise as found).
- The JSON structure should be intuitive and based on the document's content and type. Do not use a predefined schema unless the document type strongly implies one (e.g., common fields for an 'invoice' like 'invoice_number', 'total_amount', 'line_items').
- If the document type is 'unknown' or 'other', be more general in your extraction, focusing on clear headings, sections, and data points.
- Always return a single, valid JSON object. If no meaningful data can be extracted, return an empty JSON object {{}}.
- For the document type '{doc_type}', prioritize extracting information that is typically important for such documents.
"""
        
        user_prompt = f"""Please extract structured data from the following document:

Filename: {filename}
Inferred Document Type: {doc_type}

Full Text Content:
{text_content}

Extracted Tables (as HTML):
{combined_tables_html}

Respond with ONLY the structured JSON output. Do not include any explanatory text before or after the JSON object.
JSON Output:
"""
        
        extracted_data = self._extract_with_llm(system_prompt, user_prompt)
        
        # Add metadata to the final result
        # The LLM's output is now the primary content of 'extracted_data'
        return {
            'filename': filename,
            'doc_type': doc_type,
            'extracted_data': extracted_data 
        }
