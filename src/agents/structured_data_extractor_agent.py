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

        system_prompt = f"""You are an advanced AI data extraction assistant. Your task is to analyze the provided document content (both 'Text Content' and 'Tables') and its inferred type ('{doc_type}'), then extract ALL meaningful information into a structured JSON format.

**Key Instructions:**
1.  **Comprehensive Analysis:** Meticulously analyze the ENTIRE 'Text Content' section for standalone data points, key-value pairs, and important information, IN ADDITION to the structured 'Tables'. Do not solely rely on tables.
2.  **Field Identification:**
    *   Identify key entities, attributes, and relationships relevant to a document of type '{doc_type}'.
    *   For common document types like 'invoice', 'purchase_order', 'delivery_note', actively look for fields such as:
        *   Document ID (e.g., Invoice ID, PO Number, Delivery Note Number)
        *   Dates (e.g., Issue Date, Due Date, Delivery Date)
        *   Names & Roles (e.g., Vendor Name, Supplier Name, Customer Name, Buyer Name, Bill To, Ship To)
        *   Addresses (e.g., Shipping Address, Billing Address, Vendor Address, Customer Address). Try to capture full addresses, including street, city, state, zip, and country if present, even if they span multiple lines.
        *   Monetary Amounts (e.g., Subtotal, Total, VAT, Tax, Discounts)
        *   Line Items / Product Details (description, quantity, unit price, total price per item).
3.  **Data Representation:**
    *   Represent tabular data as a list of objects, where each object corresponds to a row, using clear header-derived keys.
    *   Ensure all extracted values are accurately represented (e.g., numbers as numbers, dates ideally in YYYY-MM-DD if clearly parsable, otherwise as found).
    *   If information is present in multiple places (e.g., an address in text and also in a table), prioritize the most complete and structured representation.
4.  **JSON Output Structure:**
    *   The JSON structure should be intuitive and based on the document's content and type.
    *   For specific document types like '{doc_type}', group related information under a main key (e.g., for an 'invoice', use a top-level key like `"invoice": {{{{...all invoice details...}}}}`).
    *   Use clear, descriptive, snake_case keys for extracted fields. For instance, an invoice's main identifier might be named `invoice_id` or `invoice_number` in the JSON, aiming to reflect the document's terminology while ensuring valid and consistent key formats. Similarly for other fields like `purchase_order_number`.
    *   If the document type is 'unknown' or 'other', be more general in your extraction, focusing on clear headings, sections, and data points.
5.  **Output Requirements:**
    *   Always return a single, valid JSON object.
    *   If NO meaningful data can be extracted from the entire document, return an empty JSON object {{}}.
    *   Do NOT invent data. If a field is not present in the document, it should not be in the JSON.

Prioritize extracting information that is typically important for a document of type '{doc_type}' from ALL provided content.
"""
        
        user_prompt = f"""Document Type: {doc_type}
Filename: {filename}

Text Content:
{text_content}

Tables (HTML format):
{combined_tables_html}

Based on the System Prompt instructions, extract all relevant information from the above content into a single, structured JSON object. Ensure the JSON is well-formed and adheres to all guidelines mentioned in the system prompt, especially regarding the analysis of both Text Content and Tables for the document type '{doc_type}'."""

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
                return {}
            
            extracted_data = json.loads(extracted_json_str)
            logger.info(f"Successfully extracted structured data for {filename} ({doc_type}) using {self.model}.")
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for {filename} ({doc_type}): {e}. Response: {extracted_json_str}", exc_info=True)
            return {"error": "Failed to parse LLM JSON response", "raw_response": extracted_json_str}
        except Exception as e:
            logger.error(f"Error during LLM data extraction for {filename} ({doc_type}): {e}", exc_info=True)
            return {"error": f"LLM API call failed: {str(e)}"}
