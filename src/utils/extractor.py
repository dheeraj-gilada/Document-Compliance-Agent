import json
import logging
from typing import Dict, Any, List

from src.agents.structured_data_extractor_agent import StructuredDataExtractorAgent

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Class for extracting structured data from document content using LLMs dynamically."""
    
    def __init__(self):
        """Initialize the document extractor with the StructuredDataExtractorAgent."""
        self.agent = StructuredDataExtractorAgent()
    
    async def extract_data(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts structured data from a document's content using the StructuredDataExtractorAgent.
        It leverages the document type (inferred by DocumentLoader) and its text/table content.

        Args:
            document: A dictionary containing document details including:
                      'filename' (str): The name of the document.
                      'doc_type' (str): The inferred document type.
                      'text_content' (str): The plain text content of the document.
                      'tables_html' (List[str]): A list of HTML strings representing tables.

        Returns:
            A dictionary containing the extracted structured data.
            If extraction fails or no content is found, it may return an empty dict or error info.
        """
        filename = document.get('filename', 'unknown_file')
        doc_type = document.get('doc_type', 'unknown')
        text_content = document.get('text_content', '')
        tables_html_list = document.get('tables_html', [])

        if not text_content and not tables_html_list:
            logger.warning(f"No text content or tables found for document {filename} ({doc_type}). Skipping LLM extraction.")
            return {
                'filename': filename,
                'doc_type': doc_type,
                'extracted_data': {},
                'status': 'No content to process'
            }

        combined_tables_html = "\n\n".join(tables_html_list) if tables_html_list else "No tables extracted."

        extracted_data = await self.agent.extract_structured_data(
            doc_type=doc_type,
            text_content=text_content,
            combined_tables_html=combined_tables_html,
            filename=filename
        )

        return {
            'filename': filename,
            'doc_type': doc_type,
            'extracted_data': extracted_data 
        }
