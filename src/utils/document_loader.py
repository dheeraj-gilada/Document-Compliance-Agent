# src/utils/document_loader.py

import os
import logging
from typing import Dict, List, Any

import openai
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text # For .txt files
from unstructured.documents.elements import Element, Table # Element is general, Table is specific

logger = logging.getLogger(__name__)

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set.")
    # You might want to raise an exception here or handle it as per your app's design
    # For now, we'll let it proceed, but OpenAI calls will fail.

class DocumentLoader:
    """Loads documents using Unstructured, with 'hi_res' strategy for PDFs."""
    
    def __init__(self, docs_dir: str):
        """
        Args:
            docs_dir: Directory containing the documents.
        """
        self.docs_dir = docs_dir

    def list_documents(self) -> List[str]:
        """
        Lists all supported documents (.pdf, .txt) in the documents directory.
        """
        if not os.path.exists(self.docs_dir):
            logger.warning(f"Documents directory {self.docs_dir} does not exist.")
            return []
        
        supported_extensions = ('.pdf', '.txt')
        return [
            f for f in os.listdir(self.docs_dir)
            if os.path.isfile(os.path.join(self.docs_dir, f)) and 
               f.lower().endswith(supported_extensions)
        ]

    async def _infer_document_type_with_llm(self, filename: str, text_sample: str) -> str:
        """Infers document type using an LLM based on filename and text sample."""
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("Cannot infer document type with LLM: OPENAI_API_KEY not set.")
            return "unknown_api_key_missing"

        prompt = f"""Analyze the following document filename and a sample of its text content to determine its primary type.
Focus on common business document types like 'invoice', 'purchase_order', 'goods_receipt_note', 'credit_note', 'debit_note', 'contract', 'policy_document', 'financial_statement', 'report', 'datasheet', 'specification', 'form', 'letter', 'memo', 'resume', 'presentation_slides', 'technical_manual', 'legal_document', 'regulatory_filing', 'other'.

Filename: {filename}

Text Sample (first 500 characters):
{text_sample[:500]}

Based on the filename and text sample, what is the most likely document type? Respond with only the document type string (e.g., 'invoice', 'purchase_order', 'unknown').
If the type is unclear or not a common business document, respond with 'unknown'.
Document Type:"""

        try:
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model="gpt-4o-mini", # Using gpt-4o-mini as per project memory
                messages=[
                    {"role": "system", "content": "You are an expert document classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=50
            )
            doc_type = response.choices[0].message.content.strip().lower().replace(" ", "_")
            logger.info(f"LLM inferred document type for {filename} as: {doc_type}")
            return doc_type if doc_type else "unknown_llm_error"
        except Exception as e:
            logger.error(f"Error during LLM document type inference for {filename}: {e}", exc_info=True)
            return "unknown_llm_error"

    async def load_document(self, filename: str) -> Dict[str, Any]:
        """
        Loads a document and extracts its elements using Unstructured.
        For PDFs, uses the 'hi_res' strategy.
        Infers document type using an LLM.
        
        Args:
            filename: Name of the document file.
            
        Returns:
            Dictionary with document metadata, extracted Unstructured elements (serialized),
            plain text, extracted tables (as HTML strings), and LLM-inferred doc_type.
        """
        filepath = os.path.join(self.docs_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"Document {filepath} not found.")
            raise FileNotFoundError(f"Document {filepath} not found")

        file_ext = os.path.splitext(filename)[1].lower()
        elements: List[Element] = []

        try:
            if file_ext == '.pdf':
                logger.info(f"Processing PDF {filename} with hi_res strategy.")
                elements = partition_pdf(
                    filepath,
                    strategy='hi_res',
                    infer_table_structure=True, 
                    extract_tables=True,
                )
            elif file_ext == '.txt':
                logger.info(f"Processing TXT {filename}.")
                elements = partition_text(filename=filepath, strategy='fast') 
            else:
                logger.warning(f"Unsupported file type: {file_ext} for {filename}")
                return {
                    'filename': filename,
                    'filepath': filepath,
                    'elements': [],
                    'text': '',
                    'tables_html': [], 
                    'doc_type': 'unsupported'
                }

            text_content = "\n\n".join([el.text for el in elements if el.text is not None])
            
            # Infer document type using LLM with filename and a sample of text content
            doc_type = await self._infer_document_type_with_llm(filename, text_content)

            tables_html = []
            for el in elements:
                if isinstance(el, Table):
                    if hasattr(el, 'metadata') and hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html:
                        tables_html.append(el.metadata.text_as_html)
                    else:
                        logger.warning(f"Table in {filename} did not have text_as_html metadata. Using raw text: {el.text[:100]}...")
                        # Fallback: wrap raw table text in pre tags if HTML is not available
                        tables_html.append(f"<pre>{el.text}</pre>")


            serialized_elements = [el.to_dict() for el in elements]

            return {
                'filename': filename,
                'filepath': filepath,
                'elements': serialized_elements, 
                'text': text_content,
                'tables_html': tables_html, 
                'doc_type': doc_type
            }

        except Exception as e:
            logger.error(f"Error processing document {filename} with Unstructured: {e}", exc_info=True)
            return {
                'filename': filename,
                'filepath': filepath,
                'elements': [],
                'text': f"Error during processing: {str(e)}",
                'tables_html': [],
                'doc_type': await self._infer_document_type_with_llm(filename, f"Error during processing: {str(e)}") # Attempt to infer even on error
            }