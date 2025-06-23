# src/utils/document_loader.py

import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Try to import unstructured library
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.text import partition_text
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.image import partition_image
    from unstructured.documents.elements import Element, Table
    UNSTRUCTURED_AVAILABLE = True
    logger.info("Unstructured library successfully imported and available")
except ImportError as e:
    logger.warning(f"Unstructured library not available: {e}")
    UNSTRUCTURED_AVAILABLE = False
    # Define dummy functions to prevent errors
    def partition_pdf(*args, **kwargs): return []
    def partition_text(*args, **kwargs): return []
    def partition_docx(*args, **kwargs): return []
    def partition_image(*args, **kwargs): return []
    Element = None
    Table = None

# Import the classifier agent with error handling
try:
    from src.agents.document_classifier_agent import DocumentTypeClassifierAgent
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import DocumentTypeClassifierAgent: {e}")
    CLASSIFIER_AVAILABLE = False
    class DocumentTypeClassifierAgent:
        async def infer_document_type(self, filename: str, text_sample: str) -> str:
            logger.warning(f"DocumentTypeClassifierAgent not available due to import error. Defaulting doc_type to 'unknown' for {filename}.")
            return "unknown"

class DocumentLoader:
    """Loads documents using Unstructured, with 'hi_res' strategy for PDFs."""
    
    def __init__(self, docs_dir: str):
        """
        Args:
            docs_dir: Directory containing the documents.
        """
        self.docs_dir = docs_dir
        if CLASSIFIER_AVAILABLE:
            self.classifier_agent = DocumentTypeClassifierAgent()
        else:
            self.classifier_agent = DocumentTypeClassifierAgent()  # Will use the dummy class

    def list_documents(self) -> List[str]:
        """
        Lists all supported documents (.pdf, .txt, .docx, .jpeg, .jpg, .png, .webp) in the documents directory.
        """
        if not os.path.exists(self.docs_dir):
            logger.warning(f"Documents directory {self.docs_dir} does not exist.")
            return []
        
        supported_extensions = ('.pdf', '.txt', '.docx', '.jpeg', '.jpg', '.png', '.webp')
        return [
            f for f in os.listdir(self.docs_dir)
            if os.path.isfile(os.path.join(self.docs_dir, f)) and 
               f.lower().endswith(supported_extensions)
        ]

    async def _infer_document_type_with_llm(self, filename: str, text_sample: str) -> str:
        """Infers document type using the DocumentTypeClassifierAgent."""
        return await self.classifier_agent.infer_document_type(filename, text_sample)

    async def load_document(self, filename: str) -> Dict[str, Any]:
        """
        Loads a document and extracts its elements using Unstructured.
        For PDFs, uses the 'hi_res' strategy.
        Infers document type using the DocumentTypeClassifierAgent.
        
        Args:
            filename: Name of the document file.
            
        Returns:
            Dictionary with document metadata, extracted Unstructured elements (serialized),
            plain text, extracted tables (as HTML strings), and inferred doc_type.
        """
        filepath = os.path.join(self.docs_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"Document {filepath} not found.")
            raise FileNotFoundError(f"Document {filepath} not found")

        if not UNSTRUCTURED_AVAILABLE:
            logger.warning(f"Cannot process document {filename}: unstructured library not available")
            return {
                'filename': filename,
                'filepath': filepath,
                'elements': [],
                'text': f'Document processing unavailable - unstructured library not installed properly',
                'tables_html': [],
                'doc_type': await self._infer_document_type_with_llm(filename, 'Error: processing unavailable'),
                'error_message': 'unstructured library not available'
            }

        file_ext = os.path.splitext(filename)[1].lower()
        elements = []

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
            elif file_ext == '.docx':
                logger.info(f"Processing DOCX {filename}.")
                elements = partition_docx(filename=filepath)
            elif file_ext in ('.jpeg', '.jpg', '.png', '.webp'):
                logger.info(f"Processing IMAGE {filename}.")
                # For images, strategy can be 'auto', 'ocr_only', or 'hi_res'. 
                # 'hi_res' attempts to detect layout and then OCR.
                # OCR requires Tesseract. If not available, text_content might be empty.
                elements = partition_image(filename=filepath, strategy='hi_res', infer_table_structure=True, extract_tables=True)
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
            
            # Infer document type using the DocumentTypeClassifierAgent with filename and a sample of text content
            doc_type = await self._infer_document_type_with_llm(filename, text_content)

            tables_html = []
            for el in elements:
                if Table is not None and isinstance(el, Table):
                    if hasattr(el, 'metadata') and hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html:
                        tables_html.append(el.metadata.text_as_html)
                    else:
                        logger.warning(f"Table in {filename} did not have text_as_html metadata. Using raw text: {el.text[:100]}...")
                        # Fallback: wrap raw table text in pre tags if HTML is not available
                        tables_html.append(f"<pre>{el.text}</pre>")

            # Handle serialization gracefully
            try:
                serialized_elements = [el.to_dict() for el in elements if hasattr(el, 'to_dict')]
            except Exception as e:
                logger.warning(f"Could not serialize elements for {filename}: {e}")
                serialized_elements = []

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