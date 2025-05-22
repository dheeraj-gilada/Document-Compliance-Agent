# src/utils/document_loader.py

import os
import logging
from typing import Dict, List, Any

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text # For .txt files
from unstructured.documents.elements import Element, Table # Element is general, Table is specific

from src.agents.document_classifier_agent import DocumentTypeClassifierAgent

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads documents using Unstructured, with 'hi_res' strategy for PDFs."""
    
    def __init__(self, docs_dir: str):
        """
        Args:
            docs_dir: Directory containing the documents.
        """
        self.docs_dir = docs_dir
        self.classifier_agent = DocumentTypeClassifierAgent()

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
            
            # Infer document type using the DocumentTypeClassifierAgent with filename and a sample of text content
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