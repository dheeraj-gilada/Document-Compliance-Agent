"""
Intelligent Document Compliance & Process Automation Agent

Main entry point for the application that orchestrates the document processing,
instruction parsing, compliance checking, and report generation workflow.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utility modules
from src.utils.config import DOCS_DIR, REPORTS_DIR, OPENAI_MODEL
from src.utils.document_loader import DocumentLoader
from src.utils.extractor import DocumentExtractor

def setup_argparse() -> argparse.Namespace:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Intelligent Document Compliance & Process Automation Agent"
    )
    parser.add_argument(
        "--docs-dir", 
        type=str, 
        default=DOCS_DIR,
        help="Directory containing input documents"
    )
    parser.add_argument(
        "--instructions", 
        type=str, 
        help="Natural language instructions for compliance checks"
    )
    parser.add_argument(
        "--instructions-file", 
        type=str, 
        help="File containing natural language instructions"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=REPORTS_DIR,
        help="Directory for output reports"
    )
    parser.add_argument(
        "--extract-only", 
        action="store_true",
        help="Only extract data from documents, don't perform compliance checks"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

async def load_and_extract_documents(docs_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load documents from the specified directory and extract structured data.
    
    Args:
        docs_dir: Directory containing the documents
        
    Returns:
        Dictionary mapping document filenames to their extracted data
    """
    # Initialize document loader
    loader = DocumentLoader(docs_dir)
    
    # List available documents
    document_files = loader.list_documents()
    logger.info(f"Found {len(document_files)} documents: {document_files}")
    
    if not document_files:
        logger.warning(f"No documents found in {docs_dir}")
        return {}
    
    # Initialize extractor
    extractor = DocumentExtractor()
    
    # Process each document
    extracted_documents = {}
    for doc_name in document_files:
        logger.info(f"Processing document: {doc_name}")
        
        try:
            # Load document
            document = await loader.load_document(doc_name)
            logger.info(f"Loaded document: {doc_name}, type: {document.get('doc_type', 'unknown')}, " 
                       f"tables: {len(document.get('tables_html', []))}")
            
            if document.get('doc_type') == 'unsupported' or document.get('doc_type') == 'error':
                logger.warning(f"Skipping extraction for {doc_name} due to loading error or unsupported type: {document.get('doc_type')}")
                extracted_documents[doc_name] = {
                    'filename': doc_name,
                    'doc_type': document.get('doc_type', 'error'),
                    'error': document.get('text', 'Failed to load document or unsupported type'),
                    'extracted_data': {}
                }
                continue

            # Extract structured data
            extracted_data = extractor.extract_data(document)
            
            # Log the top-level keys from extracted_data for quick review
            if 'extracted_data' in extracted_data and isinstance(extracted_data['extracted_data'], dict):
                logger.info(f"Extracted data keys for {doc_name}: {list(extracted_data['extracted_data'].keys())}")
            elif 'error' in extracted_data:
                logger.error(f"Extraction error for {doc_name}: {extracted_data['error']}")
            else:
                logger.warning(f"Unexpected extraction result structure for {doc_name}: {extracted_data}")

            extracted_documents[doc_name] = extracted_data
            
            # Print key extracted fields for quick verification
            print(f"\n--- Extracted Data Summary for {doc_name} ---")
            if 'doc_type' in extracted_data:
                print(f"Document Type: {extracted_data['doc_type']}")
            
            if 'invoice_number' in extracted_data:
                print(f"Invoice Number: {extracted_data.get('invoice_number')}")
                print(f"PO Number: {extracted_data.get('po_number')}")
                print(f"Vendor: {extracted_data.get('vendor_name')}")
                print(f"Total Amount: {extracted_data.get('total_amount')} {extracted_data.get('currency')}")
                print(f"Line Items: {len(extracted_data.get('line_items', []))}")
            
            elif 'po_number' in extracted_data:
                print(f"PO Number: {extracted_data.get('po_number')}")
                print(f"Vendor: {extracted_data.get('vendor_name')}")
                print(f"Total Amount: {extracted_data.get('total_amount')} {extracted_data.get('currency')}")
                print(f"Line Items: {len(extracted_data.get('line_items', []))}")
            
            elif 'grn_number' in extracted_data:
                print(f"GRN Number: {extracted_data.get('grn_number')}")
                print(f"PO Number: {extracted_data.get('po_number')}")
                print(f"Receipt Date: {extracted_data.get('receipt_date')}")
                print(f"Line Items: {len(extracted_data.get('line_items', []))}")
            
            print("-----------------------------------\n")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_name}: {str(e)}")
    
    # Save all extracted data to a JSON file for reference
    extracted_dir = Path("extracted_data")
    extracted_dir.mkdir(exist_ok=True)
    
    with open(extracted_dir / "all_extracted_data.json", 'w', encoding='utf-8') as f:
        json.dump(extracted_documents, f, indent=2)
    
    logger.info(f"Extracted data for {len(extracted_documents)} documents saved to extracted_data/all_extracted_data.json")
    
    return extracted_documents

def get_instructions(args: argparse.Namespace) -> Optional[str]:
    """
    Get compliance instructions from command line args or file.
    
    Args:
        args: Command line arguments
        
    Returns:
        Instructions string or None if not provided
    """
    if args.instructions:
        return args.instructions
    
    if args.instructions_file:
        if not os.path.exists(args.instructions_file):
            logger.error(f"Instructions file not found: {args.instructions_file}")
            return None
        
        with open(args.instructions_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    return None

async def main():
    """Main entry point for the application."""
    args = setup_argparse()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1-2: Load and extract data from documents
    logger.info(f"Starting document loading and extraction from: {args.docs_dir}")
    processed_docs = await load_and_extract_documents(args.docs_dir)
    
    if not processed_docs:
        logger.error("No documents were successfully processed. Exiting.")
        return 1
    
    # If extract-only mode, exit here
    if args.extract_only:
        logger.info("Extract-only mode. Skipping compliance checks.")
        return 0
    
    # Get compliance instructions
    instructions = get_instructions(args)
    if not instructions:
        logger.error("No compliance instructions provided. Use --instructions or --instructions-file.")
        return 1
    
    logger.info("Compliance instructions received:")
    print(f"\n--- Compliance Instructions ---")
    print(instructions)
    print("------------------------------\n")
    
    # Phase 3-5: To be implemented in future phases
    logger.info("Phases 3-5 (Instruction Parsing, Compliance Engine, Report Generation) will be implemented next")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
