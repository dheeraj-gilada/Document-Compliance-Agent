"""
Intelligent Document Compliance & Process Automation Agent

Main entry point for the application that orchestrates the document processing,
instruction parsing, compliance checking, and report generation workflow.
"""
import asyncio
import argparse
import json
import logging
import os
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utility modules
from src.utils.config import REPORTS_DIR, DOCS_DIR
from src.workflows.document_processing_workflow import create_document_processing_graph, DocumentProcessingState

async def run_document_processing(docs_dir_to_process: str) -> List[Dict[str, Any]]:
    """Invokes the document processing LangGraph workflow and returns processed documents."""
    logger.info(f"Starting document processing workflow for directory: {docs_dir_to_process}")
    
    # Create and compile the graph
    app = create_document_processing_graph()
    
    initial_state: DocumentProcessingState = {
        "docs_dir": docs_dir_to_process,
        "document_paths": [],
        "current_document_path": None,
        "loaded_document_data": None,
        "extracted_data_single_doc": None,
        "processed_documents": [],
        "error_messages": []
    }
    
    final_state = await app.ainvoke(initial_state)
    
    processed_docs_list = final_state.get('processed_documents', [])
    workflow_errors = final_state.get('error_messages', [])

    if workflow_errors:
        logger.error("Workflow encountered errors:")
        for err in workflow_errors:
            logger.error(f"- {err}")

    # Log summary of processed documents from the workflow perspective
    logger.info("Workflow processing summary:")
    for doc_summary in processed_docs_list:
        status = doc_summary.get('status', 'processed')
        filename = doc_summary.get('filename', 'unknown_file')
        doc_type = doc_summary.get('doc_type', 'N/A')
        error_info = doc_summary.get('error')
        log_msg = f"  File: {filename}, Type: {doc_type}, Status: {status}"
        if error_info:
            log_msg += f", Error: {error_info}"
        logger.info(log_msg)

    return processed_docs_list

async def main():
    parser = argparse.ArgumentParser(description="Intelligent Document Compliance & Process Automation Agent")
    parser.add_argument("--docs_dir", type=str, default=DOCS_DIR, 
                        help="Directory containing documents to process.")
    parser.add_argument("--output_file", type=str, 
                        default=os.path.join("extracted_data", "all_extracted_data.json"), 
                        help="File to save all extracted data.")
    parser.add_argument("--instructions", type=str, help="Natural language compliance instructions.")
    parser.add_argument("--instructions-file", type=str, help="Path to a file containing compliance instructions.")

    args = parser.parse_args()

    logger.info(f"Starting document loading and extraction from: {args.docs_dir}")

    # Invoke the new workflow-based processing
    processed_docs_data = await run_document_processing(args.docs_dir)

    # Save all extracted data to a single JSON file
    # The structure from the workflow's 'processed_documents' list should be suitable
    # It's a list of dicts, where each dict is the full data for one document.
    # We might want to transform it into the previous format (dict keyed by filename)
    # For now, let's save the list directly.
    
    output_data_for_json = {}
    successful_extractions = 0
    for doc_data in processed_docs_data:
        filename = doc_data.get('filename')
        if filename and doc_data.get('status') != 'error_loading' and doc_data.get('status') != 'error_extracting' and doc_data.get('extracted_data'):
            # Use the structure that was previously saved, if desired
            # { "filename": { "filename": ..., "doc_type": ..., "extracted_data": ... } }
            output_data_for_json[filename] = {
                'filename': filename,
                'doc_type': doc_data.get('doc_type'),
                'extracted_data': doc_data.get('extracted_data')
            }
            successful_extractions +=1
        elif filename: # Even if there was an error, record it
             output_data_for_json[filename] = doc_data # Save the error status

    if not os.path.exists(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        logger.info(f"Created directory for output file: {os.path.dirname(args.output_file)}")

    with open(args.output_file, 'w') as f:
        json.dump(output_data_for_json, f, indent=2)
    
    logger.info(f"Extracted data for {successful_extractions} documents saved to {args.output_file}")

    # --- Placeholder for Phase 3: Compliance Checks ---
    compliance_instructions = args.instructions
    if args.instructions_file:
        try:
            with open(args.instructions_file, 'r') as f_instr:
                compliance_instructions = f_instr.read()
            logger.info(f"Loaded compliance instructions from: {args.instructions_file}")
        except FileNotFoundError:
            logger.error(f"Instruction file not found: {args.instructions_file}")
            compliance_instructions = None # Or handle as critical error
    
    if not compliance_instructions:
        logger.error("No compliance instructions provided. Use --instructions or --instructions-file.")
        # Depending on requirements, might exit or just skip compliance
    else:
        logger.info("Compliance instructions received. Processing would start here.")
        # TODO: Initialize and run the compliance workflow (Phase 3)
        # compliance_results = await run_compliance_workflow(processed_docs_data, compliance_instructions)
        # logger.info(f"Compliance check results: {compliance_results}")

if __name__ == "__main__":
    asyncio.run(main())
