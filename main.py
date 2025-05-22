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
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utility modules
from src.utils.config import DOCS_DIR
from src.workflows.document_processing_workflow import create_document_processing_graph, DocumentProcessingState

async def run_document_processing(docs_dir_to_process: str, compliance_instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    """Invokes the document processing LangGraph workflow and returns processed documents."""
    logger.info(f"Starting document processing workflow for directory: {docs_dir_to_process}")
    if compliance_instructions:
        logger.info("Compliance instructions provided and will be used in the workflow.")
    else:
        logger.info("No compliance instructions provided; compliance checks will be skipped.")
    
    graph = create_document_processing_graph() # Get the StateGraph instance
    app = graph.compile() # Compile the graph to get the runnable application
    
    initial_state: DocumentProcessingState = {
        "docs_dir": docs_dir_to_process,
        "compliance_instructions": compliance_instructions,
        "initial_document_paths": [], 
        "document_queue": [],      
        "current_document_path": None,
        "loaded_document_data": None,
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

    logger.info("Workflow processing summary:")
    for doc_summary in processed_docs_list:
        status = doc_summary.get('status', 'processed')
        filename = doc_summary.get('filename', 'unknown_file')
        doc_type = doc_summary.get('doc_type', 'N/A')
        error_info = doc_summary.get('error')
        log_msg = f"  File: {filename}, Type: {doc_type}, Status: {status}"
        if error_info:
            log_msg += f", Error: {error_info}"
        
        compliance_details = doc_summary.get('compliance_details')
        if compliance_details:
            log_msg += f", Compliance: {compliance_details.get('compliance_status', 'N/A')}"
        logger.info(log_msg)

    return processed_docs_list

async def main():
    parser = argparse.ArgumentParser(description="Intelligent Document Compliance & Process Automation Agent")
    parser.add_argument("--docs_dir", type=str, default=DOCS_DIR, 
                        help="Directory containing documents to process.")
    parser.add_argument("--output_file", type=str, 
                        default=os.path.join("extracted_data", "all_extracted_data.json"), 
                        help="File to save all processed data, including extraction and compliance results.")
    parser.add_argument("--instructions", type=str, help="Natural language compliance instructions (as a string).")
    parser.add_argument("--instructions-file", type=str, help="Path to a file containing compliance instructions.")

    args = parser.parse_args()

    compliance_instructions_text: Optional[str] = args.instructions
    if args.instructions_file:
        try:
            with open(args.instructions_file, 'r') as f_instr:
                compliance_instructions_text = f_instr.read()
            logger.info(f"Loaded compliance instructions from: {args.instructions_file}")
        except FileNotFoundError:
            logger.error(f"Instruction file not found: {args.instructions_file}. Proceeding without file-based instructions.")
            if not compliance_instructions_text: 
                 logger.warning("No compliance instructions will be used as file was not found and --instructions flag was empty.")
        except Exception as e:
            logger.error(f"Error reading instruction file {args.instructions_file}: {e}. Proceeding without file-based instructions.")
            if not compliance_instructions_text:
                 logger.warning("No compliance instructions will be used due to error in reading file and --instructions flag was empty.")

    if not compliance_instructions_text:
        logger.info("No compliance instructions provided (either via --instructions or --instructions-file). Compliance checks will be skipped.")
    else:
        logger.info("Compliance instructions are prepared.")

    logger.info(f"Starting document processing from: {args.docs_dir}")

    processed_docs_data = await run_document_processing(args.docs_dir, compliance_instructions_text)

    output_data_for_json = {}
    successful_outcomes = 0
    for doc_data in processed_docs_data:
        filename = doc_data.get('filename')
        if filename:
            output_data_for_json[filename] = doc_data  
            status = doc_data.get('status', '')
            if 'error' not in status and status not in ['skipped_extraction_no_content', 'error_loading']:
                 successful_outcomes += 1
        else:
            logger.warning(f"Processed data item found without a filename: {doc_data}. This item will not be in the output JSON keyed by filename.")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created directory for output file: {output_dir}")

    with open(args.output_file, 'w') as f:
        json.dump(output_data_for_json, f, indent=2)
    
    logger.info(f"Processed data for {len(output_data_for_json)} documents (with {successful_outcomes} considered successful outcomes) saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
