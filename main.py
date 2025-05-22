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

def load_rules_from_file(filepath: str) -> Optional[str]:
    """Loads consolidated rules from a file."""
    if not filepath:
        logger.warning("No rules file path provided.")
        return None
    try:
        with open(filepath, 'r') as f_rules:
            return f_rules.read()
    except FileNotFoundError:
        logger.error(f"Rules file not found: {filepath}.")
        return None
    except Exception as e:
        logger.error(f"Error loading rules from {filepath}: {e}")
        return None

async def run_document_processing(docs_dir_to_process: str, consolidated_rules_input: Optional[str] = None) -> Dict[str, Any]:
    """Invokes the document processing LangGraph workflow and returns processed documents and all compliance findings."""
    logger.info(f"Starting document processing workflow for directory: {docs_dir_to_process}")
    if consolidated_rules_input:
        logger.info("Consolidated compliance rules provided and will be used in the workflow.")
    else:
        logger.info("No consolidated compliance rules provided; universal compliance checks might be limited or skipped.")
    
    graph = create_document_processing_graph()
    app = graph.compile()
    
    initial_state: DocumentProcessingState = {
        "docs_dir": docs_dir_to_process,
        "consolidated_rules_content": consolidated_rules_input, # Use the new consolidated rules content
        "initial_document_paths": [],
        "document_queue": [],
        "current_document_path": None,
        "loaded_document_data": None,
        "processed_documents": [],
        "error_messages": [],
        "all_compliance_findings": [] # Initialize for universal agent results
    }

    final_graph_state = None
    try:
        async for event in app.astream_events(initial_state, version="v1"):
            kind = event["event"]
            if kind == "on_chain_end":
                if event["name"] == "LangGraph": # Check for the end of the main graph execution
                    final_graph_state = event["data"]["output"]
                    logger.info("Document processing workflow completed.")
            elif kind == "on_chain_error":
                logger.error(f"Error in workflow: {event['data']}")
                # Potentially capture or handle specific errors if needed

        if final_graph_state:
            return final_graph_state
        else:
            logger.error("Workflow did not complete successfully or final state not captured.")
            # Return a structure indicating failure or partial results based on last known good state if available
            # For now, returning a minimal dict to avoid downstream errors, but this could be improved.
            return {"processed_documents": initial_state.get("processed_documents", []), "all_compliance_findings": [], "error_messages": initial_state.get("error_messages", []) + ["Workflow did not reach final state."]}

    except Exception as e:
        logger.error(f"Exception during document processing workflow execution: {e}", exc_info=True)
        return {"processed_documents": [], "all_compliance_findings": [], "error_messages": [f"Workflow execution failed: {str(e)}"]}

async def main():
    parser = argparse.ArgumentParser(description="Intelligent Document Compliance & Process Automation Agent")
    parser.add_argument("--docs_dir", type=str, default=DOCS_DIR, 
                        help="Directory containing documents to process.")
    parser.add_argument("--output_file", type=str, 
                        default=os.path.join("extracted_data", "all_extracted_data.json"), 
                        help="File to save all processed data, including extraction and compliance results.")
    parser.add_argument("--rules-file", type=str, required=True, help="Path to the consolidated compliance rules file.")

    args = parser.parse_args()

    consolidated_rules_text = load_rules_from_file(args.rules_file)
    if not consolidated_rules_text:
        logger.error(f"Failed to load rules from {args.rules_file}. Exiting as rules are required.")
        return

    logger.info(f"Consolidated rules loaded successfully from {args.rules_file}.")
    logger.info(f"Starting document processing from: {args.docs_dir}")

    # Pass the loaded consolidated rules to the workflow
    processed_workflow_output = await run_document_processing(args.docs_dir, consolidated_rules_text)

    # --- BEGIN DEBUG LOGS (Revised) ---
    logger.info(f"DEBUG main: processed_workflow_output type: {type(processed_workflow_output)}")
    actual_state_data = None

    if isinstance(processed_workflow_output, dict) or hasattr(processed_workflow_output, 'keys'):
        output_top_keys = list(processed_workflow_output.keys())
        logger.info(f"DEBUG main: processed_workflow_output top-level keys: {output_top_keys}")

        # Check if the state is nested under the last node's name
        if 'universal_compliance_check' in output_top_keys and \
           (isinstance(processed_workflow_output.get('universal_compliance_check'), dict) or \
            hasattr(processed_workflow_output.get('universal_compliance_check'), 'keys')):
            
            logger.info("DEBUG main: Assuming nested state under 'universal_compliance_check'. Accessing it.")
            actual_state_data = processed_workflow_output['universal_compliance_check']
            
            # Verify if this nested dict contains the expected keys from the node's return
            if isinstance(actual_state_data, dict) or hasattr(actual_state_data, 'keys'):
                nested_keys = list(actual_state_data.keys())
                logger.info(f"DEBUG main: Keys within processed_workflow_output['universal_compliance_check']: {nested_keys}")
                if not ('processed_documents' in nested_keys and 'all_compliance_findings' in nested_keys):
                    logger.warning("DEBUG main: Nested dict from 'universal_compliance_check' is missing expected state keys. This might indicate an issue in the workflow node's return value.")
                    # Fallback or further investigation might be needed if this occurs.
            else:
                logger.warning("DEBUG main: Value under 'universal_compliance_check' is not dictionary-like as expected.")
                actual_state_data = {} # Prevent crash, but data is missing

        elif 'processed_documents' in output_top_keys and 'all_compliance_findings' in output_top_keys:
            logger.info("DEBUG main: processed_workflow_output appears to be a flat state dictionary already.")
            actual_state_data = processed_workflow_output
        else:
            logger.warning("DEBUG main: Could not determine structure of processed_workflow_output. Expected keys ('processed_documents', 'all_compliance_findings') not found at top-level or standard nested location. Output keys: %s", output_top_keys)
            actual_state_data = {} # Fallback to empty dict to prevent crash
    else:
        logger.error("DEBUG main: processed_workflow_output is not dictionary-like. Cannot retrieve state.")
        actual_state_data = {}

    logger.info(f"DEBUG main: actual_state_data type after processing: {type(actual_state_data)}")
    if isinstance(actual_state_data, dict) or hasattr(actual_state_data, 'keys'):
        logger.info(f"DEBUG main: actual_state_data final keys for processing: {list(actual_state_data.keys())}")
    # --- END DEBUG LOGS ---

    output_data_for_json = {}
    successful_outcomes = 0
    
    # Ensure 'processed_documents' is a list, even if workflow had issues or data is missing
    processed_documents_list = actual_state_data.get('processed_documents', [])
    if not isinstance(processed_documents_list, list):
        logger.warning(f"'processed_documents' in actual_state_data is not a list: {processed_documents_list}. Defaulting to empty list.")
        processed_documents_list = []

    for doc_data in processed_documents_list:
        filename = doc_data.get('filename')
        if filename:
            output_data_for_json[filename] = doc_data  
            status = doc_data.get('status', '')
            if 'error' not in status and status not in ['skipped_extraction_no_content', 'error_loading_document', 'error_file_not_found', 'error_classifying_document', 'error_extracting_data']:
                 successful_outcomes += 1
        else:
            logger.warning(f"Processed data item found without a filename: {doc_data}. This item will not be in the output JSON keyed by filename.")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created directory for output file: {output_dir}")

    # Use 'all_compliance_findings' from the actual_state_data
    all_findings = actual_state_data.get('all_compliance_findings', [])
    if not isinstance(all_findings, list):
        logger.warning(f"'all_compliance_findings' in actual_state_data is not a list: {all_findings}. Defaulting to empty list.")
        all_findings = []

    final_json_output = {
        "individual_document_results": output_data_for_json,
        "all_compliance_findings": all_findings # Updated key for universal findings
    }

    with open(args.output_file, 'w') as f:
        json.dump(final_json_output, f, indent=2)
    
    logger.info(f"Processed data for {len(output_data_for_json)} documents (with {successful_outcomes} considered successful outcomes) and all compliance findings saved to {args.output_file}")

    # Print all compliance findings to console
    if all_findings:
        logger.info("\nUniversal Compliance Findings:")
        for finding in all_findings:
            # Adjust logging to match the typical structure of universal findings
            logger.info(f"  Rule ID: {finding.get('rule_id', 'N/A')}, Rule Text: {finding.get('rule_text', 'N/A')}, Status: {finding.get('status', 'N/A')}, Details: {finding.get('details', 'N/A')}, Involved: {finding.get('involved_documents', [])}")
    elif processed_workflow_output.get("error_messages"):
        logger.warning("Workflow completed with errors, and no compliance findings were generated.")
        for err_msg in processed_workflow_output.get("error_messages", []):
            logger.error(f"  Workflow Error: {err_msg}")
    else:
        logger.info("No compliance findings were generated (this may be normal if no rules were applicable or an error occurred upstream).")

if __name__ == "__main__":
    asyncio.run(main())
