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
from src.agents.universal_compliance_agent import UniversalComplianceAgent

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

def ensure_dir_exists(filepath: str):
    """Ensures the directory for the given filepath exists."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

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
        "consolidated_rules_content": consolidated_rules_input, 
        "initial_document_paths": [],
        "document_queue": [],
        "current_document_path": None,
        "loaded_document_data": None,
        "processed_documents": [],
        "error_messages": [],
        "all_compliance_findings": [] 
    }

    final_graph_state = None
    try:
        async for event in app.astream_events(initial_state, version="v1"):
            kind = event["event"]
            if kind == "on_chain_end":
                if event["name"] == "LangGraph": 
                    final_graph_state = event["data"]["output"]
                    logger.info("Document processing workflow completed.")
            elif kind == "on_chain_error":
                logger.error(f"Error in workflow: {event['data']}")
                # Potentially capture or handle specific errors if needed

        if final_graph_state:
            return final_graph_state
        else:
            logger.error("Workflow did not complete successfully or final state not captured.")
            return {"processed_documents": initial_state.get("processed_documents", []), "all_compliance_findings": [], "error_messages": initial_state.get("error_messages", []) + ["Workflow did not reach final state."]}

    except Exception as e:
        logger.error(f"Exception during document processing workflow execution: {e}", exc_info=True)
        return {"processed_documents": [], "all_compliance_findings": [], "error_messages": [f"Workflow execution failed: {str(e)}"]}

async def main():
    parser = argparse.ArgumentParser(description="Intelligent Document Compliance & Process Automation Agent")
    parser.add_argument("--docs_dir", type=str, default=DOCS_DIR, 
                        help="Directory containing documents to process. Used in 'extract' and 'full' modes.")
    parser.add_argument("--mode", type=str, choices=['extract', 'compliance', 'full'], default='full',
                        help="Operation mode: 'extract' (only data extraction), 'compliance' (only compliance checks on existing data), 'full' (both).")
    parser.add_argument("--extraction-output-file", type=str, 
                        default=os.path.join("extracted_data", "all_extracted_data.json"), 
                        help="File to save extracted document data. Used in 'extract' and 'full' modes.")
    parser.add_argument("--extracted-data-input", type=str, 
                        default=os.path.join("extracted_data", "all_extracted_data.json"), 
                        help="File to load extracted document data from. Used in 'compliance' mode.")
    parser.add_argument("--compliance-output-file", type=str, 
                        default=os.path.join("reports", "compliance_checks.json"), 
                        help="File to save compliance findings. Used in 'compliance' and 'full' modes.")
    parser.add_argument("--rules-file", type=str, help="Path to the consolidated compliance rules file. Required for 'compliance' and 'full' modes.")

    args = parser.parse_args()

    if args.mode in ['compliance', 'full'] and not args.rules_file:
        parser.error("--rules-file is required for 'compliance' and 'full' modes.")

    processed_documents_list = []
    all_findings = []
    actual_state_data = {}

    if args.mode == 'extract' or args.mode == 'full':
        logger.info(f"Starting document extraction from: {args.docs_dir}")
        rules_for_workflow = None
        if args.mode == 'full':
            rules_for_workflow = load_rules_from_file(args.rules_file)
            if not rules_for_workflow:
                logger.error(f"Failed to load rules from {args.rules_file} for 'full' mode. Exiting.")
                return
            logger.info(f"Consolidated rules loaded for 'full' mode workflow.")
        
        processed_workflow_output = await run_document_processing(args.docs_dir, rules_for_workflow)
        
        logger.info(f"DEBUG main: processed_workflow_output type: {type(processed_workflow_output)}")
        if isinstance(processed_workflow_output, dict) or hasattr(processed_workflow_output, 'keys'):
            output_top_keys = list(processed_workflow_output.keys())
            logger.info(f"DEBUG main: processed_workflow_output top-level keys: {output_top_keys}")
            if 'universal_compliance_check' in output_top_keys and \
               (isinstance(processed_workflow_output.get('universal_compliance_check'), dict) or \
                hasattr(processed_workflow_output.get('universal_compliance_check'), 'keys')):
                actual_state_data = processed_workflow_output['universal_compliance_check']
            elif 'processed_documents' in output_top_keys: 
                actual_state_data = processed_workflow_output
            else:
                logger.warning("DEBUG main: Could not determine structure of processed_workflow_output.")
                actual_state_data = {}
        else:
            logger.error("DEBUG main: processed_workflow_output is not dictionary-like.")
            actual_state_data = {}

        processed_documents_list = actual_state_data.get('processed_documents', [])
        if not isinstance(processed_documents_list, list):
            logger.warning(f"'processed_documents' in actual_state_data is not a list. Defaulting to empty list.")
            processed_documents_list = []

        ensure_dir_exists(args.extraction_output_file)
        with open(args.extraction_output_file, 'w') as f_extract:
            json.dump(processed_documents_list, f_extract, indent=2)
        logger.info(f"Extracted data for {len(processed_documents_list)} documents saved to {args.extraction_output_file}")

        if args.mode == 'full':
            all_findings = actual_state_data.get('all_compliance_findings', [])
            if not isinstance(all_findings, list):
                logger.warning(f"'all_compliance_findings' in actual_state_data is not a list. Defaulting to empty list.")
                all_findings = []

    if args.mode == 'compliance':
        logger.info(f"Starting compliance check mode.")
        try:
            with open(args.extracted_data_input, 'r') as f_input_data:
                processed_documents_list = json.load(f_input_data)
            if not isinstance(processed_documents_list, list):
                logger.error(f"Data in {args.extracted_data_input} is not a list. Exiting compliance mode.")
                return
            logger.info(f"Successfully loaded {len(processed_documents_list)} processed document records from {args.extracted_data_input}.")
        except FileNotFoundError:
            logger.error(f"Extracted data input file not found: {args.extracted_data_input}. Exiting compliance mode.")
            return
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {args.extracted_data_input}. Exiting compliance mode.")
            return
        
        consolidated_rules_text = load_rules_from_file(args.rules_file)
        if not consolidated_rules_text:
            logger.error(f"Failed to load rules from {args.rules_file} for 'compliance' mode. Exiting.")
            return
        logger.info(f"Consolidated rules loaded for 'compliance' mode.")

        input_for_universal_agent = []
        for doc_data in processed_documents_list:
            if (doc_data.get("status") in ["data_extracted", "classified"] and 
                doc_data.get("filename") and 
                doc_data.get("doc_type") and 
                doc_data.get("extracted_data") is not None):
                input_for_universal_agent.append({
                    "filename": doc_data["filename"],
                    "doc_type": doc_data["doc_type"],
                    "extracted_data": doc_data["extracted_data"]
                })
            elif (doc_data.get("status") == "classified" and doc_data.get("extracted_data") is None):
                 input_for_universal_agent.append({
                    "filename": doc_data["filename"],
                    "doc_type": doc_data["doc_type"],
                    "extracted_data": {} 
                })

        if not input_for_universal_agent and processed_documents_list:
            logger.warning("No suitable documents found for compliance check after filtering from loaded data.")
        
        agent = UniversalComplianceAgent()
        try:
            logger.info(f"Performing compliance checks for {len(input_for_universal_agent)} documents.")
            all_findings = await agent.check_all_compliance(
                input_for_universal_agent,
                consolidated_rules_text
            )
            logger.info(f"Compliance check completed. {len(all_findings)} findings generated.")
        except Exception as e:
            logger.error(f"Error during compliance agent invocation: {e}", exc_info=True)
            all_findings = [] 

    if args.mode in ['compliance', 'full']:
        ensure_dir_exists(args.compliance_output_file)
        with open(args.compliance_output_file, 'w') as f_compliance:
            json.dump(all_findings, f_compliance, indent=2)
        logger.info(f"Compliance findings saved to {args.compliance_output_file}")

        if all_findings:
            logger.info("\nUniversal Compliance Findings:")
            for finding in all_findings:
                logger.info(f"  Rule ID: {finding.get('rule_id', 'N/A')}, Rule Text: {finding.get('rule_text', 'N/A')}, Status: {finding.get('status', 'N/A')}, Details: {finding.get('details', 'N/A')}, Involved: {finding.get('involved_documents', [])}")
        elif args.mode == 'compliance': 
             logger.info("No compliance findings were generated in 'compliance' mode.")
        # For 'full' mode, workflow errors would be logged by run_document_processing or the debug block above.

    logger.info(f"Process completed for mode: {args.mode}.")


if __name__ == "__main__":
    asyncio.run(main())
