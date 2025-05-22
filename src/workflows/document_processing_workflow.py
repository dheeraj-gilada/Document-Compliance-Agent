# src/workflows/document_processing_workflow.py
import os
import logging
from typing import List, Dict, TypedDict, Any, Optional

from langgraph.graph import StateGraph, END

from src.utils.document_loader import DocumentLoader
from src.utils.extractor import DocumentExtractor
from src.agents.compliance_checker_agent import ComplianceCheckerAgent

logger = logging.getLogger(__name__)

# --- State Definition (Revised) ---
class DocumentProcessingState(TypedDict):
    docs_dir: str
    initial_document_paths: List[str]  # All paths found initially
    document_queue: List[str]          # Paths remaining to be processed
    current_document_path: Optional[str]
    loaded_document_data: Optional[Dict[str, Any]]
    # extracted_data_single_doc is ephemeral, its result is added to processed_documents
    processed_documents: List[Dict[str, Any]] # Accumulator for all outcomes
    error_messages: List[str]
    compliance_instructions: Optional[str]  # Added for compliance phase

# --- Node Functions (Revised) ---

async def list_document_files_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Lists all supported document files and initializes the processing queue."""
    docs_dir = state["docs_dir"]
    loader = DocumentLoader(docs_dir)
    try:
        document_filenames = loader.list_documents()
        initial_paths = [os.path.join(docs_dir, fname) for fname in document_filenames]
        logger.info(f"[Workflow] Found {len(initial_paths)} documents in {docs_dir}.")
        return {
            "initial_document_paths": initial_paths,
            "document_queue": list(initial_paths), # Create a mutable copy for the queue
            "processed_documents": [],
            "error_messages": []
        }
    except Exception as e:
        logger.error(f"[Workflow] Error listing documents in {docs_dir}: {e}", exc_info=True)
        return {
            "initial_document_paths": [],
            "document_queue": [],
            "processed_documents": [],
            "error_messages": [f"Failed to list documents: {str(e)}"]
        }

async def get_next_document_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Gets the next document from the queue."""
    doc_queue = list(state.get("document_queue", [])) # Operate on a copy
    if doc_queue:
        next_doc_path = doc_queue.pop(0)
        logger.info(f"[Workflow] Next document to process: {next_doc_path}")
        return {
            "current_document_path": next_doc_path,
            "document_queue": doc_queue,
            "loaded_document_data": None # Reset for the new document
        }
    else:
        logger.info("[Workflow] Document queue empty. No more documents to process.")
        return {"current_document_path": None, "document_queue": []}

async def load_and_classify_document_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Loads and classifies the current document. Updates processed_documents on loading failure."""
    current_document_path = state["current_document_path"]
    if not current_document_path:
        # This case should ideally not be reached if get_next_document_node works correctly
        return {"error_messages": state.get("error_messages", []) + ["No current document path to load."]}

    docs_dir = os.path.dirname(current_document_path)
    filename = os.path.basename(current_document_path)
    loader = DocumentLoader(docs_dir)
    
    try:
        logger.info(f"[Workflow] Loading and classifying: {filename}")
        document_data = await loader.load_document(filename)
        logger.info(f"[Workflow] Loaded document: {filename}, type: {document_data.get('doc_type')}")
        return {"loaded_document_data": document_data} # Success, no update to processed_documents yet
    except Exception as e:
        logger.error(f"[Workflow] Error loading/classifying {filename}: {e}", exc_info=True)
        error_entry = {
            'filename': filename,
            'status': 'error_loading',
            'error': str(e)
        }
        updated_processed_documents = state.get("processed_documents", []) + [error_entry]
        return {
            "loaded_document_data": None, 
            "processed_documents": updated_processed_documents,
            "error_messages": state.get("error_messages", []) + [f"Failed to load/classify {filename}: {str(e)}"]
        }

async def record_skipped_extraction_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Records that extraction was skipped for the current document (e.g., no text content)."""
    current_document_path = state["current_document_path"]
    loaded_data = state.get("loaded_document_data", {})
    filename = os.path.basename(current_document_path if current_document_path else "unknown_file")
    doc_type = loaded_data.get('doc_type', 'unknown_type_skipped_extraction')

    logger.warning(f"[Workflow] Skipping extraction for {filename} (type: {doc_type}) due to missing text content or other load issue not causing an exception.")
    
    skipped_entry = {
        'filename': filename,
        'doc_type': doc_type,
        'extracted_data': {},
        'status': 'skipped_extraction_no_content'
    }
    updated_processed_documents = state.get("processed_documents", []) + [skipped_entry]
    return {"processed_documents": updated_processed_documents}

async def extract_document_data_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Extracts data and updates processed_documents with the result (success or error)."""
    loaded_document_data = state.get("loaded_document_data") # Use .get for safety
    current_document_path = state.get("current_document_path", "unknown_file_at_extraction")
    filename = os.path.basename(current_document_path)

    # Safety check: ensure loaded_document_data exists and has either 'text' or 'tables_html'
    if not loaded_document_data or (not loaded_document_data.get('text') and not loaded_document_data.get('tables_html')):
        logger.error(f"[Workflow] extract_document_data_node called for {filename} but loaded_document_data is missing or lacks 'text' and 'tables_html'. This indicates a logic error in graph routing.")
        error_entry = {
            'filename': filename,
            'doc_type': loaded_document_data.get('doc_type', 'error_in_routing_to_extraction') if loaded_document_data else 'unknown',
            'extracted_data': {},
            'status': 'error_logic_extraction_called_improperly',
            'error': 'Extraction called without valid content (text or tables_html)'
        }
        updated_processed_documents = state.get("processed_documents", []) + [error_entry]
        return {"processed_documents": updated_processed_documents}

    extractor = DocumentExtractor()
    try:
        logger.info(f"[Workflow] Extracting data from: {filename}")
        # extractor.extract_data is expected to return the full document structure including filename, type, and extracted_data
        # It should be able to handle loaded_document_data that might have only text, only tables, or both.
        extracted_full_data = await extractor.extract_data(loaded_document_data) 
        logger.info(f"[Workflow] Extracted data for {filename}. Keys: {list(extracted_full_data.get('extracted_data', {}).keys()) if extracted_full_data and extracted_full_data.get('extracted_data') else 'None'}")
        
        # Ensure the status from extractor is preserved, or set a default if not present
        if 'status' not in extracted_full_data:
            extracted_full_data['status'] = 'success_extraction'
            
        updated_processed_documents = state.get("processed_documents", []) + [extracted_full_data]
        return {"processed_documents": updated_processed_documents}
    except Exception as e:
        logger.error(f"[Workflow] Error extracting data from {filename}: {e}", exc_info=True)
        error_entry = {
            'filename': filename,
            'doc_type': loaded_document_data.get('doc_type', 'unknown_extraction_error'),
            'extracted_data': {},
            'status': 'error_extracting',
            'error': str(e)
        }
        updated_processed_documents = state.get("processed_documents", []) + [error_entry]
        return {
            "processed_documents": updated_processed_documents,
            "error_messages": state.get("error_messages", []) + [f"Failed to extract data from {filename}: {str(e)}"]
        }

# --- New Node for Compliance Check ---
async def compliance_check_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Performs compliance check on the extracted data for the current document."""
    current_document_path = state.get("current_document_path")
    compliance_instructions = state.get("compliance_instructions")
    processed_docs_list = list(state.get("processed_documents", [])) # Operate on a copy
    filename = os.path.basename(current_document_path if current_document_path else "unknown_file_compliance")

    if not current_document_path:
        logger.error("[Workflow] Compliance check called without a current_document_path.")
        return {"error_messages": state.get("error_messages", []) + ["Compliance check: No current document path."]}

    if not compliance_instructions:
        logger.info(f"[Workflow] No compliance instructions provided. Skipping compliance check for {filename}.")
        # Find the document and ensure its status reflects skipped compliance if needed,
        # though decide_after_extraction should prevent this node from being called.
        return {} # No change to processed_documents if instructions are missing (should be caught by conditional edge)

    # Find the successfully extracted data for the current document
    doc_to_check = None
    doc_idx = -1
    for i, doc_entry in reversed(list(enumerate(processed_docs_list))):
        if doc_entry.get('filename') == filename and doc_entry.get('status') == 'success_extraction':
            doc_to_check = doc_entry
            doc_idx = i
            break

    if not doc_to_check:
        logger.error(f"[Workflow] Compliance check for {filename}: Successfully extracted data not found in processed_documents. This might be a logic error.")
        # Update status for the latest entry of this file if it exists, or add a new error entry
        found_existing_entry_for_update = False
        for i, doc_entry in reversed(list(enumerate(processed_docs_list))):
            if doc_entry.get('filename') == filename:
                processed_docs_list[i]['status'] = 'error_compliance_data_not_found'
                processed_docs_list[i]['compliance_details'] = {'compliance_status': 'error', 'findings': [{'rule_checked': 'N/A', 'status': 'error', 'details': 'Extracted data not found for compliance check.'}]}
                found_existing_entry_for_update = True
                break
        if not found_existing_entry_for_update:
             processed_docs_list.append({
                'filename': filename,
                'doc_type': 'unknown',
                'status': 'error_compliance_data_not_found',
                'compliance_details': {'compliance_status': 'error', 'findings': [{'rule_checked': 'N/A', 'status': 'error', 'details': 'Extracted data not found for compliance check.'}]}
            })
        return {"processed_documents": processed_docs_list, "error_messages": state.get("error_messages", []) + [f"Compliance check for {filename}: Extracted data not found."]}

    agent = ComplianceCheckerAgent()
    try:
        logger.info(f"[Workflow] Performing compliance check for: {filename} (type: {doc_to_check.get('doc_type')})")
        # The ComplianceCheckerAgent expects the 'extracted_data' field within the passed dictionary.
        # doc_to_check already has this structure from DocumentExtractor
        compliance_result = await agent.check_compliance(
            extracted_data=doc_to_check, # Pass the whole doc_to_check which contains 'extracted_data' field
            document_type=doc_to_check.get('doc_type', 'unknown'),
            compliance_instructions=compliance_instructions
        )
        
        # Update the document's entry in processed_documents
        processed_docs_list[doc_idx]['compliance_details'] = compliance_result
        # Determine overall status based on compliance_result
        if compliance_result.get('compliance_status') == 'compliant':
            processed_docs_list[doc_idx]['status'] = 'compliance_passed'
        elif compliance_result.get('compliance_status') == 'non-compliant':
            processed_docs_list[doc_idx]['status'] = 'compliance_failed'
        elif compliance_result.get('compliance_status') == 'warning':
            processed_docs_list[doc_idx]['status'] = 'compliance_warning'
        else: # error or other status from agent
            processed_docs_list[doc_idx]['status'] = f"compliance_{compliance_result.get('compliance_status', 'unknown_state')}"
        
        logger.info(f"[Workflow] Compliance check for {filename} completed. Status: {processed_docs_list[doc_idx]['status']}")
        return {"processed_documents": processed_docs_list}

    except Exception as e:
        logger.error(f"[Workflow] Error during compliance check for {filename}: {e}", exc_info=True)
        processed_docs_list[doc_idx]['compliance_details'] = {
            'compliance_status': 'error_agent_call',
            'findings': [{'rule_checked': 'Agent Execution', 'status': 'error', 'details': str(e)}]
        }
        processed_docs_list[doc_idx]['status'] = 'error_compliance_check'
        return {
            "processed_documents": processed_docs_list,
            "error_messages": state.get("error_messages", []) + [f"Compliance check failed for {filename}: {str(e)}"]
        }

# --- Graph Assembly (Revised) ---

def create_document_processing_graph():
    """Creates and returns the LangGraph for document processing."""
    workflow = StateGraph(DocumentProcessingState)

    # Add nodes
    workflow.add_node("list_files", list_document_files_node)
    workflow.add_node("get_next_document", get_next_document_node)
    workflow.add_node("load_classify_doc", load_and_classify_document_node)
    workflow.add_node("record_skipped_extraction", record_skipped_extraction_node)
    workflow.add_node("extract_data", extract_document_data_node)
    workflow.add_node("compliance_check", compliance_check_node) # New node

    # Define entry and connections
    workflow.set_entry_point("list_files")
    workflow.add_edge("list_files", "get_next_document")
    
    workflow.add_conditional_edges(
        "get_next_document",
        lambda state: "load_classify_doc" if state.get("current_document_path") else END,
        {
            "load_classify_doc": "load_classify_doc",
            END: END
        }
    )

    def decide_after_load(state: DocumentProcessingState) -> str:
        loaded_data = state.get("loaded_document_data")
        current_path = state.get('current_document_path')
        if not loaded_data:
            logger.info(f"[Workflow] Loading failed for {current_path}, proceeding to next document.")
            return "get_next_document"
        elif not loaded_data.get("text") and not loaded_data.get("tables_html"):
            logger.info(f"[Workflow] No text or tables_html for {current_path}, recording skipped extraction.")
            return "record_skipped_extraction"
        else:
            logger.info(f"[Workflow] Content (text or tables_html) found for {current_path}, proceeding to extraction.")
            return "extract_data"

    workflow.add_conditional_edges(
        "load_classify_doc",
        decide_after_load,
        {
            "get_next_document": "get_next_document",
            "record_skipped_extraction": "record_skipped_extraction",
            "extract_data": "extract_data"
        }
    )
    
    workflow.add_edge("record_skipped_extraction", "get_next_document")

    # New conditional logic after extraction
    def decide_after_extraction(state: DocumentProcessingState) -> str:
        current_document_path = state.get("current_document_path")
        filename = os.path.basename(current_document_path if current_document_path else "unknown_file")
        
        if not state.get("compliance_instructions"):
            logger.info(f"[Workflow] No compliance instructions. Skipping compliance check for {filename}.")
            return "get_next_document"

        # Check status of the current document in processed_documents
        # Find the latest entry for the current document
        latest_entry_for_doc = None
        for doc_entry in reversed(state.get("processed_documents", [])):
            if doc_entry.get('filename') == filename:
                latest_entry_for_doc = doc_entry
                break
        
        if latest_entry_for_doc and latest_entry_for_doc.get('status') == 'success_extraction':
            logger.info(f"[Workflow] Extracted data found for {filename}. Proceeding to compliance check.")
            return "compliance_check"
        else:
            status_found = latest_entry_for_doc.get('status') if latest_entry_for_doc else 'no_entry_found'
            logger.info(f"[Workflow] Compliance check for {filename} skipped. Reason: No compliance instructions or extraction not successful (status: {status_found}).")
            return "get_next_document"

    workflow.add_conditional_edges(
        "extract_data",
        decide_after_extraction,
        {
            "compliance_check": "compliance_check",
            "get_next_document": "get_next_document"
        }
    )

    workflow.add_edge("compliance_check", "get_next_document")
    
    logger.info("[Workflow] Document processing graph compiled with compliance check stage.")
    return workflow

# Example of how to run (for testing, typically called from main.py)
async def run_workflow_example(docs_dir_path):
    app = create_document_processing_graph()
    # Initial state needs to reflect the new DocumentProcessingState structure
    initial_state: DocumentProcessingState = {
        "docs_dir": docs_dir_path,
        "initial_document_paths": [],
        "document_queue": [],
        "current_document_path": None,
        "loaded_document_data": None,
        "processed_documents": [],
        "error_messages": [],
        "compliance_instructions": "example_instructions" # Added for compliance phase
    }
    final_state = await app.ainvoke(initial_state)
    
    print("\n--- Workflow Final State ---")
    for doc_summary in final_state.get('processed_documents', []):
        print(f"  File: {doc_summary.get('filename')}, Type: {doc_summary.get('doc_type', 'N/A')}, Status: {doc_summary.get('status', 'processed')}")
        if 'error' in doc_summary:
            print(f"    Error: {doc_summary['error']}")
        if 'compliance_details' in doc_summary:
            print(f"    Compliance Status: {doc_summary['compliance_details'].get('compliance_status', 'N/A')}")
            for finding in doc_summary['compliance_details'].get('findings', []):
                print(f"      Rule: {finding.get('rule_checked', 'N/A')}, Status: {finding.get('status', 'N/A')}, Details: {finding.get('details', 'N/A')}")
    if final_state.get('error_messages'):
        print("\nOverall Errors during workflow:")
        for err in final_state['error_messages']:
            print(f"  - {err}")
    return final_state

if __name__ == '__main__':
    import asyncio
    import json 
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_docs_dir = os.path.join(project_root, "docs")
    
    asyncio.run(run_workflow_example(test_docs_dir))
