# src/workflows/document_processing_workflow.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

from langgraph.graph import StateGraph, END

from src.utils.document_loader import DocumentLoader # Assuming this handles loading from paths
from src.utils.extractor import DocumentExtractor # For text extraction
from src.agents.document_classifier_agent import DocumentTypeClassifierAgent # Corrected import
from src.agents.structured_data_extractor_agent import StructuredDataExtractorAgent
from src.agents.universal_compliance_agent import UniversalComplianceAgent # New Universal Agent

logger = logging.getLogger(__name__)

# --- State Definition (Revised) ---
class DocumentProcessingState(dict):
    docs_dir: str
    consolidated_rules_content: Optional[str] # Renamed from cross_document_rules_content, holds all rules
    initial_document_paths: List[str]
    document_queue: List[str]
    current_document_path: Optional[str]
    loaded_document_data: Optional[Dict[str, Any]] # Holds content and metadata for current doc
    processed_documents: List[Dict[str, Any]] # Accumulator for all outcomes (classification, extraction)
    error_messages: List[str]
    all_compliance_findings: List[Dict[str, Any]] # Renamed from cross_document_findings, holds all findings from Universal Agent

# --- Node Functions (Revised) ---

async def list_files_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Lists all supported document files and initializes the processing queue."""
    docs_dir = state["docs_dir"]
    loader = DocumentLoader(docs_dir)
    try:
        document_filenames = loader.list_documents()
        initial_paths = [os.path.join(docs_dir, fname) for fname in document_filenames]
        logger.info(f"[Workflow] Found {len(initial_paths)} documents in {docs_dir}.")
        # No individual compliance instructions needed in state anymore
        # state['compliance_instructions'] = state.get('compliance_instructions')
        state['consolidated_rules_content'] = state.get('consolidated_rules_content')
        state['all_compliance_findings'] = [] # Initialize
        return {
            "initial_document_paths": initial_paths,
            "document_queue": initial_paths.copy(), # Initialize the queue
            "processed_documents": [],
            "error_messages": [],
            "all_compliance_findings": [] # Ensure it's initialized
        }
    except Exception as e:
        logger.error(f"[Workflow] Error listing documents in {docs_dir}: {e}", exc_info=True)
        return {
            "initial_document_paths": [],
            "document_queue": [],
            "processed_documents": [],
            "error_messages": [f"Failed to list documents: {str(e)}"],
            "all_compliance_findings": [] # Ensure it's initialized
        }

async def get_next_document_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Gets the next document from the queue."""
    doc_queue = list(state.get("document_queue", [])) # Operate on a copy
    if doc_queue:
        next_doc_path = doc_queue.pop(0)
        logger.info(f"[Workflow] Next document to process: {next_doc_path}")
        return {"current_document_path": next_doc_path, "document_queue": doc_queue}
    else:
        logger.info("[Workflow] Document queue empty. No more documents to process.")
        return {"current_document_path": None, "document_queue": []}

async def load_classify_document_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Loads and classifies the current document using DocumentLoader. Updates processed_documents."""
    current_doc_path = state.get("current_document_path")
    if not current_doc_path:
        logger.error("[Workflow] No current document path found for loading.")
        return {"error_messages": state["error_messages"] + ["No current document path for loading."]}

    doc_filename = os.path.basename(current_doc_path)
    docs_dir = os.path.dirname(current_doc_path)
    
    logger.info(f"[Workflow] Attempting to load and classify: {doc_filename} from {docs_dir}")
    loader = DocumentLoader(docs_dir) 

    doc_data_for_state = {
        "filename": doc_filename,
        "path": current_doc_path,
        "status": "initialization",
        "doc_type": None,
        "extracted_text_content": None,
        "extracted_tables_html": [], # Added for tables
        "extracted_data": None, # To be filled by the next node
        "error_message": None
    }

    try:
        # DocumentLoader.load_document handles loading, text extraction, table extraction, and classification
        loaded_info = await loader.load_document(doc_filename) 

        if not loaded_info or not loaded_info.get("text"):
            # Check if 'text' is None or empty, or if loaded_info itself is problematic
            logger.error(f"[Workflow] Failed to load or extract text from document: {doc_filename}. Skipping.")
            doc_data_for_state["status"] = "error_loading_document"
            doc_data_for_state["error_message"] = loaded_info.get("error_message", "Failed to load document content or extract text.")
        else:
            doc_data_for_state["doc_type"] = loaded_info.get("doc_type", "unknown")
            doc_data_for_state["extracted_text_content"] = loaded_info.get("text")
            doc_data_for_state["extracted_tables_html"] = loaded_info.get("tables_html", [])
            doc_data_for_state["status"] = "classified" # Document is loaded, text/tables extracted, and classified
            logger.info(f"[Workflow] Document {doc_filename} processed by DocumentLoader. Type: {doc_data_for_state['doc_type']}. Text length: {len(doc_data_for_state['extracted_text_content'] or '')}")

    except FileNotFoundError as e:
        logger.error(f"[Workflow] File not found for {doc_filename}: {e}", exc_info=True)
        doc_data_for_state["status"] = "error_file_not_found"
        doc_data_for_state["error_message"] = str(e)
    except Exception as e:
        logger.error(f"[Workflow] Error during load/classification of {doc_filename} by DocumentLoader: {e}", exc_info=True)
        doc_data_for_state["status"] = "error_load_classify"
        doc_data_for_state["error_message"] = str(e)
    
    processed_docs_list = list(state.get("processed_documents", []))
    processed_docs_list.append(doc_data_for_state)
    
    # Update error messages in state if any error occurred for this document
    current_errors = list(state.get("error_messages", []))
    if doc_data_for_state["error_message"]:
        current_errors.append(f"{doc_data_for_state['status']} for {doc_filename}: {doc_data_for_state['error_message']}")

    return {"processed_documents": processed_docs_list, "error_messages": current_errors}

async def extract_document_data_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Extracts data and updates processed_documents with the result (success or error)."""
    current_doc_path = state.get("current_document_path")
    processed_docs_list = list(state.get("processed_documents", []))
    current_errors = list(state.get("error_messages", []))

    if not current_doc_path:
        logger.error("[Workflow] No current document path in extract_document_data_node.")
        return {"error_messages": current_errors + ["Internal error: No current_doc_path in extract_data_node"]}

    filename = os.path.basename(current_doc_path)
    
    doc_entry_to_update = None
    entry_index_in_list = -1 # Initialize with a value indicating not found

    for entry_idx, entry in reversed(list(enumerate(processed_docs_list))):
        if entry.get('filename') == filename:
            doc_entry_to_update = entry
            entry_index_in_list = entry_idx
            break

    if not doc_entry_to_update or doc_entry_to_update.get("status") in ["error_loading_document", "error_file_not_found", "error_load_classify"]:
        logger.warning(f"[Workflow] Skipping data extraction for {filename} due to prior error ('{doc_entry_to_update.get('status') if doc_entry_to_update else 'No entry'}') or no entry.")
        return {} # No change to state, error already recorded or document skipped.

    doc_type = doc_entry_to_update.get("doc_type", "unknown")
    text_content = doc_entry_to_update.get("extracted_text_content", "")
    tables_html_list = doc_entry_to_update.get("extracted_tables_html", [])
    combined_tables_html = "\n\n".join(tables_html_list) # Combine list of HTML tables into a single string

    if not text_content and not combined_tables_html:
        logger.warning(f"[Workflow] No text content or tables for {filename} (type: {doc_type}). Skipping structured data extraction.")
        doc_entry_to_update["status"] = "skipped_extraction_no_content"
        if entry_index_in_list != -1: # Ensure entry was found before trying to update
            processed_docs_list[entry_index_in_list] = doc_entry_to_update
        return {"processed_documents": processed_docs_list}

    data_extractor_agent = StructuredDataExtractorAgent()
    try:
        logger.info(f"[Workflow] Extracting structured data from: {filename} (type: {doc_type})")
        extracted_data = await data_extractor_agent.extract_structured_data(
            doc_type=doc_type, 
            text_content=text_content, 
            combined_tables_html=combined_tables_html, 
            filename=filename
        )
        doc_entry_to_update["extracted_data"] = extracted_data
        doc_entry_to_update["status"] = "data_extracted"
        logger.info(f"[Workflow] Structured data extracted for {filename}.")
        if entry_index_in_list != -1:
            processed_docs_list[entry_index_in_list] = doc_entry_to_update
        return {"processed_documents": processed_docs_list}
    except Exception as e:
        logger.error(f"[Workflow] Error extracting structured data from {filename}: {e}", exc_info=True)
        doc_entry_to_update["status"] = "error_extraction"
        doc_entry_to_update["error_message"] = str(e)
        if entry_index_in_list != -1:
            processed_docs_list[entry_index_in_list] = doc_entry_to_update
            current_errors.append(f"Data extraction error for {filename}: {str(e)}") # Add specific error
        return {"processed_documents": processed_docs_list, "error_messages": current_errors}

# Renamed and Revised Node for Universal Compliance Check
async def universal_compliance_check_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Performs universal compliance check on the extracted data from all processed documents."""
    processed_docs_input_list = state.get("processed_documents", [])
    consolidated_rules = state.get("consolidated_rules_content")
    current_errors = state.get("error_messages", [])

    if not consolidated_rules:
        logger.info("[Workflow] No consolidated compliance rules provided. Skipping universal compliance check.")
        return {
            "all_compliance_findings": [],
            "processed_documents": processed_docs_input_list, # Pass through
            "error_messages": current_errors # Pass through
        }

    # Prepare data for the agent: List of {'filename': str, 'doc_type': str, 'extracted_data': Dict}
    # Only include documents that have relevant data for compliance.
    input_for_universal_agent = []
    for doc_data in processed_docs_input_list:
        if (doc_data.get("status") in ["data_extracted", "classified"] and # Include if data extracted or at least classified
            doc_data.get("filename") and 
            doc_data.get("doc_type") and 
            doc_data.get("extracted_data") is not None): # Ensure extracted_data is present
            input_for_universal_agent.append({
                "filename": doc_data["filename"],
                "doc_type": doc_data["doc_type"],
                "extracted_data": doc_data["extracted_data"]
            })
        elif (doc_data.get("status") == "classified" and doc_data.get("extracted_data") is None):
             input_for_universal_agent.append({
                "filename": doc_data["filename"],
                "doc_type": doc_data["doc_type"],
                "extracted_data": {} # Pass empty dict if classified but no data extracted (e.g. non-text doc)
            })
        else:
            logger.debug(f"[Workflow] Document {doc_data.get('filename', 'Unknown')} with status {doc_data.get('status', 'N/A')} will not be included in universal compliance check due to missing critical data.")

    if not input_for_universal_agent and processed_docs_input_list: # If there were docs but none suitable
        logger.warning("[Workflow] No suitable documents found for universal compliance check after filtering, though documents were processed.")
        # Still call agent, it might have rules that are 'not_applicable' universally
    elif not processed_docs_input_list:
        logger.info("[Workflow] No documents were processed at all. Universal compliance check will reflect this.")
        # Agent will be called with empty list, should handle it.

    agent = UniversalComplianceAgent()
    try:
        logger.info(f"[Workflow] Performing universal compliance checks for {len(input_for_universal_agent)} documents against consolidated rules.")
        all_findings = await agent.check_all_compliance(
            input_for_universal_agent,
            consolidated_rules
        )
        logger.info(f"[Workflow] Universal compliance check completed. {len(all_findings)} findings generated.")
        return {
            "all_compliance_findings": all_findings,
            "processed_documents": processed_docs_input_list, # Pass through existing
            "error_messages": current_errors # Pass through existing
        }
    except Exception as e:
        logger.error(f"[Workflow] Error during universal compliance check: {e}", exc_info=True)
        error_finding = {
            "rule_id": "N/A",
            "rule_checked": "Universal Compliance Agent Invocation Error", 
            "status": "error", 
            "details": f"Universal compliance agent failed: {str(e)}",
            "involved_documents": [d['filename'] for d in input_for_universal_agent if 'filename' in d]
        }
        return {
            "all_compliance_findings": [error_finding],
            "processed_documents": processed_docs_input_list, # Pass through existing
            "error_messages": current_errors + [f"Universal compliance check failed: {str(e)}"]
        }


# --- Graph Assembly (Revised) ---
def create_document_processing_graph():
    """Creates and returns the LangGraph for document processing with universal compliance."""
    workflow = StateGraph(DocumentProcessingState)

    # Define nodes
    workflow.add_node("list_files", list_files_node)
    workflow.add_node("get_next_document", get_next_document_node)
    workflow.add_node("load_classify_doc", load_classify_document_node)
    workflow.add_node("extract_data", extract_document_data_node)
    workflow.add_node("universal_compliance_check", universal_compliance_check_node) # Renamed node

    # Define entry and connections
    workflow.set_entry_point("list_files")
    workflow.add_edge("list_files", "get_next_document")

    # Conditional edge after get_next_document
    def should_continue_processing(state: DocumentProcessingState) -> str:
        if state.get("current_document_path"): # If get_next_document provided a path
            return "process_current_document"
        else: # No more documents in queue
            return "finalize_processing"

    workflow.add_conditional_edges(
        "get_next_document",
        should_continue_processing,
        {
            "process_current_document": "load_classify_doc",
            "finalize_processing": "universal_compliance_check" # If no more docs, proceed to universal compliance
        }
    )

    # After loading and classifying, decide if data extraction is possible/needed
    def should_extract_data(state: DocumentProcessingState) -> str:
        current_doc_path = state.get("current_document_path")
        if not current_doc_path: return "skip_extraction_to_next" # Should not happen here
        
        filename = os.path.basename(current_doc_path)
        latest_entry_for_doc = None
        for entry in reversed(state.get("processed_documents", [])):
            if entry.get('filename') == filename:
                latest_entry_for_doc = entry
                break
        
        if latest_entry_for_doc and latest_entry_for_doc.get("status") not in ["error_loading_document", "error_load_classify"] and latest_entry_for_doc.get("extracted_text_content"):
            return "proceed_to_extraction"
        else:
            logger.info(f"[Workflow] Skipping data extraction for {filename} due to prior error or no text content.")
            return "skip_extraction_to_next" # Go to next document processing cycle

    workflow.add_conditional_edges(
        "load_classify_doc",
        should_extract_data,
        {
            "proceed_to_extraction": "extract_data",
            "skip_extraction_to_next": "get_next_document"
        }
    )
    
    # After data extraction, always go to get_next_document to loop or finish
    workflow.add_edge("extract_data", "get_next_document")

    # After universal compliance check, the workflow ends.
    workflow.add_edge("universal_compliance_check", END)

    logger.info("[Workflow] Document processing graph compiled with universal compliance check stage.")
    return workflow

# Example of how to run (for testing, typically called from main.py)
async def example_run():
    app = create_document_processing_graph()
    initial_state: DocumentProcessingState = {
        "docs_dir": "./docs", # Example directory
        "initial_document_paths": [], 
        "document_queue": [],      
        "current_document_path": None,
        "loaded_document_data": None,
        "processed_documents": [],
        "error_messages": [],
        "consolidated_rules_content": "1. Test rule 1.\n2. Test rule 2 involving Document A and Document B.", # Example rules
        "all_compliance_findings": []
    }
    final_state = await app.ainvoke(initial_state)
    
    print("\n--- Final Workflow State ---")
    # Avoid printing very large data directly like extracted_text_content
    for key, value in final_state.items():
        if key == 'processed_documents':
            print(f"\n{key}:")
            for doc_summary in value:
                summary_copy = doc_summary.copy()
                summary_copy.pop('extracted_text_content', None) # Remove large field for printing
                summary_copy.pop('raw_content', None) # Remove large field for printing
                print(f"  - {summary_copy}")
        elif key == 'all_compliance_findings':
            print(f"\n{key}:")
            for finding in value:
                print(f"  - {finding}")
        else:
            print(f"{key}: {value}")

    if final_state.get('error_messages'):
        print("\nOverall Errors during workflow:")
        for err in final_state['error_messages']:
            print(f"  - {err}")
    return final_state

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_run())
