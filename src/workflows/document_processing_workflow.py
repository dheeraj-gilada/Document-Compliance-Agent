# src/workflows/document_processing_workflow.py
import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from langgraph.graph import StateGraph, END

from src.utils.document_loader import DocumentLoader 
from src.utils.extractor import DocumentExtractor 
from src.agents.document_classifier_agent import DocumentTypeClassifierAgent 
from src.agents.structured_data_extractor_agent import StructuredDataExtractorAgent
from src.agents.universal_compliance_agent import UniversalComplianceAgent 

logger = logging.getLogger(__name__)

# --- State Definition (Revised) ---
class DocumentProcessingState(dict):
    docs_dir: str
    consolidated_rules_content: Optional[str]
    initial_document_paths: List[str]
    document_queue: List[str]
    current_document_path: Optional[str]
    loaded_document_data: Optional[Dict[str, Any]]
    processed_documents: List[Dict[str, Any]]
    error_messages: List[str]
    # New state fields for rule-level evaluation
    rule_queue: List[Tuple[str, str]] 
    current_rule: Optional[Tuple[str, str]]
    aggregated_compliance_findings: List[Dict[str, Any]] 

# --- Utility Function Placeholder ---
def _parse_rules(rules_content: str) -> List[Tuple[str, str]]:
    """Parses the consolidated rules string into a list of (rule_id, rule_text) tuples."""
    parsed_rules = []
    if not rules_content:
        return parsed_rules
    
    rule_pattern = re.compile(r"^\s*([\w\.]+)\.\s+(.+)$", re.MULTILINE)
    matches = rule_pattern.findall(rules_content)
    
    for match in matches:
        rule_id = match[0].strip()
        rule_text = match[1].strip()
        parsed_rules.append((rule_id, rule_text))
        
    if not parsed_rules and rules_content.strip(): 
        logger.warning("Rule parsing with regex found no rules, but content exists. Treating as single rule or check format.")
        # Potentially treat the whole block as one rule or log an error
        # For now, if regex fails but content exists, we might need a more robust parser or clearer rule format.

    logger.info(f"Parsed {len(parsed_rules)} rules from content.")
    return parsed_rules

# --- Node Functions (Revised) ---

async def list_files_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Lists all supported document files and initializes the processing queue."""
    docs_dir = state["docs_dir"]
    loader = DocumentLoader(docs_dir)
    try:
        document_filenames = loader.list_documents()
        initial_paths = [os.path.join(docs_dir, fname) for fname in document_filenames]
        logger.info(f"[Workflow] Found {len(initial_paths)} documents in {docs_dir}.")
        state['consolidated_rules_content'] = state.get('consolidated_rules_content')
        state['rule_queue'] = []
        state['current_rule'] = None
        state['aggregated_compliance_findings'] = []
        return {
            "initial_document_paths": initial_paths,
            "document_queue": initial_paths.copy(), 
            "processed_documents": [],
            "error_messages": [],
            "rule_queue": [],
            "current_rule": None,
            "aggregated_compliance_findings": []
        }
    except Exception as e:
        logger.error(f"[Workflow] Error listing documents in {docs_dir}: {e}", exc_info=True)
        return {
            "initial_document_paths": [],
            "document_queue": [],
            "processed_documents": [],
            "error_messages": [f"Failed to list documents: {str(e)}"],
            "rule_queue": [],
            "current_rule": None,
            "aggregated_compliance_findings": []
        }

async def get_next_document_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Gets the next document from the queue."""
    doc_queue = list(state.get("document_queue", [])) 
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
        "extracted_tables_html": [], 
        "extracted_data": None, 
        "error_message": None
    }

    try:
        loaded_info = await loader.load_document(doc_filename) 

        if not loaded_info or not loaded_info.get("text"):
            logger.error(f"[Workflow] Failed to load or extract text from document: {doc_filename}. Skipping.")
            doc_data_for_state["status"] = "error_loading_document"
            doc_data_for_state["error_message"] = loaded_info.get("error_message", "Failed to load document content or extract text.")
        else:
            doc_data_for_state["doc_type"] = loaded_info.get("doc_type", "unknown")
            doc_data_for_state["extracted_text_content"] = loaded_info.get("text")
            doc_data_for_state["extracted_tables_html"] = loaded_info.get("tables_html", [])
            doc_data_for_state["status"] = "classified" 
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
    entry_index_in_list = -1 

    for entry_idx, entry in reversed(list(enumerate(processed_docs_list))):
        if entry.get('filename') == filename:
            doc_entry_to_update = entry
            entry_index_in_list = entry_idx
            break

    if not doc_entry_to_update or doc_entry_to_update.get("status") in ["error_loading_document", "error_file_not_found", "error_load_classify"]:
        logger.warning(f"[Workflow] Skipping data extraction for {filename} due to prior error ('{doc_entry_to_update.get('status') if doc_entry_to_update else 'No entry'}') or no entry.")
        return {} 

    doc_type = doc_entry_to_update.get("doc_type", "unknown")
    text_content = doc_entry_to_update.get("extracted_text_content", "")
    tables_html_list = doc_entry_to_update.get("extracted_tables_html", [])
    combined_tables_html = "\n\n".join(tables_html_list) 

    if not text_content and not combined_tables_html:
        logger.warning(f"[Workflow] No text content or tables for {filename} (type: {doc_type}). Skipping structured data extraction.")
        doc_entry_to_update["status"] = "skipped_extraction_no_content"
        if entry_index_in_list != -1: 
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
            current_errors.append(f"Data extraction error for {filename}: {str(e)}") 
        return {"processed_documents": processed_docs_list, "error_messages": current_errors}

async def initialize_rule_processing_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Parses rules and initializes the rule queue for evaluation."""
    rules_content = state.get("consolidated_rules_content")
    if not rules_content:
        logger.info("[Workflow] No rules content found. Skipping rule processing.")
        return {"rule_queue": [], "aggregated_compliance_findings": state.get("aggregated_compliance_findings", [])}

    parsed_rules = _parse_rules(rules_content)
    if not parsed_rules:
        logger.info("[Workflow] No rules were parsed from the content. Skipping rule processing.")
        return {"rule_queue": [], "aggregated_compliance_findings": state.get("aggregated_compliance_findings", [])}
        
    logger.info(f"[Workflow] Initializing rule processing with {len(parsed_rules)} rules.")
    return {
        "rule_queue": parsed_rules,
        "aggregated_compliance_findings": [], # Reset for this new round of rule evaluation
        "current_rule": None
    }

async def get_next_rule_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Gets the next rule from the queue to be processed."""
    rule_queue = list(state.get("rule_queue", []))
    if rule_queue:
        next_rule = rule_queue.pop(0)
        logger.info(f"[Workflow] Next rule to evaluate: ID '{next_rule[0]}'")
        return {"current_rule": next_rule, "rule_queue": rule_queue}
    else:
        logger.info("[Workflow] Rule queue empty. No more rules to evaluate.")
        return {"current_rule": None, "rule_queue": []}

async def evaluate_single_rule_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Evaluates the current rule using the UniversalComplianceAgent."""
    current_rule = state.get("current_rule")
    processed_documents_state = state.get("processed_documents", [])
    aggregated_findings = list(state.get("aggregated_compliance_findings", []))

    if not current_rule:
        logger.error("[Workflow] evaluate_single_rule_node called without a current_rule.")
        return {"error_messages": state.get("error_messages", []) + ["evaluate_single_rule_node called without current_rule"]}

    rule_id, rule_text = current_rule

    # Prepare documents data for the agent - only include relevant fields and suitable documents
    documents_for_rule_evaluation = []
    for doc_data in processed_documents_state:
        if doc_data.get("filename") and doc_data.get("doc_type") and doc_data.get("status") not in ["error_loading_document", "error_file_not_found", "error_load_classify"]:
            # Include documents that are at least classified, even if data extraction failed or yielded no data.
            # The agent's prompt for single rule evaluation is designed to handle this.
            data_for_agent = {
                "filename": doc_data["filename"],
                "doc_type": doc_data["doc_type"],
                "extracted_data": doc_data.get("extracted_data") if doc_data.get("extracted_data") is not None else {}
            }
            documents_for_rule_evaluation.append(data_for_agent)
        else:
            logger.debug(f"[Workflow] Document {doc_data.get('filename', 'N/A')} with status {doc_data.get('status', 'N/A')} will not be included in rule evaluation for rule {rule_id}.")

    if not documents_for_rule_evaluation and processed_documents_state:
        logger.warning(f"[Workflow] No suitable documents from processed_documents for rule {rule_id} evaluation, though some documents were processed. Rule will be evaluated with an empty document list.")
    elif not processed_documents_state:
        logger.info(f"[Workflow] No documents were processed at all. Rule {rule_id} will be evaluated with an empty document list.")

    try:
        agent = UniversalComplianceAgent() # Consider if API key needs to be passed if not globally set via env
        finding = await agent.evaluate_single_rule(
            rule_id=rule_id,
            rule_text=rule_text,
            all_documents_data=documents_for_rule_evaluation
        )
        aggregated_findings.append(finding)
        logger.info(f"[Workflow] Rule ID {rule_id} evaluated. Status: {finding.get('status')}")
        return {"aggregated_compliance_findings": aggregated_findings}
    except Exception as e:
        logger.error(f"[Workflow] Error evaluating rule ID {rule_id} with UniversalComplianceAgent: {e}", exc_info=True)
        error_finding = {
            "rule_id": rule_id,
            "rule_checked": rule_text,
            "status": "error",
            "details": f"Agent failed to evaluate rule: {str(e)}",
            "involved_documents": [doc.get('filename', 'unknown') for doc in documents_for_rule_evaluation]
        }
        aggregated_findings.append(error_finding)
        return {
            "aggregated_compliance_findings": aggregated_findings,
            "error_messages": state.get("error_messages", []) + [f"Error evaluating rule {rule_id}: {str(e)}"]
        }

# --- Conditional Edges (Revised) ---

def should_continue_processing(state: DocumentProcessingState) -> str:
    """Determines if there are more documents to process or if we should move to rule evaluation."""
    if state.get("current_document_path"):
        return "continue_processing_current_document"
    else: # Document queue is empty
        # Check if there are rules to process. _parse_rules will return empty list if no rules.
        if state.get("consolidated_rules_content") and _parse_rules(state["consolidated_rules_content"]):
            logger.info("[Workflow Decision] All documents processed. Proceeding to initialize rule evaluation.")
            return "initialize_rule_evaluation"
        else:
            logger.info("[Workflow Decision] All documents processed. No valid rules found or no rules content. Ending workflow.")
            return "end_workflow"

def should_continue_rule_evaluation(state: DocumentProcessingState) -> str:
    """Determines if there is a current rule to evaluate or if rule processing is finished."""
    if state.get("current_rule"):
        logger.info(f"[Workflow Decision] Current rule '{state['current_rule'][0]}' set. Proceeding to evaluate.")
        return "evaluate_current_rule"
    else:
        logger.info("[Workflow Decision] No current rule. Finishing rule evaluation.")
        return "finish_rule_evaluation"

# --- Graph Assembly (Revised) ---
def create_document_processing_graph():
    workflow = StateGraph(DocumentProcessingState)

    # Document processing loop nodes
    workflow.add_node("list_files", list_files_node)
    workflow.add_node("get_next_document", get_next_document_node)
    workflow.add_node("load_and_classify", load_classify_document_node)
    workflow.add_node("extract_data", extract_document_data_node)

    # New rule processing nodes
    workflow.add_node("initialize_rule_evaluation", initialize_rule_processing_node)
    workflow.add_node("get_next_rule", get_next_rule_node)
    workflow.add_node("evaluate_single_rule", evaluate_single_rule_node)

    workflow.set_entry_point("list_files")

    workflow.add_edge("list_files", "get_next_document")
    workflow.add_edge("load_and_classify", "extract_data")
    workflow.add_edge("extract_data", "get_next_document") # Loop back for next document

    # Conditional branching after getting the next document
    workflow.add_conditional_edges(
        "get_next_document",
        should_continue_processing, 
        {
            "continue_processing_current_document": "load_and_classify",
            "initialize_rule_evaluation": "initialize_rule_evaluation",
            "end_workflow": END
        }
    )

    # Rule processing flow
    workflow.add_edge("initialize_rule_evaluation", "get_next_rule")

    workflow.add_conditional_edges(
        "get_next_rule",
        should_continue_rule_evaluation,
        {
            "evaluate_current_rule": "evaluate_single_rule",
            "finish_rule_evaluation": END # All rules processed
        }
    )
    # Loop back from evaluation to get next rule
    workflow.add_edge("evaluate_single_rule", "get_next_rule")
    
    logger.info("Document processing graph with rule-level evaluation structure created.")
    return workflow

# Example of how to run (for testing, typically called from main.py)
async def example_run():
    app = create_document_processing_graph()
    initial_state: DocumentProcessingState = {
        "docs_dir": "./docs", 
        "initial_document_paths": [], 
        "document_queue": [],      
        "current_document_path": None,
        "loaded_document_data": None,
        "processed_documents": [],
        "error_messages": [],
        "consolidated_rules_content": "1. Test rule 1.\n2. Test rule 2 involving Document A and Document B.", 
        "rule_queue": [],
        "current_rule": None,
        "aggregated_compliance_findings": []
    }
    final_state = await app.ainvoke(initial_state)
    
    print("\n--- Final Workflow State ---")
    for key, value in final_state.items():
        if key == 'processed_documents':
            print(f"\n{key}:")
            for doc_summary in value:
                summary_copy = doc_summary.copy()
                summary_copy.pop('extracted_text_content', None) 
                summary_copy.pop('raw_content', None) 
                print(f"  - {summary_copy}")
        elif key == 'aggregated_compliance_findings':
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
