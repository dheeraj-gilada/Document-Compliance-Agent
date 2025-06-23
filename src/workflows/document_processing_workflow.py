# src/workflows/document_processing_workflow.py
import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from langgraph.graph import StateGraph, END

from src.utils.document_loader import DocumentLoader 
from src.agents.structured_data_extractor_agent import StructuredDataExtractorAgent
from src.agents.universal_compliance_agent import UniversalComplianceAgent 

logger = logging.getLogger(__name__)

# --- State Definition (Revised for Parallel Processing) ---
from typing_extensions import TypedDict

class DocumentProcessingState(TypedDict):
    docs_dir: str
    consolidated_rules_content: Optional[str]
    initial_document_paths: List[str]
    # document_queue, current_document_path, and loaded_document_data are removed for parallel model
    processed_documents: List[Dict[str, Any]]
    error_messages: List[str]
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

# --- New Helper for Parallel Processing ---
async def _process_single_document(doc_path: str) -> Dict[str, Any]:
    """Helper to load, classify, and extract data from a single document."""
    doc_filename = os.path.basename(doc_path)
    docs_dir = os.path.dirname(doc_path)
    logger.info(f"[Parallel Worker] Starting processing for: {doc_filename}")

    loader = DocumentLoader(docs_dir)

    doc_data_for_state = {
        "filename": doc_filename,
        "path": doc_path,
        "status": "initialization",
        "doc_type": None,
        "extracted_text_content": None, # Kept for potential debugging but can be removed from final state
        "extracted_tables_html": [],
        "extracted_data": None,
        "error_message": None
    }

    try:
        # --- Logic from former load_classify_document_node ---
        loaded_info = await loader.load_document(doc_filename)

        if not loaded_info or not loaded_info.get("text"):
            doc_data_for_state["status"] = "error_loading_document"
            doc_data_for_state["error_message"] = loaded_info.get("error_message", "Failed to load or extract text.")
            logger.error(f"[Parallel Worker] Failed to load/extract text from {doc_filename}. Error: {doc_data_for_state['error_message']}")
            return doc_data_for_state # Stop processing this doc

        doc_data_for_state["doc_type"] = loaded_info.get("doc_type", "unknown")
        doc_data_for_state["extracted_text_content"] = loaded_info.get("text")
        doc_data_for_state["extracted_tables_html"] = loaded_info.get("tables_html", [])
        doc_data_for_state["status"] = "classified"
        logger.info(f"[Parallel Worker] Document {doc_filename} classified. Type: {doc_data_for_state['doc_type']}.")

        # --- Logic from former extract_document_data_node ---
        text_content = doc_data_for_state.get("extracted_text_content", "")
        tables_html_list = doc_data_for_state.get("extracted_tables_html", [])
        combined_tables_html = "\n\n".join(tables_html_list)

        if not text_content and not combined_tables_html:
            doc_data_for_state["status"] = "skipped_extraction_no_content"
            logger.warning(f"[Parallel Worker] No text/tables for {doc_filename}. Skipping data extraction.")
            return doc_data_for_state

        data_extractor_agent = StructuredDataExtractorAgent()
        logger.info(f"[Parallel Worker] Extracting structured data from: {doc_filename}")
        extracted_data = await data_extractor_agent.extract_structured_data(
            doc_type=doc_data_for_state["doc_type"],
            text_content=text_content,
            combined_tables_html=combined_tables_html,
            filename=doc_filename
        )
        doc_data_for_state["extracted_data"] = extracted_data
        doc_data_for_state["status"] = "data_extracted"
        logger.info(f"[Parallel Worker] Structured data extracted for {doc_filename}.")

    except FileNotFoundError as e:
        logger.error(f"[Parallel Worker] File not found for {doc_filename}: {e}", exc_info=True)
        doc_data_for_state["status"] = "error_file_not_found"
        doc_data_for_state["error_message"] = str(e)
    except Exception as e:
        logger.error(f"[Parallel Worker] Unhandled error processing {doc_filename}: {e}", exc_info=True)
        doc_data_for_state["status"] = "error_processing"
        doc_data_for_state["error_message"] = str(e)

    return doc_data_for_state

# --- Node Functions (Revised for Parallel Processing) ---

async def list_files_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Lists all supported document files and initializes the processing state."""
    docs_dir = state["docs_dir"]
    loader = DocumentLoader(docs_dir)
    try:
        document_filenames = loader.list_documents()
        initial_paths = [os.path.join(docs_dir, fname) for fname in document_filenames]
        logger.info(f"[Workflow] Found {len(initial_paths)} documents in {docs_dir}.")
        
        # Initialize the state for a new run
        return {
            "initial_document_paths": initial_paths,
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
            "processed_documents": [],
            "error_messages": [f"Failed to list documents: {str(e)}"],
            "rule_queue": [],
            "current_rule": None,
            "aggregated_compliance_findings": []
        }

async def process_all_documents_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Loads, classifies, and extracts data for all documents in parallel."""
    initial_paths = state.get("initial_document_paths", [])
    if not initial_paths:
        logger.info("[Workflow] No document paths found to process.")
        return {"processed_documents": [], "error_messages": state.get("error_messages", [])}

    logger.info(f"[Workflow] Starting parallel processing for {len(initial_paths)} documents.")
    
    processing_tasks = [_process_single_document(path) for path in initial_paths]
    
    # Use return_exceptions=True to handle errors gracefully within each task
    processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

    final_processed_docs = []
    error_messages = list(state.get("error_messages", []))
    
    for i, result in enumerate(processed_results):
        if isinstance(result, Exception):
            # This catches errors within asyncio.gather or unhandled exceptions in the helper
            error_msg = f"Unhandled exception processing document {os.path.basename(initial_paths[i])}: {result}"
            logger.error(f"[Workflow] {error_msg}", exc_info=result)
            error_messages.append(error_msg)
        elif result: # Ensure result is not None
            final_processed_docs.append(result)
            if result.get("error_message"):
                error_msg = f"Error processing {result['filename']}: {result['error_message']}"
                error_messages.append(error_msg)
    
    succeeded_count = len([doc for doc in final_processed_docs if not doc.get('error_message')])
    logger.info(f"[Workflow] Parallel document processing complete. Succeeded: {succeeded_count}. Failed: {len(initial_paths) - succeeded_count}.")
    
    return {
        "processed_documents": final_processed_docs,
        "error_messages": error_messages
    }

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

# --- Conditional Edges (Revised for Parallel Processing) ---

def decide_after_processing(state: DocumentProcessingState) -> str:
    """Decides whether to proceed to rule evaluation or end the workflow."""
    # After parallel processing, check if there are rules to evaluate.
    rules_content = state.get("consolidated_rules_content")
    if rules_content and _parse_rules(rules_content):
        logger.info("[Workflow Decision] Document processing complete. Proceeding to rule evaluation.")
        return "initialize_rule_evaluation"
    else:
        logger.info("[Workflow Decision] Document processing complete. No rules to evaluate. Ending workflow.")
        return "end_workflow"

def should_continue_rule_evaluation(state: DocumentProcessingState) -> str:
    """Determines if there is a current rule to evaluate or if rule processing is finished."""
    if state.get("current_rule"):
        logger.info(f"[Workflow Decision] Current rule '{state['current_rule'][0]}' set. Proceeding to evaluate.")
        return "evaluate_current_rule"
    else:
        logger.info("[Workflow Decision] No current rule. Finishing rule evaluation.")
        return "finish_rule_evaluation"

# --- Graph Assembly (Revised for Parallel Processing) ---
def create_document_processing_graph():
    from langgraph.graph import START
    
    workflow = StateGraph(DocumentProcessingState)

    # Document processing nodes (now parallel)
    workflow.add_node("list_files", list_files_node)
    workflow.add_node("process_all_documents", process_all_documents_node)

    # Rule processing nodes (remain sequential per rule)
    workflow.add_node("initialize_rule_evaluation", initialize_rule_processing_node)
    workflow.add_node("get_next_rule", get_next_rule_node)
    workflow.add_node("evaluate_single_rule", evaluate_single_rule_node)

    # Use START to define the entry point
    workflow.add_edge(START, "list_files")

    # Document processing flow
    workflow.add_edge("list_files", "process_all_documents")
    
    # Conditional branching after all documents are processed
    workflow.add_conditional_edges(
        "process_all_documents",
        decide_after_processing, 
        {
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
    # Loop back from evaluation to get the next rule
    workflow.add_edge("evaluate_single_rule", "get_next_rule")
    
    logger.info("Document processing graph with PARALLEL document loading created.")
    return workflow.compile()

# Example of how to run (for testing, typically called from main.py)
async def example_run():
    app = create_document_processing_graph()
    # Initial state now simplified for the parallel model
    initial_state: DocumentProcessingState = {
        "docs_dir": "./docs", 
        "initial_document_paths": [], 
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
                summary_copy.pop('extracted_tables_html', None) # Also remove tables for cleaner summary
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
