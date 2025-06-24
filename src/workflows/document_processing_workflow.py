# src/workflows/document_processing_workflow.py
import os
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from langgraph.graph import StateGraph, END

from src.utils.document_loader import DocumentLoader 
from src.agents.structured_data_extractor_agent import StructuredDataExtractorAgent
from src.agents.universal_compliance_agent import UniversalComplianceAgent 
from src.utils.cache_manager import DocumentCacheManager
from src.utils.compliance_cache import ComplianceCacheManager

logger = logging.getLogger(__name__)

# --- State Definition (Revised for Parallel Processing) ---
from typing_extensions import TypedDict

class DocumentProcessingState(TypedDict):
    docs_dir: str
    consolidated_rules_content: Optional[str]
    initial_document_paths: List[str]
    # Removed old sequential fields: document_queue, current_document_path, loaded_document_data, rule_queue, current_rule
    processed_documents: List[Dict[str, Any]]
    error_messages: List[str]
    batch_rules_text: Optional[str]  # New field for batch rule processing
    aggregated_compliance_findings: List[Dict[str, Any]]
    cache_stats: Optional[Dict[str, Any]]  # New field for cache statistics

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

# --- Cache-Aware Helper for Document Processing ---
async def _process_single_document_with_cache(doc_path: str, cache_manager: DocumentCacheManager) -> Dict[str, Any]:
    """Helper to load, classify, and extract data from a single document with caching."""
    doc_filename = os.path.basename(doc_path)
    docs_dir = os.path.dirname(doc_path)
    logger.info(f"[Parallel Worker] Starting cache-aware processing for: {doc_filename}")

    doc_data_for_state = {
        "filename": doc_filename,
        "path": doc_path,
        "status": "initialization",
        "doc_type": None,
        "extracted_text_content": None, # Kept for potential debugging but can be removed from final state
        "extracted_tables_html": [],
        "extracted_data": None,
        "error_message": None,
        "cached": False,
        "processing_time": 0.0
    }

    processing_start_time = time.time()

    try:
        # Check cache first
        if cache_manager.is_cached_and_valid(doc_path):
            cached_data = cache_manager.get_cached_data(doc_path)
            if cached_data:
                # Use cached data
                doc_data_for_state.update(cached_data)
                doc_data_for_state["cached"] = True
                doc_data_for_state["processing_time"] = time.time() - processing_start_time
                logger.info(f"[Parallel Worker] Used cached data for {doc_filename}")
                return doc_data_for_state

        # If not cached or cache invalid, process normally
        loader = DocumentLoader(docs_dir)

        # --- Logic from former load_classify_document_node ---
        loaded_info = await loader.load_document(doc_filename)

        if not loaded_info or not loaded_info.get("text"):
            doc_data_for_state["status"] = "error_loading_document"
            doc_data_for_state["error_message"] = loaded_info.get("error_message", "Failed to load or extract text.")
            logger.error(f"[Parallel Worker] Failed to load/extract text from {doc_filename}. Error: {doc_data_for_state['error_message']}")
            doc_data_for_state["processing_time"] = time.time() - processing_start_time
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
            doc_data_for_state["processing_time"] = time.time() - processing_start_time
            # Cache even failed extractions to avoid re-processing
            cache_manager.cache_document_data(doc_path, doc_data_for_state, doc_data_for_state["processing_time"])
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
        doc_data_for_state["processing_time"] = time.time() - processing_start_time
        
        # Cache the processed data
        cache_manager.cache_document_data(doc_path, doc_data_for_state, doc_data_for_state["processing_time"])
        
        logger.info(f"[Parallel Worker] Structured data extracted for {doc_filename} in {doc_data_for_state['processing_time']:.2f}s")

    except FileNotFoundError as e:
        logger.error(f"[Parallel Worker] File not found for {doc_filename}: {e}", exc_info=True)
        doc_data_for_state["status"] = "error_file_not_found"
        doc_data_for_state["error_message"] = str(e)
        doc_data_for_state["processing_time"] = time.time() - processing_start_time
    except Exception as e:
        logger.error(f"[Parallel Worker] Unhandled error processing {doc_filename}: {e}", exc_info=True)
        doc_data_for_state["status"] = "error_processing"
        doc_data_for_state["error_message"] = str(e)
        doc_data_for_state["processing_time"] = time.time() - processing_start_time

    return doc_data_for_state



# --- Node Functions (Revised for Parallel Processing with Caching) ---

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
            "batch_rules_text": None,
            "aggregated_compliance_findings": [],
            "cache_stats": {}
        }
    except Exception as e:
        logger.error(f"[Workflow] Error listing documents in {docs_dir}: {e}", exc_info=True)
        return {
            "initial_document_paths": [],
            "processed_documents": [],
            "error_messages": [f"Failed to list documents: {str(e)}"],
            "batch_rules_text": None,
            "aggregated_compliance_findings": [],
            "cache_stats": {}
        }

async def process_all_documents_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Loads, classifies, and extracts data for all documents in parallel with caching."""
    # Get list of document files
    initial_paths = state.get("initial_document_paths", [])
    if not initial_paths:
        logger.warning("[Workflow] No document paths found to process.")
        return {"processed_documents": [], "error_messages": []}
    
    logger.info(f"[Workflow] Starting cached parallel processing for {len(initial_paths)} documents.")
    
    # Initialize cache manager
    # Use global cache manager if available (for Streamlit session consistency)
    from src.utils.cache_manager import get_global_cache_manager
    global_cache = get_global_cache_manager()
    cache_manager = global_cache if global_cache else DocumentCacheManager()
    
    # Use cache-aware processing
    processing_tasks = [_process_single_document_with_cache(path, cache_manager) for path in initial_paths]
    
    # Use return_exceptions=True to handle errors gracefully within each task
    processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

    final_processed_docs = []
    error_messages = list(state.get("error_messages", []))
    
    # Collect cache statistics
    cache_hits = 0
    cache_misses = 0
    total_time_saved = 0.0
    
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
            
            # Track cache statistics
            if result.get("cached", False):
                cache_hits += 1
                total_time_saved += result.get("processing_time", 0.0)
            else:
                cache_misses += 1
    
    succeeded_count = len([doc for doc in final_processed_docs if not doc.get('error_message')])
    
    # Get cache manager statistics
    cache_stats = cache_manager.get_cache_stats()
    cache_stats.update({
        "session_cache_hits": cache_hits,
        "session_cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / len(initial_paths) * 100 if initial_paths else 0,
        "estimated_time_saved": total_time_saved
    })
    
    logger.info(f"[Workflow] Cached parallel processing complete. Succeeded: {succeeded_count}. Failed: {len(initial_paths) - succeeded_count}.")
    logger.info(f"[Workflow] Cache performance: {cache_hits} hits, {cache_misses} misses ({cache_stats['cache_hit_rate']:.1f}% hit rate)")
    if total_time_saved > 0:
        logger.info(f"[Workflow] ⚡ Time saved by caching: {total_time_saved:.2f}s")
    
    return {
        "processed_documents": final_processed_docs,
        "error_messages": error_messages,
        "cache_stats": cache_stats
    }

async def initialize_rule_processing_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Parses rules and prepares for batch compliance evaluation."""
    rules_content = state.get("consolidated_rules_content")
    if not rules_content:
        logger.info("[Workflow] No rules content found. Skipping rule processing.")
        return {"aggregated_compliance_findings": state.get("aggregated_compliance_findings", [])}

    parsed_rules = _parse_rules(rules_content)
    if not parsed_rules:
        logger.info("[Workflow] No rules were parsed from the content. Skipping rule processing.")
        return {"aggregated_compliance_findings": state.get("aggregated_compliance_findings", [])}
        
    logger.info(f"[Workflow] Preparing batch compliance evaluation for {len(parsed_rules)} rules.")
    
    # Format rules for batch processing
    rules_text = "\n".join([f"{rule_id}. {rule_text}" for rule_id, rule_text in parsed_rules])
    
    return {
        "batch_rules_text": rules_text,
        "aggregated_compliance_findings": [] # Reset for this new round of rule evaluation
    }

async def evaluate_all_rules_batch_node(state: DocumentProcessingState) -> Dict[str, Any]:
    """Evaluates ALL rules in a single batch using the UniversalComplianceAgent with caching."""
    batch_rules_text = state.get("batch_rules_text")
    processed_documents_state = state.get("processed_documents", [])

    if not batch_rules_text:
        logger.info("[Workflow] No rules to evaluate in batch.")
        return {"aggregated_compliance_findings": []}

    # Prepare documents data for the agent - only include relevant fields and suitable documents
    documents_for_batch_evaluation = []
    for doc_data in processed_documents_state:
        if doc_data.get("filename") and doc_data.get("doc_type") and doc_data.get("status") not in ["error_loading_document", "error_file_not_found", "error_load_classify"]:
            # Include documents that are at least classified, even if data extraction failed or yielded no data.
            data_for_agent = {
                "filename": doc_data["filename"],
                "doc_type": doc_data["doc_type"],
                "extracted_data": doc_data.get("extracted_data") if doc_data.get("extracted_data") is not None else {}
            }
            documents_for_batch_evaluation.append(data_for_agent)

    if not documents_for_batch_evaluation and processed_documents_state:
        logger.warning(f"[Workflow] No suitable documents from processed_documents for batch rule evaluation, though some documents were processed. Rules will be evaluated with an empty document list.")
    elif not processed_documents_state:
        logger.info(f"[Workflow] No documents were processed at all. Rules will be evaluated with an empty document list.")

    try:
        # Initialize compliance cache manager
        compliance_cache = ComplianceCacheManager()
        
        # Check compliance cache first
        cache_start_time = time.time()
        logger.info(f"[Workflow] Checking compliance cache for {len(documents_for_batch_evaluation)} documents...")
        
        if compliance_cache.is_cached_and_valid(documents_for_batch_evaluation, batch_rules_text):
            cached_compliance_results = compliance_cache.get_cached_results(documents_for_batch_evaluation, batch_rules_text)
            logger.info(f"[Workflow] Cache result type: {type(cached_compliance_results)}")
            
            if cached_compliance_results and isinstance(cached_compliance_results, dict):
                logger.info(f"[Workflow] 📋 Using cached compliance results (saved significant processing time)")
                processed_findings = cached_compliance_results.get("aggregated_compliance_findings", [])
                
                # Ensure consistent format for LangGraph state
                for finding in processed_findings:
                    if 'rule_checked' not in finding and 'rule_text' in finding:
                        finding['rule_checked'] = finding['rule_text']
                
                # Update cache stats if available
                current_cache_stats = state.get("cache_stats") or {}
                current_cache_stats.update({
                    "compliance_cache_hit": True,
                    "compliance_cache_time_saved": time.time() - cache_start_time
                })
                
                return {
                    "aggregated_compliance_findings": processed_findings,
                    "cache_stats": current_cache_stats
                }
        
        # If not cached, perform compliance evaluation
        from src.agents.universal_compliance_agent import UniversalComplianceAgent
        agent = UniversalComplianceAgent()
        
        logger.info(f"[Workflow] Starting batch compliance evaluation for {len(documents_for_batch_evaluation)} documents")
        
        compliance_start_time = time.time()
        
        # Use the fast batch method instead of single rule evaluation
        raw_findings = await agent.check_all_compliance(
            all_documents_data=documents_for_batch_evaluation,
            consolidated_rules=batch_rules_text
        )
        
        compliance_processing_time = time.time() - compliance_start_time
        
        # Process findings to match expected format
        processed_findings = []
        if raw_findings is None:
            logger.warning("[Workflow] Raw findings is None, using empty list")
            raw_findings = []
            
        for raw_finding in raw_findings:
            # Skip None findings
            if raw_finding is None:
                logger.warning("[Workflow] Skipping None raw_finding")
                continue
                
            # Ensure raw_finding is a dictionary
            if not isinstance(raw_finding, dict):
                logger.warning(f"[Workflow] Skipping non-dict raw_finding: {type(raw_finding)}")
                continue
                
            # Ensure consistent format for LangGraph state
            processed_finding = {
                'rule_id': raw_finding.get('rule_id', 'N/A'),
                'rule_checked': raw_finding.get('rule_checked', raw_finding.get('rule_text', 'N/A')),
                'status': raw_finding.get('status', 'error'),
                'details': raw_finding.get('details', 'No details provided.'),
                'involved_documents': raw_finding.get('involved_documents', [])
            }
            processed_findings.append(processed_finding)
        
        # Cache the compliance results
        compliance_results_to_cache = {
            "processed_documents": documents_for_batch_evaluation,
            "aggregated_compliance_findings": processed_findings,
            "error_messages": []
        }
        
        compliance_cache.cache_compliance_results(
            documents_for_batch_evaluation, 
            batch_rules_text, 
            compliance_results_to_cache, 
            compliance_processing_time
        )
        
        # Update cache stats
        current_cache_stats = state.get("cache_stats") or {}
        current_cache_stats.update({
            "compliance_cache_hit": False,
            "compliance_processing_time": compliance_processing_time
        })
        
        logger.info(f"[Workflow] Batch compliance evaluation completed in {compliance_processing_time:.2f}s. {len(processed_findings)} findings generated.")
        return {
            "aggregated_compliance_findings": processed_findings,
            "cache_stats": current_cache_stats
        }
        
    except Exception as e:
        logger.error(f"[Workflow] Error during batch compliance evaluation: {e}", exc_info=True)
        error_finding = {
            "rule_id": "batch_error",
            "rule_checked": "Batch compliance evaluation",
            "status": "error",
            "details": f"Batch compliance evaluation failed: {str(e)}",
            "involved_documents": [doc.get('filename', 'unknown') for doc in documents_for_batch_evaluation]
        }
        return {
            "aggregated_compliance_findings": [error_finding],
            "error_messages": state.get("error_messages", []) + [f"Batch compliance evaluation error: {str(e)}"]
        }

# --- Conditional Edges (Revised for Batch Processing) ---

def decide_after_processing(state: DocumentProcessingState) -> str:
    """Decides whether to proceed to rule evaluation or end the workflow."""
    # After parallel processing, check if there are rules to evaluate.
    rules_content = state.get("consolidated_rules_content")
    if rules_content and _parse_rules(rules_content):
        logger.info("[Workflow Decision] Document processing complete. Proceeding to batch rule evaluation.")
        return "initialize_rule_evaluation"
    else:
        logger.info("[Workflow Decision] Document processing complete. No rules to evaluate. Ending workflow.")
        return "end_workflow"

# --- Graph Assembly (Revised for Batch Processing) ---
def create_document_processing_graph():
    from langgraph.graph import START
    
    workflow = StateGraph(DocumentProcessingState)

    # Document processing nodes (parallel)
    workflow.add_node("list_files", list_files_node)
    workflow.add_node("process_all_documents", process_all_documents_node)

    # Rule processing nodes (now batch instead of sequential)
    workflow.add_node("initialize_rule_evaluation", initialize_rule_processing_node)
    workflow.add_node("evaluate_all_rules_batch", evaluate_all_rules_batch_node)

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

    # Rule processing flow (simplified to batch)
    workflow.add_edge("initialize_rule_evaluation", "evaluate_all_rules_batch")
    workflow.add_edge("evaluate_all_rules_batch", END)
    
    logger.info("Document processing graph with PARALLEL document loading and BATCH rule evaluation created.")
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
        "batch_rules_text": None,
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


