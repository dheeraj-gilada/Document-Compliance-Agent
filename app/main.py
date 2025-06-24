"""
Intelligent Document Compliance & Process Automation Agent

Main entry point for the application that orchestrates the document processing,
instruction parsing, compliance checking, and report generation workflow.
"""
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

# --- Path Correction ---
# Add the project root to the Python path to allow imports from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utility modules
from src.workflows.document_processing_workflow import create_document_processing_graph, DocumentProcessingState


def load_rules_from_file(filepath: str) -> Optional[str]:
    """
    Load compliance rules from a text file.
    
    Args:
        filepath: Path to the rules file
        
    Returns:
        String content of the rules file, or None if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else None
    except FileNotFoundError:
        logging.error(f"Rules file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading rules file {filepath}: {e}")
        return None

def ensure_dir_exists(filepath: str):
    """Create directory for filepath if it doesn't exist"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

async def run_document_processing(docs_dir_to_process: str, consolidated_rules_input: Optional[str] = None) -> Dict[str, Any]:
    """
    Invokes the cached LangGraph workflow for complete document processing pipeline.
    
    This function runs the full pipeline (extraction + compliance) with caching enabled.
    Uses the enhanced LangGraph workflow with document and compliance caching.
    
    Args:
        docs_dir_to_process: Directory containing documents to process
        consolidated_rules_input: Optional compliance rules string
        
    Returns:
        Dictionary containing processed documents, compliance findings, cache stats, and errors
    """
    logger.info(f"Starting cached document processing workflow for directory: {docs_dir_to_process}")
    if consolidated_rules_input:
        logger.info("Consolidated compliance rules provided and will be used in the workflow.")
    else:
        logger.info("No consolidated compliance rules provided; universal compliance checks might be limited or skipped.")
    
    app = create_document_processing_graph()
    
    initial_state: DocumentProcessingState = {
        "docs_dir": docs_dir_to_process,
        "consolidated_rules_content": consolidated_rules_input,
        "initial_document_paths": [],
        "processed_documents": [],
        "error_messages": [],
        "batch_rules_text": None,
        "aggregated_compliance_findings": [],
        "cache_stats": {}
    }

    final_graph_state = None
    try:
        logger.info(f"Invoking cached document processing workflow with initial state")
        # Use ainvoke to get the final state directly
        final_graph_state = await app.ainvoke(initial_state, config={"recursion_limit": 100})
        
        logger.info(f"DEBUG run_document_processing: ainvoke completed. Raw output type: {type(final_graph_state)}")
        # Some LangServe setups might wrap the output in a list
        if isinstance(final_graph_state, list) and final_graph_state and isinstance(final_graph_state[0], dict):
            final_graph_state = final_graph_state[0]
            logger.info("Extracted final state dictionary from the first element of a list after ainvoke.")
        elif not isinstance(final_graph_state, dict):
            logger.error(f"ainvoke did not return a dictionary or a list containing a dictionary. Type: {type(final_graph_state)}, Content: {str(final_graph_state)[:500]}...")
            # Return a default error state if the output is not as expected
            return {"processed_documents": [], "aggregated_compliance_findings": [], "error_messages": [f"Workflow returned unexpected data type: {type(final_graph_state)}"]}

        # Log cache performance if available
        cache_stats = final_graph_state.get("cache_stats", {})
        if cache_stats:
            doc_cache_hit_rate = cache_stats.get("cache_hit_rate", 0)
            time_saved = cache_stats.get("estimated_time_saved", 0)
            compliance_cache_hit = cache_stats.get("compliance_cache_hit", False)
            
            logger.info(f"📊 Cache Performance Summary:")
            logger.info(f"   Document cache hit rate: {doc_cache_hit_rate:.1f}%")
            if time_saved > 0:
                logger.info(f"   Time saved by document caching: {time_saved:.2f}s")
            if compliance_cache_hit:
                logger.info(f"   Compliance cache: HIT (significant time saved)")
            elif 'compliance_processing_time' in cache_stats:
                logger.info(f"   Compliance processing: {cache_stats['compliance_processing_time']:.2f}s")

        logger.info("Cached document processing workflow completed.")
        return final_graph_state

    except Exception as e:
        logger.error(f"Exception during document processing workflow execution (ainvoke): {e}", exc_info=True)
        return {"processed_documents": [], "aggregated_compliance_findings": [], "error_messages": [f"Workflow execution failed: {str(e)}"]}


async def extract_document_data(docs_dir_to_process: str) -> Dict[str, Any]:
    """
    Extracts data from documents using the cached LangGraph workflow.
    
    This is a wrapper around the LangGraph workflow for document extraction only.
    Uses caching to avoid reprocessing unchanged documents.
    
    Args:
        docs_dir_to_process: Directory containing documents to process
        
    Returns:
        Dictionary containing processed documents data and cache stats
    """
    logger.info(f"Starting cached document extraction using LangGraph workflow: {docs_dir_to_process}")
    
    # Use LangGraph workflow without rules (extraction only)
    result = await run_document_processing(docs_dir_to_process, consolidated_rules_input=None)
    
    # Return only the document processing results (no compliance findings)
    return {
        "processed_documents": result.get("processed_documents", []),
        "error_messages": result.get("error_messages", []),
        "cache_stats": result.get("cache_stats", {})
    }


async def run_compliance_check(extracted_documents: List[Dict[str, Any]], rules_content: str) -> Dict[str, Any]:
    """
    Runs compliance checks using the cached LangGraph workflow.
    
    This function directly invokes the compliance portion of the workflow with caching.
    Bypasses document processing by injecting pre-extracted data.
    
    Args:
        extracted_documents: List of previously extracted document data
        rules_content: String containing the compliance rules
        
    Returns:
        Dictionary containing compliance findings and cache performance stats
    """
    logger.info(f"Starting cached compliance checks using LangGraph workflow for {len(extracted_documents)} documents")
    
    try:
        # Import workflow functions for direct compliance checking
        from src.workflows.document_processing_workflow import initialize_rule_processing_node, evaluate_all_rules_batch_node
        
        # Create initial state with pre-processed documents and rules
        initial_state: DocumentProcessingState = {
            "docs_dir": "",  # Not needed since we're injecting processed documents
            "consolidated_rules_content": rules_content,
            "initial_document_paths": [doc.get("filename", f"doc_{i}") for i, doc in enumerate(extracted_documents)],
            "processed_documents": extracted_documents,  # Inject pre-processed documents
            "error_messages": [],
            "batch_rules_text": None,
            "aggregated_compliance_findings": [],
            "cache_stats": {}
        }
        
        # Step 1: Initialize rule processing
        rule_init_result = await initialize_rule_processing_node(initial_state)
        initial_state.update(rule_init_result)
        
        # Step 2: Run cached batch compliance evaluation
        final_result = await evaluate_all_rules_batch_node(initial_state)
        initial_state.update(final_result)
        
        # Process findings for Streamlit compatibility - add is_compliant field
        findings = initial_state.get("aggregated_compliance_findings", [])
        for finding in findings:
            # Determine the boolean 'is_compliant' status for Streamlit compatibility
            is_compliant_bool = finding.get('status', '').lower() in ['pass', 'compliant']
            finding['is_compliant'] = is_compliant_bool
            
            # Ensure 'reason' field exists for Streamlit compatibility
            if 'reason' not in finding:
                finding['reason'] = finding.get('details', 'No details provided.')

        # Log cache performance
        cache_stats = initial_state.get("cache_stats") or {}
        if cache_stats.get("compliance_cache_hit"):
            logger.info("✅ Used cached compliance results")
        elif cache_stats.get("compliance_processing_time"):
            logger.info(f"🔄 Compliance processed in {cache_stats['compliance_processing_time']:.2f}s")
        
        return {
            "processed_documents": extracted_documents,
            "aggregated_compliance_findings": findings,
            "error_messages": initial_state.get("error_messages", []),
            "cache_stats": cache_stats or {}
        }
        
    except Exception as e:
        logger.error(f"Exception during LangGraph compliance checking: {e}", exc_info=True)
        return {
            "processed_documents": extracted_documents,
            "aggregated_compliance_findings": [],
            "error_messages": [f"LangGraph compliance checking failed: {str(e)}"],
            "cache_stats": {}
        }



