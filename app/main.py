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
from app.config import DOCS_DIR
from src.workflows.document_processing_workflow import create_document_processing_graph, DocumentProcessingState
from src.agents.universal_compliance_agent import UniversalComplianceAgent
from src.utils.cache_manager import DocumentCacheManager
from src.utils.compliance_cache import ComplianceCacheManager


async def process_single_document_parallel(doc_path: str, docs_dir: str, cache_manager: DocumentCacheManager = None) -> Dict[str, Any]:
    """
    Process a single document in parallel - loads, classifies, and extracts data.
    
    Args:
        doc_path: Full path to the document
        docs_dir: Directory containing the document
        cache_manager: Optional cache manager for caching results
        
    Returns:
        Dictionary containing processed document data
    """
    from src.utils.document_loader import DocumentLoader
    from src.agents.structured_data_extractor_agent import StructuredDataExtractorAgent
    
    doc_filename = os.path.basename(doc_path)
    
    # Check cache first
    if cache_manager and cache_manager.is_cached_and_valid(doc_path):
        cached_data = cache_manager.get_cached_data(doc_path)
        if cached_data:
            return cached_data
    
    # Track processing time for caching
    import time
    start_time = time.time()
    
    # Initialize document data structure
    doc_data = {
        "filename": doc_filename,
        "path": doc_path,
        "status": "initialization",
        "doc_type": None,
        "extracted_text_content": None,
        "extracted_tables_html": [], 
        "extracted_data": None, 
        "error_message": None
    }
    
    try:
        # Step 1: Load and classify document
        logger.info(f"[Parallel] Processing document: {doc_filename}")
        loader = DocumentLoader(docs_dir)
        loaded_info = await loader.load_document(doc_filename)
        
        if not loaded_info or not loaded_info.get("text"):
            doc_data["status"] = "error_loading_document"
            doc_data["error_message"] = loaded_info.get("error_message", "Failed to load document content or extract text.")
            return doc_data
        
        # Update with loaded info
        doc_data["doc_type"] = loaded_info.get("doc_type", "unknown")
        doc_data["extracted_text_content"] = loaded_info.get("text")
        doc_data["extracted_tables_html"] = loaded_info.get("tables_html", [])
        doc_data["status"] = "classified"
        
        logger.info(f"[Parallel] Document {doc_filename} classified as: {doc_data['doc_type']}")
        
        # Step 2: Extract structured data
        text_content = doc_data.get("extracted_text_content", "")
        tables_html_list = doc_data.get("extracted_tables_html", [])
        combined_tables_html = "\n\n".join(tables_html_list)
        
        if not text_content and not combined_tables_html:
            doc_data["status"] = "skipped_extraction_no_content"
            logger.warning(f"[Parallel] No content to extract from {doc_filename}")
            return doc_data
        
        # Extract structured data
        data_extractor_agent = StructuredDataExtractorAgent()
        extracted_data = await data_extractor_agent.extract_structured_data(
            doc_type=doc_data["doc_type"], 
            text_content=text_content, 
            combined_tables_html=combined_tables_html, 
            filename=doc_filename
        )
        
        doc_data["extracted_data"] = extracted_data
        doc_data["status"] = "data_extracted"
        logger.info(f"[Parallel] Data extracted from {doc_filename}")
        
        # Cache the successful result
        if cache_manager:
            processing_time = time.time() - start_time
            cache_manager.cache_document_data(doc_path, doc_data, processing_time)
        
        return doc_data
        
    except FileNotFoundError as e:
        logger.error(f"[Parallel] File not found: {doc_filename}: {e}")
        doc_data["status"] = "error_file_not_found"
        doc_data["error_message"] = str(e)
        return doc_data
    except Exception as e:
        logger.error(f"[Parallel] Error processing {doc_filename}: {e}", exc_info=True)
        doc_data["status"] = "error_processing"
        doc_data["error_message"] = str(e)
        return doc_data


async def extract_document_data_parallel(docs_dir_to_process: str) -> Dict[str, Any]:
    """
    Parallel document processing - processes all documents simultaneously.
    
    This is a major performance optimization that processes multiple documents
    in parallel instead of sequentially, dramatically reducing processing time.
    
    Args:
        docs_dir_to_process: Directory containing documents to process
        
    Returns:
        Dictionary containing processed documents data
    """
    logger.info(f"🚀 PARALLEL PROCESSING: Starting document extraction from: {docs_dir_to_process}")
    
    # Initialize cache manager
    cache_manager = DocumentCacheManager()
    
    try:
        # Step 1: List all documents
        from src.utils.document_loader import DocumentLoader
        loader = DocumentLoader(docs_dir_to_process)
        document_filenames = loader.list_documents()
        document_paths = [os.path.join(docs_dir_to_process, fname) for fname in document_filenames]
        
        if not document_paths:
            logger.info("No documents found to process")
            return {
                "processed_documents": [],
                "error_messages": []
            }
        
        logger.info(f"📊 Found {len(document_paths)} documents for parallel processing")
        
        # Step 2: Process all documents in parallel
        import time
        start_time = time.time()
        
        # Create parallel tasks for all documents (with caching)
        tasks = [
            process_single_document_parallel(doc_path, docs_dir_to_process, cache_manager) 
            for doc_path in document_paths
        ]
        
        # Execute all tasks in parallel
        logger.info(f"⚡ Processing {len(tasks)} documents in parallel...")
        processed_documents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_documents = []
        error_messages = []
        
        for i, result in enumerate(processed_documents):
            if isinstance(result, Exception):
                error_msg = f"Error processing {document_filenames[i]}: {str(result)}"
                logger.error(error_msg)
                error_messages.append(error_msg)
                # Create error document entry
                final_documents.append({
                    "filename": document_filenames[i],
                    "path": document_paths[i],
                    "status": "error_parallel_processing",
                    "error_message": str(result)
                })
            else:
                final_documents.append(result)
        
        # Performance logging
        total_time = time.time() - start_time
        successful_docs = len([d for d in final_documents if d.get("status") == "data_extracted"])
        
        logger.info(f"✅ PARALLEL PROCESSING COMPLETED:")
        logger.info(f"   📈 Total time: {total_time:.2f}s")
        logger.info(f"   🎯 Documents processed: {successful_docs}/{len(document_paths)}")
        logger.info(f"   ⚡ Average time per document: {total_time/len(document_paths):.2f}s")
        logger.info(f"   🚀 Speed improvement: ~{len(document_paths)}x faster than sequential")
        
        return {
            "processed_documents": final_documents,
            "error_messages": error_messages
        }
        
    except Exception as e:
        logger.error(f"Exception during parallel document processing: {e}", exc_info=True)
        return {
            "processed_documents": [],
            "error_messages": [f"Parallel document processing failed: {str(e)}"]
        }


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

async def extract_document_data(docs_dir_to_process: str) -> Dict[str, Any]:
    """
    Extracts data from documents without running compliance checks.
    
    Now uses parallel processing for dramatically improved performance.
    
    Args:
        docs_dir_to_process: Directory containing documents to process
        
    Returns:
        Dictionary containing processed documents data
    """
    # Use the new parallel processing function
    return await extract_document_data_parallel(docs_dir_to_process)


async def run_compliance_check(extracted_documents: List[Dict[str, Any]], rules_content: str) -> Dict[str, Any]:
    """
    Runs compliance checks on already extracted document data.
    
    This function is optimized for the Streamlit app to run compliance checks
    on previously extracted document data without reprocessing the documents.
    
    Args:
        extracted_documents: List of previously extracted document data
        rules_content: String containing the compliance rules
        
    Returns:
        Dictionary containing compliance findings
    """
    import time
    
    logger.info(f"Running compliance checks with {len(extracted_documents)} documents")
    
    # Initialize compliance cache
    compliance_cache = ComplianceCacheManager()
    
    # Check cache first
    if compliance_cache.is_cached_and_valid(extracted_documents, rules_content):
        cached_results = compliance_cache.get_cached_results(extracted_documents, rules_content)
        if cached_results:
            return cached_results
    
    try:
        start_time = time.time()
        
        # Parse the rules
        from src.workflows.document_processing_workflow import _parse_rules
        parsed_rules = _parse_rules(rules_content)
        logger.info(f"Parsed {len(parsed_rules)} rules for compliance checking")
        
        # If no rules were parsed, return early with a helpful message
        if not parsed_rules:
            logger.warning("No rules were parsed from the provided rules content")
            return {
                "processed_documents": extracted_documents,
                "aggregated_compliance_findings": [],
                "error_messages": ["No compliance rules were parsed. Please check rule format."]
            }
        
        # Create compliance agent
        compliance_agent = UniversalComplianceAgent()
        
        # Format rules for batch processing
        rules_text = "\n".join([f"{rule_id}. {rule_text}" for rule_id, rule_text in parsed_rules])
        
        # Run all compliance checks in a single API call
        raw_findings = await compliance_agent.check_all_compliance(
            all_documents_data=extracted_documents,
            consolidated_rules=rules_text
        )
        
        # Process findings for Streamlit compatibility
        findings = []
        for raw_finding in raw_findings:
            # Determine the boolean 'is_compliant' status
            is_compliant_bool = raw_finding.get('status', '').lower() in ['pass', 'compliant']
            
            # Create the final finding dictionary with the 'is_compliant' key
            processed_finding = {
                'rule_id': raw_finding.get('rule_id', 'N/A'),
                'rule_checked': raw_finding.get('rule_checked', 'N/A'),
                'status': raw_finding.get('status', 'error'),
                'is_compliant': is_compliant_bool,
                'reason': raw_finding.get('details', 'No details provided.'),
                'details': raw_finding.get('details', 'No details provided.'),
                'involved_documents': raw_finding.get('involved_documents', [])
            }
            findings.append(processed_finding)
        
        # Prepare results
        results = {
            "processed_documents": extracted_documents,
            "aggregated_compliance_findings": findings
        }
        
        # Cache the results
        processing_time = time.time() - start_time
        compliance_cache.cache_compliance_results(
            extracted_documents, rules_content, results, processing_time
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Exception during compliance checking: {e}", exc_info=True)
        return {
            "processed_documents": extracted_documents,
            "aggregated_compliance_findings": [],
            "error_messages": [f"Compliance checking failed: {str(e)}"]
        }


async def run_document_processing(docs_dir_to_process: str, consolidated_rules_input: Optional[str] = None) -> Dict[str, Any]:
    """Invokes the document processing LangGraph workflow and returns processed documents and all compliance findings.
    
    This function runs the full pipeline (extraction + compliance) and is maintained for backward compatibility.
    For Streamlit, consider using extract_document_data and run_compliance_check separately for better efficiency.
    """
    logger.info(f"Starting full document processing workflow for directory: {docs_dir_to_process}")
    if consolidated_rules_input:
        logger.info("Consolidated compliance rules provided and will be used in the workflow.")
    else:
        logger.info("No consolidated compliance rules provided; universal compliance checks might be limited or skipped.")
    
    app = create_document_processing_graph()
    
    initial_state: DocumentProcessingState = {
        "docs_dir": docs_dir_to_process,
        "consolidated_rules_content": consolidated_rules_input, 
        "initial_document_paths": [],
        "document_queue": [],
        "current_document_path": None,
        "loaded_document_data": None,
        "processed_documents": [],
        "error_messages": [],
        "aggregated_compliance_findings": [] 
    }

    final_graph_state = None
    try:
        logger.info(f"Invoking document processing workflow with initial state: {initial_state}")
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

        logger.info("Document processing workflow completed.")
        return final_graph_state

    except Exception as e:
        logger.error(f"Exception during document processing workflow execution (ainvoke): {e}", exc_info=True)
        return {"processed_documents": [], "aggregated_compliance_findings": [], "error_messages": [f"Workflow execution failed: {str(e)}"]}


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
        
        logger.info(f"DEBUG main: processed_workflow_output received. Type: {type(processed_workflow_output)}")

        if isinstance(processed_workflow_output, dict):
            actual_state_data = processed_workflow_output
            logger.info(f"DEBUG main: actual_state_data assigned directly from processed_workflow_output (dict). Keys: {list(actual_state_data.keys())}")
        else:
            logger.warning(f"DEBUG main: processed_workflow_output is not a dictionary. Type: {type(processed_workflow_output)}. Content: {str(processed_workflow_output)[:500]}... Assigning empty dict to actual_state_data.") # Log content snippet
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
            all_findings = actual_state_data.get('aggregated_compliance_findings', [])
            if not isinstance(all_findings, list):
                logger.warning(f"'aggregated_compliance_findings' in actual_state_data is not a list. Defaulting to empty list.")
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
