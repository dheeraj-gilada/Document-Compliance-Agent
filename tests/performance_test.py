#!/usr/bin/env python3
"""
Advanced Performance Testing for the Intelligent Document Compliance Agent.

This module provides comprehensive performance analysis including timing,
memory usage, and optimization recommendations.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List

# --- Path Correction ---
# Add the project root to the Python path to allow imports from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---

from src.agents.universal_compliance_agent import UniversalComplianceAgent
from src.utils.document_loader import DocumentLoader
from src.workflows.document_processing_workflow import create_document_processing_graph, DocumentProcessingState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from app.main import extract_document_data_parallel

async def simulate_sequential_document_processing(docs_dir: str) -> Dict[str, Any]:
    """Simulate the old sequential document processing approach"""
    logger.info("🐌 TESTING SEQUENTIAL DOCUMENT PROCESSING")
    start_time = time.time()
    
    # Use the original workflow approach
    app = create_document_processing_graph()
    
    initial_state: DocumentProcessingState = {
        "docs_dir": docs_dir,
        "consolidated_rules_content": None,  # No rules for extraction-only
        "initial_document_paths": [],
        "document_queue": [],
        "current_document_path": None,
        "loaded_document_data": None,
        "processed_documents": [],
        "error_messages": [],
        "aggregated_compliance_findings": [] 
    }
    
    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 100})
    
    duration = time.time() - start_time
    processed_docs = final_state.get('processed_documents', [])
    successful_docs = len([d for d in processed_docs if d.get("status") == "data_extracted"])
    
    logger.info(f"🐌 SEQUENTIAL COMPLETED: {duration:.2f}s, {successful_docs} documents")
    return {
        "approach": "sequential",
        "duration": duration,
        "documents_processed": len(processed_docs),
        "successful_documents": successful_docs,
        "processed_documents": processed_docs
    }

async def test_parallel_document_processing(docs_dir: str) -> Dict[str, Any]:
    """Test the new parallel document processing approach"""
    logger.info("🚀 TESTING PARALLEL DOCUMENT PROCESSING")
    start_time = time.time()
    
    result = await extract_document_data_parallel(docs_dir)
    
    duration = time.time() - start_time
    processed_docs = result.get('processed_documents', [])
    successful_docs = len([d for d in processed_docs if d.get("status") == "data_extracted"])
    
    logger.info(f"🚀 PARALLEL COMPLETED: {duration:.2f}s, {successful_docs} documents")
    return {
        "approach": "parallel",
        "duration": duration,
        "documents_processed": len(processed_docs),
        "successful_documents": successful_docs,
        "processed_documents": processed_docs
    }

async def simulate_old_compliance_approach(agent: UniversalComplianceAgent, documents: List[Dict], rules: List[tuple]) -> List[Dict]:
    """Simulate the old single-rule approach"""
    logger.info("🐌 TESTING OLD APPROACH (Single Rule Processing)")
    start_time = time.time()
    
    findings = []
    for i, (rule_id, rule_text) in enumerate(rules, 1):
        rule_start = time.time()
        finding = await agent.evaluate_single_rule(
            rule_id=rule_id,
            rule_text=rule_text,
            all_documents_data=documents
        )
        rule_duration = time.time() - rule_start
        findings.append(finding)
        logger.info(f"  Rule {i}/{len(rules)}: {rule_duration:.2f}s")
    
    total_duration = time.time() - start_time
    logger.info(f"🐌 OLD APPROACH COMPLETED: {total_duration:.2f}s total, {len(findings)} rules")
    return findings

async def test_new_compliance_approach(agent: UniversalComplianceAgent, documents: List[Dict], rules: List[tuple]) -> List[Dict]:
    """Test the new batch compliance approach"""
    logger.info("🚀 TESTING NEW APPROACH (Batch Rule Processing)")
    start_time = time.time()
    
    # Format rules for batch processing
    rules_text = "\n".join([f"{rule_id}. {rule_text}" for rule_id, rule_text in rules])
    
    findings = await agent.check_all_compliance(
        all_documents_data=documents,
        consolidated_rules=rules_text
    )
    
    total_duration = time.time() - start_time
    logger.info(f"🚀 NEW APPROACH COMPLETED: {total_duration:.2f}s total, {len(findings)} rules")
    return findings

async def run_comprehensive_performance_test():
    """Run comprehensive performance tests"""
    logger.info("=" * 80)
    logger.info("🧪 COMPREHENSIVE PERFORMANCE TEST STARTING")
    logger.info("=" * 80)
    
    # Test documents directory
    docs_dir = "./docs"
    
    # Test rules
    test_rules = [
        ("1", "All invoices must have a total amount greater than $0 and include tax information."),
        ("2", "Purchase orders must contain valid supplier information and delivery addresses."),
        ("3", "All financial documents must have matching date formats."),
        ("4", "Delivery notes must reference corresponding purchase order numbers when available."),
        ("5", "All documents must contain complete contact information."),
        ("6", "Invoice amounts should not exceed $10,000 without additional authorization."),
        ("7", "All documents must be dated within the last 2 years."),
        ("8", "Purchase orders and invoices must have consistent vendor names."),
        ("9", "All monetary amounts must be clearly specified with currency symbols."),
        ("10", "Documents must not contain any missing or incomplete required fields.")
    ]
    
    try:
        # =============================================================================
        # DOCUMENT PROCESSING PERFORMANCE TEST
        # =============================================================================
        logger.info("\n" + "=" * 50)
        logger.info("📄 DOCUMENT PROCESSING PERFORMANCE TEST")
        logger.info("=" * 50)
        
        # Test 1: Sequential Document Processing
        sequential_result = await simulate_sequential_document_processing(docs_dir)
        
        # Test 2: Parallel Document Processing  
        parallel_result = await test_parallel_document_processing(docs_dir)
        
        # Calculate document processing improvement
        if sequential_result["duration"] > 0:
            doc_speedup = sequential_result["duration"] / parallel_result["duration"]
            doc_time_saved = sequential_result["duration"] - parallel_result["duration"]
        else:
            doc_speedup = 1.0
            doc_time_saved = 0.0
        
        logger.info(f"\n📊 DOCUMENT PROCESSING RESULTS:")
        logger.info(f"   🐌 Sequential: {sequential_result['duration']:.2f}s")
        logger.info(f"   🚀 Parallel:   {parallel_result['duration']:.2f}s")
        logger.info(f"   ⚡ Speedup:    {doc_speedup:.1f}x faster")
        logger.info(f"   💰 Time saved: {doc_time_saved:.2f}s")
        
        # =============================================================================
        # COMPLIANCE PROCESSING PERFORMANCE TEST
        # =============================================================================
        logger.info("\n" + "=" * 50)
        logger.info("✅ COMPLIANCE PROCESSING PERFORMANCE TEST")
        logger.info("=" * 50)
        
        # Use parallel processing results for compliance testing
        documents = parallel_result["processed_documents"]
        successful_docs = [d for d in documents if d.get("status") == "data_extracted"]
        
        if not successful_docs:
            logger.warning("No successfully processed documents for compliance testing")
            return
        
        # Prepare documents for compliance agent
        compliance_docs = []
        for doc in successful_docs:
            compliance_docs.append({
                "filename": doc["filename"],
                "doc_type": doc["doc_type"],
                "extracted_data": doc.get("extracted_data", {})
            })
        
        agent = UniversalComplianceAgent()
        
        # Test 3: Old Compliance Approach (Single Rule)
        old_compliance_start = time.time()
        old_findings = await simulate_old_compliance_approach(agent, compliance_docs, test_rules)
        old_compliance_duration = time.time() - old_compliance_start
        
        # Test 4: New Compliance Approach (Batch)
        new_compliance_start = time.time()
        new_findings = await test_new_compliance_approach(agent, compliance_docs, test_rules)
        new_compliance_duration = time.time() - new_compliance_start
        
        # Calculate compliance improvement
        if old_compliance_duration > 0:
            compliance_speedup = old_compliance_duration / new_compliance_duration
            compliance_time_saved = old_compliance_duration - new_compliance_duration
        else:
            compliance_speedup = 1.0
            compliance_time_saved = 0.0
        
        logger.info(f"\n📊 COMPLIANCE PROCESSING RESULTS:")
        logger.info(f"   🐌 Single Rule: {old_compliance_duration:.2f}s")
        logger.info(f"   🚀 Batch:       {new_compliance_duration:.2f}s")
        logger.info(f"   ⚡ Speedup:     {compliance_speedup:.1f}x faster")
        logger.info(f"   💰 Time saved:  {compliance_time_saved:.2f}s")
        
        # =============================================================================
        # OVERALL PERFORMANCE SUMMARY
        # =============================================================================
        logger.info("\n" + "=" * 60)
        logger.info("🏆 OVERALL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        old_total = sequential_result["duration"] + old_compliance_duration
        new_total = parallel_result["duration"] + new_compliance_duration
        overall_speedup = old_total / new_total if new_total > 0 else 1.0
        overall_time_saved = old_total - new_total
        
        logger.info(f"\n📈 COMPLETE PIPELINE COMPARISON:")
        logger.info(f"   🐌 Old Pipeline:  {old_total:.2f}s (Sequential Docs + Single Rules)")
        logger.info(f"   🚀 New Pipeline:  {new_total:.2f}s (Parallel Docs + Batch Rules)")
        logger.info(f"   ⚡ Overall Speedup: {overall_speedup:.1f}x faster")
        logger.info(f"   💰 Total Time Saved: {overall_time_saved:.2f}s")
        logger.info(f"   📊 Efficiency Gain: {((overall_speedup - 1) * 100):.1f}% improvement")
        
        # Performance breakdown
        logger.info(f"\n🔍 PERFORMANCE BREAKDOWN:")
        logger.info(f"   📄 Document Processing Improvement: {doc_speedup:.1f}x")
        logger.info(f"   ✅ Compliance Processing Improvement: {compliance_speedup:.1f}x")
        logger.info(f"   🎯 Documents Processed: {len(successful_docs)}")
        logger.info(f"   📋 Rules Evaluated: {len(test_rules)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PERFORMANCE TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_comprehensive_performance_test()) 