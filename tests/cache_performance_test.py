#!/usr/bin/env python3
"""
Cache Performance Test
Demonstrates the enhanced caching capabilities of the document compliance system.
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import CacheStatusManager
from src.workflows.document_processing_workflow import create_document_processing_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_cache_performance():
    """Test and demonstrate cache performance improvements."""
    
    print("🧪 Testing Enhanced Cache Performance")
    print("="*50)
    
    # Initialize cache manager
    cache_manager = CacheStatusManager()
    
    # Reset session stats for clean test
    cache_manager.reset_session_stats()
    
    # Test data
    docs_dir = str(project_root / "data" / "documents")
    rules_file = str(project_root / "data" / "rules" / "consolidated_compliance_rules.txt")
    
    if not os.path.exists(docs_dir) or not os.path.exists(rules_file):
        print("❌ Test data not found. Please ensure documents and rules are in data/ directory.")
        return
    
    print(f"📁 Documents directory: {docs_dir}")
    print(f"📋 Rules file: {rules_file}")
    print()
    
    # Create the workflow
    workflow = create_document_processing_graph()
    
    # Test 1: First run (should be cache misses)
    print("🔄 Test 1: First run (cache misses expected)")
    start_time = time.time()
    
    try:
        result1 = await workflow.ainvoke({
            "docs_dir": docs_dir,
            "consolidated_rules_file": rules_file
        })
    except Exception as e:
        # If consolidated_rules_file is not supported, try reading the file directly
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_content = f.read()
        
        result1 = await workflow.ainvoke({
            "docs_dir": docs_dir,
            "consolidated_rules_content": rules_content
        })
    
    first_run_time = time.time() - start_time
    print(f"⏱️  First run completed in: {first_run_time:.2f}s")
    print()
    
    # Show cache stats after first run
    print("📊 Cache Stats After First Run:")
    cache_manager.print_cache_summary()
    
    # Test 2: Second run (should be cache hits)
    print("\n🔄 Test 2: Second run (cache hits expected)")
    start_time = time.time()
    
    try:
        result2 = await workflow.ainvoke({
            "docs_dir": docs_dir,
            "consolidated_rules_file": rules_file
        })
    except Exception as e:
        # If consolidated_rules_file is not supported, try reading the file directly
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_content = f.read()
        
        result2 = await workflow.ainvoke({
            "docs_dir": docs_dir,
            "consolidated_rules_content": rules_content
        })
    
    second_run_time = time.time() - start_time
    print(f"⏱️  Second run completed in: {second_run_time:.2f}s")
    print()
    
    # Show cache stats after second run
    print("📊 Cache Stats After Second Run:")
    cache_manager.print_cache_summary()
    
    # Performance analysis
    time_saved = first_run_time - second_run_time
    speed_improvement = (time_saved / first_run_time) * 100 if first_run_time > 0 else 0
    
    print(f"\n🚀 PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"📈 Time Improvement: {time_saved:.2f}s saved ({speed_improvement:.1f}% faster)")
    print(f"📊 First run: {first_run_time:.2f}s")
    print(f"📊 Second run: {second_run_time:.2f}s")
    
    # Verify results are identical
    if result1 and result2:
        compliance1 = result1.get("aggregated_compliance_findings", [])
        compliance2 = result2.get("aggregated_compliance_findings", [])
        
        if len(compliance1) == len(compliance2):
            print(f"✅ Results consistency: Both runs processed {len(compliance1)} compliance rules")
        else:
            print(f"⚠️  Results inconsistency: {len(compliance1)} vs {len(compliance2)} rules")
    
    # Test 3: Cache optimization
    print(f"\n🔧 Test 3: Cache Optimization")
    orphaned_files = cache_manager.optimize_all_caches()
    print(f"🗑️  Removed {orphaned_files} orphaned cache files")
    
    print("\n✅ Cache Performance Test Complete!")
    return {
        "first_run_time": first_run_time,
        "second_run_time": second_run_time,
        "time_saved": time_saved,
        "speed_improvement": speed_improvement
    }

def test_bulk_operations():
    """Test bulk cache operations."""
    print("\n🔄 Testing Bulk Cache Operations")
    print("="*40)
    
    cache_manager = CacheStatusManager()
    
    # Test document cache bulk check
    test_files = [
        str(project_root / "data" / "documents" / "AI Engineer Challenge Invoice.PDF"),
        str(project_root / "data" / "documents" / "DELIVERY_NOTE_GROK.pdf"),
        str(project_root / "data" / "documents" / "purchase_order_GROK.pdf"),
    ]
    
    # Filter to only existing files
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print(f"📋 Testing bulk cache check for {len(existing_files)} files...")
        
        start_time = time.time()
        cache_results = cache_manager.doc_cache.bulk_check_cache_status(existing_files)
        bulk_time = time.time() - start_time
        
        cached_count = sum(1 for cached in cache_results.values() if cached)
        print(f"⚡ Bulk check completed in {bulk_time:.3f}s")
        print(f"📊 {cached_count}/{len(existing_files)} files are cached")
        
        for file_path, is_cached in cache_results.items():
            filename = os.path.basename(file_path)
            status = "✅ CACHED" if is_cached else "❌ NOT CACHED"
            print(f"   • {filename}: {status}")
    else:
        print("⚠️  No test files found for bulk operations")

def demonstrate_cache_features():
    """Demonstrate advanced cache features."""
    print("\n🎯 Demonstrating Advanced Cache Features")
    print("="*45)
    
    cache_manager = CacheStatusManager()
    
    # Show compliance history
    print("📊 Recent Compliance History:")
    history = cache_manager.compliance_cache.get_compliance_history(limit=5)
    
    if history:
        for i, entry in enumerate(history, 1):
            print(f"   {i}. {entry['processed_at'][:16]} - {entry['compliance_summary']} "
                  f"({entry['compliance_score']:.1f}% compliant)")
    else:
        print("   No compliance history found")
    
    # Show unified statistics
    print(f"\n📈 Unified Cache Statistics:")
    stats = cache_manager.get_unified_stats()
    
    if stats:
        overall = stats["overall_performance"]
        print(f"   • Overall Hit Rate: {overall['overall_cache_hit_rate']:.1f}%")
        print(f"   • Total Time Saved: {overall['total_time_saved_seconds']:.1f}s")
        print(f"   • Session Time Saved: {overall['session_time_saved_seconds']:.1f}s")

async def main():
    """Main async function to run all tests."""
    try:
        # Run comprehensive cache performance test
        performance_results = await test_cache_performance()
        
        # Test bulk operations
        test_bulk_operations()
        
        # Demonstrate advanced features
        demonstrate_cache_features()
        
        print(f"\n🎉 All cache tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        print(f"❌ Cache test failed: {e}")
        sys.exit(1)

# Entry point removed - use from within application or tests only 