#!/usr/bin/env python3
"""
Streamlit Cache Integration Demo
Demonstrates how the caching system works with Streamlit uploads and processing.
"""

import streamlit as st
import os
import sys
import tempfile
import time
import shutil
import hashlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import CacheStatusManager, DocumentCacheManager, ComplianceCacheManager

st.set_page_config(page_title="Cache Demo", layout="wide")

def calculate_file_hash(uploaded_file):
    """Calculate hash of uploaded file for cache validation."""
    uploaded_file.seek(0)  # Reset file pointer
    content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset again for reuse
    return hashlib.sha256(content).hexdigest()

def simulate_document_processing(file_path, processing_time=2.0):
    """Simulate document processing with artificial delay."""
    time.sleep(processing_time)
    return {
        "filename": os.path.basename(file_path),
        "doc_type": "Demo Document", 
        "extracted_data": {"sample": "data"},
        "status": "data_extracted"
    }

def main():
    st.title("🚀 Streamlit Cache Integration Demo")
    st.markdown("This demo shows **exactly** how caching works with Streamlit file uploads.")
    
    # Initialize session state
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="streamlit_cache_demo_")
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    
    # Cache managers
    cache_manager = CacheStatusManager()
    doc_cache = DocumentCacheManager()
    
    # Sidebar - Cache Statistics
    with st.sidebar:
        st.header("📊 Real-time Cache Status")
        
        # Display cache statistics
        cache_stats = cache_manager.get_unified_stats()
        
        if cache_stats:
            overall = cache_stats.get("overall_performance", {})
            doc_cache_stats = cache_stats.get("document_cache", {})
            
            # Overall performance
            st.subheader("🎯 Overall Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hit Rate", f"{overall.get('overall_cache_hit_rate', 0):.1f}%")
                st.metric("Total Saved", f"{overall.get('total_time_saved_seconds', 0):.1f}s")
            with col2:
                st.metric("Cache Hits", overall.get('total_cache_hits', 0))
                st.metric("Cache Misses", overall.get('total_cache_misses', 0))
            
            # Document cache details
            st.subheader("📁 Document Cache")
            st.metric("Cached Documents", doc_cache_stats.get('total_cached_documents', 0))
            st.metric("Session Hit Rate", f"{doc_cache_stats.get('cache_hit_rate', 0):.1f}%")
            st.metric("Time Saved", f"{doc_cache_stats.get('total_processing_time_saved', 0):.1f}s")
            
            # Cache management
            st.subheader("🛠️ Cache Management")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All"):
                    cache_manager.clear_all_caches()
                    st.success("Cache cleared!")
                    st.rerun()
            with col2:
                if st.button("🧹 Optimize"):
                    cache_manager.optimize_all_caches()
                    st.success("Cache optimized!")
                    st.rerun()
        
        # Reset session button
        if st.button("🔄 Reset Demo Session"):
            cache_manager.reset_session_stats()
            st.session_state.processed_files = {}
            st.success("Session reset!")
            st.rerun()
    
    # Main content
    st.header("📁 File Upload & Cache Validation")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents to test caching", 
        type=['pdf', 'docx', 'txt', 'jpg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.subheader("🔍 Cache Validation Process")
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            file_hash = calculate_file_hash(uploaded_file)
            
            # Create columns for detailed analysis
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**📄 {filename}**")
                st.write(f"Size: {uploaded_file.size:,} bytes")
                st.write(f"Hash: `{file_hash[:16]}...`")
            
            # Save file to temp directory for processing
            file_path = os.path.join(st.session_state.temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with col2:
                # Check cache status
                is_cached = doc_cache.is_cached_and_valid(file_path)
                
                if is_cached:
                    st.success("✅ **CACHE HIT**")
                    st.write("File found in cache")
                    
                    # Get cached data
                    start_time = time.time()
                    cached_data = doc_cache.get_cached_data(file_path)
                    cache_load_time = time.time() - start_time
                    
                    if cached_data:
                        st.write(f"⚡ Loaded in {cache_load_time:.3f}s")
                        st.session_state.processed_files[filename] = {
                            "data": cached_data,
                            "processing_time": 0,
                            "from_cache": True
                        }
                else:
                    st.warning("❌ **CACHE MISS**")
                    st.write("File not in cache")
                    
                    if filename not in st.session_state.processed_files:
                        st.write("🔄 Processing needed...")
            
            with col3:
                # Process button for cache misses
                if not is_cached and filename not in st.session_state.processed_files:
                    # Create unique key using hash to avoid duplicates
                    unique_key = f"process_{file_hash[:8]}_{filename.replace('.', '_')}"
                    if st.button(f"Process", key=unique_key):
                        with st.spinner("Processing..."):
                            start_time = time.time()
                            
                            # Simulate processing
                            processed_data = simulate_document_processing(file_path, processing_time=3.0)
                            processing_time = time.time() - start_time
                            
                            # Cache the result
                            doc_cache.cache_document_data(file_path, processed_data, processing_time)
                            
                            st.session_state.processed_files[filename] = {
                                "data": processed_data,
                                "processing_time": processing_time,
                                "from_cache": False
                            }
                        
                        st.rerun()
                elif is_cached or filename in st.session_state.processed_files:
                    st.success("✅ Ready")
    
    # Results section
    if st.session_state.processed_files:
        st.header("📊 Processing Results")
        
        # Summary metrics
        total_files = len(st.session_state.processed_files)
        cached_files = sum(1 for f in st.session_state.processed_files.values() if f.get('from_cache'))
        total_processing_time = sum(f.get('processing_time', 0) for f in st.session_state.processed_files.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("From Cache", cached_files)
        with col3:
            st.metric("Processing Time", f"{total_processing_time:.1f}s")
        with col4:
            cache_hit_rate = (cached_files / total_files * 100) if total_files > 0 else 0
            st.metric("Hit Rate", f"{cache_hit_rate:.1f}%")
        
        # Detailed results
        st.subheader("📋 File Details")
        for filename, result in st.session_state.processed_files.items():
            with st.expander(f"📄 {filename}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Source:**", "🔥 Cache" if result['from_cache'] else "🔄 Processing")
                    st.write("**Processing Time:**", f"{result['processing_time']:.2f}s")
                with col2:
                    st.write("**Document Type:**", result['data'].get('doc_type', 'Unknown'))
                    st.write("**Status:**", result['data'].get('status', 'Unknown'))
    
    # Educational section
    st.header("🎓 How Caching Works with Streamlit")
    
    with st.expander("📚 Cache Validation Process", expanded=False):
        st.markdown("""
        ### 🔍 **Cache Identification Process**
        
        1. **File Hash Calculation**: SHA-256 hash of file content
        2. **Metadata Comparison**: File size and modification time
        3. **Version Compatibility**: Model and workflow versions
        4. **File Existence**: Cached data file still exists
        
        ### ⚡ **Cache Hit Criteria**
        - ✅ **Identical file hash** (content unchanged)
        - ✅ **Matching file size** (quick validation)
        - ✅ **Compatible versions** (model/workflow)
        - ✅ **Cache file exists** (not deleted)
        
        ### 🗑️ **Cache Clearing Triggers**
        - **Manual clearing** (user button)
        - **Version mismatch** (automatic invalidation)
        - **File changes** (hash/size difference)
        - **Age-based cleanup** (configurable)
        """)
    
    with st.expander("🔄 Streamlit Integration Details", expanded=False):
        st.markdown("""
        ### 📁 **File Upload Handling**
        
        1. **File Upload**: Streamlit `file_uploader()` widget
        2. **Hash Calculation**: Immediate file content hashing
        3. **Temp File Creation**: Save to temporary directory
        4. **Cache Check**: Validate against cached versions
        5. **Process or Load**: Either process new or load cached
        
        ### 💾 **Session State Management**
        - `st.session_state.processed_files`: Track processed files
        - `st.session_state.temp_dir`: Temporary file storage
        - **Automatic cleanup**: On session end
        
        ### 🚀 **Performance Benefits**
        - **Instant loading**: Cached files load in <1ms
        - **Cross-session**: Cache persists between Streamlit sessions
        - **Multi-user**: Shared cache across users (same files)
        - **Scalable**: Handles large document collections
        """)
    
    with st.expander("⚙️ Cache Configuration & Management", expanded=False):
        st.markdown("""
        ### 🛠️ **Cache Storage**
        - **Location**: `.cache/` directory
        - **Database**: SQLite for metadata
        - **Files**: JSON for extracted data
        - **Size**: Automatic size monitoring
        
        ### 🔧 **Cache Management Options**
        - **Clear All**: Remove all cached data
        - **Clear Old**: Remove entries older than N days
        - **Optimize**: Remove orphaned files, compact DB
        - **Reset Stats**: Reset session performance metrics
        
        ### 📊 **Performance Monitoring**
        - **Hit Rate**: Percentage of cache hits
        - **Time Saved**: Total processing time saved
        - **Storage Usage**: Cache database size
        - **Recent Activity**: Files cached in last 24h
        """)

    # Cleanup on app end
    if st.button("🧹 Cleanup Demo Files"):
        if os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = tempfile.mkdtemp(prefix="streamlit_cache_demo_")
        st.session_state.processed_files = {}
        st.success("Demo files cleaned up!")
        st.rerun()

# Entry point removed - use from within application or tests only 