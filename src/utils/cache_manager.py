"""
Document Processing Cache Manager
Implements intelligent caching to avoid reprocessing unchanged documents.
Enhanced for LangGraph workflow integration with performance optimizations.
"""

import os
import json
import hashlib
import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Global cache manager instance for cross-module sharing
_global_cache_manager = None

def get_global_cache_manager():
    """Get the global cache manager instance if available."""
    return _global_cache_manager

class DocumentCacheManager:
    """Manages caching of processed documents to avoid expensive reprocessing."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # SQLite database for metadata and fast lookups
        self.db_path = self.cache_dir / "document_cache.db"
        self.init_database()
        
        # JSON storage for extracted data
        self.data_dir = self.cache_dir / "extracted_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self._session_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "time_saved": 0.0,
            "files_processed": 0
        }
        
    def init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_cache (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    last_modified REAL NOT NULL,
                    processed_at TIMESTAMP NOT NULL,
                    doc_type TEXT,
                    processing_time REAL,
                    data_file_path TEXT,
                    extraction_success BOOLEAN DEFAULT TRUE,
                    model_version TEXT DEFAULT 'v1',
                    workflow_version TEXT DEFAULT 'v1'
                )
            """)
            
            # Enhanced indexes for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash 
                ON document_cache(file_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed_at 
                ON document_cache(processed_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_extraction_success 
                ON document_cache(extraction_success)
            """)

    async def _get_file_hash_async(self, file_path: str) -> str:
        """Generate SHA-256 hash of file content asynchronously."""
        hash_sha256 = hashlib.sha256()
        try:
            # Use asyncio to avoid blocking on large files
            def _hash_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):  # Larger chunk size
                        hash_sha256.update(chunk)
                return hash_sha256.hexdigest()
            
            return await asyncio.get_event_loop().run_in_executor(None, _hash_file)
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash of file content (synchronous)."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):  # Larger chunk size
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata for cache validation."""
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "hash": self._get_file_hash(file_path)
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return {}

    async def _get_file_metadata_async(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata for cache validation asynchronously."""
        try:
            stat = os.stat(file_path)
            file_hash = await self._get_file_hash_async(file_path)
            return {
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "hash": file_hash
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return {}
    
    def is_cached_and_valid(self, file_path: str, model_version: str = "v1", workflow_version: str = "v1") -> bool:
        """Check if document is cached and cache is still valid using content hash."""
        try:
            # Get file hash first
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                self._session_stats["cache_misses"] += 1
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                # First, check if the new columns exist
                cursor = conn.execute("PRAGMA table_info(document_cache)")
                columns = [row[1] for row in cursor.fetchall()]
                has_version_columns = "model_version" in columns and "workflow_version" in columns
                
                if has_version_columns:
                    # Use new schema with version checking - search by hash instead of path
                    cursor = conn.execute("""
                        SELECT file_path, file_size, last_modified, data_file_path, model_version, workflow_version
                        FROM document_cache 
                        WHERE file_hash = ?
                    """, (file_hash,))
                else:
                    # Use old schema for backward compatibility - search by hash
                    logger.info("Using backward-compatible cache lookup (no version columns)")
                    cursor = conn.execute("""
                        SELECT file_path, file_size, last_modified, data_file_path
                        FROM document_cache 
                        WHERE file_hash = ?
                    """, (file_hash,))
                
                result = cursor.fetchone()
                if not result:
                    logger.info(f"❌ Cache MISS for {os.path.basename(file_path)} (hash: {file_hash[:8]}...)")
                    self._session_stats["cache_misses"] += 1
                    return False
                
                if has_version_columns:
                    cached_path, cached_size, cached_modified, data_file, cached_model_ver, cached_workflow_ver = result
                    # Check version compatibility
                    if cached_model_ver != model_version or cached_workflow_ver != workflow_version:
                        logger.info(f"❌ Cache MISS for {os.path.basename(file_path)} (version mismatch)")
                        self._session_stats["cache_misses"] += 1
                        return False
                else:
                    cached_path, cached_size, cached_modified, data_file = result
                    logger.info(f"🔄 Cache lookup for {os.path.basename(file_path)} (legacy mode)")
                
                # Check if data file exists
                if not os.path.exists(data_file):
                    logger.info(f"Cache data file missing for {file_path}")
                    self._session_stats["cache_misses"] += 1
                    return False
                
                # Get current file metadata
                current_meta = self._get_file_metadata(file_path)
                if not current_meta:
                    self._session_stats["cache_misses"] += 1
                    return False
                
                # Validate cache - only check size (hash already matches)
                is_valid = current_meta["size"] == cached_size
                
                if is_valid:
                    logger.info(f"✅ Cache HIT for {os.path.basename(file_path)} (hash: {file_hash[:8]}...)")
                    self._session_stats["cache_hits"] += 1
                else:
                    logger.info(f"❌ Cache MISS for {os.path.basename(file_path)} (file size changed)")
                    self._session_stats["cache_misses"] += 1
                
                return is_valid
                
        except Exception as e:
            logger.error(f"Error checking cache for {file_path}: {e}")
            self._session_stats["cache_misses"] += 1
            return False
    
    def get_cached_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached document data using content hash lookup."""
        try:
            # Get file hash for lookup
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                return None
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data_file_path, doc_type, processing_time
                    FROM document_cache 
                    WHERE file_hash = ?
                """, (file_hash,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                data_file, doc_type, processing_time = result
                
                # Load the cached data
                with open(data_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Track time saved
                self._session_stats["time_saved"] += processing_time
                
                logger.info(f"📁 Loaded cached data for {os.path.basename(file_path)} (saved {processing_time:.2f}s)")
                return cached_data
                
        except Exception as e:
            logger.error(f"Error retrieving cached data for {file_path}: {e}")
            return None
    
    def cache_document_data(self, file_path: str, document_data: Dict[str, Any], processing_time: float, 
                          model_version: str = "v1", workflow_version: str = "v1"):
        """Cache processed document data with version tracking."""
        try:
            # Get file metadata
            file_meta = self._get_file_metadata(file_path)
            if not file_meta:
                logger.error(f"Cannot cache {file_path} - failed to get metadata")
                return
            
            # Create unique filename for cached data
            file_hash = file_meta["hash"][:16]  # First 16 chars of hash
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"{filename}_{file_hash}_{timestamp}.json"
            data_file_path = self.data_dir / data_filename
            
            # Save document data
            with open(data_file_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, default=str)
            
            # Update database (backward compatible)
            with sqlite3.connect(self.db_path) as conn:
                # Check if new columns exist
                cursor = conn.execute("PRAGMA table_info(document_cache)")
                columns = [row[1] for row in cursor.fetchall()]
                has_version_columns = "model_version" in columns and "workflow_version" in columns
                
                if has_version_columns:
                    # Use new schema
                    conn.execute("""
                        INSERT OR REPLACE INTO document_cache 
                        (file_path, file_hash, file_size, last_modified, processed_at, 
                         doc_type, processing_time, data_file_path, extraction_success,
                         model_version, workflow_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_path,
                        file_meta["hash"],
                        file_meta["size"],
                        file_meta["modified"],
                        datetime.now().isoformat(),
                        document_data.get("doc_type", "unknown"),
                        processing_time,
                        str(data_file_path),
                        document_data.get("status") == "data_extracted",
                        model_version,
                        workflow_version
                    ))
                else:
                    # Use old schema for backward compatibility
                    logger.info("Using backward-compatible cache storage (no version columns)")
                    conn.execute("""
                        INSERT OR REPLACE INTO document_cache 
                        (file_path, file_hash, file_size, last_modified, processed_at, 
                         doc_type, processing_time, data_file_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_path,
                        file_meta["hash"],
                        file_meta["size"],
                        file_meta["modified"],
                        datetime.now().isoformat(),
                        document_data.get("doc_type", "unknown"),
                        processing_time,
                        str(data_file_path)
                    ))
            
            self._session_stats["files_processed"] += 1
            logger.info(f"💾 Cached data for {filename} (processing time: {processing_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error caching document data for {file_path}: {e}")

    def bulk_check_cache_status(self, file_paths: List[str], model_version: str = "v1", workflow_version: str = "v1") -> Dict[str, bool]:
        """Efficiently check cache status for multiple files at once."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ','.join('?' * len(file_paths))
                cursor = conn.execute(f"""
                    SELECT file_path, file_hash, file_size, last_modified, data_file_path, model_version, workflow_version
                    FROM document_cache 
                    WHERE file_path IN ({placeholders})
                """, file_paths)
                
                cache_results = {}
                cached_files = {row[0]: row[1:] for row in cursor.fetchall()}
                
                for file_path in file_paths:
                    if file_path not in cached_files:
                        cache_results[file_path] = False
                        continue
                    
                    cached_hash, cached_size, cached_modified, data_file, cached_model_ver, cached_workflow_ver = cached_files[file_path]
                    
                    # Quick checks
                    if not os.path.exists(data_file):
                        cache_results[file_path] = False
                        continue
                    
                    if cached_model_ver != model_version or cached_workflow_ver != workflow_version:
                        cache_results[file_path] = False
                        continue
                    
                    # File metadata check (most expensive)
                    current_meta = self._get_file_metadata(file_path)
                    if not current_meta:
                        cache_results[file_path] = False
                        continue
                    
                    is_valid = (
                        current_meta["hash"] == cached_hash and
                        current_meta["size"] == cached_size and
                        abs(current_meta["modified"] - cached_modified) < 1.0
                    )
                    
                    cache_results[file_path] = is_valid
                
                return cache_results
                
        except Exception as e:
            logger.error(f"Error in bulk cache check: {e}")
            return {file_path: False for file_path in file_paths}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics including session performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_cached,
                        SUM(processing_time) as total_time_saved,
                        AVG(processing_time) as avg_processing_time,
                        COUNT(CASE WHEN extraction_success = 1 THEN 1 END) as successful_extractions,
                        COUNT(CASE WHEN processed_at > datetime('now', '-24 hours') THEN 1 END) as recent_cached
                    FROM document_cache
                """)
                
                result = cursor.fetchone()
                base_stats = {}
                if result:
                    base_stats = {
                        "total_cached_documents": result[0],
                        "total_processing_time_saved": round(result[1] or 0, 2),
                        "average_processing_time": round(result[2] or 0, 2),
                        "successful_extractions": result[3],
                        "recent_cached_24h": result[4],
                        "cache_directory": str(self.cache_dir),
                        "database_size_mb": round(os.path.getsize(self.db_path) / 1024 / 1024, 2) if os.path.exists(self.db_path) else 0
                    }
                
                # Add session statistics
                total_checks = self._session_stats["cache_hits"] + self._session_stats["cache_misses"]
                hit_rate = (self._session_stats["cache_hits"] / total_checks * 100) if total_checks > 0 else 0
                
                base_stats.update({
                    "session_cache_hits": self._session_stats["cache_hits"],
                    "session_cache_misses": self._session_stats["cache_misses"],
                    "cache_hit_rate": round(hit_rate, 1),
                    "estimated_time_saved": round(self._session_stats["time_saved"], 2),
                    "files_processed_this_session": self._session_stats["files_processed"]
                })
                
                return base_stats
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
        
        return self._session_stats.copy()
    
    def clear_cache(self, older_than_days: int = None, failed_extractions_only: bool = False):
        """Clear cache entries with enhanced filtering options."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if failed_extractions_only:
                    # Clear only failed extractions
                    cursor = conn.execute("""
                        SELECT data_file_path FROM document_cache 
                        WHERE extraction_success = 0
                    """)
                    files_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    conn.execute("DELETE FROM document_cache WHERE extraction_success = 0")
                    deleted_count = cursor.rowcount
                    
                elif older_than_days:
                    cutoff_date = datetime.now() - timedelta(days=older_than_days)
                    
                    # Get files to delete
                    cursor = conn.execute("""
                        SELECT data_file_path FROM document_cache 
                        WHERE processed_at < ?
                    """, (cutoff_date.isoformat(),))
                    
                    files_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    # Delete database entries
                    cursor = conn.execute("""
                        DELETE FROM document_cache WHERE processed_at < ?
                    """, (cutoff_date.isoformat(),))
                    
                    deleted_count = cursor.rowcount
                    
                else:
                    # Clear all
                    cursor = conn.execute("SELECT data_file_path FROM document_cache")
                    files_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    conn.execute("DELETE FROM document_cache")
                    deleted_count = cursor.rowcount
                
                # Delete data files
                for file_path in files_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not delete cache file {file_path}: {e}")
                
                action_desc = "failed extractions" if failed_extractions_only else f"entries older than {older_than_days} days" if older_than_days else "all entries"
                logger.info(f"🗑️ Cleared {deleted_count} cache {action_desc}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def optimize_cache(self):
        """Optimize cache by removing orphaned files and compacting database."""
        try:
            orphaned_files = 0
            
            # Find and remove orphaned cache files
            for cache_file in self.data_dir.glob("*.json"):
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM document_cache 
                        WHERE data_file_path = ?
                    """, (str(cache_file),))
                    
                    if cursor.fetchone()[0] == 0:
                        try:
                            cache_file.unlink()
                            orphaned_files += 1
                        except Exception as e:
                            logger.warning(f"Could not remove orphaned cache file {cache_file}: {e}")
            
            # Vacuum database to reclaim space
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
            
            logger.info(f"🔧 Cache optimization complete: removed {orphaned_files} orphaned files, compacted database")
            return orphaned_files
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
            return 0

    def reset_session_stats(self):
        """Reset session statistics."""
        self._session_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "time_saved": 0.0,
            "files_processed": 0
        } 