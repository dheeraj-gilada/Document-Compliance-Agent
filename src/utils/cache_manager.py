"""
Document Processing Cache Manager
Implements intelligent caching to avoid reprocessing unchanged documents.
"""

import os
import json
import hashlib
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

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
                    extraction_success BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash 
                ON document_cache(file_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed_at 
                ON document_cache(processed_at)
            """)
            
    def _get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
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
    
    def is_cached_and_valid(self, file_path: str) -> bool:
        """Check if document is cached and cache is still valid."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_hash, file_size, last_modified, data_file_path
                    FROM document_cache 
                    WHERE file_path = ?
                """, (file_path,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                cached_hash, cached_size, cached_modified, data_file = result
                
                # Check if data file exists
                if not os.path.exists(data_file):
                    logger.info(f"Cache data file missing for {file_path}")
                    return False
                
                # Get current file metadata
                current_meta = self._get_file_metadata(file_path)
                if not current_meta:
                    return False
                
                # Validate cache
                is_valid = (
                    current_meta["hash"] == cached_hash and
                    current_meta["size"] == cached_size and
                    abs(current_meta["modified"] - cached_modified) < 1.0  # 1 second tolerance
                )
                
                if is_valid:
                    logger.info(f"✅ Cache HIT for {os.path.basename(file_path)}")
                else:
                    logger.info(f"❌ Cache MISS for {os.path.basename(file_path)} (file changed)")
                
                return is_valid
                
        except Exception as e:
            logger.error(f"Error checking cache for {file_path}: {e}")
            return False
    
    def get_cached_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached document data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data_file_path, doc_type, processing_time
                    FROM document_cache 
                    WHERE file_path = ?
                """, (file_path,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                data_file, doc_type, processing_time = result
                
                # Load the cached data
                with open(data_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                logger.info(f"📁 Loaded cached data for {os.path.basename(file_path)} (saved {processing_time:.2f}s)")
                return cached_data
                
        except Exception as e:
            logger.error(f"Error retrieving cached data for {file_path}: {e}")
            return None
    
    def cache_document_data(self, file_path: str, document_data: Dict[str, Any], processing_time: float):
        """Cache processed document data."""
        try:
            # Get file metadata
            file_meta = self._get_file_metadata(file_path)
            if not file_meta:
                logger.error(f"Cannot cache {file_path} - failed to get metadata")
                return
            
            # Create unique filename for cached data
            file_hash = file_meta["hash"][:16]  # First 16 chars of hash
            filename = os.path.basename(file_path)
            data_filename = f"{filename}_{file_hash}.json"
            data_file_path = self.data_dir / data_filename
            
            # Save document data
            with open(data_file_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, default=str)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO document_cache 
                    (file_path, file_hash, file_size, last_modified, processed_at, 
                     doc_type, processing_time, data_file_path, extraction_success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path,
                    file_meta["hash"],
                    file_meta["size"],
                    file_meta["modified"],
                    datetime.now().isoformat(),
                    document_data.get("doc_type", "unknown"),
                    processing_time,
                    str(data_file_path),
                    document_data.get("status") == "data_extracted"
                ))
            
            logger.info(f"💾 Cached data for {filename} (processing time: {processing_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error caching document data for {file_path}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_cached,
                        SUM(processing_time) as total_time_saved,
                        AVG(processing_time) as avg_processing_time,
                        COUNT(CASE WHEN extraction_success = 1 THEN 1 END) as successful_extractions
                    FROM document_cache
                """)
                
                result = cursor.fetchone()
                if result:
                    return {
                        "total_cached_documents": result[0],
                        "total_processing_time_saved": round(result[1] or 0, 2),
                        "average_processing_time": round(result[2] or 0, 2),
                        "successful_extractions": result[3],
                        "cache_directory": str(self.cache_dir),
                        "database_size_mb": round(os.path.getsize(self.db_path) / 1024 / 1024, 2)
                    }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
        
        return {}
    
    def clear_cache(self, older_than_days: int = None):
        """Clear cache entries, optionally only those older than specified days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if older_than_days:
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
                
                logger.info(f"🗑️ Cleared {deleted_count} cache entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0 