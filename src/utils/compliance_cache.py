"""
Compliance Results Cache Manager
Caches compliance check results for identical document sets and rules.
"""

import hashlib
import json
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ComplianceCacheManager:
    """Manages caching of compliance check results to avoid expensive re-evaluations."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # SQLite database for compliance cache
        self.db_path = self.cache_dir / "compliance_cache.db"
        self.init_database()
        
        # JSON storage for compliance results
        self.results_dir = self.cache_dir / "compliance_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database for compliance cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_cache (
                    cache_key TEXT PRIMARY KEY,
                    documents_hash TEXT NOT NULL,
                    rules_hash TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    rules_count INTEGER NOT NULL,
                    processed_at TIMESTAMP NOT NULL,
                    processing_time REAL NOT NULL,
                    results_file_path TEXT NOT NULL,
                    compliance_summary TEXT
                )
            """)
            
            # Indexes for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_hash 
                ON compliance_cache(documents_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rules_hash 
                ON compliance_cache(rules_hash)
            """)
    
    def _generate_documents_hash(self, documents: List[Dict[str, Any]]) -> str:
        """Generate hash for document set based on extracted data."""
        # Sort documents by filename for consistent hashing
        sorted_docs = sorted(documents, key=lambda x: x.get('filename', ''))
        
        # Create hash based on relevant document data
        hash_data = []
        for doc in sorted_docs:
            doc_hash_data = {
                'filename': doc.get('filename', ''),
                'doc_type': doc.get('doc_type', ''),
                'extracted_data': doc.get('extracted_data', {})
            }
            hash_data.append(doc_hash_data)
        
        # Generate SHA-256 hash
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _generate_rules_hash(self, rules_content: str) -> str:
        """Generate hash for rules content."""
        # Normalize rules content (remove extra whitespace, etc.)
        normalized_rules = '\n'.join(line.strip() for line in rules_content.strip().split('\n') if line.strip())
        return hashlib.sha256(normalized_rules.encode()).hexdigest()
    
    def _generate_cache_key(self, documents_hash: str, rules_hash: str) -> str:
        """Generate unique cache key for document set + rules combination."""
        combined = f"{documents_hash}:{rules_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def is_cached_and_valid(self, documents: List[Dict[str, Any]], rules_content: str, max_age_hours: int = 24) -> bool:
        """Check if compliance results are cached and still valid."""
        try:
            documents_hash = self._generate_documents_hash(documents)
            rules_hash = self._generate_rules_hash(rules_content)
            cache_key = self._generate_cache_key(documents_hash, rules_hash)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT processed_at, results_file_path
                    FROM compliance_cache 
                    WHERE cache_key = ?
                """, (cache_key,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                processed_at_str, results_file = result
                
                # Check if results file exists
                if not os.path.exists(results_file):
                    logger.info(f"Compliance cache results file missing: {results_file}")
                    return False
                
                # Check age
                processed_at = datetime.fromisoformat(processed_at_str)
                age_hours = (datetime.now() - processed_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    logger.info(f"Compliance cache expired (age: {age_hours:.1f}h > {max_age_hours}h)")
                    return False
                
                logger.info(f"✅ Compliance cache HIT (age: {age_hours:.1f}h)")
                return True
                
        except Exception as e:
            logger.error(f"Error checking compliance cache: {e}")
            return False
    
    def get_cached_results(self, documents: List[Dict[str, Any]], rules_content: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached compliance results."""
        try:
            documents_hash = self._generate_documents_hash(documents)
            rules_hash = self._generate_rules_hash(rules_content)
            cache_key = self._generate_cache_key(documents_hash, rules_hash)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT results_file_path, processing_time, compliance_summary
                    FROM compliance_cache 
                    WHERE cache_key = ?
                """, (cache_key,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                results_file, processing_time, compliance_summary = result
                
                # Load the cached results
                with open(results_file, 'r', encoding='utf-8') as f:
                    cached_results = json.load(f)
                
                logger.info(f"📁 Loaded cached compliance results (saved {processing_time:.2f}s)")
                logger.info(f"📊 Summary: {compliance_summary}")
                return cached_results
                
        except Exception as e:
            logger.error(f"Error retrieving cached compliance results: {e}")
            return None
    
    def cache_compliance_results(self, documents: List[Dict[str, Any]], rules_content: str, 
                               compliance_results: Dict[str, Any], processing_time: float):
        """Cache compliance check results."""
        try:
            documents_hash = self._generate_documents_hash(documents)
            rules_hash = self._generate_rules_hash(rules_content)
            cache_key = self._generate_cache_key(documents_hash, rules_hash)
            
            # Create unique filename for cached results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"compliance_{cache_key[:16]}_{timestamp}.json"
            results_file_path = self.results_dir / results_filename
            
            # Save compliance results
            with open(results_file_path, 'w', encoding='utf-8') as f:
                json.dump(compliance_results, f, indent=2, default=str)
            
            # Generate compliance summary
            findings = compliance_results.get('aggregated_compliance_findings', [])
            compliant_count = len([f for f in findings if f.get('status', '').lower() in ['compliant', 'pass']])
            total_rules = len(findings)
            compliance_summary = f"{compliant_count}/{total_rules} rules passed"
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO compliance_cache 
                    (cache_key, documents_hash, rules_hash, document_count, rules_count,
                     processed_at, processing_time, results_file_path, compliance_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    documents_hash,
                    rules_hash,
                    len(documents),
                    total_rules,
                    datetime.now().isoformat(),
                    processing_time,
                    str(results_file_path),
                    compliance_summary
                ))
            
            logger.info(f"💾 Cached compliance results (processing time: {processing_time:.2f}s)")
            logger.info(f"📊 Summary: {compliance_summary}")
            
        except Exception as e:
            logger.error(f"Error caching compliance results: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get compliance cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_cached,
                        SUM(processing_time) as total_time_saved,
                        AVG(processing_time) as avg_processing_time,
                        AVG(document_count) as avg_document_count,
                        AVG(rules_count) as avg_rules_count
                    FROM compliance_cache
                """)
                
                result = cursor.fetchone()
                if result:
                    return {
                        "total_cached_compliance_checks": result[0],
                        "total_compliance_time_saved": round(result[1] or 0, 2),
                        "average_compliance_time": round(result[2] or 0, 2),
                        "average_documents_per_check": round(result[3] or 0, 1),
                        "average_rules_per_check": round(result[4] or 0, 1),
                        "cache_directory": str(self.cache_dir),
                        "database_size_mb": round(os.path.getsize(self.db_path) / 1024 / 1024, 2)
                    }
        except Exception as e:
            logger.error(f"Error getting compliance cache stats: {e}")
        
        return {}
    
    def clear_cache(self, older_than_hours: int = None):
        """Clear compliance cache entries, optionally only those older than specified hours."""
        try:
            import os
            with sqlite3.connect(self.db_path) as conn:
                if older_than_hours:
                    cutoff_date = datetime.now() - timedelta(hours=older_than_hours)
                    
                    # Get files to delete
                    cursor = conn.execute("""
                        SELECT results_file_path FROM compliance_cache 
                        WHERE processed_at < ?
                    """, (cutoff_date.isoformat(),))
                    
                    files_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    # Delete database entries
                    cursor = conn.execute("""
                        DELETE FROM compliance_cache WHERE processed_at < ?
                    """, (cutoff_date.isoformat(),))
                    
                    deleted_count = cursor.rowcount
                    
                else:
                    # Clear all
                    cursor = conn.execute("SELECT results_file_path FROM compliance_cache")
                    files_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    conn.execute("DELETE FROM compliance_cache")
                    deleted_count = cursor.rowcount
                
                # Delete results files
                for file_path in files_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not delete compliance cache file {file_path}: {e}")
                
                logger.info(f"🗑️ Cleared {deleted_count} compliance cache entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error clearing compliance cache: {e}")
            return 0 