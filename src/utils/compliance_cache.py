"""
Compliance Results Cache Manager
Caches compliance check results for identical document sets and rules.
Enhanced for LangGraph workflow integration with performance optimizations.
"""

import hashlib
import json
import sqlite3
import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
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
        
        # Performance tracking
        self._session_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
            "checks_performed": 0
        }
    
    def init_database(self):
        """Initialize SQLite database for compliance cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            # Create the main table
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
                    compliance_summary TEXT,
                    model_version TEXT DEFAULT 'v1',
                    workflow_version TEXT DEFAULT 'v1',
                    compliance_score REAL DEFAULT 0.0
                )
            """)
            
            # Create indexes safely - check if columns exist first
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_hash 
                    ON compliance_cache(documents_hash)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rules_hash 
                    ON compliance_cache(rules_hash)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_processed_at 
                    ON compliance_cache(processed_at)
                """)
                
                # Check if compliance_score column exists before creating index
                cursor = conn.execute("PRAGMA table_info(compliance_cache)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if "compliance_score" in columns:
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_compliance_score 
                        ON compliance_cache(compliance_score)
                    """)
                else:
                    logger.info("Compliance cache: compliance_score column not found, skipping index creation")
                    
            except Exception as e:
                logger.warning(f"Error creating compliance cache indexes: {e}")
                # Continue without indexes if there's an issue
    
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

    def _calculate_compliance_score(self, compliance_results: Dict[str, Any]) -> float:
        """Calculate compliance score (percentage of rules passed)."""
        findings = compliance_results.get('aggregated_compliance_findings', [])
        if not findings:
            return 0.0
        
        compliant_count = len([f for f in findings if f.get('status', '').lower() in ['compliant', 'pass']])
        return (compliant_count / len(findings)) * 100.0
    
    def is_cached_and_valid(self, documents: List[Dict[str, Any]], rules_content: str, 
                          max_age_hours: int = 24, model_version: str = "v1", workflow_version: str = "v1") -> bool:
        """Check if compliance results are cached and still valid."""
        try:
            documents_hash = self._generate_documents_hash(documents)
            rules_hash = self._generate_rules_hash(rules_content)
            cache_key = self._generate_cache_key(documents_hash, rules_hash)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT processed_at, results_file_path, model_version, workflow_version
                    FROM compliance_cache 
                    WHERE cache_key = ?
                """, (cache_key,))
                
                result = cursor.fetchone()
                if not result:
                    self._session_stats["cache_misses"] += 1
                    return False
                
                processed_at_str, results_file, cached_model_ver, cached_workflow_ver = result
                
                # Check if results file exists
                if not os.path.exists(results_file):
                    logger.info(f"Compliance cache results file missing: {results_file}")
                    self._session_stats["cache_misses"] += 1
                    return False

                # Check version compatibility
                if cached_model_ver != model_version or cached_workflow_ver != workflow_version:
                    logger.info(f"❌ Compliance cache MISS (version mismatch)")
                    self._session_stats["cache_misses"] += 1
                    return False
                
                # Check age
                processed_at = datetime.fromisoformat(processed_at_str)
                age_hours = (datetime.now() - processed_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    logger.info(f"❌ Compliance cache expired (age: {age_hours:.1f}h > {max_age_hours}h)")
                    self._session_stats["cache_misses"] += 1
                    return False
                
                logger.info(f"✅ Compliance cache HIT (age: {age_hours:.1f}h)")
                self._session_stats["cache_hits"] += 1
                return True
                
        except Exception as e:
            logger.error(f"Error checking compliance cache: {e}")
            self._session_stats["cache_misses"] += 1
            return False
    
    def get_cached_results(self, documents: List[Dict[str, Any]], rules_content: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached compliance results."""
        try:
            documents_hash = self._generate_documents_hash(documents)
            rules_hash = self._generate_rules_hash(rules_content)
            cache_key = self._generate_cache_key(documents_hash, rules_hash)
            
            with sqlite3.connect(self.db_path) as conn:
                # Check if compliance_score and compliance_summary columns exist
                cursor = conn.execute("PRAGMA table_info(compliance_cache)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if "compliance_score" in columns and "compliance_summary" in columns:
                    cursor = conn.execute("""
                        SELECT results_file_path, processing_time, compliance_summary, compliance_score
                        FROM compliance_cache 
                        WHERE cache_key = ?
                    """, (cache_key,))
                else:
                    # Fallback for older schema
                    cursor = conn.execute("""
                        SELECT results_file_path, processing_time
                        FROM compliance_cache 
                        WHERE cache_key = ?
                    """, (cache_key,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                if len(result) >= 4:
                    results_file, processing_time, compliance_summary, compliance_score = result
                    logger.info(f"📊 Summary: {compliance_summary} ({compliance_score:.1f}% compliant)")
                else:
                    results_file, processing_time = result[:2]
                    compliance_summary = "N/A"
                    compliance_score = 0.0
                
                # Check if results file exists
                if not os.path.exists(results_file):
                    logger.warning(f"Compliance cache results file missing: {results_file}")
                    return None
                
                # Load the cached results
                with open(results_file, 'r', encoding='utf-8') as f:
                    cached_results = json.load(f)
                
                # Validate that the loaded results is a dictionary
                if not isinstance(cached_results, dict):
                    logger.warning(f"Invalid compliance cache format - expected dict, got {type(cached_results)}")
                    return None
                
                # Track time saved
                self._session_stats["total_time_saved"] += processing_time
                
                logger.info(f"📁 Loaded cached compliance results (saved {processing_time:.2f}s)")
                return cached_results
                
        except Exception as e:
            logger.error(f"Error retrieving cached compliance results: {e}")
            return None
    
    def cache_compliance_results(self, documents: List[Dict[str, Any]], rules_content: str, 
                               compliance_results: Dict[str, Any], processing_time: float,
                               model_version: str = "v1", workflow_version: str = "v1"):
        """Cache compliance check results with enhanced metadata."""
        try:
            documents_hash = self._generate_documents_hash(documents)
            rules_hash = self._generate_rules_hash(rules_content)
            cache_key = self._generate_cache_key(documents_hash, rules_hash)
            
            # Create unique filename for cached results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_key_short = cache_key[:16]
            results_filename = f"compliance_{cache_key_short}_{timestamp}.json"
            results_file_path = self.results_dir / results_filename
            
            # Save compliance results
            with open(results_file_path, 'w', encoding='utf-8') as f:
                json.dump(compliance_results, f, indent=2, default=str)
            
            # Generate compliance summary and score
            findings = compliance_results.get('aggregated_compliance_findings', [])
            compliant_count = len([f for f in findings if f.get('status', '').lower() in ['compliant', 'pass']])
            total_rules = len(findings)
            compliance_summary = f"{compliant_count}/{total_rules} rules passed"
            compliance_score = self._calculate_compliance_score(compliance_results)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO compliance_cache 
                    (cache_key, documents_hash, rules_hash, document_count, rules_count,
                     processed_at, processing_time, results_file_path, compliance_summary,
                     model_version, workflow_version, compliance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    documents_hash,
                    rules_hash,
                    len(documents),
                    total_rules,
                    datetime.now().isoformat(),
                    processing_time,
                    str(results_file_path),
                    compliance_summary,
                    model_version,
                    workflow_version,
                    compliance_score
                ))
            
            self._session_stats["checks_performed"] += 1
            logger.info(f"💾 Cached compliance results (processing time: {processing_time:.2f}s)")
            logger.info(f"📊 Summary: {compliance_summary} ({compliance_score:.1f}% compliant)")
            
        except Exception as e:
            logger.error(f"Error caching compliance results: {e}")

    async def bulk_cache_check(self, document_rule_pairs: List[Tuple[List[Dict[str, Any]], str]],
                             max_age_hours: int = 24, model_version: str = "v1", workflow_version: str = "v1") -> List[bool]:
        """Efficiently check cache status for multiple document-rule combinations."""
        try:
            cache_keys = []
            for documents, rules_content in document_rule_pairs:
                documents_hash = self._generate_documents_hash(documents)
                rules_hash = self._generate_rules_hash(rules_content)
                cache_key = self._generate_cache_key(documents_hash, rules_hash)
                cache_keys.append(cache_key)
            
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ','.join('?' * len(cache_keys))
                cursor = conn.execute(f"""
                    SELECT cache_key, processed_at, results_file_path, model_version, workflow_version
                    FROM compliance_cache 
                    WHERE cache_key IN ({placeholders})
                """, cache_keys)
                
                cached_results = {row[0]: row[1:] for row in cursor.fetchall()}
                
                results = []
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                
                for i, cache_key in enumerate(cache_keys):
                    if cache_key not in cached_results:
                        results.append(False)
                        continue
                    
                    processed_at_str, results_file, cached_model_ver, cached_workflow_ver = cached_results[cache_key]
                    
                    # Check file existence, version compatibility, and age
                    if (os.path.exists(results_file) and
                        cached_model_ver == model_version and
                        cached_workflow_ver == workflow_version and
                        datetime.fromisoformat(processed_at_str) > cutoff_time):
                        results.append(True)
                    else:
                        results.append(False)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in bulk compliance cache check: {e}")
            return [False] * len(document_rule_pairs)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get compliance cache statistics including session performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_cached,
                        SUM(processing_time) as total_time_saved,
                        AVG(processing_time) as avg_processing_time,
                        AVG(document_count) as avg_document_count,
                        AVG(rules_count) as avg_rules_count,
                        AVG(compliance_score) as avg_compliance_score,
                        COUNT(CASE WHEN processed_at > datetime('now', '-24 hours') THEN 1 END) as recent_cached
                    FROM compliance_cache
                """)
                
                result = cursor.fetchone()
                base_stats = {}
                if result:
                    base_stats = {
                        "total_cached_compliance_checks": result[0],
                        "total_compliance_time_saved": round(result[1] or 0, 2),
                        "average_compliance_time": round(result[2] or 0, 2),
                        "average_documents_per_check": round(result[3] or 0, 1),
                        "average_rules_per_check": round(result[4] or 0, 1),
                        "average_compliance_score": round(result[5] or 0, 1),
                        "recent_cached_24h": result[6],
                        "cache_directory": str(self.cache_dir),
                        "database_size_mb": round(os.path.getsize(self.db_path) / 1024 / 1024, 2) if os.path.exists(self.db_path) else 0
                    }
                
                # Add session statistics
                total_checks = self._session_stats["cache_hits"] + self._session_stats["cache_misses"]
                hit_rate = (self._session_stats["cache_hits"] / total_checks * 100) if total_checks > 0 else 0
                
                base_stats.update({
                    "session_cache_hits": self._session_stats["cache_hits"],
                    "session_cache_misses": self._session_stats["cache_misses"],
                    "compliance_cache_hit_rate": round(hit_rate, 1),
                    "session_time_saved": round(self._session_stats["total_time_saved"], 2),
                    "checks_performed_this_session": self._session_stats["checks_performed"]
                })
                
                return base_stats
                
        except Exception as e:
            logger.error(f"Error getting compliance cache stats: {e}")
        
        return self._session_stats.copy()

    def get_compliance_history(self, document_hashes: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical compliance results for analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if document_hashes:
                    placeholders = ','.join('?' * len(document_hashes))
                    cursor = conn.execute(f"""
                        SELECT documents_hash, rules_hash, processed_at, compliance_summary, 
                               compliance_score, document_count, rules_count
                        FROM compliance_cache 
                        WHERE documents_hash IN ({placeholders})
                        ORDER BY processed_at DESC 
                        LIMIT ?
                    """, document_hashes + [limit])
                else:
                    cursor = conn.execute("""
                        SELECT documents_hash, rules_hash, processed_at, compliance_summary, 
                               compliance_score, document_count, rules_count
                        FROM compliance_cache 
                        ORDER BY processed_at DESC 
                        LIMIT ?
                    """, (limit,))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        "documents_hash": row[0][:16],  # Shortened for display
                        "rules_hash": row[1][:16],
                        "processed_at": row[2],
                        "compliance_summary": row[3],
                        "compliance_score": row[4],
                        "document_count": row[5],
                        "rules_count": row[6]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting compliance history: {e}")
            return []
    
    def clear_cache(self, older_than_hours: int = None, low_compliance_only: bool = False, 
                   compliance_threshold: float = 50.0):
        """Clear compliance cache entries with enhanced filtering options."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if low_compliance_only:
                    # Clear only low compliance score entries
                    cursor = conn.execute("""
                        SELECT results_file_path FROM compliance_cache 
                        WHERE compliance_score < ?
                    """, (compliance_threshold,))
                    files_to_delete = [row[0] for row in cursor.fetchall()]
                    
                    cursor = conn.execute("""
                        DELETE FROM compliance_cache WHERE compliance_score < ?
                    """, (compliance_threshold,))
                    deleted_count = cursor.rowcount
                    
                elif older_than_hours:
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
                
                action_desc = (f"low compliance entries (<{compliance_threshold}%)" if low_compliance_only 
                             else f"entries older than {older_than_hours} hours" if older_than_hours 
                             else "all entries")
                logger.info(f"🗑️ Cleared {deleted_count} compliance cache {action_desc}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error clearing compliance cache: {e}")
            return 0

    def optimize_cache(self):
        """Optimize cache by removing orphaned files and compacting database."""
        try:
            orphaned_files = 0
            
            # Find and remove orphaned compliance result files
            for result_file in self.results_dir.glob("*.json"):
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM compliance_cache 
                        WHERE results_file_path = ?
                    """, (str(result_file),))
                    
                    if cursor.fetchone()[0] == 0:
                        try:
                            result_file.unlink()
                            orphaned_files += 1
                        except Exception as e:
                            logger.warning(f"Could not remove orphaned compliance result file {result_file}: {e}")
            
            # Vacuum database to reclaim space
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
            
            logger.info(f"🔧 Compliance cache optimization complete: removed {orphaned_files} orphaned files, compacted database")
            return orphaned_files
            
        except Exception as e:
            logger.error(f"Error optimizing compliance cache: {e}")
            return 0

    def reset_session_stats(self):
        """Reset session statistics."""
        self._session_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
            "checks_performed": 0
        } 