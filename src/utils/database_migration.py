"""
Database Migration Utility
Handles database schema updates for cache managers.
"""

import sqlite3
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def migrate_compliance_cache_db(db_path: str):
    """Migrate compliance cache database to include new columns."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Check if compliance_score column exists
            cursor = conn.execute("PRAGMA table_info(compliance_cache)")
            columns = [row[1] for row in cursor.fetchall()]
            
            migrations_needed = []
            
            if 'model_version' not in columns:
                migrations_needed.append(
                    "ALTER TABLE compliance_cache ADD COLUMN model_version TEXT DEFAULT 'v1'"
                )
            
            if 'workflow_version' not in columns:
                migrations_needed.append(
                    "ALTER TABLE compliance_cache ADD COLUMN workflow_version TEXT DEFAULT 'v1'"
                )
            
            if 'compliance_score' not in columns:
                migrations_needed.append(
                    "ALTER TABLE compliance_cache ADD COLUMN compliance_score REAL DEFAULT 0.0"
                )
            
            # Execute migrations
            for migration in migrations_needed:
                conn.execute(migration)
                logger.info(f"Executed migration: {migration}")
            
            # Add new indexes if they don't exist
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_processed_at 
                    ON compliance_cache(processed_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_score 
                    ON compliance_cache(compliance_score)
                """)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
            
            conn.commit()
            
            if migrations_needed:
                logger.info(f"Compliance cache database migrated successfully: {len(migrations_needed)} changes applied")
            else:
                logger.info("Compliance cache database is already up to date")
                
    except Exception as e:
        logger.error(f"Error migrating compliance cache database: {e}")
        raise

def migrate_document_cache_db(db_path: str):
    """Migrate document cache database to include new columns."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Check if new columns exist
            cursor = conn.execute("PRAGMA table_info(document_cache)")
            columns = [row[1] for row in cursor.fetchall()]
            
            migrations_needed = []
            
            if 'model_version' not in columns:
                migrations_needed.append(
                    "ALTER TABLE document_cache ADD COLUMN model_version TEXT DEFAULT 'v1'"
                )
            
            if 'workflow_version' not in columns:
                migrations_needed.append(
                    "ALTER TABLE document_cache ADD COLUMN workflow_version TEXT DEFAULT 'v1'"
                )
            
            # Execute migrations
            for migration in migrations_needed:
                conn.execute(migration)
                logger.info(f"Executed migration: {migration}")
            
            # Add new indexes if they don't exist
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_extraction_success 
                    ON document_cache(extraction_success)
                """)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
            
            conn.commit()
            
            if migrations_needed:
                logger.info(f"Document cache database migrated successfully: {len(migrations_needed)} changes applied")
            else:
                logger.info("Document cache database is already up to date")
                
    except Exception as e:
        logger.error(f"Error migrating document cache database: {e}")
        raise

def migrate_all_cache_databases(cache_dir: str = ".cache"):
    """Migrate all cache databases in the cache directory."""
    cache_path = Path(cache_dir)
    
    # Migrate document cache
    doc_cache_db = cache_path / "document_cache.db"
    if doc_cache_db.exists():
        logger.info(f"Migrating document cache database: {doc_cache_db}")
        migrate_document_cache_db(str(doc_cache_db))
    
    # Migrate compliance cache
    compliance_cache_db = cache_path / "compliance_cache.db"
    if compliance_cache_db.exists():
        logger.info(f"Migrating compliance cache database: {compliance_cache_db}")
        migrate_compliance_cache_db(str(compliance_cache_db))
    
    logger.info("All cache database migrations completed")

# Entry point removed - use from within application or tests only 