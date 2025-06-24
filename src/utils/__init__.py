from .document_loader import DocumentLoader
from .extractor import DocumentExtractor
from .cache_manager import DocumentCacheManager
from .compliance_cache import ComplianceCacheManager

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CacheStatusManager:
    """Unified cache status and management utility for the document compliance system."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.doc_cache = DocumentCacheManager(cache_dir)
        self.compliance_cache = ComplianceCacheManager(cache_dir)
        
    def get_unified_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from both cache systems."""
        try:
            doc_stats = self.doc_cache.get_cache_stats()
            compliance_stats = self.compliance_cache.get_cache_stats()
            
            # Calculate combined metrics
            total_time_saved = (
                doc_stats.get("total_processing_time_saved", 0) + 
                compliance_stats.get("total_compliance_time_saved", 0)
            )
            
            session_time_saved = (
                doc_stats.get("estimated_time_saved", 0) + 
                compliance_stats.get("session_time_saved", 0)
            )
            
            total_cache_hits = (
                doc_stats.get("session_cache_hits", 0) + 
                compliance_stats.get("session_cache_hits", 0)
            )
            
            total_cache_misses = (
                doc_stats.get("session_cache_misses", 0) + 
                compliance_stats.get("session_cache_misses", 0)
            )
            
            total_checks = total_cache_hits + total_cache_misses
            overall_hit_rate = (total_cache_hits / total_checks * 100) if total_checks > 0 else 0
            
            return {
                "overall_performance": {
                    "total_time_saved_seconds": round(total_time_saved, 2),
                    "session_time_saved_seconds": round(session_time_saved, 2),
                    "overall_cache_hit_rate": round(overall_hit_rate, 1),
                    "total_cache_hits": total_cache_hits,
                    "total_cache_misses": total_cache_misses
                },
                "document_cache": doc_stats,
                "compliance_cache": compliance_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting unified cache stats: {e}")
            return {}
    
    def print_cache_summary(self):
        """Print a formatted summary of cache performance."""
        stats = self.get_unified_stats()
        
        if not stats:
            print("❌ Unable to retrieve cache statistics")
            return
        
        overall = stats.get("overall_performance", {})
        doc_cache = stats.get("document_cache", {})
        compliance_cache = stats.get("compliance_cache", {})
        
        print("\n" + "="*60)
        print("🚀 CACHE PERFORMANCE SUMMARY")
        print("="*60)
        
        # Overall performance
        print(f"📊 Overall Performance:")
        print(f"   • Total Time Saved: {overall.get('total_time_saved_seconds', 0):.1f}s")
        print(f"   • Session Time Saved: {overall.get('session_time_saved_seconds', 0):.1f}s")
        print(f"   • Overall Hit Rate: {overall.get('overall_cache_hit_rate', 0):.1f}%")
        print()
        
        # Document cache
        print(f"📁 Document Cache:")
        print(f"   • Cached Documents: {doc_cache.get('total_cached_documents', 0)}")
        print(f"   • Session Hit Rate: {doc_cache.get('cache_hit_rate', 0):.1f}%")
        print(f"   • Time Saved: {doc_cache.get('total_processing_time_saved', 0):.1f}s")
        print(f"   • Recent (24h): {doc_cache.get('recent_cached_24h', 0)} documents")
        print()
        
        # Compliance cache
        print(f"⚖️ Compliance Cache:")
        print(f"   • Cached Checks: {compliance_cache.get('total_cached_compliance_checks', 0)}")
        print(f"   • Session Hit Rate: {compliance_cache.get('compliance_cache_hit_rate', 0):.1f}%")
        print(f"   • Time Saved: {compliance_cache.get('total_compliance_time_saved', 0):.1f}s")
        print(f"   • Avg Compliance Score: {compliance_cache.get('average_compliance_score', 0):.1f}%")
        print(f"   • Recent (24h): {compliance_cache.get('recent_cached_24h', 0)} checks")
        print()
        
        # Storage info
        doc_db_size = doc_cache.get('database_size_mb', 0)
        compliance_db_size = compliance_cache.get('database_size_mb', 0)
        total_db_size = doc_db_size + compliance_db_size
        
        print(f"💾 Storage:")
        print(f"   • Document DB: {doc_db_size:.2f} MB")
        print(f"   • Compliance DB: {compliance_db_size:.2f} MB")
        print(f"   • Total Cache Size: {total_db_size:.2f} MB")
        print("="*60)
    
    def optimize_all_caches(self):
        """Optimize both cache systems."""
        print("🔧 Optimizing caches...")
        
        doc_orphaned = self.doc_cache.optimize_cache()
        compliance_orphaned = self.compliance_cache.optimize_cache()
        
        total_orphaned = doc_orphaned + compliance_orphaned
        print(f"✅ Cache optimization complete: removed {total_orphaned} total orphaned files")
        
        return total_orphaned
    
    def clear_all_caches(self, older_than_days: int = None):
        """Clear both cache systems with the same age criteria."""
        print(f"🗑️ Clearing caches...")
        
        doc_cleared = self.doc_cache.clear_cache(older_than_days=older_than_days)
        compliance_cleared = self.compliance_cache.clear_cache(
            older_than_hours=older_than_days * 24 if older_than_days else None
        )
        
        total_cleared = doc_cleared + compliance_cleared
        age_desc = f" older than {older_than_days} days" if older_than_days else ""
        print(f"✅ Cleared {total_cleared} total cache entries{age_desc}")
        
        return total_cleared
    
    def reset_session_stats(self):
        """Reset session statistics for both cache systems."""
        self.doc_cache.reset_session_stats()
        self.compliance_cache.reset_session_stats()
        logger.info("🔄 Session statistics reset for both cache systems")

__all__ = [
    'DocumentLoader', 
    'DocumentExtractor', 
    'DocumentCacheManager', 
    'ComplianceCacheManager',
    'CacheStatusManager'
]
