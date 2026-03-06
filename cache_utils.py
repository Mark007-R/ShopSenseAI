"""
Caching utilities for performance optimization.
Handles caching of similarity matrices, recommendations, and expensive computations.
"""

import pickle
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


class CachedMatrix:
    """Caches large matrices to disk to avoid recomputation."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, key: str) -> Path:
        """Generate consistent cache file path."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl"
    
    def get(self, key: str) -> Any:
        """Retrieve cached matrix."""
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache retrieval failed for {key}: {e}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache matrix to disk."""
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Cache write failed for {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Cache deletion failed: {e}")


class RecommendationCache:
    """LRU-style cache for recommendation results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Any:
        """Get value from recommendation cache."""
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in recommendation cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO for now)
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': round(hit_rate, 2)
        }


def timed_cache(func: Callable) -> Callable:
    """Decorator for timing and caching function calls."""
    import time
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.4f}s")
        return result
    
    return wrapper


# Global cache instances
matrix_cache = CachedMatrix()
recommendation_cache = RecommendationCache()
