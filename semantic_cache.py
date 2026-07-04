import numpy as np
from collections import OrderedDict
from typing import Dict, List, Any

class DynamicSemanticCache:
    def __init__(self, capacity: int = 1000, gamma: float = 1.0, default_base_threshold: float = 0.85, cluster_densities: dict = None):
        """
        Initializes the semantic cache from first principles using Approach 1.
        
        Args:
            capacity: Maximum number of entries before LRU eviction kicks in.
            gamma: The core TUNABLE PARAMETER. Scales the base cluster thresholds up or down.
            default_base_threshold: Fallback threshold if a cluster's density is unknown.
            cluster_densities: Precomputed base thresholds per cluster based on their variance.
        """
        self.capacity = capacity
        self.gamma = gamma 
        self.default_base_threshold = default_base_threshold
        self.cluster_densities = cluster_densities or {}
        
        # OrderedDict provides O(1) LRU eviction.
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Cluster partitions for O(1) lookup routing. Maps cluster_id -> List of query strings
        self.partitions: Dict[int, List[str]] = {}
        
        # API Stats tracking
        self.total_hits = 0
        self.total_misses = 0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def check_cache(self, query_text: str, query_vector: np.ndarray, target_clusters: list[int]) -> tuple[bool, dict]:
        """
        Checks the cache using Nucleus Sampling (Top-p) partitioned lookups.
        Aggregates cached queries from multiple clusters if the model is uncertain.
        """
        # 1. Aggregate candidates from all target clusters
        candidate_queries = []
        for cluster_id in target_clusters:
            if cluster_id in self.partitions:
                candidate_queries.extend(self.partitions[cluster_id])
                
        candidate_queries = list(set(candidate_queries))

        # If all target buckets are empty, it's an automatic miss
        if not candidate_queries:
            self.total_misses += 1
            return False, {}

        similarities = []
        # 2. Calculate similarity against this dynamically sized pool
        for cached_q in candidate_queries:
            cached_vector = self.cache[cached_q]['vector']
            sim = self._cosine_similarity(query_vector, cached_vector)
            similarities.append((sim, cached_q))

        best_sim, best_query = max(similarities, key=lambda x: x[0])

        # 3. Calculate Dynamic Threshold based on Cluster Density
        best_query_cluster = self.cache[best_query]['cluster']
        base_threshold = self.cluster_densities.get(best_query_cluster, self.default_base_threshold)
        
        dynamic_threshold = base_threshold * self.gamma
        
        # Cap it to a logical maximum so it never exceeds 1.0
        dynamic_threshold = min(0.95, dynamic_threshold)

        # 4. Evaluate Hit or Miss
        if best_sim >= dynamic_threshold:
            self.total_hits += 1
            self.cache.move_to_end(best_query)
            return True, {
                "matched_query": best_query,
                "similarity_score": round(float(best_sim), 4),
                "result": self.cache[best_query]['result'],
                "dynamic_threshold_used": round(float(dynamic_threshold), 4)
            }
            
        self.total_misses += 1
        return False, {}

    def add_to_cache(self, query_text: str, query_vector: np.ndarray, result: str, dominant_cluster: int):
        """
        Adds a new computed result to the cache, managing LRU capacity.
        """
        # Prevent identical strings from overwriting keys and inflating miss count
        if query_text in self.cache:
            return 
            
        # LRU Eviction: If full, pop the first (oldest/least recently used) item
        if len(self.cache) >= self.capacity:
            oldest_query, oldest_data = self.cache.popitem(last=False)
            old_cluster = oldest_data['cluster']
            # Clean up the partition reference
            if old_cluster in self.partitions and oldest_query in self.partitions[old_cluster]:
                self.partitions[old_cluster].remove(oldest_query)

        # Insert new data
        self.cache[query_text] = {
            'vector': query_vector,
            'result': result,
            'cluster': dominant_cluster
        }
        
        # Update partition routing
        if dominant_cluster not in self.partitions:
            self.partitions[dominant_cluster] = []
        if query_text not in self.partitions[dominant_cluster]:
            self.partitions[dominant_cluster].append(query_text)

    def get_stats(self) -> dict:
        """Returns required statistics for the API endpoint."""
        total_requests = self.total_hits + self.total_misses
        hit_rate = self.total_hits / total_requests if total_requests > 0 else 0.0
        return {
            "total_entries": len(self.cache),
            "hit_count": self.total_hits,
            "miss_count": self.total_misses,
            "hit_rate": round(hit_rate, 3)
        }
        
    def flush(self):
        """Clears the cache entirely."""
        self.cache.clear()
        self.partitions.clear()
        self.total_hits = 0
        self.total_misses = 0
