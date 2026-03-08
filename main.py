import joblib
import numpy as np
import hdbscan
import chromadb
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from semantic_cache import DynamicSemanticCache

app_state = {}

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None = None
    similarity_score: float | None = None
    result: str
    dominant_cluster: int

class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Loading models into memory...")
    
    app_state["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    app_state["collection"] = client.get_collection(name="newsgroups_collection")
    
    app_state["umap_model"] = joblib.load("./models/umap_model.joblib")
    app_state["hdbscan_model"] = joblib.load("./models/hdbscan_model.joblib")
    
    cluster_densities = joblib.load("./models/cluster_densities.joblib")
    
    app_state["cache"] = DynamicSemanticCache(
        capacity=1000, 
        gamma=1.0, # The Tunable Parameter
        default_base_threshold=0.85,
        cluster_densities=cluster_densities
    )
    
    print("All systems loaded successfully!")
    yield
    print("Shutting down gracefully...")
    app_state.clear()

app = FastAPI(title="Trademarkia Semantic Search", lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Accepts a JSON body: { "query": "<natural language query>" }[cite: 41, 42].
    Embeds the query, checks the semantic cache, and returns the result[cite: 41, 42].
    """
    query_text = request.query
    query_vector = app_state["embedder"].encode(query_text)
    reduced_vector = app_state["umap_model"].transform([query_vector])
    
    # 1. Predict the distribution
    _, _ = hdbscan.approximate_predict(
        app_state["hdbscan_model"], 
        reduced_vector
    )
    distribution = hdbscan.membership_vector(app_state["hdbscan_model"], reduced_vector)[0]
    
    # 2. Nucleus Sampling (Top-p) Logic 
    target_p = 0.85 
    sorted_indices = np.argsort(distribution)[::-1] 
    
    target_clusters = []
    cumulative_prob = 0.0
    
    for idx in sorted_indices:
        target_clusters.append(int(idx))
        cumulative_prob += distribution[idx]
        if cumulative_prob >= target_p:
            break
            
    # API contract requires a single integer for the dominant cluster
    dominant_cluster = target_clusters[0] if sum(distribution) > 0 else 999
    
    # Ensure dominant cluster is in the search pool
    if dominant_cluster not in target_clusters:
        target_clusters.append(dominant_cluster)
    
    # 3. Check Semantic Cache 
    cache = app_state["cache"]
    is_hit, hit_data = cache.check_cache(query_text, query_vector, target_clusters)
    
    if is_hit:
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=hit_data["matched_query"],
            similarity_score=hit_data["similarity_score"],
            result=hit_data["result"],
            dominant_cluster=dominant_cluster
        )
        
    # 4. On Miss: Compute Result from Vector DB
    db_results = app_state["collection"].query(
        query_embeddings=[query_vector.tolist()],
        n_results=1 
    )
    
    if db_results['documents'] and db_results['documents'][0]:
        computed_result = db_results['documents'][0][0]
    else:
        computed_result = "No relevant context found in the corpus."
        
    # 5. On a miss, compute and store the result before returning[cite: 52].
    cache.add_to_cache(query_text, query_vector, computed_result, dominant_cluster)
    
    return QueryResponse(
        query=query_text,
        cache_hit=False,
        result=computed_result,
        dominant_cluster=dominant_cluster
    )

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Returns current cache state statistics[cite: 53, 54]."""
    stats = app_state["cache"].get_stats()
    return CacheStatsResponse(**stats)

@app.delete("/cache")
async def flush_cache():
    """Flushes the cache entirely and resets all stats."""
    app_state["cache"].flush()
    return {"message": "Cache flushed successfully"}