import numpy as np
import chromadb
import hdbscan
import umap
import joblib
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "newsgroups_collection"
MODELS_DIR = "./models"

def fetch_embeddings_from_db():
    """Retrieves all document embeddings from the ChromaDB collection."""
    print("Connecting to ChromaDB to fetch embeddings...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    db_data = collection.get(include=["embeddings", "documents", "metadatas"])
    
    embeddings = np.array(db_data["embeddings"])
    ids = db_data["ids"]
    print(f"Retrieved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
    
    return embeddings, ids, db_data["documents"], db_data["metadatas"]

def train_fuzzy_clusters(embeddings):
    """
    Trains UMAP for dimensionality reduction and HDBSCAN for fuzzy clustering.
    Justification for design: HDBSCAN algorithmically determines the optimal 
    number of clusters based on density, fulfilling the requirement for evidence-based k.
    """
    print("Step 1: Running UMAP Dimensionality Reduction...")
    # Reduce 384 dimensions to 15. This makes density-based clustering mathematically stable.
    umap_model = umap.UMAP(
        n_neighbors=15, 
        n_components=15, 
        metric='cosine', 
        random_state=42
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    print("Step 2: Fitting HDBSCAN for Fuzzy Clustering...")
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=50,
        metric='euclidean',
        prediction_data=True ,
        core_dist_n_jobs=1 
    )
    hdbscan_model.fit(reduced_embeddings)
    
    soft_clusters = hdbscan.all_points_membership_vectors(hdbscan_model)
    
    num_clusters = soft_clusters.shape[1]
    print(f"Evidence-based clustering complete! Discovered {num_clusters} latent semantic clusters.")
    
    return umap_model, hdbscan_model, soft_clusters

def calculate_cluster_densities(embeddings: np.ndarray, soft_clusters: np.ndarray) -> dict:
    """
    Calculates the base semantic threshold for each cluster based on its density.
    """
    print("Calculating cluster densities for dynamic cache thresholds...")
    
    hard_labels = np.argmax(soft_clusters, axis=1)
    unique_clusters = np.unique(hard_labels)
    
    cluster_densities = {}
    
    for cluster_id in unique_clusters:
        if cluster_id == -1 or cluster_id == 999:
            continue
            
        # 1. Get all embeddings belonging to this cluster
        cluster_embeddings = embeddings[hard_labels == cluster_id]
        
        if len(cluster_embeddings) == 0:
            continue
            
        # 2. Calculate the Centroid (mean vector)
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        
        # 3. Calculate how tightly packed the documents are around the center
        similarities = cosine_similarity(cluster_embeddings, centroid)
        avg_similarity = float(np.mean(similarities))
        
        # Set the baseline threshold slightly below the cluster's average internal similarity
        base_threshold = avg_similarity - 0.02 
        
        # USABILITY SAFEGUARDS
        # While mathematically avg_similarity (and hence base_threshold) can't exceed 1.0, we bound it for following reasons:
        # 1. The Floor (0.75): Prevents "garbage hits" in extremely sparse/broad clusters.
        #    Even if a topic is messy, queries still need a baseline semantic match to trigger a hit.
        # 2. The Ceiling (0.92): Prevents the "perfect match" trap in hyper-dense clusters.
        #    Ensures the cache forgives minor paraphrasing, rather than demanding 0.98+ exact string matches.
        base_threshold = min(0.92, max(0.75, base_threshold))
        
        cluster_densities[int(cluster_id)] = round(base_threshold, 4)
        
    print(f"Calculated densities for {len(cluster_densities)} clusters.")
    return cluster_densities

def save_models(umap_model, hdbscan_model, cluster_desnsities):
    """Persists the trained models so the FastAPI service can load them quickly."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("Saving models to disk...")
    joblib.dump(umap_model, os.path.join(MODELS_DIR, "umap_model.joblib"))
    joblib.dump(hdbscan_model, os.path.join(MODELS_DIR, "hdbscan_model.joblib"))
    joblib.dump(cluster_densities, os.path.join(MODELS_DIR, "cluster_densities.joblib"))
    print("Models saved successfully!")

if __name__ == "__main__":
    # 1. Fetch the data we processed
    embeddings, ids, docs, metadata = fetch_embeddings_from_db()
    
    # 2. Train the fuzzy clustering pipeline
    umap_model, hdbscan_model, soft_clusters = train_fuzzy_clusters(embeddings)
    cluster_densities = calculate_cluster_densities(embeddings, soft_clusters)
    
    # 3. Save the models
    save_models(umap_model, hdbscan_model, cluster_densities)
    
    # Print a sample distribution
    print("\nSample Fuzzy Distribution for Document 0:")
    print(np.round(soft_clusters[0], 3))