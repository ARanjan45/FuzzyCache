import os
import joblib
import numpy as np
import chromadb
import hdbscan
import umap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

CHROMA_DB_DIR = "./chroma_db"
MODELS_DIR = "./models"
PLOT_OUTPUT = "cluster_analysis.png"

def load_data_and_models():
    print("Loading database and models...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name="newsgroups_collection")
    db_data = collection.get(include=["embeddings", "documents", "metadatas"])
    
    embeddings = np.array(db_data["embeddings"])
    documents = db_data["documents"]
    
    hdbscan_model = joblib.load(os.path.join(MODELS_DIR, "hdbscan_model.joblib"))
    umap_model = joblib.load(os.path.join(MODELS_DIR, "umap_model.joblib"))
    cluster_densities = joblib.load(os.path.join(MODELS_DIR, "cluster_densities.joblib"))
    
    return embeddings, documents, umap_model, hdbscan_model, cluster_densities

def analyze_and_plot(embeddings, documents, umap_model, hdbscan_model, cluster_densities):
    print("Generating 2D projection for visualization...")
    reducer_2d = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
    embeddings_2d = reducer_2d.fit_transform(embeddings)
    
    print("Calculating fuzzy distributions and uncertainties...")
    soft_clusters = hdbscan.all_points_membership_vectors(hdbscan_model)
    
    dominant_clusters = np.argmax(soft_clusters, axis=1)
    max_probabilities = np.max(soft_clusters, axis=1)
    
    # Identify boundary cases (highest probability < 40%)
    uncertain_mask = max_probabilities < 0.40
    uncertain_indices = np.where(uncertain_mask)[0]
    
    # --- 1. Generate the Visual Plot ---
    plt.figure(figsize=(12, 8))
    
    plt.scatter(
        embeddings_2d[~uncertain_mask, 0], 
        embeddings_2d[~uncertain_mask, 1], 
        c=dominant_clusters[~uncertain_mask], 
        cmap='tab20', 
        s=10, 
        alpha=0.7,
        label='Confident Assignments'
    )
    
    plt.scatter(
        embeddings_2d[uncertain_mask, 0], 
        embeddings_2d[uncertain_mask, 1], 
        c='black', 
        s=15, 
        alpha=0.9,
        marker='x',
        label='Boundary / Uncertain Cases'
    )
    
    plt.title("2D UMAP Projection of Semantic Clusters with Boundary Cases Highlighted")
    plt.legend()
    plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"✅ Visual plot saved to {PLOT_OUTPUT}")
    
    # --- 2. Prove Semantic Meaning (Text Analysis) ---
    print("\n" + "="*60)
    print("CLUSTER SEMANTIC ANALYSIS REPORT")
    print("="*60)
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    unique_clusters = np.unique(dominant_clusters)
    
    print(f"Total Documents: {len(documents)}")
    print(f"Noise Documents (Unclustered): {np.sum(dominant_clusters == 0) if 0 in unique_clusters and max_probabilities[np.where(dominant_clusters==0)[0][0]] == 0.0 else 'Calculated dynamically'}")
    print(f"Total Boundary/Uncertain Cases: {len(uncertain_indices)}\n")

    # Analyze clusters
    for cluster_id in unique_clusters:
        if cluster_id == -1 or cluster_id == 999: continue
            
        cluster_docs_idx = np.where(dominant_clusters == cluster_id)[0]
        if len(cluster_docs_idx) == 0: continue
            
        avg_tfidf = np.asarray(X_tfidf[cluster_docs_idx].mean(axis=0)).flatten()
        top_word_indices = avg_tfidf.argsort()[-5:][::-1]
        top_words = feature_names[top_word_indices]
        
        density = cluster_densities.get(int(cluster_id), "N/A")
        
        print(f"Cluster {cluster_id} | Size: {len(cluster_docs_idx)} docs | Base Cache Threshold: {density}")
        print(f"Top Keywords: {', '.join(top_words)}")
        # Print first 100 characters of a sample document, replacing newlines for clean output
        sample_doc = documents[cluster_docs_idx[0]][:100].replace('\n', ' ')
        print(f"Sample: \"{sample_doc}...\"\n")

    # --- 3. Boundary Case Deep Dive ---
    print("="*60)
    print("GENUINE UNCERTAINTY: BOUNDARY CASE ANALYSIS")
    print("="*60)
    
    # Print up to 3 uncertain cases
    for i in range(min(3, len(uncertain_indices))):
        idx = uncertain_indices[i]
        doc_text = documents[idx][:200].replace('\n', ' ')
        
        # Get top 3 cluster probabilities for this document
        probs = soft_clusters[idx]
        top_3_idx = np.argsort(probs)[-3:][::-1]
        
        print(f"\nBoundary Document {i+1}:")
        print(f"Text Snippet: \"{doc_text}...\"")
        print("Fuzzy Distribution:")
        for c_idx in top_3_idx:
            if probs[c_idx] > 0.01:
                print(f"  -> {probs[c_idx]*100:.1f}% match for Cluster {c_idx}")

if __name__ == "__main__":
    emb, docs, u_mod, h_mod, densities = load_data_and_models()
    analyze_and_plot(emb, docs, u_mod, h_mod, densities)