import chromadb
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "newsgroups_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

def load_and_clean_data():
    """
    Fetches the 20 Newsgroups dataset and removes noisy metadata.
    """
    print("Fetching and cleaning 20 Newsgroups dataset...")
    # We deliberately remove 'headers', 'footers', and 'quotes' to ensure 
    # the embeddings capture semantic meaning rather than metadata or reply chains.
    dataset = fetch_20newsgroups(
        subset='all', 
        remove=('headers', 'footers', 'quotes')
    )
    
    documents = dataset.data
    labels = dataset.target
    target_names = dataset.target_names
    
    # Filter out empty documents or extremely short noise (e.g., just punctuation)
    cleaned_docs = []
    cleaned_labels = []
    
    for doc, label in zip(documents, labels):
        # Keep documents with at least 20 characters to ensure meaningful embeddings
        if doc and len(doc.strip()) > 20:
            cleaned_docs.append(doc.strip())
            cleaned_labels.append(target_names[label])
            
    print(f"Retained {len(cleaned_docs)} viable documents out of {len(documents)}.")
    return cleaned_docs, cleaned_labels

def setup_vector_database(documents, labels):
    """
    Embeds the documents and stores them in a local ChromaDB instance.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Create or reset the collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Existing collection deleted. Starting fresh.")
    except Exception:
        pass
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    print("Embedding and indexing documents (this may take a few minutes)...")
    for i in tqdm(range(0, len(documents), BATCH_SIZE)):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_labels = labels[i : i + BATCH_SIZE]
        
        batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_docs))]
        
        batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=[{"true_label": label} for label in batch_labels]
        )
        
    print("Vector database setup complete!")

if __name__ == "__main__":
    docs, lbls = load_and_clean_data()
    setup_vector_database(docs, lbls)