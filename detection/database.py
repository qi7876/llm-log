import chromadb
import os

class VectorDatabase:
    """Vector database for RAG applications using ChromaDB."""
    
    def __init__(self, db_dir: str = "chroma_db", collection_name: str = "documents"):
        """
        Initialize the vector database from a local directory.
        
        Args:
            db_dir: Directory of the database
            collection_name: Name of the collection
        """
        if not os.path.exists(db_dir):
            raise FileNotFoundError(f"Unable to find database directory: {db_dir}")
        
        self.client = chromadb.PersistentClient(path=db_dir)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(f"Unable to load collection: {collection_name}: {e}")
            
        print(f"Loaded database form {db_dir}")
        
    def query(self, query_text: str, n_results: int = 5) -> str:
        """
        Query the database for similar content.
        
        Args:
            query_text: Text to query
            n_results: Number of results to return
            
        Returns:
            String formatted for LLM consumption, or empty string if no results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            return ""
            
        # Format results for LLM
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if 'distances' in results else "N/A"
            metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
            
            source = metadata.get('source', 'Unknown')
            formatted_doc = f"Document {i+1} (Distance: {distance}):\nSource: {source}\n{doc}\n"
            formatted_results.append(formatted_doc)
            
        return "\n".join(formatted_results)
