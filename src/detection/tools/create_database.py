import chromadb
import os
import shutil
from datetime import datetime
import uuid

def create_database(
    file_path: str,
    db_dir: str = "chroma_db",
    collection_name: str = "documents",
    backup: bool = True,
) -> str:
    """
    Create a vector database from a local file.
    
    Args:
        file_path: Path to the file to be ingested
        db_dir: Directory to store the database
        collection_name: Name of the collection
        backup: Whether to backup existing database
        
    Returns:
        Path to the created database
    """
    # Backup if needed
    if os.path.exists(db_dir) and backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{db_dir}_backup_{timestamp}"
        shutil.copytree(db_dir, backup_dir)
        print(f"Backup the old database to {backup_dir}")
        
    # Create client
    client = chromadb.PersistentClient(path=db_dir)
    
    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Get the existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(name=collection_name)
        print(f"Create new collection: {collection_name}")
    
    # Read the file content
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Unable to find file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content into chunks (simple splitting by newlines for example)
    chunks = content.split("\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Generate IDs for documents
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Add documents to the collection
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[{"source": file_path} for _ in chunks]
    )
    
    print(f"Add {len(chunks)} document chunks to the collection.")
    return db_dir

if __name__ == "__main__":
    file_path = "../rag_dataset/phi-4_liberty2.txt"
    chroma_db_dir = "../chroma_db"
    collection_name = "documents"
    backup = False

    create_database(file_path, db_dir=chroma_db_dir, collection_name=collection_name, backup=backup)