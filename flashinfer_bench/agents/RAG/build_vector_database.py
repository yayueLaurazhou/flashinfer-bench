import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, CollectionSchema, FieldSchema, DataType, Function, FunctionType
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def truncate_long_documents(documents: List[Dict[str, Any]], max_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Truncate documents longer than max_length into 2 files.
    Returns list of documents with long ones split in half.

    Args:
        documents: List of document dictionaries
        max_length: Maximum length threshold for truncation

    Returns:
        List of documents with long ones split into two parts
    """
    truncated_docs = []
    for doc in documents:
        content = doc["text"]
        if len(content) > max_length:
            # Split into two parts
            mid_point = len(content) // 2
            # Find a good split point (look for newline near midpoint)
            split_idx = content.rfind('\n', mid_point - 100, mid_point + 100)
            if split_idx == -1:
                split_idx = mid_point

            # Create first document
            doc1 = doc.copy()
            doc1["text"] = content[:split_idx]
            doc1["metadata"] = doc["metadata"].copy()
            doc1["metadata"]["part"] = "1/2"
            doc1["id"] = doc["id"] + "_part1"

            # Create second document
            doc2 = doc.copy()
            doc2["text"] = content[split_idx:]
            doc2["metadata"] = doc["metadata"].copy()
            doc2["metadata"]["part"] = "2/2"
            doc2["id"] = doc["id"] + "_part2"

            truncated_docs.extend([doc1, doc2])
        else:
            truncated_docs.append(doc)

    logger.info(f"Document truncation: {len(documents)} docs -> {len(truncated_docs)} docs")
    return truncated_docs


class VectorDatabaseBuilder:
    """Builder class for creating and populating a Milvus vector database."""

    def __init__(
        self,
        cuda_folder: str,
        db_path: str = "./milvus_lite.db",
        collection_name: str = "cuda_documentation",
        model_name: str = "nvidia/NV-Embed-v2",
        max_doc_length: int = 5000
    ):
        """
        Initialize the vector database builder using Milvus Lite (no Docker required).

        Args:
            cuda_folder: Path to the cuda folder containing markdown files
            db_path: Local path for Milvus Lite database file (default: ./milvus_lite.db)
            collection_name: Name of the collection to create
            model_name: Model name for embeddings (default: nvidia/nv-embed-v2)
            max_doc_length: Maximum length for document truncation
        
        """
        self.cuda_folder = Path(cuda_folder)
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.max_doc_length = max_doc_length

        logger.info(f"Initializing Milvus Lite with database: {db_path}")
        self.client = MilvusClient(uri=db_path)

        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def get_markdown_files(self) -> List[Path]:
        """
        Recursively get all markdown files from cuda folder, excluding index.json files.

        Returns:
            List of Path objects pointing to markdown files
        """
        md_files = []
        for file_path in self.cuda_folder.rglob("*.md"):
            # Skip index.json references, only process .md files
            if file_path.suffix == ".md":
                md_files.append(file_path)

        logger.info(f"Found {len(md_files)} markdown files")
        return md_files

    def read_markdown_file(self, file_path: Path) -> Dict[str, str]:
        """
        Read a markdown file and extract its content.

        Args:
            file_path: Path to the markdown file

        Returns:
            Dictionary with file metadata and content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relative_path = file_path.relative_to(self.cuda_folder)
            return {
                "path": str(relative_path),
                "content": content,
                "file_name": file_path.name,
            }
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    def create_documents(self) -> List[Dict[str, Any]]:
        """
        Create document objects from all markdown files.
        Automatically truncates long documents

        Returns:
            List of document dictionaries with id, text, and metadata
        """
        documents = []
        md_files = self.get_markdown_files()

        for idx, file_path in enumerate(md_files):
            file_data = self.read_markdown_file(file_path)
            if file_data:
                # Create a unique ID based on file path
                doc_id = hashlib.md5(file_data["path"].encode()).hexdigest()

                documents.append(
                    {
                        "id": doc_id,
                        "text": file_data["content"],
                        "metadata": {
                            "path": file_data["path"],
                            "file_name": file_data["file_name"],
                            "source": "cuda_documentation",
                        },
                    }
                )

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} files")

        logger.info(f"Created {len(documents)} documents (before truncation)")

        # Truncate long documents
        documents = truncate_long_documents(documents, max_length=self.max_doc_length)

        logger.info(f"Final document count: {len(documents)}")
        return documents

    def setup_collection(self):
        """Create or recreate the Milvus collection with hybrid search support (dense + BM25 sparse)."""
        # Drop existing collection if it exists
        if self.client.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)

        # Define schema with both dense and sparse vector fields
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)

        # Primary key
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        # Text field for BM25
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        # Dense embedding vector
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        # Sparse vector (auto-populated by BM25 function)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # Metadata fields
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=256)

        # Add BM25 function to auto-generate sparse vectors from text
        bm25_function = Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # Create collection with schema
        logger.info(f"Creating hybrid collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # Create indexes for both vector fields
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="FLAT",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        logger.info("Hybrid collection created with dense (COSINE) + sparse (BM25) indexes")

    def generate_embeddings(self, texts: List[str], batch_size: int = 4) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation

        Returns:
            List of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Generated embeddings for {min(i + batch_size, len(texts))} texts")

        return embeddings

    def build_database(self, batch_size: int = 4):
        """
        Build the complete vector database.

        Args:
            batch_size: Batch size for embedding generation and insertion
        """
        logger.info("Starting database build process...")

        # Setup collection
        self.setup_collection()

        # Create documents
        documents = self.create_documents()

        if not documents:
            logger.warning("No documents found to index")
            return

        # Extract texts and prepare data for insertion
        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)

        # Insert data into Milvus in batches
        logger.info(f"Inserting {len(documents)} documents into Milvus...")
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_texts = texts[i:batch_end]
            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]

            try:
                self.client.insert(
                    collection_name=self.collection_name,
                    data=[
                        {
                            "id": batch_ids[j],
                            "text": batch_texts[j],
                            "dense_vector": batch_embeddings[j],
                            "path": batch_metadatas[j]["path"],
                            "file_name": batch_metadatas[j]["file_name"],
                            "source": batch_metadatas[j]["source"],
                        }
                        for j in range(len(batch_ids))
                    ],
                )

                if (i + batch_size) % (batch_size * 10) == 0 or batch_end == len(documents):
                    logger.info(f"Inserted {batch_end}/{len(documents)} documents")
            except Exception as e:
                logger.error(f"Error inserting batch {i}-{batch_end}: {e}")

        logger.info("Database build complete!")
        logger.info(f"Collection '{self.collection_name}' now contains {len(documents)} documents")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense (semantic) and sparse (BM25 keyword) retrieval
        with Reciprocal Rank Fusion (RRF) for result merging.

        Args:
            query: Query text
            top_k: Number of final results to return

        Returns:
            List of search results with scores and metadata
        """
        # Generate dense embedding for query
        query_embedding = self.model.encode(query).tolist()

        # Dense search request (semantic similarity)
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
        )

        # Sparse search request (BM25 keyword matching)
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=top_k,
        )

        # Hybrid search with RRF fusion
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=["path", "file_name", "source"],
        )

        formatted_results = []
        for result in results[0]:
            formatted_results.append(
                {
                    "id": result.id,
                    "distance": result.distance,
                    "path": result.fields.get("path", ""),
                    "file_name": result.fields.get("file_name", ""),
                    "source": result.fields.get("source", ""),
                }
            )

        return formatted_results


def main():
    """Main function to build the vector database."""
    # Get the cuda folder path
    current_dir = Path(__file__).parent
    cuda_folder = current_dir / "cuda"

    if not cuda_folder.exists():
        logger.error(f"Cuda folder not found at {cuda_folder}")
        return

    # Create builder and build database
    db_path = current_dir / "milvus_lite.db"
    builder = VectorDatabaseBuilder(
        cuda_folder=str(cuda_folder),
        db_path=str(db_path),
        collection_name="cuda_documentation",
        model_name="nvidia/nv-embed-v2",
    )

    builder.build_database(batch_size=1)

    # Example search
    logger.info("\nExample search:")
    query = "GPU memory management"
    results = builder.search(query, top_k=3)
    logger.info(f"Search results for '{query}':")
    for i, result in enumerate(results, 1):
        logger.info(
            f"  {i}. {result['path']} (distance: {result['distance']:.4f})"
        )


if __name__ == "__main__":
    main()
