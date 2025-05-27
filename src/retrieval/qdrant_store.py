from qdrant_client import QdrantClient
from qdrant_client.http import models
from uuid import uuid4

import src.constants.constants as constants

# Create client to connect to Qdrant server running through Docker
qdrant_client = QdrantClient(host=constants.QDRANT_HOST, port=constants.QDRANT_PORT)


def generate_id():
    # Generate ID for Qdrant
    # ID must be unique in collection
    # Can use UUID or hash function to create unique ID
    return str(uuid4())


# Function to create collection in Qdrant
def create_collection(collection_name: str):
    try:
        # Check if collection already exists
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        # If collection doesn't exist, create new one
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                distance=models.Distance.COSINE,
                size=constants.VECTOR_SIZE,
            ),
        )


# Function to delete collection in Qdrant
def delete_collection(collection_name: str):
    try:
        # Check if collection already exists
        qdrant_client.get_collection(collection_name=collection_name)
        # If exists, delete collection
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        # If collection doesn't exist, do nothing
        pass


# Function to upload embedded text to Qdrant
# We will store original text, metadata, and embedding vector together
# So that the data can be used in the future
def add_embedding(
    collection_name: str,
    vector: list,
    original_text: str,
    expanded_text: str,
    metadata: dict,
):
    # Create collection if it doesn't exist yet
    create_collection(collection_name)

    # Add data to Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=generate_id(),
                vector=vector,
                payload={
                    "original_text": original_text,
                    "expanded_text": expanded_text,
                    "metadata": metadata,
                },
            )
        ],
    )


def get_all_embedding(collection_name: str, limit: int = 10):
    try:
        # Check if collection already exists
        qdrant_client.get_collection(collection_name=collection_name)
        # If exists, retrieve all data
        response = qdrant_client.scroll(
            collection_name=collection_name,
            limit=limit,
        )
        return response
    except Exception as e:
        # If collection doesn't exist, do nothing
        print(f"Error getting data from Qdrant: {e}")
        return []


def similarity_search(collection_name: str, query_vector: list, limit: int = 10):
    try:
        # Check if collection already exists
        qdrant_client.get_collection(collection_name=collection_name)

        # If exists, search for similar data
        response = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_vectors=False,
            with_payload=True,
        )
        return response
    except Exception as e:
        # If collection doesn't exist, do nothing
        print(f"Error searching data in Qdrant: {e}")
        return []
