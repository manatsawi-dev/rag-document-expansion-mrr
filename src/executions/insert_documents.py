import os
import time
import json
from pathlib import Path

from src.embeddings.text_embeddings import dense_embedding, sparse_embedding
from src.retrieval.qdrant_store import add_embedding
from src.models.generated_document import (
    GeneratedDocument,
    ExpandedDocument,
    QueryDocument,
)
from src.generation.google_gemini_ai import gemini_generate_content
from src.utils.file_utils import write_file, read_jsonl_file

import src.constants.constants as constants


DATASET_DIR = Path("data/data_set")
EXPANDED_DATA_DIR = Path("data/expanded")
EXPANDED_FILE_NAME = "expanded_documents.jsonl"
MAX_QUERY_PER_DOCUMENT = 5
RATE_LIMIT_DELAY = 5


def load_and_combine_documents():
    files = [
        f
        for f in os.listdir(DATASET_DIR)
        if os.path.isfile(os.path.join(DATASET_DIR, f)) and f.endswith(".jsonl")
    ]

    documents = []
    for file in files:
        with open(os.path.join(DATASET_DIR, file), "r") as f:
            for line in f:
                documents.append(line.strip())
    return documents


def generate_expanded_documents(documents):
    # If the expanded file already exists, load it
    expanded_file_path = f"{EXPANDED_DATA_DIR}/{EXPANDED_FILE_NAME}"
    if os.path.exists(expanded_file_path):
        data = read_jsonl_file(expanded_file_path)
        expanded_documents = [ExpandedDocument(**item) for item in data]
        return expanded_documents

    # If the expanded file does not exist, generate it via Gemini
    expanded_documents = []
    for document in documents:
        doc = GeneratedDocument.model_validate_json(document)

        prompt = f"""Generate query prediction for {MAX_QUERY_PER_DOCUMENT} queries (Doc2Query technique) for the document: {document}

        When answering
        - Do not include any other text
"""
        response = gemini_generate_content(prompt, response_schema=list[QueryDocument])

        queries: list[QueryDocument] = []
        json_data = json.loads(response)
        queries = [QueryDocument(**item) for item in json_data]

        query_texts = [query.query for query in queries]
        expanded_documents.append(
            ExpandedDocument(
                original_text=doc.original_text,
                title=doc.title,
                expanded_text=", ".join(query_texts),
            )
        )

        print(f"Progressing: {len(expanded_documents)}/{len(documents)}")
        time.sleep(RATE_LIMIT_DELAY)

    jsonl_content = "\n".join(doc.model_dump_json() for doc in expanded_documents)

    # Save the expanded documents to a file
    write_file(EXPANDED_DATA_DIR, EXPANDED_FILE_NAME, jsonl_content)

    return expanded_documents


def insert_original_documents():
    collection_name = constants.ORIGINAL_TEXT_COLLECTION_NAME

    documents = load_and_combine_documents()

    for idx, document in enumerate(documents):
        doc = GeneratedDocument.model_validate_json(document)

        embedding_vector = dense_embedding(doc.original_text)
        metadata = {
            "title": doc.title,
        }

        add_embedding(
            collection_name=collection_name,
            vector=embedding_vector,
            original_text=doc.original_text,
            expanded_text="",
            metadata=metadata,
        )

        print(
            f"Inserted document into collection: {collection_name} : Progress {idx+1}/{len(documents)}"
        )


def insert_original_and_expanded_documents():
    collection_name = constants.ORIGINAL_TEXT_AND_EXPANDED_COLLECTION_NAME

    documents = load_and_combine_documents()

    expanded_documents = generate_expanded_documents(documents)

    for idx, document in enumerate(expanded_documents):
        text = f"{document.original_text} {document.expanded_text}"
        embedding_vector = dense_embedding(text)
        metadata = {
            "title": document.title,
        }

        add_embedding(
            collection_name=collection_name,
            vector=embedding_vector,
            original_text=document.original_text,
            expanded_text=document.expanded_text,
            metadata=metadata,
        )

        print(
            f"Inserted document into collection: {collection_name} : Progress {idx+1}/{len(expanded_documents)}"
        )


def insert_expanded_documents():
    collection_name = constants.ONLY_EXPANDED_COLLECTION_NAME

    documents = load_and_combine_documents()

    expanded_documents = generate_expanded_documents(documents)

    for idx, document in enumerate(expanded_documents):
        embedding_vector = dense_embedding(document.expanded_text)
        metadata = {
            "title": document.title,
        }

        add_embedding(
            collection_name=collection_name,
            vector=embedding_vector,
            original_text=document.original_text,
            expanded_text=document.expanded_text,
            metadata=metadata,
        )

        print(
            f"Inserted document into collection: {collection_name} : Progress {idx+1}/{len(expanded_documents)}"
        )
