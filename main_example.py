from src.embeddings.text_embeddings import (
    dense_embedding,
    dense_embedding_list,
    sparse_embedding,
    sparse_embedding_list,
)

from src.retrieval.qdrant_store import (
    get_all_embedding,
    add_embedding,
    qdrant_client,
    similarity_search,
)
from src.generation.google_gemini_ai import gemini_generate_content
from src.models.example import ExampleModel
import src.constants.constants as constants

example_documents = [
    "How to request vacation leave?",
    "Company leave policy explained.",
    "Guide to submitting a leave request form.",
]


def test_dense_embedding():
    print("Testing dense embedding...")
    for doc in example_documents:
        print(f"Document: {doc}")
        print(f"Dense embedding: {dense_embedding(doc)}")


def test_dense_embedding_list():
    print("Testing dense embedding list...")
    embeddings = dense_embedding_list(example_documents)
    for doc, embedding in zip(example_documents, embeddings):
        print(f"Document: {doc}")
        print(f"Dense embedding: {embedding}")


def test_sparse_embedding():
    print("Testing sparse embedding...")
    for doc in example_documents:
        print(f"Document: {doc}")
        print(f"Sparse embedding: {sparse_embedding(doc)}")


def test_sparse_embedding_list():
    print("Testing sparse embedding list...")
    embeddings = sparse_embedding_list(example_documents)
    for doc, embedding in zip(example_documents, embeddings):
        print(f"Document: {doc}")
        print(f"Sparse embedding: {embedding}")


# ทดสอบการเชื่อมต่อกับ Qdrant
def test_qdrant_connection():
    try:
        # ตรวจสอบการเชื่อมต่อ
        qdrant_client.get_collections()
        print("Qdrant connection is successful.")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")


# ทดสอบการเพิ่มข้อมูลลงใน Qdrant
def test_insert_documents():
    embedding = dense_embedding(example_documents[2])
    add_embedding(
        constants.TEST_COLLECTION_NAME,
        embedding,
        example_documents[2],
        {"doc_info": "Example document info"},
    )


# ทดสอบการดึงข้อมูลทั้งหมดจาก Qdrant
def test_get_all_embedding():
    print(f"All embeddings in {constants.TEST_COLLECTION_NAME}:")
    data = get_all_embedding(constants.TEST_COLLECTION_NAME)
    print(data)


# ทดสอบการค้นหาความคล้ายคลึง (Similarity search)
def test_similarity_search():
    query = "How to request vacation leave?"
    query_vector = dense_embedding(query)
    search_result = similarity_search(constants.TEST_COLLECTION_NAME, query_vector)
    print(f"Similarity search result for query : {query}")
    print("Search results:")
    print(search_result)


# ทดสอบการสร้างเนื้อหาจาก LLM
# โดยใช้ Google Gemini
def test_llm():
    query = "Who is Albert Einstein"
    response = gemini_generate_content(query)
    print(f"LLM response for query '{query}': {response}")


def test_llm_with_model():
    query = "Who is Albert Einstein and His/Her characteristics?"
    response = gemini_generate_content(query, response_schema=ExampleModel)

    example_parser = ExampleModel.model_validate_json(response)
    print(f"Name: {example_parser.name}")
    print(f"Answer: {example_parser.answer}")
    print(f"Characteristics: {example_parser.characteristics}")


def main():
    print("Running main.py...")
    # Uncomment the tests you want to run
    # test_dense_embedding()
    # test_dense_embedding_list()
    # test_sparse_embedding()
    # test_sparse_embedding_list()

    # test_qdrant_connection()
    # test_insert_documents()
    # test_get_all_embedding()
    # test_similarity_search()
    test_llm_with_model()


if __name__ == "__main__":
    main()
