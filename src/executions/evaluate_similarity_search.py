import time
import json

from src.utils.document_utils import (
    load_and_combine_jsonl_documents,
    format_documents_as_context,
)
from src.generation.google_gemini_ai import gemini_generate_content
from src.retrieval.qdrant_store import similarity_search
from src.embeddings.text_embeddings import dense_embedding
from src.models.generated_document import QuestionDocument
from src.models.mrr_document import MRRResult, MRRDataset, Relevance
from src.utils.file_utils import write_file

QUESTIONS_PATH = "data/question"
MRR_RESULTS_PATH = "data/mrr_results/preprocessing"
RATE_LIMIT = 5


def evaluate_similarity_search(
    collection_name: str,
):

    questions = load_and_combine_jsonl_documents(QUESTIONS_PATH)

    mrr_results: list[MRRDataset] = []

    for idx, question in enumerate(questions):

        question_text = QuestionDocument.model_validate_json(question).question

        query_vector = dense_embedding(question_text)
        similarity_results = similarity_search(
            collection_name=collection_name, query_vector=query_vector
        )

        formatted_documents = format_documents_as_context(similarity_results)

        prompt = f"""
Based on the following query and context, please analyze and evaluate the relevance of each document. 

Evaluation instructions:
1. Examine each document in the context separately and document number is provided in the context
2. For each document, determine if it contains information relevant to answering the query
3. Provide a result for each document indicating its relevance by using the following scale:
    relevant - It is relevant to the query
    not_relevant - It is not relevant to the query
    ambiguous - It is unclear if it is relevant to the query

query: {question_text}
context: {formatted_documents}
"""
        response_list: list[MRRResult] = []
        retrieved: list[str] = []
        relevant: set[str] = set()

        response = gemini_generate_content(
            prompt,
            response_schema=list[MRRResult],
        )

        data = json.loads(response)

        parsed_response = [MRRResult(**item) for item in data]
        response_list.extend(parsed_response)

        sorted_response_list = sorted(response_list, key=lambda x: x.document_number)

        for _, response in enumerate(sorted_response_list):
            doc_name = f"doc_{response.document_number}"
            retrieved.append(doc_name)
            if response.relevance == Relevance.RELEVANT:
                relevant.add(doc_name)

        mrr_results.append(
            MRRDataset(
                query_id=str(idx + 1),
                retrieved=retrieved,
                relevant=relevant,
            )
        )

        print(f"Progressing {idx + 1}/{len(questions)}")
        time.sleep(RATE_LIMIT)

    jsonl_content = ""
    for item in mrr_results:
        json_str = item.model_dump_json()
        jsonl_content += json_str + "\n"

    write_file(
        MRR_RESULTS_PATH,
        f"{collection_name}_mrr_results.jsonl",
        jsonl_content,
    )
