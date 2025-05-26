import json
import time
import os
from pathlib import Path

from src.generation.google_gemini_ai import gemini_generate_content
from src.models.generated_document import GeneratedDocument, QuestionDocument
from src.utils.file_utils import read_jsonl_file, write_file
from src.utils.document_utils import load_and_combine_jsonl_documents


DATASET_DIR = Path("data/data_set")
OUTPUT_DIR = Path("data/question")
QUESTION_FILE_NAME = "questions.jsonl"
TOTAL_QUESTIONS = 2
RATE_LIMIT_DELAY = 10


def generate_questions():
    # If file is present, return the file
    questions_file_path = f"{OUTPUT_DIR}/{QUESTION_FILE_NAME}"
    if os.path.exists(questions_file_path):
        data = read_jsonl_file(questions_file_path)
        question_documents = [QuestionDocument(**item) for item in data]
        return question_documents

    # Load and combine documents from the dataset directory
    original_documents = load_and_combine_jsonl_documents(DATASET_DIR)

    question_documents = []
    # Generate questions for each document
    for idx, document in enumerate(original_documents):
        doc = GeneratedDocument.model_validate_json(document)
        prompt = f"""Given the following document, generate {TOTAL_QUESTIONS} high-quality search queries that users might use to find this information. The document content is: {doc.original_text}

        Generate diverse questions following these requirements:
        1. Semantic Variation: Create questions that use different words/synonyms than those in the original text, but mean the same thing
        2. Question Types:
           - Direct questions about specific facts
           - Questions about relationships between concepts
           - Questions that require understanding the context
           - Questions using alternative terminology
        3. Complexity Levels:
           - Include both simple and complex queries
           - Some questions should combine multiple aspects from the document
           - Some questions should test semantic understanding rather than keyword matching
        4. Natural Language Patterns:
           - Use natural language as real users would ask
           - Vary question formats (what, how, why, when, etc.)
           - Include both formal and conversational writing styles
        5. Search Relevance:
           - Questions should have clear answers in the document
           - Avoid questions that are too generic or could apply to any document
           - Include edge cases that test the search system's semantic understanding

        Format each question as a QuestionDocument object with appropriate metadata.
        """

        response = gemini_generate_content(
            prompt,
            response_schema=list[QuestionDocument],
        )

        documents: list[QuestionDocument] = []
        json_data = json.loads(response)
        documents = [QuestionDocument(**item) for item in json_data]

        question_documents.extend(documents)

        print(f"Progressing: {idx + 1}/{len(original_documents)}")
        time.sleep(
            RATE_LIMIT_DELAY
        )  #  Sleep for {RATE_LIMIT_DELAY} second to avoid hitting the rate limit

    jsonl_content = "\n".join([doc.model_dump_json() for doc in question_documents])

    write_file(OUTPUT_DIR, QUESTION_FILE_NAME, jsonl_content)

    return question_documents
