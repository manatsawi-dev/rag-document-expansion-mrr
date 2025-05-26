import json
import time
from pathlib import Path

from src.generation.google_gemini_ai import gemini_generate_content
from src.models.generated_document import GeneratedDocument
from src.utils.file_utils import write_file


OUTPUT_DIR = Path("data/data_set")
BATCH_SIZE = 10
RATE_LIMIT_DELAY = 10


SUBTOPICS = [
    "Leave Policy",
    "Salary & Compensation",
    "Benefits",
    "Performance Review",
    "Health and Safety",
]

FILE_NAME_MAPPING = {
    "Leave Policy": "leave_policy",
    "Salary & Compensation": "salary_compensation",
    "Benefits": "benefits",
    "Performance Review": "performance_review",
    "Health and Safety": "health_safety",
}


def generate_data_set():
    for subtopic in SUBTOPICS:
        prompt = f"""
You are a helpful assistant generating realistic internal knowledge base documents for testing an AI document retrieval system.

Topic: HR Knowledge Base 

Objective:
Generate {BATCH_SIZE} unique, documents that resemble internal HR or company knowledge base entries.

Instructions:
Each document must include:
	•	A short and clear title
	•	A original text field (10-15 sentences), written in professional, informative tone
	•	Content should appear as if it were published on a company intranet or internal wiki

Coverage (Subtopics to include):
	•	{subtopic}

Constraints:
	•	Do not repeat or reuse exact sentences between documents
	•	Vary wording, phrasing, and examples across documents
	•	Tone should be clear, helpful, and consistent with corporate internal communication
	•	Do not generate queries — only documents
"""

        response = gemini_generate_content(
            prompt,
            response_schema=list[GeneratedDocument],
        )

        documents: list[GeneratedDocument] = []
        json_data = json.loads(response)
        documents = [GeneratedDocument(**item) for item in json_data]

        jsonl_content = "\n".join(doc.model_dump_json() for doc in documents)

        # Write the JSONL content to a file
        write_file(OUTPUT_DIR, f"{FILE_NAME_MAPPING[subtopic]}.jsonl", jsonl_content)
        time.sleep(
            RATE_LIMIT_DELAY
        )  # Sleep for {RATE_LIMIT_DELAY} second to avoid hitting the rate limit


if __name__ == "__main__":
    # Generate the data set
    generate_data_set()
