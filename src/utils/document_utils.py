import os


def load_and_combine_jsonl_documents(dir_path: str):
    files = [
        f
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".jsonl")
    ]

    documents = []
    for file in files:
        with open(os.path.join(dir_path, file), "r") as f:
            for line in f:
                documents.append(line.strip())
    return documents


def format_documents_as_context(documents: list):
    formatted = []
    for i, doc in enumerate(documents):

        payload = doc.payload or dict()
        original_text = payload.get("original_text", "")

        formatted.append(
            f"--- Document {i+1} ---\n"
            f"{original_text}\n"
            f"--- End of Document {i+1} ---"
        )

    return "\n\n".join(formatted)
