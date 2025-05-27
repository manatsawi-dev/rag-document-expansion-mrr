from fastembed.embedding import TextEmbedding

# Create encoder (you can specify the model name you want to use)
embedding_model: TextEmbedding = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Sample text list
documents = [
    "How to request vacation leave?",
    "Company leave policy explained.",
    "Guide to submitting a leave request form.",
]

# Convert text to dense embedding
embeddings = list(embedding_model.embed(documents))

# Display results
for i, doc in enumerate(documents):
    print(f"Document {i}: {doc}")
    print(f"Embedding: {embeddings[i][:5]}")
    print()

# Example output (first 5 values of embedding)
# Document 0: How to request vacation leave?
# Embedding: [-0.03820893 -0.02452314  0.00697538 -0.05776384  0.0553261, ...]

# Document 1: Company leave policy explained.
# Embedding: [-0.02869109  0.01727294 -0.01712723 -0.05012331  0.05791203, ...]

# Document 2: Guide to submitting a leave request form.
# Embedding: [-0.07710513 -0.0187871   0.04183139 -0.02038685  0.04148764, ...]
