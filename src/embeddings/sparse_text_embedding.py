from fastembed import SparseTextEmbedding

# Create encoder (you can specify the model name you want to use)
embedding_model: SparseTextEmbedding = SparseTextEmbedding(model_name="Qdrant/bm25")

# Sample text list
documents = [
    "How to request vacation leave?",
    "Company leave policy explained.",
    "Guide to submitting a leave request form.",
]


# Convert text to sparse embedding
embeddings = list(embedding_model.embed(documents))


# Display results
for i, doc in enumerate(documents):
    print(f"Document {i}: {doc}")
    print(f"Embedding: {embeddings[i]}")
    print()

# Example output
# Document 0: How to request vacation leave?
# Embedding: SparseEmbedding(values=array([1.67868852, 1.67868852, 1.67868852]), indices=array([2064885619, 1253263062, 1943443462]))

# Document 1: Company leave policy explained.
# Embedding: SparseEmbedding(values=array([1.67419738, 1.67419738, 1.67419738, 1.67419738]), indices=array([1442710396, 1943443462,  203139330, 1971389377]))

# Document 2: Guide to submitting a leave request form.
# Embedding: SparseEmbedding(values=array([1.66973021, 1.66973021, 1.66973021, 1.66973021, 1.66973021]), indices=array([1209733406, 1546460417, 1943443462, 2064885619,   14784032]))
