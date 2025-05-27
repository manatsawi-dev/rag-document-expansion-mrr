from fastembed import TextEmbedding, SparseTextEmbedding
import src.constants.constants as constants


# Create encoder (you can specify the model name you want to use)
dense_text_embedding = TextEmbedding(model_name=constants.FASTEMBED_DENSE_MODEL_NAME)
sparse_text_embedding = SparseTextEmbedding(
    model_name=constants.FASTEMBED_BM25_MODEL_NAME
)


# Dense embedding for single text
def dense_embedding(document: str) -> list:
    # Return [0] because fastembed returns a list of lists
    # But we only want a single list
    # So we return the first list of the first list
    return list(dense_text_embedding.embed([document]))[0]


# Dense embedding for multiple texts
def dense_embedding_list(documents: list) -> list:
    return list(dense_text_embedding.embed(documents))


# Sparse embedding for single text
def sparse_embedding(document: str) -> list:
    # Return [0] because fastembed returns a list of lists
    # But we only want a single list
    # So we return the first list of the first list
    return list(sparse_text_embedding.embed([document]))[0]


# Sparse embedding for multiple texts
def sparse_embedding_list(documents: list) -> list:
    return list(sparse_text_embedding.embed(documents))
