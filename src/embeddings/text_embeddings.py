from fastembed import TextEmbedding, SparseTextEmbedding
import src.constants.constants as constants


# สร้างตัว encoder (สามารถระบุชื่อโมเดลที่ต้องการใช้)
dense_text_embedding = TextEmbedding(model_name=constants.FASTEMBED_DENSE_MODEL_NAME)
sparse_text_embedding = SparseTextEmbedding(
    model_name=constants.FASTEMBED_BM25_MODEL_NAME
)


# dense embedding แบบทีละข้อความ
def dense_embedding(document: str) -> list:
    # return [0] เพราะว่า fastembed จะ return เป็น list ของ list
    # แต่เราต้องการแค่ list เดียว
    # ดังนั้นเราจะ return list แรกของ list แรก
    return list(dense_text_embedding.embed([document]))[0]


# dense embedding แบบหลายข้อความ
def dense_embedding_list(documents: list) -> list:
    return list(dense_text_embedding.embed(documents))


# sparse embedding แบบทีละข้อความ
def sparse_embedding(document: str) -> list:
    # return [0] เพราะว่า fastembed จะ return เป็น list ของ list
    # แต่เราต้องการแค่ list เดียว
    # ดังนั้นเราจะ return list แรกของ list แรก
    return list(sparse_text_embedding.embed([document]))[0]


# sparse embedding แบบหลายข้อความ
def sparse_embedding_list(documents: list) -> list:
    return list(sparse_text_embedding.embed(documents))
