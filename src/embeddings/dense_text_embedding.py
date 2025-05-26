from fastembed.embedding import TextEmbedding

# สร้างตัว encoder (สามารถระบุชื่อโมเดลที่ต้องการใช้)
embedding_model: TextEmbedding = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# รายการข้อความตัวอย่าง
documents = [
    "How to request vacation leave?",
    "Company leave policy explained.",
    "Guide to submitting a leave request form.",
]

# แปลงข้อความเป็น dense embedding
embeddings = list(embedding_model.embed(documents))

# แสดงผลลัพธ์
for i, doc in enumerate(documents):
    print(f"Document {i}: {doc}")
    print(f"Embedding: {embeddings[i][:5]}")
    print()

# ตัวอย่าง output (5 ค่าแรกของ embedding)
# Document 0: How to request vacation leave?
# Embedding: [-0.03820893 -0.02452314  0.00697538 -0.05776384  0.0553261, ...]

# Document 1: Company leave policy explained.
# Embedding: [-0.02869109  0.01727294 -0.01712723 -0.05012331  0.05791203, ...]

# Document 2: Guide to submitting a leave request form.
# Embedding: [-0.07710513 -0.0187871   0.04183139 -0.02038685  0.04148764, ...]
