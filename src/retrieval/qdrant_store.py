from qdrant_client import QdrantClient
from qdrant_client.http import models
from uuid import uuid4

import src.constants.constants as constants

# สร้าง client เชื่อมต่อไปยัง Qdrant server ที่รันผ่าน Docker
qdrant_client = QdrantClient(host=constants.QDRANT_HOST, port=constants.QDRANT_PORT)


def generate_id():
    # สร้าง ID สำหรับ Qdrant
    # ID จะต้องไม่ซ้ำกันใน collection
    # สามารถใช้ UUID หรือ hash function เพื่อสร้าง ID ที่ไม่ซ้ำกัน
    return str(uuid4())


# ฟังก์ชันสำหรับสร้าง collection ใน Qdrant
def create_collection(collection_name: str):
    try:
        # ตรวจสอบว่า collection มีอยู่แล้วหรือไม่
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        # ถ้าไม่มี collection ให้สร้างใหม่
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                distance=models.Distance.COSINE,
                size=constants.VECTOR_SIZE,
            ),
        )


# ฟังก์ชันสำหรับลบ collection ใน Qdrant
def delete_collection(collection_name: str):
    try:
        # ตรวจสอบว่า collection มีอยู่แล้วหรือไม่
        qdrant_client.get_collection(collection_name=collection_name)
        # ถ้ามีให้ลบ collection
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        # ถ้าไม่มี collection ไม่ต้องทำอะไร
        pass


# ฟังก์ชันสำหรับอัปโหลด ข้อขวามที่ถูก embedding แล้วไปยัง Qdrant
# เราจะเก็บข้อความต้นฉบับ metadata และ embedding vector ไว้ด้วยกัน
# เพื่อให้สามามารถนำข้อมูลไปใช้ได้ในอนาคต
def add_embedding(
    collection_name: str,
    vector: list,
    original_text: str,
    expanded_text: str,
    metadata: dict,
):
    # สร้าง collection ถ้ายังไม่มี
    create_collection(collection_name)

    # เพิ่มข้อมูลลง Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=generate_id(),
                vector=vector,
                payload={
                    "original_text": original_text,
                    "expanded_text": expanded_text,
                    "metadata": metadata,
                },
            )
        ],
    )


def get_all_embedding(collection_name: str, limit: int = 10):
    try:
        # ตรวจสอบว่า collection มีอยู่แล้วหรือไม่
        qdrant_client.get_collection(collection_name=collection_name)
        # ถ้ามีให้ดึงข้อมูลทั้งหมด
        response = qdrant_client.scroll(
            collection_name=collection_name,
            limit=limit,
        )
        return response
    except Exception as e:
        # ถ้าไม่มี collection ไม่ต้องทำอะไร
        print(f"Error getting data from Qdrant: {e}")
        return []


def similarity_search(collection_name: str, query_vector: list, limit: int = 10):
    try:
        # ตรวจสอบว่า collection มีอยู่แล้วหรือไม่
        qdrant_client.get_collection(collection_name=collection_name)

        # ถ้ามีให้ค้นหาข้อมูลที่คล้ายกัน
        response = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_vectors=False,
            with_payload=True,
        )
        return response
    except Exception as e:
        # ถ้าไม่มี collection ไม่ต้องทำอะไร
        print(f"Error searching data in Qdrant: {e}")
        return []
