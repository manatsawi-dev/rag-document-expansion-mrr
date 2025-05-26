from pydantic import BaseModel
from enum import Enum


class Relevance(Enum):
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    AMBIGUOUS = "ambiguous"


class MRRResult(BaseModel):
    document_number: int
    relevance: Relevance


class MRRDataset(BaseModel):
    query_id: str
    retrieved: list[str]
    relevant: set[str]


class MRRDatasetInfo(BaseModel):
    dataset_name: str
    note: str
    embedding_type: str
