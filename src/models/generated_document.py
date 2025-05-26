from pydantic import BaseModel


class GeneratedDocument(BaseModel):
    title: str
    original_text: str


class QueryDocument(BaseModel):
    query: str


class ExpandedDocument(GeneratedDocument):
    expanded_text: str


class QuestionDocument(BaseModel):
    question: str
