from pydantic import BaseModel


class ExampleModel(BaseModel):
    name: str
    answer: str
    characteristics: str
