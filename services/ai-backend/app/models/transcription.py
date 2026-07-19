from pydantic import BaseModel

class TranscriptionResult(BaseModel):
    model_config = {"extra": "allow"}

class TranscriptionSegment(BaseModel):
    model_config = {"extra": "allow"}

class TranscriptionWord(BaseModel):
    model_config = {"extra": "allow"}
