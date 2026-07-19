from pydantic import BaseModel

class TTSRequest(BaseModel):
    model_config = {"extra": "allow"}
