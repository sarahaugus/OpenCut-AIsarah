from pydantic import BaseModel

class EnhancePromptRequest(BaseModel):
    model_config = {"extra": "allow"}

class ImageGenParams(BaseModel):
    model_config = {"extra": "allow"}

class InfographicRequest(BaseModel):
    model_config = {"extra": "allow"}
