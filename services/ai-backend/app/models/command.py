from pydantic import BaseModel

class CommandRequest(BaseModel):
    model_config = {"extra": "allow"}

class CommandResponse(BaseModel):
    model_config = {"extra": "allow"}

class EditorAction(BaseModel):
    model_config = {"extra": "allow"}
