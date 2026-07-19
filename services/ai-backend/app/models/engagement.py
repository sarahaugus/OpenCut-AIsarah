from pydantic import BaseModel

class EngagementScore(BaseModel):
    model_config = {"extra": "allow"}

class ScoreBatchRequest(BaseModel):
    model_config = {"extra": "allow"}

class ScoreClipRequest(BaseModel):
    model_config = {"extra": "allow"}

class CuriosityScore(BaseModel):
    model_config = {"extra": "allow"}

class EmotionalArcScore(BaseModel):
    model_config = {"extra": "allow"}

class EnhancementSuggestion(BaseModel):
    model_config = {"extra": "allow"}

class EnergyScore(BaseModel):
    model_config = {"extra": "allow"}

class AudioSyncScore(BaseModel):
    model_config = {"extra": "allow"}

class FacePresenceScore(BaseModel):
    model_config = {"extra": "allow"}

class HookScore(BaseModel):
    model_config = {"extra": "allow"}

class ViralityScore(BaseModel):
    model_config = {"extra": "allow"}

class ScoredClip(BaseModel):
    model_config = {"extra": "allow"}

class JobStatus(BaseModel):
    model_config = {"extra": "allow"}

class YouTubeVideoMeta(BaseModel):
    model_config = {"extra": "allow"}
