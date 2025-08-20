from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class SampleStats(BaseModel):
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency: Optional[float] = None


class ZeroShotOutput(BaseModel):
    label: str = Field(..., description="The predicted classification label.")
    stats: Optional[SampleStats] = None
    person_key: Optional[str] = None
    role_playing: Optional[Literal["active", "passive", "none"]] = None


class FewShotOutput(BaseModel):
    label: str = Field(..., description="The predicted classification label.")
    stats: Optional[SampleStats] = None
    person_key: Optional[str] = None
    role_playing: Optional[Literal["active", "passive", "none"]] = None


class ReasoningStep(BaseModel):
    step_number: int
    content: str


class CoTResult(BaseModel):
    reasoning_steps: List[ReasoningStep]
    final_reasoning: str
    final_label: str
    confidence: Optional[float] = None
    stats: Optional[SampleStats] = None
    person_key: Optional[str] = None
    role_playing: Optional[Literal["active", "passive", "none"]] = None

