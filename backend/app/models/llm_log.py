"""LLMLog ORM model — one row per LLM inference call (LLMOps observability)."""

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class LLMLog(SQLModel, table=True):
    __tablename__ = "llm_logs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    complaint_id: Optional[uuid.UUID] = Field(default=None, foreign_key="complaints.id", index=True)

    operation: str = Field(max_length=100)      # e.g. "classify", "resolve", "guardrail_check"
    model_used: str = Field(max_length=100)     # e.g. "qwen2.5-3b-resolveai", "llama3-70b-groq"
    provider: str = Field(max_length=50)        # "ollama" | "groq" | "openai"

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None            # None for local Ollama calls

    was_fallback: bool = Field(default=False)   # True if cloud was used as fallback

    created_at: datetime = Field(default_factory=datetime.utcnow)
