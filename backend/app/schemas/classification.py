"""
Shared structured schema for complaint classifications.

Single source of truth for every component that produces a classification:
the teacher LLM during label engineering (Phase 2 Day 7-8), the
fine-tuned Qwen2.5-3B served via Ollama (Day 13), and any cloud
fallback. Centralizing the contract means the same validation runs at
every layer and a vocabulary change in one place propagates everywhere.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Closed vocabularies — used as Pydantic Literal so the JSON schema we
# hand to instructor/openai is exact. Adding a value here is a schema
# change and should bump a migration if persisted.

Sentiment = Literal["neutral", "negative", "extreme_negative"]

Intent = Literal[
    "information_request",
    "dispute_resolution",
    "account_action",
    "fraud_report",
    "regulatory_complaint",
]

EntityType = Literal[
    "company",
    "product",
    "issue",
    "regulation",
    "amount",
    "person",
    "account_type",
    "other",
]


class Entity(BaseModel):
    """A named entity lifted from the complaint narrative."""

    entity: str = Field(
        min_length=1,
        max_length=255,
        description="Surface form as it appears in the text",
    )
    type: EntityType = Field(description="Coarse semantic category")


class ComplaintClassification(BaseModel):
    """Structured classification of a single complaint.

    Produced by the teacher LLM during labeling and by the fine-tuned
    classifier in production. Every field is required; instructor retries
    the LLM call when validation fails so downstream consumers never see
    a partial result.
    """

    sentiment: Sentiment = Field(
        description="Tone of the complaint relative to a neutral baseline",
    )
    intent: Intent = Field(
        description="What the consumer wants from the company or regulator",
    )
    urgency: int = Field(
        ge=1,
        le=5,
        description="1=informational, 5=acute financial or safety harm",
    )
    key_entities: list[Entity] = Field(
        default_factory=list,
        max_length=20,
        description="Companies, products, issues, regulations, amounts named in the text",
    )
    reasoning: str = Field(
        min_length=10,
        max_length=2000,
        description="Short justification used for human audit and as agent context downstream",
    )
