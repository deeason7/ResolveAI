"""
Prompt templates for the resolution agent.

Kept apart from the orchestration logic so the wording can be iterated and
diffed without touching control flow, and so tests can assert on the rendered
text. The drafting prompt mirrors the agent template in the engineering spec.
"""

from __future__ import annotations

from app.schemas.agent import (
    CompanyHistoryResult,
    DraftResponseInput,
    PrecedentResult,
    RegulationResult,
)

SYSTEM_PROMPT = (
    "You are a consumer financial protection specialist. You draft resolution "
    "responses for consumer complaints. You must be:\n"
    "- EMPATHETIC: acknowledge the consumer's frustration and situation.\n"
    "- ACCURATE: only reference regulations and precedents you have been given. "
    "Never invent regulation names, citations, or case outcomes.\n"
    "- ACTIONABLE: give specific next steps the consumer can take.\n"
    "- PROFESSIONAL: keep a respectful, firm tone. Do NOT give legal advice "
    "(never tell the consumer to sue or file a lawsuit) and do NOT admit legal "
    "liability on the company's behalf.\n\n"
    "Respond with JSON only, matching the requested schema."
)


def _format_precedents(precedents: list[PrecedentResult]) -> str:
    if not precedents:
        return "No similar past cases were found."
    lines = []
    for i, p in enumerate(precedents, 1):
        outcome = p.company_response or "outcome unknown"
        preview = p.narrative_preview or "(no preview)"
        lines.append(f"{i}. [{p.similarity_score:.2f} similar] {preview} -> resolved as: {outcome}")
    return "\n".join(lines)


def _format_regulations(regulations: list[RegulationResult]) -> str:
    if not regulations:
        return "No specific regulations were matched for this product/issue."
    lines = []
    for r in regulations:
        provisions = "; ".join(r.key_provisions[:3]) if r.key_provisions else "see summary"
        lines.append(
            f"- {r.title} ({r.cfr_reference}): {r.summary} "
            f"Key provisions: {provisions}. Relevance: {r.relevance}"
        )
    return "\n".join(lines)


def _format_company(profile: CompanyHistoryResult | None) -> str:
    if profile is None:
        return "No company history is available."
    parts = [
        f"{profile.company_name}: {profile.total_complaints} total complaints",
    ]
    if profile.risk_score is not None:
        parts.append(f"risk score {profile.risk_score:.2f}/1.0")
    if profile.top_products:
        parts.append("most complained-about products: " + ", ".join(profile.top_products[:3]))
    if profile.violations:
        parts.append("linked violations: " + ", ".join(profile.violations))
    if profile.repeat_offender:
        parts.append("flagged as a repeat offender")
    return "; ".join(parts) + "."


def build_draft_prompt(inp: DraftResponseInput) -> str:
    """Render the user turn for the drafting call from all gathered context."""
    cls = inp.classification
    entities = (
        ", ".join(f"{e.entity} ({e.type})" for e in cls.key_entities)
        if cls.key_entities
        else "none extracted"
    )
    return (
        f"COMPLAINT:\n{inp.complaint_narrative.strip()}\n\n"
        "CLASSIFICATION:\n"
        f"- Sentiment: {cls.sentiment}\n"
        f"- Intent: {cls.intent}\n"
        f"- Urgency: {cls.urgency}/5\n"
        f"- Key entities: {entities}\n\n"
        f"SIMILAR PAST CASES:\n{_format_precedents(inp.precedents)}\n\n"
        f"APPLICABLE REGULATIONS:\n{_format_regulations(inp.regulations)}\n\n"
        f"COMPANY PROFILE:\n{_format_company(inp.company_profile)}\n\n"
        "Draft a response that:\n"
        "1. Acknowledges the specific issue the consumer raised.\n"
        "2. References applicable consumer rights, citing the regulations above by title.\n"
        "3. Suggests concrete resolution steps.\n"
        "4. Provides an escalation path if the company does not respond.\n"
        "5. Stays empathetic but firm.\n\n"
        "Output JSON with fields: response_text, tone, cited_regulations, "
        "recommended_actions, confidence."
    )


def build_regeneration_prompt(previous_draft: str, feedback: str) -> str:
    """Render the user turn for a retry after guardrails reject a draft."""
    return (
        "Your previous draft failed validation and must be revised.\n\n"
        f"PREVIOUS DRAFT:\n{previous_draft}\n\n"
        f"VALIDATION FEEDBACK (address every point):\n{feedback}\n\n"
        "Produce a corrected response in the same JSON schema. Keep what worked "
        "and fix only what the feedback flags."
    )
