"""Tests for the shared ComplaintClassification schema."""

import pytest
from pydantic import ValidationError

from app.schemas.classification import ComplaintClassification, Entity


def _good_payload(**overrides) -> dict:
    base = {
        "sentiment": "neutral",
        "intent": "information_request",
        "urgency": 1,
        "key_entities": [{"entity": "Wells Fargo", "type": "company"}],
        "reasoning": "Plain factual request for a statement copy.",
    }
    base.update(overrides)
    return base


class TestComplaintClassificationHappyPath:
    def test_round_trip(self):
        payload = _good_payload()
        cc = ComplaintClassification.model_validate(payload)
        assert cc.sentiment == "neutral"
        assert cc.intent == "information_request"
        assert cc.urgency == 1
        assert cc.key_entities[0].entity == "Wells Fargo"
        assert cc.key_entities[0].type == "company"

    def test_empty_entities_allowed(self):
        cc = ComplaintClassification.model_validate(_good_payload(key_entities=[]))
        assert cc.key_entities == []

    def test_serializes_to_json(self):
        cc = ComplaintClassification.model_validate(_good_payload())
        round_tripped = ComplaintClassification.model_validate_json(cc.model_dump_json())
        assert round_tripped == cc


class TestComplaintClassificationValidation:
    @pytest.mark.parametrize(
        "field,bad_value,expected_error_type",
        [
            ("sentiment", "very_negative", "literal_error"),
            ("intent", "complaint", "literal_error"),
            ("urgency", 0, "greater_than_equal"),
            ("urgency", 6, "less_than_equal"),
            ("reasoning", "no", "string_too_short"),
        ],
    )
    def test_field_rejects_bad_value(self, field, bad_value, expected_error_type):
        with pytest.raises(ValidationError) as exc:
            ComplaintClassification.model_validate(_good_payload(**{field: bad_value}))
        errors = exc.value.errors()
        assert any(field in str(e["loc"]) and e["type"] == expected_error_type for e in errors), (
            f"expected {expected_error_type} on {field}; got {errors}"
        )

    def test_invalid_entity_type_rejected(self):
        with pytest.raises(ValidationError):
            ComplaintClassification.model_validate(
                _good_payload(key_entities=[{"entity": "X", "type": "made_up"}])
            )

    def test_reasoning_max_length_enforced(self):
        too_long = "a" * 2001
        with pytest.raises(ValidationError):
            ComplaintClassification.model_validate(_good_payload(reasoning=too_long))

    def test_entity_max_list_length(self):
        too_many = [{"entity": f"E{i}", "type": "other"} for i in range(21)]
        with pytest.raises(ValidationError):
            ComplaintClassification.model_validate(_good_payload(key_entities=too_many))


class TestEntity:
    def test_blank_surface_form_rejected(self):
        with pytest.raises(ValidationError):
            Entity.model_validate({"entity": "", "type": "company"})

    def test_all_known_entity_types_accepted(self):
        for t in [
            "company",
            "product",
            "issue",
            "regulation",
            "amount",
            "person",
            "account_type",
            "other",
        ]:
            Entity.model_validate({"entity": "X", "type": t})
