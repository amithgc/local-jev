"""The phase-1 exit test: the official, unmodified typesafe-sdk talking to a running local-jev.

    local-jev serve --port 8765 &
    LOCAL_JEV_URL=http://127.0.0.1:8765 pytest tests/test_sdk_dropin.py
"""
import os

import pytest

URL = os.environ.get("LOCAL_JEV_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set LOCAL_JEV_URL to a running local-jev server")

STATE = ("Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. "
         "I'm losing sales. Please help ASAP.")


@pytest.fixture()
def client(monkeypatch):
    from typesafe_sdk import TypeSafeClient
    monkeypatch.setenv("TYPESAFE_BASE_URL", URL)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local")
    return TypeSafeClient()


def test_quickstart_example(client):
    from typesafe_sdk import Choice, Noul, Score
    result = client.system_one(state=STATE, questions={
        "department": Choice(instructions="Which team should handle this", criteria={
            "billing": "Payment or subscription issues", "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions"}),
        "frustration": Score(instructions="How frustrated the customer appears", criteria=[
            "Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]),
        "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity"),
    })
    dept = result.choices["department"]
    assert dept.choice in {"billing", "technical", "sales"}
    assert abs(sum(dept.probabilities.values()) - 1) < 0.01 and 0 <= dept.confidence <= 1
    frustration = result.scores["frustration"]
    assert 0 <= frustration.score <= 2 and len(frustration.probabilities) == 3
    assert 0 <= result.nouls["is_urgent"].noul <= 1
    assert result.request_id


def test_models(client):
    names = [m.name for m in client.models.list().models]
    assert "jev-latest" in names


def test_validation_error_is_a_422(client):
    from typesafe_sdk import TypeSafeAPIError
    with pytest.raises(TypeSafeAPIError) as err:
        too_many = {f"option_{i}": None for i in range(300)}          # the limit is 255
        client.system_one(state="x", questions={"q": {"type": "choice", "criteria": too_many}})
    assert err.value.status == 422
