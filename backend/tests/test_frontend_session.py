"""HTTP client contracts; no browser, server or TLS handshake is simulated here."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def frontend(monkeypatch):
    state = {}
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(session_state=state, cache_data=lambda **kwargs: lambda fn: fn),
    )
    spec = importlib.util.spec_from_file_location(
        "session_api_client", Path(__file__).resolve().parents[2] / "frontend/api_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module, state
    module.clear_session()


def transport(module, monkeypatch, handler, *, scheme="https"):
    clients = []

    def make_client():
        if module._CLIENT_KEY not in module.st.session_state:
            client = httpx.Client(
                base_url=f"{scheme}://api.test:8443/api/v1",
                transport=httpx.MockTransport(handler),
                timeout=module.TIMEOUT_S,
            )
            clients.append(client)
            module.st.session_state[module._CLIENT_KEY] = client
        return module.st.session_state[module._CLIENT_KEY]

    monkeypatch.setattr(module, "_http", make_client)
    return clients


@pytest.mark.parametrize("operation", ["login", "register"])
@pytest.mark.parametrize("exception", [httpx.ConnectError, httpx.ReadTimeout])
def test_auth_transport_failures_are_bounded_api_errors(
    frontend, monkeypatch, operation, exception
):
    module, state = frontend

    def fail(request):
        assert request.extensions["timeout"]["connect"] == 5.0
        assert request.extensions["timeout"]["read"] == 10.0
        raise exception("private infrastructure detail", request=request)

    transport(module, monkeypatch, fail)
    with pytest.raises(module.ApiError) as error:
        if operation == "login":
            module.login("synthetic@example.com", "password")
        else:
            module.register("synthetic@example.com", "Synthetic", "password")
    assert error.value.status_code == 0
    assert "private infrastructure" not in error.value.detail
    assert not module.is_authenticated()
    assert module._USER_KEY not in state


@pytest.mark.parametrize("profile_failure", ["timeout", "401", "json"])
def test_login_profile_failure_discards_partial_token_and_cookie(
    frontend, monkeypatch, profile_failure
):
    module, state = frontend

    def handler(request):
        if request.url.path.endswith("/login"):
            return httpx.Response(
                200,
                json={"access_token": "new-token"},
                headers={
                    "set-cookie": "refresh_token=synthetic; Secure; HttpOnly; Path=/api/v1/auth"
                },
            )
        assert request.headers["authorization"] == "Bearer new-token"
        assert module._TOKEN_KEY not in state
        if profile_failure == "timeout":
            raise httpx.ReadTimeout("offline", request=request)
        if profile_failure == "401":
            return httpx.Response(401, json={"detail": "Inactive user"})
        return httpx.Response(200, text="invalid")

    clients = transport(module, monkeypatch, handler)
    with pytest.raises(module.ApiError):
        module.login("synthetic@example.com", "password")
    assert state == {}
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("scheme", ["https", "http"])
def test_real_cookie_jar_only_sends_secure_refresh_cookie_over_https(frontend, monkeypatch, scheme):
    module, state = frontend
    refresh_requests = []

    def handler(request):
        path = request.url.path
        if path.endswith("/login"):
            return httpx.Response(
                200,
                json={"access_token": "initial"},
                headers={"set-cookie": "refresh_token=first; Secure; HttpOnly; Path=/api/v1/auth"},
            )
        if path.endswith("/me"):
            assert "cookie" not in request.headers or scheme == "https"
            return httpx.Response(200, json={"role": "viewer"})
        if path.endswith("/refresh"):
            refresh_requests.append(request)
            if "refresh_token=first" not in request.headers.get("cookie", ""):
                return httpx.Response(401, json={"detail": "Missing refresh cookie"})
            return httpx.Response(
                200,
                json={"access_token": "rotated"},
                headers={"set-cookie": "refresh_token=second; Secure; HttpOnly; Path=/api/v1/auth"},
            )
        assert "cookie" not in request.headers  # refresh path never leaks to complaints
        if request.headers.get("authorization") == "Bearer rotated":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(401, json={"detail": "Expired token"})

    transport(module, monkeypatch, handler, scheme=scheme)
    module.login("synthetic@example.com", "password")
    if scheme == "https":
        assert module.triage_queue() == {"items": []}
        assert state[module._TOKEN_KEY] == "rotated"
        assert module._http().cookies.get("refresh_token") == "second"
    else:
        with pytest.raises(module.ApiError) as error:
            module.triage_queue()
        assert error.value.status_code == 401
        assert state == {}
    assert len(refresh_requests) == 1
    assert ("refresh_token=first" in refresh_requests[0].headers.get("cookie", "")) == (
        scheme == "https"
    )


def test_transient_refresh_failure_preserves_session(frontend, monkeypatch):
    module, state = frontend
    state[module._TOKEN_KEY] = "existing"
    state[module._USER_KEY] = {"role": "viewer"}

    def handler(request):
        if request.url.path.endswith("/refresh"):
            return httpx.Response(
                503, json={"detail": "Authentication service temporarily unavailable"}
            )
        return httpx.Response(401, json={"detail": "Expired token"})

    transport(module, monkeypatch, handler)
    with pytest.raises(module.ApiError) as error:
        module.triage_queue()
    assert error.value.status_code == 503
    assert state[module._TOKEN_KEY] == "existing"
    assert module.current_user() == {"role": "viewer"}


def test_production_http_configuration_fails_before_creating_client(frontend, monkeypatch):
    module, state = frontend
    monkeypatch.setattr(module, "ENVIRONMENT", "production")
    monkeypatch.setattr(module, "API_URL", "http://api:8000")
    with pytest.raises(module.ApiError, match="HTTPS API_URL"):
        module._http()
    assert state == {}


def test_error_payload_can_be_non_object_json(frontend):
    module, _ = frontend
    assert module._detail(httpx.Response(502, json=["bad gateway"])) == '["bad gateway"]'


@pytest.mark.parametrize("failure", ["timeout", "503"])
def test_uncertain_job_request_reuses_key_until_success(frontend, monkeypatch, failure):
    module, state = frontend
    keys = []

    def handler(request):
        keys.append(request.headers["Idempotency-Key"])
        if len(keys) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("synthetic timeout", request=request)
            return httpx.Response(503, json={"detail": "Unavailable"})
        return httpx.Response(201, json={"id": "same-case"})

    transport(module, monkeypatch, handler)
    with pytest.raises(module.ApiError):
        module._job_request("/complaints/", {"narrative": "Synthetic fee dispute"})
    assert len(state["pending_job_requests"]) == 1
    assert module._job_request("/complaints/", {"narrative": "Synthetic fee dispute"}) == {
        "id": "same-case"
    }
    assert keys[0] == keys[1]
    assert state["pending_job_requests"] == {}


def test_rejected_submissions_do_not_exhaust_pending_key_limit(frontend, monkeypatch):
    module, state = frontend
    transport(module, monkeypatch, lambda _: httpx.Response(422, json={"detail": "Invalid"}))
    for index in range(130):
        with pytest.raises(module.ApiError) as error:
            module._job_request("/complaints/", {"narrative": str(index)})
        assert error.value.status_code == 422
    assert state["pending_job_requests"] == {}


def test_refresh_replay_preserves_job_key_and_rotates_access_token(frontend, monkeypatch):
    module, state = frontend
    state[module._TOKEN_KEY] = "old"
    requests = []

    def handler(request):
        if request.url.path.endswith("/refresh"):
            return httpx.Response(200, json={"access_token": "new"})
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(401, json={"detail": "Expired"})
        return httpx.Response(202, json={"job_id": "same-job"})

    transport(module, monkeypatch, handler)
    assert module._job_request("/resolutions/synthetic/generate") == {"job_id": "same-job"}
    assert requests[0].headers["Idempotency-Key"] == requests[1].headers["Idempotency-Key"]
    assert requests[1].headers["Authorization"] == "Bearer new"
