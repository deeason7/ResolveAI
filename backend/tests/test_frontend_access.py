"""Client contracts with an in-memory session and HTTP transport, without Streamlit runtime."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def frontend_client(monkeypatch):
    state = {}
    streamlit = SimpleNamespace(session_state=state, cache_data=lambda **kwargs: lambda fn: fn)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)
    spec = importlib.util.spec_from_file_location(
        "access_api_client", Path(__file__).resolve().parents[2] / "frontend" / "api_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module, state
    module.clear_session()


def _transport(module, monkeypatch, handler):
    def make_client():
        if module._CLIENT_KEY not in module.st.session_state:
            module.st.session_state[module._CLIENT_KEY] = httpx.Client(
                base_url="https://api.test/api/v1",
                transport=httpx.MockTransport(handler),
            )
        return module.st.session_state[module._CLIENT_KEY]

    monkeypatch.setattr(module, "_http", make_client)


def test_demo_401_never_registers(frontend_client, monkeypatch):
    module, state = frontend_client
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(401, json={"detail": "Invalid credentials"})

    _transport(module, monkeypatch, handler)
    with pytest.raises(module.ApiError) as error:
        module.start_demo()
    assert error.value.status_code == 401
    assert requests == [("POST", "/api/v1/auth/login")]
    assert not module.is_authenticated()
    assert state == {}


@pytest.mark.parametrize("role", ["admin", "analyst", None])
def test_demo_rejects_nonviewer_and_clears_tokens_and_cookies(frontend_client, monkeypatch, role):
    module, state = frontend_client
    clients = []

    def handler(request):
        clients.append(module._http())
        if request.url.path.endswith("/login"):
            return httpx.Response(
                200,
                json={"access_token": "synthetic-token"},
                headers={
                    "set-cookie": "refresh_token=synthetic-cookie; Secure; HttpOnly",
                },
            )
        return httpx.Response(200, json={"role": role})

    _transport(module, monkeypatch, handler)
    with pytest.raises(module.ApiError) as error:
        module.start_demo()
    assert error.value.status_code == 403
    assert state == {}
    assert all(client.is_closed for client in clients)


def test_demo_accepts_provisioned_viewer(frontend_client, monkeypatch):
    module, state = frontend_client

    def handler(request):
        if request.url.path.endswith("/login"):
            return httpx.Response(200, json={"access_token": "synthetic-token"})
        return httpx.Response(200, json={"role": "viewer"})

    _transport(module, monkeypatch, handler)
    module.start_demo()
    assert module.is_demo()
    assert module.is_authenticated()
    assert state[module._USER_KEY]["role"] == "viewer"


def test_registration_capability_comes_from_server(frontend_client, monkeypatch):
    module, _ = frontend_client
    _transport(
        module,
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"public_registration_enabled": False},
        ),
    )
    assert module.auth_capabilities() == {"public_registration_enabled": False}
