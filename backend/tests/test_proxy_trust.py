"""The Space image's forwarded-header trust setting.

These pin a deployment constant that no unit test would otherwise reach, and
one specific way of getting it wrong.

uvicorn wraps ``ProxyHeadersMiddleware`` unconditionally (``proxy_headers``
defaults to True) but only consumes ``X-Forwarded-For`` when the immediate peer
is trusted, and ``forwarded_allow_ips`` defaults to ``127.0.0.1`` alone. HF's
ingress is not localhost, so without ``FORWARDED_ALLOW_IPS`` every caller
collapses into a single identity -- the rate limiter keys on the proxy and the
audit log records it too.

The value has to stay a network, not ``*``: uvicorn's
``get_trusted_client_address`` walks the forwarded chain in reverse and returns
the first untrusted hop, but ``always_trust`` (which is what ``*`` sets)
short-circuits to ``hosts[0]`` -- the end the client controls. ``*`` therefore
lets any caller choose their own rate-limit bucket, which is the opposite of
what setting this is for.
"""

import ipaddress
import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"


@pytest.fixture(scope="module")
def forwarded_allow_ips() -> str:
    text = DOCKERFILE.read_text()
    match = re.search(r"FORWARDED_ALLOW_IPS=(\S+)", text)
    assert match, "the Space image must set FORWARDED_ALLOW_IPS"
    return match.group(1).rstrip("\\").strip()


def test_forwarded_allow_ips_is_set(forwarded_allow_ips: str) -> None:
    assert forwarded_allow_ips


def test_forwarded_allow_ips_is_not_wildcard(forwarded_allow_ips: str) -> None:
    """`*` makes uvicorn take the client-controlled end of the chain."""
    assert forwarded_allow_ips != "*", (
        "FORWARDED_ALLOW_IPS='*' sets always_trust, which returns "
        "x_forwarded_for_hosts[0] -- the entry a caller can inject. Use a CIDR "
        "so uvicorn walks the chain in reverse to the first untrusted hop."
    )


def test_forwarded_allow_ips_is_a_private_network(forwarded_allow_ips: str) -> None:
    """Only HF's own infrastructure can reach the container, and it is RFC1918.

    Trusting a public range would mean trusting a hop we do not operate.
    """
    network = ipaddress.ip_network(forwarded_allow_ips)
    assert network.is_private, f"{forwarded_allow_ips} is not a private network"
