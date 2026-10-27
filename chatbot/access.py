"""Who may use the app and what they may do: sign-in policy, roles, daily quotas and an audit log.

Sign-in itself is Streamlit's OIDC support (`st.login`) against any provider configured in `.streamlit/secrets.toml`
(Google, Microsoft Entra ID, Okta, Auth0, ...). This module only decides what a signed-in identity is allowed to do,
so it can be tested without a browser or an identity provider.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass

READER, CONTRIBUTOR, ADMIN = "reader", "contributor", "admin"
PERMISSIONS = {
    READER: frozenset({"ask"}),
    CONTRIBUTOR: frozenset({"ask", "upload", "advanced"}),
    ADMIN: frozenset({"ask", "upload", "advanced", "usage"}),
}

audit = logging.getLogger("chatbot.audit")
if not audit.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s audit %(message)s"))
    audit.addHandler(_handler)
    audit.setLevel(logging.INFO)


def log_event(event: str, **fields) -> None:
    audit.info(json.dumps({"event": event, **fields}))


def auth_configured() -> bool:
    import streamlit as st

    try:
        return "auth" in st.secrets
    except Exception:  # no secrets.toml at all
        return False


def read_identity() -> dict | None:
    """Claims of the signed-in user, or None. Tests replace this to simulate an identity provider."""
    import streamlit as st

    return st.user.to_dict() if st.user.is_logged_in else None


def _patterns(value: str) -> tuple[str, ...]:
    return tuple(p.strip().lower() for p in value.split(",") if p.strip())


def matches(email: str, patterns: tuple[str, ...]) -> bool:
    """`alice@example.com` matches itself; `@example.com` matches any address at exactly that domain."""
    email = email.strip().lower()
    domain = "@" + email.rpartition("@")[2] if "@" in email else None
    return any(p == email or (p.startswith("@") and p == domain) for p in patterns)


@dataclass(frozen=True)
class Principal:
    email: str
    name: str
    role: str

    @property
    def permissions(self) -> frozenset[str]:
        return PERMISSIONS[self.role]

    def can(self, permission: str) -> bool:
        return permission in self.permissions


LOCAL_USER = Principal("local", "Local user", CONTRIBUTOR)


@dataclass(frozen=True)
class AccessPolicy:
    required: bool = False
    provider: str | None = None
    allowed: tuple[str, ...] = ()
    contributors: tuple[str, ...] = ()
    admins: tuple[str, ...] = ()
    questions_per_day: int = 50
    uploads_per_day: int = 5

    @classmethod
    def from_env(cls) -> "AccessPolicy":
        env = os.environ
        return cls(
            required=env.get("AUTH_REQUIRED", "0").lower() in ("1", "true", "yes"),
            provider=env.get("AUTH_PROVIDER") or None,
            allowed=_patterns(env.get("AUTH_ALLOWED", "")),
            contributors=_patterns(env.get("AUTH_CONTRIBUTORS", "")),
            admins=_patterns(env.get("AUTH_ADMINS", "")),
            questions_per_day=int(env.get("QUOTA_QUESTIONS_PER_DAY", "50")),
            uploads_per_day=int(env.get("QUOTA_UPLOADS_PER_DAY", "5")),
        )

    def principal(self, identity: dict) -> Principal | None:
        """Map OIDC claims to a role, or None when the account may not use the app.

        Accounts whose provider says the email is unverified are refused, since anyone can type an address into
        some providers. Admins and contributors are allowed even if `allowed` is set and does not list them.
        """
        email = str(identity.get("email") or identity.get("preferred_username") or "").strip().lower()
        if "@" not in email or identity.get("email_verified") is False:
            return None
        name = str(identity.get("name") or email)
        if matches(email, self.admins):
            return Principal(email, name, ADMIN)
        if matches(email, self.contributors):
            return Principal(email, name, CONTRIBUTOR)
        if not self.allowed or matches(email, self.allowed):
            return Principal(email, name, READER)
        return None


class QuotaExceeded(Exception):
    pass


class UsageLedger:
    """Per-user daily counters shared by every browser session of the server process.

    Unlike a per-session limit, refreshing the page or opening a new tab does not reset it.
    """

    def __init__(self, clock=time.time):
        self.clock = clock
        self._counts: dict[tuple[str, str, int], int] = {}
        self._lock = threading.Lock()

    def _day(self) -> int:
        return int(self.clock() // 86400)

    def consume(self, user: str, kind: str, limit: int) -> int:
        """Record one use and return how many are left today; raise QuotaExceeded at the limit (0 = unlimited)."""
        with self._lock:
            key = (user, kind, self._day())
            used = self._counts.get(key, 0)
            if limit > 0 and used >= limit:
                raise QuotaExceeded(f"Daily limit of {limit} {kind}s reached; it resets at 00:00 UTC.")
            self._counts[key] = used + 1
            return max(limit - used - 1, 0) if limit > 0 else -1

    def today(self) -> list[dict]:
        day = self._day()
        with self._lock:
            rows: dict[str, dict] = {}
            for (user, kind, d), n in self._counts.items():
                if d == day:
                    rows.setdefault(user, {"user": user, "question": 0, "upload": 0})[kind] = n
        return sorted(rows.values(), key=lambda r: -r["question"])
