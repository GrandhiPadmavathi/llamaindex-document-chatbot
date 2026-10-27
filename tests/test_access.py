import json
import logging
import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from chatbot import access, indexing, llm
from chatbot.access import ADMIN, CONTRIBUTOR, READER, AccessPolicy, QuotaExceeded, UsageLedger, matches
from chatbot.answering import FakeLLM
from chatbot.embeddings import configure_embed_model
from chatbot.loaders import load_documents_from_directory

APP = str(Path(__file__).resolve().parents[1] / "app.py")
AUTH_SECRETS = {"auth": {"redirect_uri": "http://localhost:8501/oauth2callback", "cookie_secret": "x" * 32}}
POLICY_ENV = {
    "AUTH_REQUIRED": "1",
    "AUTH_ALLOWED": "@example.com",
    "AUTH_CONTRIBUTORS": "carol@example.com",
    "AUTH_ADMINS": "ada@ops.example.org",
}


def identity(email, verified=True, **extra):
    return {"email": email, "email_verified": verified, "name": email.split("@")[0].title(), **extra}


def test_email_and_domain_matching_is_exact():
    patterns = ("alice@example.com", "@corp.example")
    assert matches("Alice@Example.com", patterns)
    assert matches("bob@corp.example", patterns)
    assert not matches("bob@evil-corp.example", patterns)
    assert not matches("bob@corp.example.attacker.io", patterns)
    assert not matches("bob@sub.corp.example", patterns)
    assert not matches("corp.example", patterns)


def test_roles_from_claims():
    policy = AccessPolicy(required=True, allowed=("@example.com",), contributors=("carol@example.com",),
                          admins=("ada@ops.example.org",))
    assert policy.principal(identity("rita@example.com")).role == READER
    assert policy.principal(identity("CAROL@example.com")).role == CONTRIBUTOR
    assert policy.principal(identity("ada@ops.example.org")).role == ADMIN  # admin outside the allowed domain
    assert policy.principal(identity("mallory@other.org")) is None
    assert policy.principal(identity("rita@example.com", verified=False)) is None
    assert policy.principal({"name": "No Email"}) is None
    entra = policy.principal({"preferred_username": "rita@example.com", "name": "Rita"})
    assert entra is not None and entra.role == READER
    assert AccessPolicy(required=True).principal(identity("anyone@gmail.com")).role == READER


def test_permissions_per_role():
    policy = AccessPolicy(required=True, contributors=("c@x.io",), admins=("a@x.io",))
    reader, contributor, admin = (policy.principal(identity(e)) for e in ("r@x.io", "c@x.io", "a@x.io"))
    assert reader.can("ask") and not reader.can("upload") and not reader.can("advanced")
    assert contributor.can("upload") and contributor.can("advanced") and not contributor.can("usage")
    assert admin.can("usage")
    assert access.LOCAL_USER.can("upload") and not access.LOCAL_USER.can("usage")


def test_policy_from_env(monkeypatch):
    for k, v in {**POLICY_ENV, "AUTH_ALLOWED": " @Example.com , bob@x.io ", "QUOTA_QUESTIONS_PER_DAY": "3"}.items():
        monkeypatch.setenv(k, v)
    policy = AccessPolicy.from_env()
    assert policy.required and policy.allowed == ("@example.com", "bob@x.io") and policy.questions_per_day == 3
    monkeypatch.delenv("AUTH_REQUIRED")
    assert not AccessPolicy.from_env().required


def test_usage_ledger_is_per_user_per_day():
    now = [10.0]
    ledger = UsageLedger(clock=lambda: now[0])
    assert ledger.consume("a", "question", 2) == 1
    assert ledger.consume("a", "question", 2) == 0
    with pytest.raises(QuotaExceeded, match="Daily limit of 2 questions"):
        ledger.consume("a", "question", 2)
    assert ledger.consume("b", "question", 2) == 1
    assert ledger.consume("a", "upload", 0) == -1  # 0 means unlimited
    assert ledger.today()[0] == {"user": "a", "question": 2, "upload": 1}
    now[0] += 86400
    assert ledger.consume("a", "question", 2) == 1 and ledger.today() == [{"user": "a", "question": 1, "upload": 0}]


def test_uploads_can_be_indexed_without_touching_the_disk_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs = load_documents_from_directory(str(Path(APP).parent / "sample_docs"))
    index = indexing.get_or_create_index(docs, "abc", 400, 50, persist=False)
    assert index.docstore.docs and not any(tmp_path.iterdir())


def test_space_deploy_never_uploads_streamlit_secrets(tmp_path, monkeypatch):
    deploy_space = pytest.importorskip("scripts.deploy_space")
    src = tmp_path / "repo"
    (src / ".streamlit").mkdir(parents=True)
    (src / ".streamlit" / "config.toml").write_text("[server]\n")
    (src / ".streamlit" / "secrets.toml").write_text('[auth]\nclient_secret = "s3cret"\n')
    for name in ("app.py", "requirements.txt", "LICENSE"):
        (src / name).write_text("")
    for name in ("chatbot", "sample_docs", "deploy/space"):
        (src / name).mkdir(parents=True)
    for name in ("Dockerfile", "README.md"):
        (src / "deploy" / "space" / name).write_text("")
    monkeypatch.setattr(deploy_space, "ROOT", src)
    out = tmp_path / "staged"
    out.mkdir()
    deploy_space.stage(out)
    assert (out / ".streamlit" / "config.toml").exists() and not list(out.rglob("secrets.toml"))


# --- the Streamlit app, with a simulated identity provider and no model server ---


@pytest.fixture
def app(monkeypatch, tmp_path):
    def fake_configure(config, temperature=0.1):
        configure_embed_model(for_testing=True)
        return FakeLLM("Card refunds take 5-7 business days [1].")

    monkeypatch.setattr(llm, "configure_llama_index", fake_configure)
    monkeypatch.setattr(indexing, "get_or_create_index",
                        lambda docs, _hash, chunk_size, chunk_overlap, **_: indexing.build_index_from_documents(
                            docs, chunk_size, chunk_overlap))
    for k, v in POLICY_ENV.items():
        monkeypatch.setenv(k, v)
    st.cache_resource.clear()

    def make(who=None, secrets=True):
        monkeypatch.setattr(access, "read_identity", lambda: who)
        at = AppTest.from_file(APP, default_timeout=120)
        if secrets:
            at.secrets = AUTH_SECRETS
        return at.run()

    yield make
    st.cache_resource.clear()


def test_sign_in_required_fails_closed_without_a_provider(app):
    at = app(secrets=False)
    assert not at.exception and "no identity provider is configured" in at.error[0].value
    assert not at.chat_input


def test_signed_out_users_only_see_sign_in(app):
    at = app()
    assert [b.label for b in at.button] == ["Sign in"] and not at.chat_input and not at.sidebar.radio


def test_accounts_outside_the_policy_are_refused_and_audited(app, caplog):
    caplog.set_level(logging.INFO, logger="chatbot.audit")
    at = app(identity("mallory@other.org"))
    assert "mallory@other.org is not allowed" in at.error[0].value and not at.chat_input
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "chatbot.audit"]
    assert events[-1]["event"] == "access_denied" and events[-1]["email"] == "mallory@other.org"


def test_reader_gets_samples_only_and_no_costly_settings(app):
    at = app(identity("rita@example.com"))
    assert not at.exception
    assert at.sidebar.radio[0].options == ["Sample documents"]
    assert all(t.disabled for t in at.sidebar.toggle if t.label != "Hybrid search (BM25 + vector)")
    assert "Signed in as **Rita** · reader" in at.sidebar.caption[0].value
    assert not any(e.label == "Usage today" for e in at.sidebar.expander)


def test_contributor_can_upload_and_admin_sees_usage(app):
    contributor = app(identity("carol@example.com"))
    assert contributor.sidebar.radio[0].options == ["Sample documents", "Upload your own"]
    assert not any(t.disabled for t in contributor.sidebar.toggle)
    admin = app(identity("ada@ops.example.org"))
    assert any(e.label == "Usage today" for e in admin.sidebar.expander)


def test_question_quota_survives_a_new_session(app, monkeypatch, caplog):
    monkeypatch.setenv("QUOTA_QUESTIONS_PER_DAY", "1")
    caplog.set_level(logging.INFO, logger="chatbot.audit")
    first = app(identity("rita@example.com"))
    first.chat_input[0].set_value("How long do card refunds take?").run()
    assert not first.exception and "5-7 business days" in first.chat_message[-1].markdown[0].value

    second = app(identity("rita@example.com"))  # new browser session, same person
    second.chat_input[0].set_value("And for bank transfers?").run()
    assert "Daily limit of 1 questions reached" in second.warning[0].value

    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "chatbot.audit"]
    asked = next(e for e in events if e["event"] == "question")
    assert asked["user"] == "rita@example.com" and asked["chars"] == len("How long do card refunds take?")
    assert "refunds" not in json.dumps(asked)
    assert any(e["event"] == "quota_exceeded" for e in events)


def test_local_mode_is_unchanged(app, monkeypatch):
    monkeypatch.delenv("AUTH_REQUIRED")
    at = app(secrets=False)
    assert not at.exception and not at.error
    assert at.sidebar.radio[0].options == ["Sample documents", "Upload your own"]
    assert not any(c.value.startswith("Signed in") for c in at.sidebar.caption)
    assert os.environ.get("AUTH_REQUIRED") is None
