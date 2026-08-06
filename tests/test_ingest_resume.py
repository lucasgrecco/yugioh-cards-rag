"""DB-free unit tests for ingest resume and commit-per-batch behavior."""

import json

import app.ingest as ingest


class _FakeScalars(list):
    """List-like result that supports the ``.all()`` call used by scalars()."""

    def all(self):
        return list(self)


class FakeSession:
    """Minimal stand-in for a SQLAlchemy Session (no DB, no ORM)."""

    def __init__(self, existing_ids=()):
        self._existing_ids = list(existing_ids)
        self.commits = 0
        self.scalars_called = 0
        self.upserts = []

    def scalars(self, stmt):
        self.scalars_called += 1
        return _FakeScalars(self._existing_ids)

    def execute(self, stmt):
        self.upserts.append(stmt)

    def commit(self):
        self.commits += 1


def _write_card(tmp_path, card_id, name="Card"):
    path = tmp_path / f"{card_id}.json"
    path.write_text(json.dumps({"id": card_id, "name": name}))
    return path


def _fake_batch_embeddings(calls):
    def fake(texts):
        calls.append(list(texts))
        return ([[0.1] * 3] * len(texts), [[0.2] * 3] * len(texts))
    return fake


def test_commit_once_per_batch(tmp_path, monkeypatch):
    """BATCH_SIZE=2 with 3 files must produce exactly 2 commits (batch granularity,
    not one-at-the-end and not per-card)."""
    for card_id in (1, 2, 3):
        _write_card(tmp_path, card_id)
    session = FakeSession()
    monkeypatch.setattr(ingest, "BATCH_SIZE", 2)
    embedding_calls = []
    monkeypatch.setattr(
        ingest, "get_both_embeddings_batch", _fake_batch_embeddings(embedding_calls)
    )

    count = ingest.process_jsons(str(tmp_path), session)

    assert count == 3
    assert session.commits == 2
    assert len(session.upserts) == 3
    assert len(embedding_calls) == 2


def test_resume_skips_existing_ids(tmp_path, monkeypatch):
    """Files whose card_json_id already exists are skipped: no embedding, no upsert."""
    for card_id in (1, 2, 3):
        _write_card(tmp_path, card_id)
    session = FakeSession(existing_ids=[1, 2])
    monkeypatch.setattr(ingest, "BATCH_SIZE", 1)
    embedding_calls = []
    monkeypatch.setattr(
        ingest, "get_both_embeddings_batch", _fake_batch_embeddings(embedding_calls)
    )

    count = ingest.process_jsons(str(tmp_path), session)

    assert count == 1
    assert session.commits == 1
    assert len(session.upserts) == 1
    assert len(embedding_calls) == 1
    assert session.scalars_called == 1


def test_force_processes_all(tmp_path, monkeypatch):
    """--force disables the resume skip: every file is processed, no existing-id query."""
    for card_id in (1, 2, 3):
        _write_card(tmp_path, card_id)
    session = FakeSession(existing_ids=[1, 2])
    monkeypatch.setattr(ingest, "BATCH_SIZE", 1)
    embedding_calls = []
    monkeypatch.setattr(
        ingest, "get_both_embeddings_batch", _fake_batch_embeddings(embedding_calls)
    )

    count = ingest.process_jsons(str(tmp_path), session, force=True)

    assert count == 3
    assert session.commits == 3
    assert len(session.upserts) == 3
    assert len(embedding_calls) == 3
    assert session.scalars_called == 0

def test_all_skipped_returns_zero_without_commit(tmp_path, monkeypatch):
    """All files already ingested -> early return 0, no embedding, no commit."""
    for card_id in (1, 2, 3):
        _write_card(tmp_path, card_id)
    session = FakeSession(existing_ids=[1, 2, 3])
    monkeypatch.setattr(ingest, "BATCH_SIZE", 1)
    embedding_calls = []
    monkeypatch.setattr(
        ingest, "get_both_embeddings_batch", _fake_batch_embeddings(embedding_calls)
    )

    count = ingest.process_jsons(str(tmp_path), session)

    assert count == 0
    assert session.commits == 0
    assert session.upserts == []
    assert embedding_calls == []

def test_stem_id_parses_numeric_and_returns_none_otherwise():
    """_stem_id: numeric stem -> int; non-numeric stem -> None."""
    from pathlib import Path

    assert ingest._stem_id(Path("10000.json")) == 10000
    assert ingest._stem_id(Path("blue-eyes.json")) is None


def test_fallback_path_commits_per_batch(tmp_path, monkeypatch):
    """When batch embedding fails, the per-card fallback still commits per batch."""
    for card_id in (1, 2, 3):
        _write_card(tmp_path, card_id)
    session = FakeSession()
    monkeypatch.setattr(ingest, "BATCH_SIZE", 1)

    def boom(texts):
        raise RuntimeError("batch embedding failed")

    monkeypatch.setattr(ingest, "get_both_embeddings_batch", boom)

    fallback_calls = []

    def fake_upsert_card(card_json, session):
        fallback_calls.append(card_json["id"])

    monkeypatch.setattr(ingest, "upsert_card", fake_upsert_card)

    count = ingest.process_jsons(str(tmp_path), session)

    assert count == 3
    assert session.commits == 3
    assert fallback_calls == [1, 2, 3]
