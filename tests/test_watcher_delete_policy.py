"""DB-free unit tests for the watcher delete policy.

Covers the policy scenarios (5 file + 3 directory) using a fake
``session_factory`` and simulated watchdog events:

a. move within the tree -> ``_handle_change(dest)`` and NOT ``_handle_delete(src)``
b. move outside the tree -> ``_handle_delete(src)``
c. move from outside into the tree -> ``_handle_change(dest)`` and NOT ``_handle_delete(src)``
d. ``on_deleted`` (file) -> ``_handle_delete(src)``
e. directory moved out -> ``_reconcile()`` deletes cards whose files are absent
f. directory deleted -> ``_reconcile()`` deletes absent cards
g. directory moved in -> ``_handle_change`` called for each card file
"""

from types import SimpleNamespace

from app.watcher import CardFileHandler


class FakeSession:
    """Minimal fake SQLAlchemy session used by ``_reconcile``."""

    def __init__(self, existing: list[int]) -> None:
        self.existing = list(existing)
        self.executed: list[object] = []
        self.committed = 0
        self.rolled_back = 0

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def scalars(self, stmt: object) -> "FakeSession":
        return self

    def all(self) -> list[int]:
        return self.existing

    def execute(self, stmt: object) -> SimpleNamespace:
        self.executed.append(stmt)
        return SimpleNamespace(rowcount=1)

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1


def _make_handler(tmp_path):
    """Handler with recorded _handle_change/_handle_delete and no debounce."""
    calls = {"change": [], "delete": []}
    handler = CardFileHandler(
        session_factory=lambda: None,
        watched_dir=tmp_path,
    )
    handler._handle_change = lambda p: calls["change"].append(p)
    handler._handle_delete = lambda p: calls["delete"].append(p)
    handler._debounce = lambda *args, **kwargs: True
    return handler, calls


def _outside(tmp_path):
    """A directory that is NOT inside the watched tree."""
    return tmp_path.parent / "outside"


# --- file scenarios -------------------------------------------------------


def test_move_within_tree_upserts_dest_and_does_not_delete_src(tmp_path):
    """Req 1: moving a JSON to a subfolder inside the tree keeps the card."""
    sub = tmp_path / "sub"
    sub.mkdir()
    handler, calls = _make_handler(tmp_path)
    event = SimpleNamespace(
        src_path=str(tmp_path / "10000.json"),
        dest_path=str(sub / "10000.json"),
        is_directory=False,
    )
    handler.on_moved(event)
    assert calls["change"] == [str(sub / "10000.json")]
    assert calls["delete"] == []


def test_rename_within_tree_upserts_dest_and_does_not_delete_src(tmp_path):
    """Req 2: renaming 10000.json -> 10001.json inside the tree keeps the card."""
    handler, calls = _make_handler(tmp_path)
    event = SimpleNamespace(
        src_path=str(tmp_path / "10000.json"),
        dest_path=str(tmp_path / "10001.json"),
        is_directory=False,
    )
    handler.on_moved(event)
    assert calls["change"] == [str(tmp_path / "10001.json")]
    assert calls["delete"] == []

def test_rename_json_to_non_json_inside_tree_deletes_src(tmp_path):
    """Renaming 10000.json -> notes.txt inside the tree removes the card
    (index mirrors the tree: the file is no longer a card JSON)."""
    handler, calls = _make_handler(tmp_path)
    event = SimpleNamespace(
        src_path=str(tmp_path / "10000.json"),
        dest_path=str(tmp_path / "notes.txt"),
        is_directory=False,
    )
    handler.on_moved(event)
    assert calls["delete"] == [str(tmp_path / "10000.json")]
    assert calls["change"] == []


def test_move_outside_tree_deletes_src(tmp_path):
    """Req 3: moving a JSON outside the tree deletes the card."""
    handler, calls = _make_handler(tmp_path)
    event = SimpleNamespace(
        src_path=str(tmp_path / "10000.json"),
        dest_path=str(_outside(tmp_path) / "10000.json"),
        is_directory=False,
    )
    handler.on_moved(event)
    assert calls["delete"] == [str(tmp_path / "10000.json")]
    assert calls["change"] == []


def test_move_into_tree_upserts_dest_and_does_not_delete_src(tmp_path):
    """Req 5: moving a JSON from outside into the tree upserts (treated as created)."""
    handler, calls = _make_handler(tmp_path)
    event = SimpleNamespace(
        src_path=str(_outside(tmp_path) / "10000.json"),
        dest_path=str(tmp_path / "10000.json"),
        is_directory=False,
    )
    handler.on_moved(event)
    assert calls["change"] == [str(tmp_path / "10000.json")]
    assert calls["delete"] == []


def test_on_deleted_file_deletes_src(tmp_path):
    """Req 4: deleting a JSON still deletes the card."""
    handler, calls = _make_handler(tmp_path)
    event = SimpleNamespace(
        src_path=str(tmp_path / "10000.json"),
        is_directory=False,
    )
    handler.on_deleted(event)
    assert calls["delete"] == [str(tmp_path / "10000.json")]


# --- directory scenarios --------------------------------------------------


def test_directory_moved_out_reconciles_absent_cards(tmp_path):
    """Req 6: directory moved out -> reconciliation deletes absent cards."""
    for i in (1, 2, 3):
        (tmp_path / f"{i}.json").write_text("{}")
    session = FakeSession([1, 2, 3, 4])
    handler = CardFileHandler(
        session_factory=lambda: session, watched_dir=tmp_path
    )
    handler._debounce = lambda *args, **kwargs: True
    event = SimpleNamespace(
        src_path=str(tmp_path / "sub"),
        dest_path=str(_outside(tmp_path) / "sub"),
        is_directory=True,
    )
    handler.on_moved(event)
    assert session.executed, "directory moved out must trigger the bulk delete"
    assert session.committed == 1
    # existing {1,2,3,4} - tree {1,2,3} -> to_delete {4}
    assert handler._reconcile() == {4}


def test_directory_deleted_reconciles_absent_cards(tmp_path):
    """Req 7: directory deleted -> reconciliation deletes absent cards."""
    for i in (1, 2, 3):
        (tmp_path / f"{i}.json").write_text("{}")
    session = FakeSession([1, 2, 3, 4])
    handler = CardFileHandler(
        session_factory=lambda: session, watched_dir=tmp_path
    )
    handler._debounce = lambda *args, **kwargs: True
    event = SimpleNamespace(
        src_path=str(tmp_path / "sub"),
        is_directory=True,
    )
    handler.on_deleted(event)
    assert session.executed, "directory deleted must trigger the bulk delete"
    assert session.committed == 1
    assert handler._reconcile() == {4}

def test_reconcile_nothing_to_delete_does_not_execute(tmp_path):
    """Reconciliation with nothing absent does not execute or commit."""
    for i in (1, 2, 3):
        (tmp_path / f"{i}.json").write_text("{}")
    session = FakeSession([1, 2, 3])
    handler = CardFileHandler(
        session_factory=lambda: session, watched_dir=tmp_path
    )
    assert handler._reconcile() == set()
    assert session.executed == []
    assert session.committed == 0


def test_directory_moved_into_tree_ingests_card_files(tmp_path):
    """Req 8: directory moved in -> each card file is upserted (dotfiles skipped)."""
    moved = tmp_path / "moved_in"
    (moved / "sub").mkdir(parents=True)
    (moved / "10000.json").write_text("{}")
    (moved / "sub" / "10001.json").write_text("{}")
    (moved / ".hidden.json").write_text("{}")
    handler, calls = _make_handler(tmp_path)
    event = SimpleNamespace(
        src_path=str(_outside(tmp_path) / "moved_in"),
        dest_path=str(moved),
        is_directory=True,
    )
    handler.on_moved(event)
    assert sorted(calls["change"]) == sorted(
        [str(moved / "10000.json"), str(moved / "sub" / "10001.json")]
    )
    assert calls["delete"] == []
