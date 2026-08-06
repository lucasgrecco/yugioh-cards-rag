"""Ingestion pipeline: reads Yu-Gi-Oh! JSON cards, generates embeddings,
and stores them in the PostgreSQL database with pgvector support.
"""

import argparse
import json
import logging
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from app.config import DATABASE_URL, DEFAULT_JSON_DIR
from app.embeddings import get_both_embeddings, get_both_embeddings_batch
from app.models import Base, Card

logger = logging.getLogger(__name__)


def clean_int(value: object) -> int | None:
    """Safely cast a value to int; returns None on failure."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def build_card_content(card_json: dict) -> str:
    """Build a concatenated text representation of a card for embedding.

    Args:
        card_json: Raw card data from a JSON file.

    Returns:
        A pipe-delimited string of the card's key fields.
    """
    parts = [f"Name: {card_json.get('name')}"]

    attribute = card_json.get("englishAttribute")
    if attribute:
        parts.append(f"Attribute: {attribute}")

    atk = clean_int(card_json.get("atk"))
    defense = clean_int(card_json.get("def"))
    level = clean_int(card_json.get("level"))

    if level is not None:
        parts.append(f"Level: {level}")
    if atk is not None:
        parts.append(f"ATK: {atk}")
    if defense is not None:
        parts.append(f"DEF: {defense}")

    properties = card_json.get("properties")
    if properties is not None:
        parts.append(f"PROPERTIES: {properties}")

    full_content = " | ".join(parts)

    effect = card_json.get("effectText", "")
    if effect:
        full_content += f" | Description: {effect}"

    return full_content

BATCH_SIZE = 100  # Safe margin below OpenAI's 2048 input limit


def upsert_card_with_vectors(
    card_json: dict,
    openai_vector: list[float] | None,
    local_vector: list[float] | None,
    session: Session,
) -> None:
    """Upsert a single card with pre-computed embedding vectors. Does NOT commit.

    Args:
        card_json: Raw card data from a JSON file.
        openai_vector: Pre-computed OpenAI embedding (or None).
        local_vector: Pre-computed local embedding (or None).
        session: An active SQLAlchemy session.

    Raises:
        ValueError: If card_json has no 'id' field.
    """
    card_id = card_json.get("id")
    if card_id is None:
        raise ValueError("Card JSON missing 'id' field")

    content = build_card_content(card_json)

    values = {
        "card_json_id": card_id,
        "name": card_json.get("name"),
        "level": clean_int(card_json.get("level")),
        "atk": clean_int(card_json.get("atk")),
        "def_": clean_int(card_json.get("def")),
        "english_attribute": card_json.get("englishAttribute"),
        "properties": card_json.get("properties"),
        "content": content,
    }
    if openai_vector is not None:
        values["embedding"] = openai_vector
    if local_vector is not None:
        values["embedding_local"] = local_vector

    stmt = pg_insert(Card).values(**values).on_conflict_do_update(
        index_elements=["card_json_id"],
        set_=values,
    )
    session.execute(stmt)


def upsert_card(card_json: dict, session: Session) -> None:
    """Upsert a single card by card_json_id. Does NOT commit.

    Generates BOTH OpenAI and local embeddings, then delegates to
    :func:`upsert_card_with_vectors`.

    Args:
        card_json: Raw card data from a JSON file.
        session: An active SQLAlchemy session.

    Raises:
        ValueError: If card_json has no 'id' field.
    """
    content = build_card_content(card_json)
    openai_vec, local_vec = get_both_embeddings(content)
    upsert_card_with_vectors(card_json, openai_vec, local_vec, session)


def _stem_id(path: Path) -> int | None:
    """Parse a JSON file's stem as an int card id; None if not numeric.

    Resume relies on the convention that card files are named ``<id>.json``
    (filename stem == the JSON ``id`` field). Files with non-numeric stems are
    never skipped by resume and are always processed.
    """
    try:
        return int(path.stem)
    except ValueError:
        return None

def process_jsons(json_dir: str, session: Session, force: bool = False) -> int:
    """Ingest all JSON card files from the given directory using batch embeddings.

    Cards are processed in batches of BATCH_SIZE to minimize OpenAI API calls.
    If a batch embedding call fails, falls back to individual processing.

    By default, files whose ``card_json_id`` already exists in the database are
    skipped before being read or embedded (resume support). Pass ``force=True``
    to re-ingest every file.

    Args:
        json_dir: Path to the directory containing ``.json`` card files.
        session: An active SQLAlchemy session.
        force: If True, re-ingest all files, ignoring existing card_json_ids.

    Returns:
        The number of cards ingested.
    """
    json_path = Path(json_dir)
    json_files = sorted(json_path.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", json_dir)
        return 0

    if not force:
        existing_ids = set(session.scalars(select(Card.card_json_id)).all())
        kept: list[Path] = []
        skipped = 0
        for json_file in json_files:
            stem_id = _stem_id(json_file)
            if stem_id is not None and stem_id in existing_ids:
                skipped += 1
                continue
            kept.append(json_file)
        if skipped:
            logger.info("Skipping %d already-ingested card(s)", skipped)
        json_files = kept

    if not json_files:
        logger.info("No new cards to ingest in %s", json_dir)
        return 0

    count = 0
    total = len(json_files)

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Ingesting cards", unit="batch"):
        batch_files = json_files[i:i + BATCH_SIZE]
        batch_cards: list[dict] = []
        batch_texts: list[str] = []

        for json_file in batch_files:
            with open(json_file, "r", encoding="utf-8") as f:
                card_json = json.load(f)
            content = build_card_content(card_json)
            batch_cards.append(card_json)
            batch_texts.append(content)

        try:
            openai_vecs, local_vecs = get_both_embeddings_batch(batch_texts)
        except Exception:
            logger.exception(
                "Batch embedding failed for batch %d, falling back to individual",
                i // BATCH_SIZE,
            )
            for card_json in batch_cards:
                try:
                    upsert_card(card_json, session)
                    count += 1
                except Exception:
                    logger.exception("Failed: %s", card_json.get("name"))
        else:
            for j, card_json in enumerate(batch_cards):
                try:
                    openai_vec = openai_vecs[j] if openai_vecs else None
                    local_vec = local_vecs[j] if local_vecs else None
                    upsert_card_with_vectors(
                        card_json, openai_vec, local_vec, session
                    )
                    count += 1
                except Exception:
                    logger.exception("Failed: %s", card_json.get("name"))

        session.commit()

    return count


def show_stats(session: Session) -> None:
    """Print current card count in the database."""
    total = session.scalar(select(func.count()).select_from(Card))
    print(f"Database contains {total} cards.")

def main() -> None:
    """Entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest Yu-Gi-Oh! card JSONs into the RAG database."
    )
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        help=f"Directory containing .json card files (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest all cards, ignoring already-ingested card_json_ids",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logger.info("Starting card ingestion from %s ...", args.json_dir)

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        count = process_jsons(args.json_dir, session, force=args.force)

    logger.info("Ingestion complete! %d cards ingested.", count)
    with session_factory() as stats_session:
        show_stats(stats_session)

if __name__ == "__main__":
    main()
