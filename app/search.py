"""Interactive semantic search over Yu-Gi-Oh! cards using RAG.

Accepts natural language queries, retrieves the most semantically similar
cards via pgvector cosine distance, and generates a natural language answer
using an LLM.
"""

import argparse
import logging

from langsmith import traceable
from sqlalchemy import create_engine
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import (
    DATABASE_URL,
    CHAT_MODEL,
    SEARCH_LIMIT,
    EMBEDDING_PROVIDER,
    CANDIDATE_POOL_SIZE,
    OPENAI_API_KEY,
)
from app.embeddings import get_embedding
from app.query_parser import prepare_search_query
from app.retrieval import hybrid_search_sync, rerank

EMBEDDING_COLUMN = "embedding" if EMBEDDING_PROVIDER == "openai" else "embedding_local"

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL)
_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """Build the OpenAI client lazily on first use.

    Raises:
        RuntimeError: If called without OPENAI_API_KEY set.
    """
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Create a .env file with your API key."
            )
        _openai_client = wrap_openai(OpenAI())
    return _openai_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
@traceable(run_type="tool", name="Chat with model")
def get_model_answer(question: str, results_db: list) -> str:
    """Generate a natural language answer from retrieved card results.

    Args:
        question: The user's original query.
        results_db: List of rows from the pgvector similarity query.

    Returns:
        A natural language answer grounded in the retrieved cards.
    """
    card_text = []
    for row in results_db:
        card_text.append(f"- {row['name']}: {row['content']}")

    context = "\n".join(card_text)

    prompt = f"""You are a Yu-Gi-Oh! card expert.
The user asked: "{question}"

Below are cards retrieved from the database. You MUST always describe at least the most relevant cards from this list — never refuse to answer when cards are present.
If the user's query is generic (e.g., "any card", "a card"), simply list and describe the top cards found.
For spell and trap cards, always describe their effects using the card description. DO NOT IGNORE THE CONTEXT PROVIDED.
Only if the list below is COMPLETELY EMPTY (0 cards), say: "I couldn't find relevant cards for that query."
Do not invent effects, stats, or lore that are not in the cards below.

CARDS FOUND:
{context}
"""

    logger.debug("=== PROMPT SENT TO MODEL ===\n%s\n=== END PROMPT ===", prompt)

    response = _get_openai_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,  # Low temperature keeps responses factual
    )

    return response.choices[0].message.content


@traceable(name="Chat Pipeline")
def search_card(
    query: str,
    min_atk: int | None = None,
    max_atk: int | None = None,
    min_def: int | None = None,
    max_def: int | None = None,
    level: int | None = None,
    attribute: str | None = None,
    card_type: str | None = None,
) -> None:
    """Execute a single hybrid search query (vector + lexical + rerank).

    ATK phrases written in the query text (e.g. "more than 4000 ATK",
    "entre 1000 e 2000", ">= 2500", "2500+") are auto-extracted and enforced
    as structured min_atk/max_atk filters, merged with any explicit bounds
    (effective min = max, effective max = min). Only ATK is parsed; DEF,
    level, attribute and card_type must still be passed explicitly.

    Args:
        query: The user's natural language search query.
        min_atk: Minimum ATK (inclusive).
        max_atk: Maximum ATK (inclusive).
        min_def: Minimum DEF (inclusive).
        max_def: Maximum DEF (inclusive).
        level: Exact card level/rank.
        attribute: Exact attribute (e.g. "light", "dark", "wind").
        card_type: A value that must appear in the card's properties/type list.
    """
    search_query, eff_min_atk, eff_max_atk = prepare_search_query(query, min_atk, max_atk)
    logger.info(
        "Parsed ATK bounds from query: eff_min_atk=%s eff_max_atk=%s (remainder=%r)",
        eff_min_atk,
        eff_max_atk,
        search_query,
    )

    logger.info("Generating query embedding...")
    query_vector = get_embedding(search_query)

    logger.info("Searching database with hybrid vector+lexical retrieval...")
    filters = {
        "min_atk": eff_min_atk,
        "max_atk": eff_max_atk,
        "min_def": min_def,
        "max_def": max_def,
        "level": level,
        "attribute": attribute,
        "card_type": card_type,
    }
    candidate_pool_size = max(CANDIDATE_POOL_SIZE, SEARCH_LIMIT * 2)

    with engine.connect() as conn:
        candidates = hybrid_search_sync(
            conn,
            search_query,
            query_vector,
            EMBEDDING_COLUMN,
            filters=filters,
            candidate_pool_size=candidate_pool_size,
        )
        results = rerank(search_query, candidates, SEARCH_LIMIT)
        if OPENAI_API_KEY:
            model_answer = get_model_answer(query, results)
        else:
            model_answer = None

    print(f"\n🏆 Top {len(results)} Results:\n")
    for i, row in enumerate(results, 1):
        print(f"{i}. {row['name']}")
        print(f"   {row['content']}")
        print("-" * 60)

    if model_answer is not None:
        print(f"\n💡 AI Answer:\n{model_answer}\n")


def interactive_loop() -> None:
    """Run the search in an interactive REPL-style loop."""
    print("\n--- 🔍 Yu-Gi-Oh! Semantic Search ---\n")
    print("Type your query or 'quit' / 'exit' to leave.\n")

    while True:
        try:
            question = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye! 👋")
            break

        try:
            search_card(question)
        except Exception:
            logger.exception("Search failed for query: %s", question)
            print("An error occurred. Please try again.\n")


def main() -> None:
    """Entry point for the search CLI."""
    parser = argparse.ArgumentParser(
        description="Semantic search over Yu-Gi-Oh! cards using RAG."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument("--min-atk", type=int, default=None, help="Minimum ATK (inclusive).")
    parser.add_argument("--max-atk", type=int, default=None, help="Maximum ATK (inclusive).")
    parser.add_argument("--min-def", type=int, default=None, help="Minimum DEF (inclusive).")
    parser.add_argument("--max-def", type=int, default=None, help="Maximum DEF (inclusive).")
    parser.add_argument("--level", type=int, default=None, help="Exact card level/rank.")
    parser.add_argument("--attribute", type=str, default=None, help="Exact attribute (e.g. light, dark).")
    parser.add_argument("--card-type", type=str, default=None, help="Value that must appear in properties (e.g. Dragon).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.query:
        search_card(
            args.query,
            min_atk=args.min_atk,
            max_atk=args.max_atk,
            min_def=args.min_def,
            max_def=args.max_def,
            level=args.level,
            attribute=args.attribute,
            card_type=args.card_type,
        )
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
