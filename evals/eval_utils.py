# ruff: noqa: I001  — import order matters: Paper must register on Base before make_database
import json
from deepeval.test_case import LLMTestCase
from src.models.paper import Paper  # noqa: F401
from src.db.factory import make_database
from src.repositories.paper import PaperRepository
from pathlib import Path
from src.services.pinecone_.factory import make_pinecone_client

DATA_DIR = Path(__file__).parent / "data"
GOLDENS_DIR = DATA_DIR / "goldens"
EVALUATION_DIR = DATA_DIR / "evaluation"


def load_context(max_papers: int = 5, chunks_per_paper=3) -> list[list[str]]:
    db = make_database()
    pinecone = make_pinecone_client()
    with db.get_session() as session:
        papers = PaperRepository(session).get_processed_papers(limit=max_papers)
        arxiv_ids = [p.arxiv_id for p in papers]
    contexts = []
    for arxiv_id in arxiv_ids:
        chunks = pinecone.get_chunks_by_paper(arxiv_id)
        good = [c["chunk_text"] for c in chunks if len(c["chunk_text"]) > 300]
        contexts.extend([[text] for text in good[:chunks_per_paper]])
    return contexts


def latest_goldens_file() -> Path:
    files = list(GOLDENS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No goldens in {GOLDENS_DIR} — run generate_goldens first")
    return max(files)


def load_latest_eval_dataset(retriever: str) -> list[dict]:
    files = list(EVALUATION_DIR.glob(f"{retriever}_*.json"))
    if not files:
        raise FileNotFoundError(f"No evaluation dataset in {EVALUATION_DIR} — run generate_evaluation_dataset file first")
    latest_file = max(files)
    with open(latest_file) as file:
        return json.load(file)


def build_test_cases(retriever: str) -> list[LLMTestCase]:
    """Turn the latest evaluation dataset into DeepEval test cases (one per golden)."""
    return [
        LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            expected_output=case["expected_output"],
            retrieval_context=case["retrieval_context"],
        )
        for case in load_latest_eval_dataset(retriever)
    ]
