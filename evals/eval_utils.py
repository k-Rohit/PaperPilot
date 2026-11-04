# ruff: noqa: I001  — import order matters: Paper must register on Base before make_database
import json
import random
from collections import defaultdict
from deepeval.test_case import LLMTestCase
from src.models.paper import Paper  # noqa: F401
from src.config import get_settings
from statistics import mean
from src.db.factory import make_database
from src.repositories.paper import PaperRepository
from pathlib import Path
from src.services.pinecone_.factory import make_pinecone_client

DATA_DIR = Path(__file__).parent / "data"
GOLDENS_DIR = DATA_DIR / "goldens"
EVALUATION_DIR = DATA_DIR / "evaluation"
REPORTS_DIR = Path(__file__).parent / "reports"
ROOT_DIR = Path(__file__).parent.parent
HISTORICAL_RUNS_JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"


def load_documents(num_papers: int = 10, seed: int = 42) -> dict[str, str]:
    """{arxiv_id: raw_text} for a random sample of papers that are parsed AND indexed in Pinecone."""
    db = make_database()
    pinecone = make_pinecone_client()
    with db.get_session() as session:
        papers = PaperRepository(session).get_processed_papers(limit=10_000)  # ALL papers
        docs = {p.arxiv_id: p.raw_text for p in papers if p.raw_text}  # skip papers with no parsed text

    random.seed(seed)  # same shuffle every run
    candidates = random.sample(sorted(docs), len(docs))  # shuffle all ids
    picked = {}
    for arxiv_id in candidates:  # walk the shuffled list...
        if pinecone.get_chunks_by_paper(arxiv_id):  # ...skip any not in Pinecone
            picked[arxiv_id] = docs[arxiv_id]
        if len(picked) == num_papers:  # ...stop at num_papers
            break
    return picked


def latest_goldens_file() -> Path:
    files = list(GOLDENS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No goldens in {GOLDENS_DIR} — run generate_goldens first")
    return max(files)


def latest_report() -> Path:
    files = list(REPORTS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No reports in {REPORTS_DIR} — run evaluate_rag_app first")
    return max(files)


def latest_eval_dataset_file(retriever: str) -> Path:
    files = list(EVALUATION_DIR.glob(f"{retriever}_*.json"))
    if not files:
        raise FileNotFoundError(f"No evaluation dataset in {EVALUATION_DIR} — run generate_evaluation_dataset file first")
    return max(files)


def load_latest_eval_dataset(retriever: str) -> list[dict]:
    with open(latest_eval_dataset_file(retriever)) as file:
        return json.load(file)


def metrics_from_result(result) -> dict[str, float]:
    scores = defaultdict(list)
    for test in result.test_results:
        for m in test.metrics_data:
            if m.score is not None:
                scores[m.name].append(m.score)
    metrics = {name: mean(vals) for name, vals in scores.items()}
    metrics["pass_rate"] = mean(t.success for t in result.test_results)
    return metrics


def collect_params(pipeline: str = "direct") -> dict:
    """Every setting that can change the eval scores, read from the same places the code reads them."""
    s = get_settings()
    goldens_file = latest_goldens_file()
    return {
        # retrieval
        "top_k": s.retrieval.top_k,
        "alpha": s.pinecone_.alpha,
        "embedding_model": s.openai_embeddings.model,
        "embedding_dims": s.openai_embeddings.dimensions,
        "sparse_model": s.pinecone_.sparse_model,
        # chunking
        "chunk_size": s.chunking.chunk_size,
        "overlap": s.chunking.overlap_size,
        "min_chunk_size": s.chunking.min_chunk_size,
        "min_section_words": s.chunking.min_section_words,
        "max_section_words": s.chunking.max_section_words,
        # generation
        "llm_model": s.openai_client.model,
        "temperature": s.openai_client.temperature,
        "max_tokens": s.openai_client.max_tokens,
        # grading
        "judge_model": s.deepeval.judge_model,
        "threshold": s.deepeval.threshold,
        # traceability
        "pipeline": pipeline,
        "goldens_file": goldens_file.stem,
        "num_goldens": len(json.loads(goldens_file.read_text())),
        "eval_dataset_file": latest_eval_dataset_file(s.deepeval.retriever).stem,
    }


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
