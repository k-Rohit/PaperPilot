import tempfile
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, EvolutionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv

from evals.eval_utils import load_documents
from src.config import get_settings

load_dotenv()

GOLDENS_DIR = Path(__file__).parent / "goldens"


def generate_goldens() -> None:
    docs = load_documents(num_papers=10)  # {arxiv_id: raw_text} read from Postgres
    print(f"Loaded {len(docs)} papers")

    evolution_config = EvolutionConfig(
        num_evolutions=2,
        evolutions={
            Evolution.REASONING: 1 / 6,
            Evolution.CONCRETIZING: 1 / 6,
            Evolution.CONSTRAINED: 1 / 6,
            Evolution.IN_BREADTH: 1 / 6,
            Evolution.MULTICONTEXT: 1 / 6,
            Evolution.COMPARATIVE: 1 / 6,
        },
    )
    context_config = ContextConstructionConfig(
        max_contexts_per_document=2,
        max_context_length=3,
        context_quality_threshold=0.5,
        context_similarity_threshold=0.5,
    )
    synthesizer = Synthesizer(model=get_settings().deepeval.synthesizer_model, evolution_config=evolution_config)

    # DeepEval only accepts file paths, so write each paper's text to a temporary .txt file.
    # The files must exist while DeepEval reads them, so the call stays inside this block.
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for arxiv_id, text in docs.items():
            path = Path(tmp) / f"{arxiv_id}.txt"  # e.g. /tmp/abc123/2607.28587v1.txt
            path.write_text(text)  # ← text from Postgres goes into the file here
            paths.append(str(path))

        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=paths,
            context_construction_config=context_config,
            max_goldens_per_context=2,
        )
    # Temp folder is deleted here; the goldens are already in memory.

    print(f"Generated {len(goldens)} goldens")
    synthesizer.save_as(file_type="json", directory=GOLDENS_DIR)


if __name__ == "__main__":
    generate_goldens()
