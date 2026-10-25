"""Run every golden's question through the /ask RAG pipeline and save the filled-in goldens.

Goldens come out of the synthesizer with `input`, `expected_output` and `context`.
This script fills `actual_output` (our answer) and `retrieval_context` (the chunks we
retrieved), which is everything the DeepEval metrics need.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden

from evals.eval_utils import latest_goldens_file
from src.config import get_settings
from src.services.embeddings.factory import make_openai_embeddings_client
from src.services.openai_.factory import make_openai_client
from src.services.pinecone_.client import PineconeClient
from src.services.pinecone_.factory import make_pinecone_client

DATA_DIR = Path(__file__).parent
EVALUATION_DIR = DATA_DIR / "evaluation"


async def answer_golden(golden: Golden, client: PineconeClient, embedder, llm, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        query_embedding = await embedder.embed_text(golden.input)
        hits = client.search(golden.input, query_embedding, top_k=get_settings().deepeval.top_k)
        chunks = [{"arxiv_id": h.get("arxiv_id", ""), "chunk_text": h.get("chunk_text", h.get("abstract", ""))} for h in hits]

        golden.retrieval_context = [c["chunk_text"] for c in chunks]
        if not chunks:
            golden.actual_output = "No relevant papers found for this query."
            return

        rag_result = await llm.generate_rag_response(query=golden.input, chunks=chunks)
        golden.actual_output = rag_result.get("answer", "")


async def main() -> None:
    goldens_file = latest_goldens_file()
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(goldens_file))
    print(f"Loaded {len(dataset.goldens)} goldens from {goldens_file}")

    retriever = get_settings().deepeval.retriever
    client = make_pinecone_client()
    embedder = make_openai_embeddings_client()
    llm = make_openai_client()
    semaphore = asyncio.Semaphore(get_settings().deepeval.max_concurrent)

    await asyncio.gather(*(answer_golden(g, client, embedder, llm, semaphore) for g in dataset.goldens))

    filename = f"{retriever}_{datetime.now():%Y%m%d_%H%M%S}"
    path = dataset.save_as(file_type="json", directory=str(EVALUATION_DIR), file_name=filename)
    print(f"Saved evaluation dataset to {path}")


if __name__ == "__main__":
    asyncio.run(main())
