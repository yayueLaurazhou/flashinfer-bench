"""
RAG (Retrieval-Augmented Generation) helper for the kernel generator.

Uses an LLM to decompose the kernel generation context into targeted search
queries, then performs async hybrid search (dense + BM25 sparse) against the
CUDA documentation Milvus database and returns aggregated reference material.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from flashinfer_bench import Definition, EvaluationStatus, Trace

from utils import LLMClient, format_definition, format_trace_logs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt for the LLM to generate search queries
# ---------------------------------------------------------------------------

QUERY_GENERATION_PROMPT = """You are an expert CUDA/GPU programming assistant.
Given the following kernel specification and context, generate exactly 3 short,
specific search queries that would retrieve the most relevant CUDA documentation
to help write or fix this kernel.

Each query should target a different aspect:
1. The core algorithm / operation (e.g. "matrix multiply tiling shared memory")
2. The data types, memory patterns, or correctness concern (e.g. "float16 accumulation precision loss")
3. The GPU architecture or performance optimization angle (e.g. "H100 thread block cluster occupancy")

Kernel Specification:
{definition}

{error_context}

Return ONLY a JSON array of exactly 3 query strings, nothing else.
Example: ["query one", "query two", "query three"]
"""


class RAGHelper:
    """Retrieval-Augmented Generation helper that enriches prompts with
    relevant CUDA documentation retrieved from a Milvus vector database."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        db_path: Optional[str] = None,
        collection_name: str = "cuda_documentation",
        embedding_model_name: str = "nvidia/NV-Embed-v2",
        llm_model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_k: int = 3,
    ):
        """
        Args:
            llm: An existing LLMClient instance. If provided, ``llm_model_name``,
                 ``api_key``, and ``base_url`` are ignored for the LLM.
            db_path: Path to the Milvus Lite database file. Defaults to the
                     one in flashinfer_bench/agents/RAG/milvus_lite.db.
            collection_name: Milvus collection to search.
            embedding_model_name: SentenceTransformer model used at index time.
            llm_model_name: LLM model for query generation (used only when
                            ``llm`` is not provided).
            api_key: OpenAI-compatible API key (falls back to LLM_API_KEY env var).
                     Used only when ``llm`` is not provided.
            base_url: Optional base URL for a non-OpenAI endpoint.
                      Used only when ``llm`` is not provided.
            top_k: Number of results to return per query.
        """
        # Lazy imports – only needed when RAG is actually used
        from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
        from sentence_transformers import SentenceTransformer

        self._AnnSearchRequest = AnnSearchRequest
        self._RRFRanker = RRFRanker

        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent.parent
                / "flashinfer_bench" / "agents" / "RAG" / "milvus_lite.db"
            )

        self.client = MilvusClient(uri=db_path)
        self.collection_name = collection_name
        self.top_k = top_k

        # Embedding model (same one used at build time)
        self.embed_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

        # LLM client — reuse an existing one or create a lightweight instance
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMClient(
                model_name=llm_model_name,
                api_key=api_key,
                base_url=base_url,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        definition: Definition,
        trace: Optional[Trace] = None,
        current_code: Optional[str] = None,
    ) -> str:
        """High-level entry point: generate queries ➜ search ➜ return formatted RAG data.

        Args:
            definition: The kernel definition / specification.
            trace: Optional evaluation trace (used to tailor queries to the error type).
            current_code: Optional current kernel source code.

        Returns:
            A formatted string of relevant documentation snippets ready to be
            injected into the ``{rag_data}`` placeholder in the optimization prompt.
        """
        logger.info("[RAG] Starting retrieval for definition '%s'", definition.name)
        t_total = time.perf_counter()

        logger.info("[RAG] Generating search queries via LLM...")
        t0 = time.perf_counter()
        queries = await self._generate_queries(definition, trace, current_code)
        logger.info("[RAG] Query generation complete (%.2fs) — queries: %s", time.perf_counter() - t0, queries)

        # Run all searches concurrently
        logger.info("[RAG] Running %d hybrid searches in parallel...", len(queries))
        t0 = time.perf_counter()
        search_tasks = [self._search(q) for q in queries]
        all_results = await asyncio.gather(*search_tasks)
        total_hits = sum(len(r) for r in all_results)
        logger.info("[RAG] Hybrid search complete (%.2fs) — %d total hits", time.perf_counter() - t0, total_hits)

        # Deduplicate by document id, preserving order
        seen_ids = set()
        unique_results = []
        for results in all_results:
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)

        formatted = self._format_results(unique_results)
        logger.info(
            "[RAG] Retrieval complete (%.2fs total) — %d unique docs, %d chars of context",
            time.perf_counter() - t_total, len(unique_results), len(formatted),
        )
        return formatted

    def retrieve_sync(
        self,
        definition: Definition,
        trace: Optional[Trace] = None,
        current_code: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper around :meth:`retrieve`."""
        return asyncio.run(self.retrieve(definition, trace, current_code))

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    async def _generate_queries(
        self,
        definition: Definition,
        trace: Optional[Trace] = None,
        current_code: Optional[str] = None,
    ) -> List[str]:
        """Use an LLM to rewrite the kernel context into 3 targeted search queries."""
        definition_str = format_definition(definition)

        # Build error context if a trace is available
        error_context = ""
        if trace is not None and trace.evaluation is not None:
            status = trace.evaluation.status
            trace_logs = format_trace_logs(trace)
            error_context = (
                f"Current Error Status: {status.value}\n"
                f"Evaluation Details:\n{trace_logs}\n"
            )
            if current_code:
                # Include a truncated version so the LLM can see what was attempted
                truncated = current_code[:2000] + ("..." if len(current_code) > 2000 else "")
                error_context += f"\nCurrent Code (truncated):\n{truncated}\n"

        prompt = QUERY_GENERATION_PROMPT.format(
            definition=definition_str,
            error_context=error_context if error_context else "No prior evaluation — this is the first generation attempt.",
        )

        try:
            raw = await self.llm.complete(prompt, temperature=0.3)

            # Parse the JSON array
            import json
            queries = json.loads(raw)
            if isinstance(queries, list) and len(queries) >= 1:
                return queries[:3]
        except Exception as e:
            logger.warning("[RAG] Query generation LLM call failed (%s), using fallback queries", e)

        # Fallback: craft generic queries from the definition
        return [
            f"CUDA kernel {definition.op_type} {definition.name}",
            f"{definition.op_type} memory optimization GPU",
            f"CUDA performance tuning {definition.name}",
        ]

    # ------------------------------------------------------------------
    # Hybrid search (dense + BM25 sparse via Milvus)
    # ------------------------------------------------------------------

    async def _search(self, query: str) -> List[Dict]:
        """Run a single hybrid search query against Milvus.

        Runs in a thread executor because the Milvus client is synchronous.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query)

    def _search_sync(self, query: str) -> List[Dict]:
        """Synchronous hybrid search combining dense + sparse retrieval."""
        query_embedding = self.embed_model.encode(query).tolist()

        dense_req = self._AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=self.top_k,
        )

        sparse_req = self._AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=self.top_k,
        )

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=self._RRFRanker(k=60),
            limit=self.top_k,
            output_fields=["path", "file_name", "source", "text"],
        )

        formatted = []
        for result in results[0]:
            formatted.append(
                {
                    "id": result.id,
                    "distance": result.distance,
                    "path": result.fields.get("path", ""),
                    "file_name": result.fields.get("file_name", ""),
                    "text": result.fields.get("text", ""),
                }
            )
        return formatted

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: List[Dict]) -> str:
        """Format search results into a readable string for prompt injection."""
        if not results:
            return ""

        sections = []
        for i, r in enumerate(results, 1):
            path = r.get("path", "unknown")
            text = r.get("text", "").strip()
            # Truncate very long documents to avoid blowing up prompt length
            if len(text) > 3000:
                text = text[:3000] + "\n... (truncated)"
            sections.append(
                f"--- Reference Document {i}: {text}"
            )

        return "\n\n".join(sections)
