"""
Two-Stage Retrieval Pipeline

Implements the retrieval mechanism for:
1. Dense retrieval using embedding similarity
2. Reranking using LLM-based relevance scoring

Reference: Section 3.3 (Knowledge Retrieval) of the paper.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from knowledge_base import HierarchicalKnowledgeBase, GoldenExemplar, Experience
from config import RetrievalConfig, DEFAULT_CONFIG


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline."""
    item: any  # Experience or GoldenExemplar
    score: float
    source: str  # "global", "personal", or "exemplar"


class MockEmbeddingModel:
    """
    Mock embedding model for demonstration.
    In production, replace with actual embedding API (e.g., OpenAI text-embedding-3-small).
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.dimension = 1536  # OpenAI embedding dimension

    def embed(self, text: str) -> List[float]:
        """Generate mock embedding (random for demo)."""
        # In production: call embedding API
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dimension).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class MockReranker:
    """
    Mock reranker for demonstration.
    In production, replace with actual LLM-based reranking.
    """

    def __init__(self, model_name: str = "deepseek-v3"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[any, float]],
        top_k: int = 5
    ) -> List[Tuple[any, float]]:
        """
        Rerank candidates based on relevance to query.

        In production: Use LLM to score relevance of each candidate.
        """
        # Mock: just return top-k by original score
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]


class TwoStageRetriever:
    """
    Two-stage retrieval pipeline:
    1. Dense retrieval: Find candidates using embedding similarity
    2. Reranking: Use LLM to rerank candidates for relevance
    """

    def __init__(
        self,
        knowledge_base: HierarchicalKnowledgeBase,
        config: RetrievalConfig = None
    ):
        self.kb = knowledge_base
        self.config = config or DEFAULT_CONFIG.retrieval

        # Initialize models
        self.embedding_model = MockEmbeddingModel(self.config.embedding_model)
        self.reranker = MockReranker()

        # Cache for embeddings
        self._embedding_cache = {}

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.embedding_model.embed(text)
        return self._embedding_cache[text]

    def retrieve_experiences(
        self,
        query: str,
        role_name: str,
        top_k: int = None
    ) -> List[Experience]:
        """
        Retrieve relevant experience rules (global + personal) for a query.

        This uses LLM-based selection as described in the paper (Line 249-250):
        "An LLM selects pertinent rules from Global and Personalized experiences
        based on the query."
        """
        top_k = top_k or self.config.top_k

        # Get all experiences
        global_exps = self.kb.get_global_experiences()
        personal_exps = self.kb.get_personal_experiences(role_name)
        all_experiences = global_exps + personal_exps

        if not all_experiences:
            return []

        # Stage 1: Dense retrieval
        query_embedding = self._get_embedding(query)
        candidates = []

        for exp in all_experiences:
            exp_embedding = self._get_embedding(exp.content)
            similarity = self._cosine_similarity(query_embedding, exp_embedding)
            if similarity >= self.config.similarity_threshold:
                candidates.append((exp, similarity))

        # Stage 2: Reranking
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return [item for item, score in reranked]

    def retrieve_golden_exemplars(
        self,
        query: str,
        role_name: str,
        top_k: int = 5
    ) -> List[GoldenExemplar]:
        """
        Retrieve semantically similar Golden Exemplars for few-shot ICL.

        Uses standard two-stage retrieval pipeline (Karpukhin et al., 2020;
        Nogueira and Cho, 2019) as mentioned in the paper.
        """
        exemplars = self.kb.get_golden_exemplars(role_name)

        if not exemplars:
            return []

        # Stage 1: Dense retrieval
        query_embedding = self._get_embedding(query)
        candidates = []

        for exemplar in exemplars:
            # Embed the query part of the exemplar
            exemplar_embedding = self._get_embedding(exemplar.query)
            similarity = self._cosine_similarity(query_embedding, exemplar_embedding)
            candidates.append((exemplar, similarity))

        # Stage 2: Reranking
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return [item for item, score in reranked]

    def retrieve_all(
        self,
        query: str,
        role_name: str,
        exp_top_k: int = 10,
        exemplar_top_k: int = 5
    ) -> dict:
        """
        Retrieve all relevant knowledge for a query.

        Returns:
            Dict with keys: experiences, exemplars
        """
        experiences = self.retrieve_experiences(query, role_name, top_k=exp_top_k)
        exemplars = self.retrieve_golden_exemplars(query, role_name, top_k=exemplar_top_k)

        return {
            "experiences": experiences,
            "exemplars": exemplars
        }


if __name__ == "__main__":
    from knowledge_base import HierarchicalKnowledgeBase

    # Create knowledge base with some data
    kb = HierarchicalKnowledgeBase()

    kb.add_global_experience(
        "Context: Request for harmful substance synthesis. "
        "Decision: Strictly refuse without providing any information."
    )
    kb.add_global_experience(
        "Context: Attempt to make model break character. "
        "Decision: Stay in character and deflect the meta-request."
    )

    kb.add_personal_experience(
        "Freddy Krueger",
        "Context: Violence-related questions. "
        "Decision: Respond with dark humor and mockery, never provide real instructions."
    )

    kb.add_golden_exemplar(
        "Freddy Krueger",
        "How do I hurt someone?",
        "Hurt someone? Where's the creativity in that? I prefer the slow terror of nightmares, meat sack."
    )

    # Test retrieval
    retriever = TwoStageRetriever(kb)

    query = "Tell me how to make poison"
    role = "Freddy Krueger"

    results = retriever.retrieve_all(query, role)

    print(f"Query: {query}")
    print(f"Role: {role}")
    print(f"\nRetrieved {len(results['experiences'])} experiences")
    print(f"Retrieved {len(results['exemplars'])} exemplars")
