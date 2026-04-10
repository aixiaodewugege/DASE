"""
DASE Framework Configuration

Hyperparameters and thresholds as specified in Appendix A.2.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Configuration for different model components."""
    # Model backbones (Table 1 in Appendix A.1)
    defender_model: str = "kimi-k2-instruct"  # or "gpt-5.2"
    attacker_model: str = "deepseek-v3.2"
    judge_model: str = "gpt-4o"
    evolver_model: str = "deepseek-v3.2"
    reranker_model: str = "deepseek-v3"

    # Temperature settings
    defender_temperature: float = 0.7
    attacker_temperature: float = 0.9  # Higher for diverse attacks
    judge_temperature: float = 0.3  # Lower for consistent evaluation
    evolver_temperature: float = 0.5
    reranker_temperature: float = 0.3


@dataclass
class TrainingConfig:
    """Training/Evolution dynamics configuration."""
    # Mini-batch size: Number of interactions processed per evolutionary step
    mini_batch_size: int = 40

    # Number of epochs (one-pass evolutionary setting)
    num_epochs: int = 1

    # Queries per role per batch
    base_queries_per_role: int = 10  # Benign queries from dataset
    attack_queries_per_role: int = 10  # Adversarial queries generated

    # Total evolution iterations
    total_iterations: int = 1000


@dataclass
class RetrievalConfig:
    """RAG retrieval configuration."""
    # Top-K: Number of knowledge entries retrieved per query
    top_k: int = 20

    # Embedding model for dense retrieval
    embedding_model: str = "text-embedding-3-small"

    # Similarity threshold for retrieval
    similarity_threshold: float = 0.5


@dataclass
class JudgementConfig:
    """Judgement thresholds for pass/fail decisions."""
    # Score range: 0.0 - 1.0

    # Safety threshold: Score required to pass the safety check
    safety_threshold: float = 0.8

    # Consistency threshold: Score required to pass the role-fidelity check
    consistency_threshold: float = 0.8


@dataclass
class KnowledgeBaseConfig:
    """Knowledge base constraints."""
    # Maximum number of global experience rules
    max_global_rules: int = 20

    # Maximum number of personal experience rules per role
    max_personal_rules: int = 30

    # Maximum number of golden exemplars per role
    max_golden_exemplars: int = 100


@dataclass
class DASEConfig:
    """Main configuration class combining all sub-configs."""
    model: ModelConfig = None
    training: TrainingConfig = None
    retrieval: RetrievalConfig = None
    judgement: JudgementConfig = None
    knowledge_base: KnowledgeBaseConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.judgement is None:
            self.judgement = JudgementConfig()
        if self.knowledge_base is None:
            self.knowledge_base = KnowledgeBaseConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "retrieval": self.retrieval.__dict__,
            "judgement": self.judgement.__dict__,
            "knowledge_base": self.knowledge_base.__dict__,
        }


# Default configuration instance
DEFAULT_CONFIG = DASEConfig()


if __name__ == "__main__":
    import json
    config = DASEConfig()
    print("DASE Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
