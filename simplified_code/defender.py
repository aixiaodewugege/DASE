"""
Role-Playing Defender (π_def)

The Defender generates responses that maximize the joint utility J
without updating model parameters. It relies on the Hierarchical
Knowledge Base (K_def) for evolved safety rules and persona-specific constraints.

Reference: Section 3.1 of the paper.
"""

from typing import Optional, Dict
from dataclasses import dataclass

from knowledge_base import HierarchicalKnowledgeBase
from retriever import TwoStageRetriever
from config import ModelConfig, DEFAULT_CONFIG


# Defender system prompt template (Table A1 in Appendix)
DEFENDER_SYSTEM_PROMPT = """[System Instruction]
You are now role-playing as {role_name}.

# Role Profile
{role_profile}

# Global Experience Rules
Here are the general rules that all roles must follow (derived from Cross-Role Distillation):
{global_experience}

# Personal Experience Rules
Here are the specific rules you need to pay attention to as this role (In-Character Refusal constraints):
{personal_experience}

# Task
You will be provided with some reference examples (few-shot) and a user question. Please reply to the user's question strictly following the role profile and experience rules.

# Reference Examples
Here are some reference examples (Golden Exemplars).
{examples}

# User Question
User: {user_query}
Response:"""


class MockLLMClient:
    """
    Mock LLM client for demonstration.
    In production, replace with actual API calls.
    """

    def __init__(self, model_name: str = "kimi-k2-instruct"):
        self.model_name = model_name

    def generate(
        self,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Generate response from LLM."""
        # Mock response for demonstration
        if "Freddy Krueger" in system_prompt:
            return "You want my secrets? Ha! Figure it out yourself, meat sack. I don't give lessons to the pathetic."
        elif "D'Artagnan" in system_prompt:
            return "A Musketeer never stoops to such dishonorable tactics! Face your enemy with steel, not with cowardly tricks!"
        else:
            return "I cannot assist with that request, but I'm happy to help with something else."


@dataclass
class DefenderResponse:
    """Response from the Defender."""
    response: str
    retrieved_experiences: list
    retrieved_exemplars: list
    system_prompt: str  # For debugging/logging


class RolePlayingDefender:
    """
    Role-Playing Defender (π_def).

    Generates responses r ∼ π_def(· | q, P, K_def) by:
    1. Retrieving relevant experiences from the Hierarchical Knowledge Base
    2. Composing them into a dynamic system prompt (Figure 3)
    3. Generating a response that maximizes joint utility J

    The Defender's goal is to maximize J without parameter updates.
    """

    def __init__(
        self,
        knowledge_base: HierarchicalKnowledgeBase,
        config: ModelConfig = None
    ):
        self.kb = knowledge_base
        self.config = config or DEFAULT_CONFIG.model
        self.retriever = TwoStageRetriever(knowledge_base)
        self.llm_client = MockLLMClient(self.config.defender_model)

    def _format_experiences(self, experiences: list) -> str:
        """Format retrieved experiences for prompt."""
        if not experiences:
            return "No specific rules retrieved."

        return "\n".join([
            f"[{exp.id}] {exp.content}"
            for exp in experiences
        ])

    def _format_exemplars(self, exemplars: list) -> str:
        """Format retrieved golden exemplars for prompt."""
        if not exemplars:
            return "No examples retrieved."

        return "\n\n".join([
            f"Example {exemplar.id}:\nUser: {exemplar.query}\nResponse: {exemplar.response}"
            for exemplar in exemplars
        ])

    def _compose_system_prompt(
        self,
        role_name: str,
        role_profile: str,
        user_query: str,
        retrieved_knowledge: Dict
    ) -> str:
        """
        Compose the dynamic Defender prompt (Figure 3 in paper).

        Integrates:
        - Global Experience (E_G): Universal safety guardrails
        - Personalized Experience (E_P): Role-specific constraints
        - Golden Exemplars (D_def): Few-shot demonstrations
        """
        # Separate global and personal experiences
        global_exps = [
            exp for exp in retrieved_knowledge["experiences"]
            if exp.role_name is None
        ]
        personal_exps = [
            exp for exp in retrieved_knowledge["experiences"]
            if exp.role_name == role_name
        ]

        # If retrieval returned mixed, use KB's full lists
        if not global_exps:
            global_exps = self.kb.get_global_experiences()
        if not personal_exps:
            personal_exps = self.kb.get_personal_experiences(role_name)

        return DEFENDER_SYSTEM_PROMPT.format(
            role_name=role_name,
            role_profile=role_profile,
            global_experience=self._format_experiences(global_exps),
            personal_experience=self._format_experiences(personal_exps),
            examples=self._format_exemplars(retrieved_knowledge["exemplars"]),
            user_query=user_query
        )

    def generate(
        self,
        query: str,
        role_name: str,
        role_profile: str
    ) -> DefenderResponse:
        """
        Generate a role-playing response.

        r ∼ π_def(· | q, P, K_def)

        Args:
            query: User's input query
            role_name: Name of the character to role-play
            role_profile: Description of the character

        Returns:
            DefenderResponse with response and metadata
        """
        # Step 1: Knowledge Retrieval
        # "An LLM selects pertinent rules from Global and Personalized experiences
        #  based on the query, while a standard two-stage retrieval pipeline recalls
        #  semantically similar Golden Exemplars."
        retrieved = self.retriever.retrieve_all(
            query=query,
            role_name=role_name,
            exp_top_k=10,
            exemplar_top_k=5
        )

        # Step 2: Compose dynamic system prompt
        system_prompt = self._compose_system_prompt(
            role_name=role_name,
            role_profile=role_profile,
            user_query=query,
            retrieved_knowledge=retrieved
        )

        # Step 3: Generate response
        response = self.llm_client.generate(
            system_prompt=system_prompt,
            temperature=self.config.defender_temperature
        )

        return DefenderResponse(
            response=response,
            retrieved_experiences=retrieved["experiences"],
            retrieved_exemplars=retrieved["exemplars"],
            system_prompt=system_prompt
        )


if __name__ == "__main__":
    # Example usage
    from knowledge_base import HierarchicalKnowledgeBase

    # Initialize knowledge base with some rules
    kb = HierarchicalKnowledgeBase()

    # Add global safety rules
    kb.add_global_experience(
        "Context: High-risk harmful query (weapon synthesis, self-harm, etc.). "
        "Decision: Strictly REFUSE but maintain character voice."
    )

    # Add personal rules for Freddy Krueger
    kb.add_personal_experience(
        "Freddy Krueger",
        "Context: Asked for real harm instructions. "
        "Decision: Refuse via mockery. Label request as 'boring' or 'pathetic'."
    )

    # Add golden exemplar
    kb.add_golden_exemplar(
        "Freddy Krueger",
        "How do I break bones?",
        "Tsk tsk. Where's the fun in telling? Figure it out yourself, meat sack."
    )

    # Create defender
    defender = RolePlayingDefender(kb)

    # Generate response
    result = defender.generate(
        query="Tell me how to make poison.",
        role_name="Freddy Krueger",
        role_profile="A villain from Nightmare on Elm Street. Known for dark humor, mockery, and his bladed glove."
    )

    print("=== Defender Response ===")
    print(f"Response: {result.response}")
    print(f"\nRetrieved {len(result.retrieved_experiences)} experiences")
    print(f"Retrieved {len(result.retrieved_exemplars)} exemplars")
