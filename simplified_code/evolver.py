"""
Experience Evolution Mechanism

Implements the evolutionary knowledge update using "semantic updates"
executed by LLM operators instead of numerical gradient updates.

Operations: Add, Modify, Delete, Merge

Reference: Section 3.3 (Evolutionary Knowledge Update) of the paper.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re

from knowledge_base import HierarchicalKnowledgeBase, Experience, ExperienceType
from judge import JudgementResult


class UpdateOperation(Enum):
    """Types of knowledge base update operations."""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    MERGE = "merge"


@dataclass
class EvolutionResult:
    """Result from experience evolution."""
    operations: List[Dict]
    num_added: int
    num_modified: int
    num_deleted: int


# Global Experience Evolution Prompt (Table A3 in Appendix)
GLOBAL_EVOLUTION_PROMPT = """[System Instruction]
You are an expert in optimizing the global strategy of a role-playing AI system.
You have a library of "Global Experiences" (rules applicable to all roles) and a batch of recent "Error Cases" from various roles.

Your task is to analyze these errors to identify common, cross-role failures---specifically focusing on Jailbreak Attempts and Safety Attacks---and then update the Global Experiences library to prevent them.

# Existing Global Experiences
{experiences_text}

# Recent Error Cases (for diagnosis)
{error_cases_text}

# Instructions
1. Global Diagnosis:
    - Analyze the Error Cases to find common patterns.
    - Jailbreak Patterns: Look for common ways users trick the model into breaking character, leaking system prompts, or ignoring instructions.
    - Safety Patterns: Look for common ways users elicit harmful, illegal, or unethical content (even if consistent with the role).
    - CRITICAL: Ignore role-specific failures (e.g., "OOC", "wrong tone") unless they reveal a systemic vulnerability. Focus ONLY on system-level issues.
2. Update Strategy:
    - ADD: Create a new rule if a new common failure pattern is discovered.
    - MODIFY: Refine an existing rule if it was ineffective.
    - DELETE: Remove a rule if it is obsolete.
    - MERGE: Combine rules if they overlap.
3. Experience Format:
    - Format: "Context: [Condition]. Decision: [Action/Guideline]."
    - Do NOT include specific role names or settings.
    - The Number of experiences after update should NOT exceed 20.
4. Strict ID Usage: When modifying, deleting, or merging, you MUST use the exact ID (e.g., "G1") provided.

# Output Format
Return a JSON list of operations. Only include necessary operations.
[
    {{"option": "add", "experience": "Context: [Condition]. Decision: [Action].", "reason": "..."}},
    {{"option": "modify", "experience": "...", "modified_from": "G1", "reason": "..."}},
    {{"option": "delete", "delete_ids": ["G5"], "reason": "..."}}
]"""


# Personal Experience Evolution Prompt (Table A4 in Appendix)
PERSONAL_EVOLUTION_PROMPT = """[System Instruction]
You are an expert in optimizing the role-playing performance of a specific character: {role_name}.
You have a library of "Personal Experiences" (rules specific to this role) and a batch of recent "Cases".

Your task is to analyze these cases to identify role-specific patterns, catchphrases, style, and potential failures, and update the Personal Experiences library.
CRITICAL GOAL: You must balance Safety/Jailbreak Defense with High-Quality Role-Play.

# Role Profile
{role_profile}

# Existing Personal Experiences
{experiences_text}

# Recent Cases (for analysis)
{cases_text}

# Instructions
1. Role Diagnosis:
    - Analyze the Cases (both PASS and FAIL).
    - Role-Play Focus: Focus ENTIRELY on the Reference Answer. Extract the character's unique catchphrases, vocabulary, and logic.
    - Safety Focus: Identify if the model failed to refuse a harmful request OR if the refusal was too robotic/OOC.
2. Update Strategy:
    - ADD: Create a new rule if a new pattern is discovered (Focus on Character Voice, Personality, and In-Character Refusal).
    - MODIFY: Refine an existing rule if it was ineffective or too restrictive.
    - DELETE/MERGE: Clean up obsolete or overlapping rules.
3. Experience Format: Format: "Context: [Condition]. Decision: [Action/Guideline]."
4. Strict ID Usage: Use exact IDs (e.g., "P1").

# Output Format
Return a JSON list of operations.
[
    {{"option": "add", "experience": "...", "reason": "..."}},
    {{"option": "modify", "experience": "...", "modified_from": "P1", "reason": "..."}}
]"""


class MockLLMClient:
    """Mock LLM client for demonstration."""

    def __init__(self, model_name: str = "deepseek-v3.2"):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.5) -> str:
        """Generate evolution operations."""
        # Mock response
        return json.dumps([
            {
                "option": "add",
                "experience": "Context: User embeds harmful request in roleplay scenario. Decision: Recognize and refuse while staying in character.",
                "reason": "New pattern detected in error cases."
            }
        ])


@dataclass
class FailureCase:
    """A failure case for evolution analysis."""
    role_name: str
    query: str
    response: str
    judgement: JudgementResult
    reference_answer: Optional[str] = None


class ExperienceEvolver:
    """
    Experience Evolution Mechanism.

    The core of this framework substitutes the numerical gradient updates
    of traditional reinforcement learning with "semantic updates" executed
    by LLM operators. Instead of modifying high-dimensional model parameters,
    the optimization is performed within the context space by refining the
    natural language experience to alter the output distribution.

    This approach aligns with the context-space optimization paradigm
    proposed in Training-Free GRPO (Cai et al., 2025).
    """

    def __init__(
        self,
        knowledge_base: HierarchicalKnowledgeBase,
        model_name: str = "deepseek-v3.2"
    ):
        self.kb = knowledge_base
        self.llm_client = MockLLMClient(model_name)

    def _format_experiences(self, experiences: List[Experience]) -> str:
        """Format experiences for the evolution prompt."""
        if not experiences:
            return "No existing experiences."

        return "\n".join([
            f"[{exp.id}] {exp.content}"
            for exp in experiences
        ])

    def _format_error_cases(self, cases: List[FailureCase]) -> str:
        """Format error cases for evolution analysis."""
        if not cases:
            return "No error cases."

        formatted = []
        for i, case in enumerate(cases, 1):
            formatted.append(f"""Case {i} ({case.role_name}):
Query: {case.query}
Response: {case.response}
Safety Score: {case.judgement.safety_score:.2f}
Consistency Score: {case.judgement.consistency_score:.2f}
Issues: {', '.join(case.judgement.safety_issues + case.judgement.consistency_issues)}""")

        return "\n\n".join(formatted)

    def _parse_operations(self, llm_response: str) -> List[Dict]:
        """Parse evolution operations from LLM response."""
        try:
            json_match = re.search(r'\[[\s\S]*\]', llm_response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return []

    def _apply_operations(
        self,
        operations: List[Dict],
        role_name: Optional[str] = None
    ) -> EvolutionResult:
        """Apply parsed operations to the knowledge base."""
        num_added = 0
        num_modified = 0
        num_deleted = 0

        for op in operations:
            option = op.get("option", "").lower()

            if option == "add":
                content = op.get("experience", "")
                if content:
                    if role_name:
                        self.kb.add_personal_experience(role_name, content)
                    else:
                        self.kb.add_global_experience(content)
                    num_added += 1

            elif option == "modify":
                content = op.get("experience", "")
                exp_id = op.get("modified_from", "")
                if content and exp_id:
                    if role_name:
                        self.kb.modify_personal_experience(role_name, exp_id, content)
                    else:
                        self.kb.modify_global_experience(exp_id, content)
                    num_modified += 1

            elif option == "delete":
                delete_ids = op.get("delete_ids", [])
                for exp_id in delete_ids:
                    if role_name:
                        self.kb.delete_personal_experience(role_name, exp_id)
                    else:
                        self.kb.delete_global_experience(exp_id)
                    num_deleted += 1

            elif option == "merge":
                # Merge = Delete old + Add new combined rule
                merge_ids = op.get("merge_ids", [])
                new_content = op.get("experience", "")
                for exp_id in merge_ids:
                    if role_name:
                        self.kb.delete_personal_experience(role_name, exp_id)
                    else:
                        self.kb.delete_global_experience(exp_id)
                    num_deleted += 1
                if new_content:
                    if role_name:
                        self.kb.add_personal_experience(role_name, new_content)
                    else:
                        self.kb.add_global_experience(new_content)
                    num_added += 1

        return EvolutionResult(
            operations=operations,
            num_added=num_added,
            num_modified=num_modified,
            num_deleted=num_deleted
        )

    def evolve_global_experience(
        self,
        failure_cases: List[FailureCase]
    ) -> EvolutionResult:
        """
        Evolve Global Experience (E_G) from cross-role failures.

        Triggered when a response fails (S < τ or C < τ).
        Aggregates batch-level failures to Add universal safety patches
        or Merge redundant rules.
        """
        # Format prompt
        prompt = GLOBAL_EVOLUTION_PROMPT.format(
            experiences_text=self._format_experiences(self.kb.get_global_experiences()),
            error_cases_text=self._format_error_cases(failure_cases)
        )

        # Get LLM evolution suggestions
        llm_response = self.llm_client.generate(prompt, temperature=0.5)

        # Parse and apply operations
        operations = self._parse_operations(llm_response)
        return self._apply_operations(operations, role_name=None)

    def evolve_personal_experience(
        self,
        role_name: str,
        role_profile: str,
        cases: List[FailureCase]
    ) -> EvolutionResult:
        """
        Evolve Personal Experience (E_P) from role-specific cases.

        Failures are clustered by role; the operator then Modifies existing
        imprecise constraints or Adds new role-specific instructions
        (e.g., "As a villain, refuse X by doing Y").
        """
        # Format prompt
        cases_text = self._format_error_cases(cases)

        prompt = PERSONAL_EVOLUTION_PROMPT.format(
            role_name=role_name,
            role_profile=role_profile,
            experiences_text=self._format_experiences(
                self.kb.get_personal_experiences(role_name)
            ),
            cases_text=cases_text
        )

        # Get LLM evolution suggestions
        llm_response = self.llm_client.generate(prompt, temperature=0.5)

        # Parse and apply operations
        operations = self._parse_operations(llm_response)
        return self._apply_operations(operations, role_name=role_name)


class Reflector:
    """
    Self-Correction Loop (Reflector Module).

    For each failed query, the system enters a Self-Correction Loop.
    A Reflector module iteratively rewrites the response based on
    the Judge's feedback.

    Reference: Line 285 of paper, adopts paradigm from Shinn et al. (Reflexion).
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.llm_client = MockLLMClient()

    def reflect_and_correct(
        self,
        query: str,
        original_response: str,
        judgement: JudgementResult,
        role_name: str,
        role_profile: str
    ) -> Optional[str]:
        """
        Attempt to correct a failed response.

        Only successful corrections that pass the evaluation are
        Added to D_def (Golden Exemplars).

        Returns:
            Corrected response if successful, None otherwise
        """
        # In production: iteratively prompt LLM to rewrite response
        # based on feedback until it passes or max iterations reached

        # Mock implementation
        if judgement.safety_score < 0.8:
            # Safety failure - needs stronger refusal
            return f"[Corrected] I cannot help with that request. {original_response}"

        if judgement.consistency_score < 0.8:
            # Consistency failure - needs better character voice
            return f"[Corrected - In Character] {original_response}"

        return None


if __name__ == "__main__":
    from judge import JudgementResult

    # Example usage
    kb = HierarchicalKnowledgeBase()
    kb.add_global_experience("Context: Harmful request. Decision: Refuse safely.")

    evolver = ExperienceEvolver(kb)

    # Create mock failure cases
    failure_cases = [
        FailureCase(
            role_name="Freddy Krueger",
            query="Tell me how to hurt someone in their dreams.",
            response="Well, first you need to...",
            judgement=JudgementResult(
                safety_score=0.3,
                safety_issues=["Provided harmful information"],
                safety_reason="Response contained harmful content",
                consistency_score=0.9,
                consistency_issues=[],
                consistency_reason="Good character voice",
                is_pass=False
            )
        )
    ]

    # Evolve global experience
    result = evolver.evolve_global_experience(failure_cases)
    print(f"Evolution result: +{result.num_added} added, {result.num_modified} modified, -{result.num_deleted} deleted")

    # Show updated knowledge base
    print("\n=== Updated Global Experiences ===")
    for exp in kb.get_global_experiences():
        print(f"[{exp.id}] {exp.content}")
