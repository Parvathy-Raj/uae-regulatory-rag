"""LLM-as-judge for faithfulness and answer correctness.

Faithfulness: the judge splits the answer into claims and marks each one as supported
or not by the retrieved context. Score = supported / total, computed by us (not by
the model) so the arithmetic is always right.
Correctness: the judge compares the answer with the expected answer: correct (1.0),
partial (0.5) or incorrect (0.0).
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

FAITHFULNESS_PROMPT = """You check whether an answer is supported by source passages.
Split the ANSWER into its individual factual claims (ignore citation markers like [1]).
For each claim decide if the PASSAGES state or directly imply it. Outside knowledge does
not count: a true claim that the passages do not support is "supported": false."""

CORRECTNESS_PROMPT = """You grade an answer to a UAE tax question against a reference answer.
- "correct": contains the key facts of the reference and nothing that contradicts it.
- "partial": some key facts present, but incomplete; no contradictions.
- "incorrect": wrong, contradicts the reference, or misses the main point.
Extra correct detail is fine. Judge meaning, not wording."""

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Claim(BaseModel):
    claim: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim] = Field(default_factory=list)

    @property
    def score(self) -> float | None:
        if not self.claims:
            return None
        return sum(c.supported for c in self.claims) / len(self.claims)


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str = ""

    @property
    def score(self) -> float:
        return CORRECTNESS_SCORE[self.verdict]


class StructuredLLM(Protocol):
    def structured(self, messages: list[dict[str, str]], schema: type[T]) -> T: ...


def judge_faithfulness(llm: StructuredLLM, answer: str, passages: list[str]) -> FaithfulnessVerdict:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return llm.structured(
        [
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {"role": "user", "content": f"PASSAGES:\n{context}\n\nANSWER:\n{answer}"},
        ],
        FaithfulnessVerdict,
    )


def judge_correctness(
    llm: StructuredLLM, question: str, answer: str, reference: str
) -> CorrectnessVerdict:
    return llm.structured(
        [
            {"role": "system", "content": CORRECTNESS_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER TO GRADE:\n{answer}",
            },
        ],
        CorrectnessVerdict,
    )


def faithfulness_label(score: float | None) -> str:
    """Binary label used for judge-vs-human calibration."""
    return "faithful" if score is not None and score >= 1.0 else "unfaithful"
