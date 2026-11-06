from conftest import scored
from uae_rag.config import GenerationConfig
from uae_rag.generate.answer import AnswerDraft, build_messages, generate_answer, map_citations


class FakeLLM:
    model = "fake"

    def __init__(self, draft: AnswerDraft):
        self.draft, self.calls = draft, 0

    def structured(self, messages, schema):
        self.calls += 1
        return self.draft


CHUNKS = [scored("doc:000:00", 0.9), scored("doc:001:00", 0.6)]


def test_context_is_numbered_with_page_and_heading():
    content = build_messages("q?", CHUNKS)[1]["content"]
    assert "[1] (Doc Title (2024), p.1, section: H)" in content
    assert "[2]" in content and "Question: q?" in content


def test_citation_labels_map_to_chunk_ids_and_invalid_are_dropped():
    draft = AnswerDraft(answer="Rate is 5% [2]. Also [7].", citations=[1])
    valid, invalid = map_citations(draft, CHUNKS)
    assert valid == ["doc:001:00", "doc:000:00"]
    assert invalid == [7]


def test_answer_with_valid_citation_is_returned():
    llm = FakeLLM(AnswerDraft(answer="It is X [1].", citations=[1]))
    ans = generate_answer("q", CHUNKS, llm, GenerationConfig(), mode="hybrid")
    assert not ans.refused and ans.citations == ["doc:000:00"]


def test_low_rerank_score_refuses_without_calling_llm():
    llm = FakeLLM(AnswerDraft(answer="x [1]"))
    low = [scored("a", 0.001)]
    ans = generate_answer(
        "q", low, llm, GenerationConfig(min_rerank_score=0.05), mode="hybrid_rerank"
    )
    assert ans.refused and ans.refusal_reason == "low_retrieval_score"
    assert llm.calls == 0


def test_model_refusal_is_respected():
    llm = FakeLLM(AnswerDraft(refused=True, refusal_reason="not in context"))
    ans = generate_answer("q", CHUNKS, llm, GenerationConfig(), mode="dense")
    assert ans.refused and ans.refusal_reason == "not in context"


def test_uncited_answer_becomes_refusal():
    llm = FakeLLM(AnswerDraft(answer="Confident but uncited.", citations=[]))
    ans = generate_answer(
        "q", CHUNKS, llm, GenerationConfig(refuse_if_no_valid_citation=True), mode="dense"
    )
    assert ans.refused and ans.refusal_reason == "no_valid_citation"


def test_no_chunks_refuses():
    ans = generate_answer(
        "q", [], FakeLLM(AnswerDraft(answer="x")), GenerationConfig(), mode="hybrid"
    )
    assert ans.refused
