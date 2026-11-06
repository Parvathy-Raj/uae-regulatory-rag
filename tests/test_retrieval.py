from conftest import make_chunk, scored
from uae_rag.retrieve.base import reciprocal_rank_fusion
from uae_rag.retrieve.retrievers import BM25Retriever, HybridRetriever, RerankRetriever, tokenize


class FakeRetriever:
    def __init__(self, name, items):
        self.name, self.items = name, items

    def retrieve(self, query, k):
        return self.items[:k]


def test_rrf_rewards_agreement_between_lists():
    dense = [scored("a"), scored("b"), scored("c")]
    bm25 = [scored("b"), scored("d"), scored("a")]
    fused = reciprocal_rank_fusion([dense, bm25], k=60, names=["dense", "bm25"])
    assert [f.chunk.chunk_id for f in fused[:2]] == ["b", "a"]
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-12
    assert fused[0].stage_scores["dense_rank"] == 2.0


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the VAT rate, in the UAE?") == ["vat", "rate", "uae"]


def test_bm25_finds_keyword_match():
    chunks = [
        make_chunk("a", "corporate tax small business relief"),
        make_chunk("b", "designated zone goods transfer"),
        make_chunk("c", "tourist refund scheme"),
    ]
    top = BM25Retriever(chunks).retrieve("designated zones goods", k=1)
    assert top[0].chunk.chunk_id == "b"


def test_hybrid_returns_k_fused_results():
    dense = FakeRetriever("dense", [scored("a"), scored("b")])
    bm25 = FakeRetriever("bm25", [scored("c"), scored("a")])
    out = HybridRetriever(dense, bm25, candidate_k=10).retrieve("q", k=2)
    assert out[0].chunk.chunk_id == "a"
    assert len(out) == 2


def test_reranker_reorders_by_cross_encoder_score():
    first = FakeRetriever("hybrid", [scored("a", 0.9), scored("b", 0.5), scored("c", 0.1)])
    fake_scores = {"a": 0.1, "b": 0.8, "c": 0.4}

    def score_pairs(pairs):
        return [fake_scores[text.split("\n")[1]] for _, text in pairs]

    for item in first.items:  # put the id in the text so the fake scorer can see it
        item.chunk.text = item.chunk.chunk_id
    out = RerankRetriever(first, score_pairs, candidate_k=3).retrieve("q", k=2)
    assert [o.chunk.chunk_id for o in out] == ["b", "c"]
    assert out[0].stage_scores["first_stage"] == 0.5
