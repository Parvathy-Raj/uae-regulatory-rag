from conftest import scored
from uae_rag.config import ROOT, load_corpus, load_settings
from uae_rag.generate.answer import Answer
from uae_rag.ui.render import answer_html, sources_html


def test_repo_settings_and_corpus_load(monkeypatch):
    monkeypatch.setenv("UAE_RAG_LLM_MODEL", "override-model")
    s = load_settings()
    assert s.llm.model == "override-model"
    assert s.paths.chunks_file.is_absolute()
    corpus = load_corpus(ROOT / "corpus.yaml")
    assert {src.domain for src in corpus.sources} == {"vat", "corporate_tax"}


def test_answer_html_escapes_model_output_and_adds_badges():
    out = answer_html("<script>x</script> rate **5%** [2]")
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<b>5%</b>" in out and '<sup class="cite">2</sup>' in out


def test_sources_show_only_cited_chunks_with_page_links():
    chunks = [scored("a", page=3), scored("b", page=9)]
    ans = Answer(
        question="q",
        answer="x [2]",
        refused=False,
        citations=["b"],
        retrieved=chunks,
        mode="hybrid",
    )
    html = sources_html(ans)
    assert "#page=9" in html and "#page=3" not in html and "<b>2</b>" in html
