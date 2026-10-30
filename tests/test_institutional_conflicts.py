"""
Institutional Navigation evidence conflicts and determinism (Milestone 4, Step 4.16-B5; test groups G and H of the Step
4.16-A test strategy).

G. Conflicting evidence never produces a silent winner: two directory lines for one office give no number (partial, the
   office name only); two units stated to do one duty give every candidate (partial); a chunk from any source other than
   the registered handbook is never read, so it can neither answer nor overrule (single-source design, DEP-4: when an
   approved second source exists, both will be cited with no winner).
H. Answers do not depend on retrieval: the specialist reads its directory lines and statements once, at construction,
   and never calls the reranker, the embedder or the index while answering, so an adversarial ranking or a second
   instance gives byte-identical results.
No model, no LLM, no network.
"""
from __future__ import annotations

import json

import pytest

from handbook_bot.agents.specialists.contracts import FindingStatus
from handbook_bot.agents.specialists.institutional import (APPROVED_SOURCE_ID, InstitutionalNavigationSpecialist,
                                                           InstitutionalResources)
from handbook_bot.text_utils import PHONE_PATTERN

from institutional_support import (contact_cases, evidence_cases, institutional_corpus, make_institutional,
                                   make_institutional_with_lines, make_institutional_with_text, run)

HR_LINE_CHUNK = 2112          # a page-54 directory line replaced in the single-entry control (contact case SYN-06)
ITC_PARAGRAPH_CHUNK = 2470    # a page-183 paragraph chunk of the IT Center lead-in


def _without_ms(value):
    if isinstance(value, dict):
        return {k: _without_ms(v) for k, v in value.items() if k != "ms"}
    if isinstance(value, list):
        return [_without_ms(v) for v in value]
    return value


@pytest.fixture(scope="module")
def sp():
    return make_institutional()


# ---------------------------------------------------------------------------
# G. Evidence conflicts
# ---------------------------------------------------------------------------
def test_g1_two_directory_lines_for_one_office_give_no_number(sp):
    assert run(sp, "What is the phone number of Human Resources?").findings[0].metadata["contact_values"] == {"telephone": "+971 6 5050023"}
    conflicted = make_institutional_with_lines({HR_LINE_CHUNK: "Human Resources +971 6 5051111 +971 6 5052222"})
    r = run(conflicted, "What is the phone number of Human Resources?")
    assert r.status is FindingStatus.PARTIAL and [f.evidence_quote for f in r.findings] == ["Human Resources"]
    assert not any(f.metadata.get("contact_values") for f in r.findings)
    assert any("more than one line" in m for m in r.missing)
    text = json.dumps(_without_ms(r.to_dict()), ensure_ascii=False)
    assert not PHONE_PATTERN.search(text)                       # neither number is output, so no winner is chosen


def test_g2_two_units_stated_to_do_one_duty_give_every_candidate(sp):
    base = run(sp, "Who maintains the Data Center?")
    assert base.status is FindingStatus.SUPPORTED and [f.metadata["office_id"] for f in base.findings] == ["information_technology_center"]
    conflicted = make_institutional_with_text(appends={ITC_PARAGRAPH_CHUNK: " The Physical Plant also maintains a cutting-edge Data Center."})
    r = run(conflicted, "Who maintains the Data Center?")
    assert r.status is FindingStatus.PARTIAL
    assert {f.metadata["office_id"] for f in r.findings} == {"physical_plant", "information_technology_center"}
    assert any("more than one unit" in m for m in r.missing)


def _with_other_source_chunks(extra):
    """The fixture corpus plus chunks copied from given chunks but labelled with another source id."""
    corpus = institutional_corpus()
    chunks, metadata = list(corpus.chunks), [dict(m) for m in corpus.metadata]
    for like, text in extra:
        meta = dict(metadata[like])
        meta["source_id"], meta["chunk_id"] = "uos_hr_portal_guide", len(chunks)
        chunks.append(text)
        metadata.append(meta)
    res = InstitutionalResources(embed=lambda t: corpus.embedder.encode([t]), rerank=lambda pairs: [0.0] * len(pairs),
                                 index=corpus.index, chunks=chunks, metadata=metadata, source=corpus.source, section_map=corpus.section_map)
    return InstitutionalNavigationSpecialist(res)


def test_g3_a_chunk_from_another_source_is_never_read(sp):
    other = _with_other_source_chunks([(HR_LINE_CHUNK, "Human Resources +971 6 5059999 +971 6 5058888"),
                                       (ITC_PARAGRAPH_CHUNK, "The Physical Plant also maintains a cutting-edge Data Center.")])
    assert len(other.statements) == len(sp.statements) and len(other.directory_rows()) == len(sp.directory_rows())
    for question in ("What is the phone number of Human Resources?", "Who maintains the Data Center?"):
        a, b = run(sp, question), run(other, question)
        assert _without_ms(a.to_dict()) == _without_ms(b.to_dict()), question
        assert all(f.source_id == APPROVED_SOURCE_ID for f in b.findings)
    assert "5059999" not in json.dumps(run(other, "What is the phone number of Human Resources?").to_dict())


# ---------------------------------------------------------------------------
# H. Determinism independent of retrieval
# ---------------------------------------------------------------------------
class _ReversedReranker:
    """Scores the candidates in the opposite order of their position: the worst ranking a retriever could give."""

    def predict(self, pairs):
        return [float(-i) for i in range(len(pairs))]


def _questions():
    seen, out = set(), []
    for case in evidence_cases() + contact_cases():
        key = (case["question"], tuple(case.get("clauses") or ()))
        if key not in seen and not case.get("append_lines") and not case.get("replace_lines"):
            seen.add(key)
            out.append(case)
    return out


def test_h1_answers_do_not_depend_on_the_ranking_or_the_instance(sp):
    other = make_institutional(reranker=_ReversedReranker())
    cases = _questions()
    assert len(cases) > 350
    for case in cases:
        a = run(sp, case["question"], clauses=case.get("clauses"))
        b = run(other, case["question"], clauses=case.get("clauses"))
        assert _without_ms(a.to_dict()) == _without_ms(b.to_dict()), case["case_id"]


def test_h2_answering_never_calls_the_reranker_the_embedder_or_the_index():
    corpus = institutional_corpus()
    calls = []

    def spy(name):
        def f(*args, **kwargs):
            calls.append(name)
            raise AssertionError("%s called while answering" % name)
        return f

    sp = InstitutionalNavigationSpecialist(InstitutionalResources(embed=spy("embed"), rerank=spy("rerank"), index=None,
                                                                  chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source,
                                                                  section_map=corpus.section_map))
    for question in ("What is the phone number of Human Resources?", "Who maintains the Data Center?", "Where is the Medical Library?",
                     "Who chairs the College Council?", "How many colleges does the university have?",
                     "Which office handles faculty housing?", "How do I log in to MyUOS?"):
        run(sp, question)
    assert calls == []


def test_h3_two_constructions_read_the_same_evidence(sp):
    other = make_institutional()
    assert other.statements == sp.statements and other.placements == sp.placements
    assert [r.text for r in other.directory_rows()] == [r.text for r in sp.directory_rows()]
