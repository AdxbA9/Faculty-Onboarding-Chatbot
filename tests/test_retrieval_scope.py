"""Retrieval scope: the optional restriction of a search to part of the corpus.

All corpora here are synthetic (deterministic vectors, real FAISS flat index),
so no model is loaded and nothing touches the network. The scope=None cases
pin the candidate order the unrestricted path produced before the scope was
added, captured on the test corpus in ``conftest.py``.
"""
from __future__ import annotations

import numpy as np
import faiss
import pytest

from handbook_bot import retrieval
from handbook_bot.config import RERANK_CANDIDATES, TOP_K_DENSE
from handbook_bot.retrieval import RetrievalScope, _search_within, classify_query, gather_candidates

from conftest import CORPUS, FakeEmbedder

HANDBOOK = "uos_faculty_handbook_2025_26"
OTHER = "unregistered_other_guide"


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def _unit(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


@pytest.fixture(scope="module")
def synthetic():
    """60 chunks: 40 handbook chunks in chapters 1-4 (10 pages each, two
    sections per chapter) and 20 chunks of an unregistered document."""
    rng = np.random.default_rng(7)
    dim, n = 32, 60
    vectors = _unit(rng.standard_normal((n, dim))).astype(np.float32)
    chunks, metadata = [], []
    for i in range(n):
        handbook = i < 40
        chapter = i // 10 + 1 if handbook else None
        section = "%d.%d" % (chapter, (i % 10) // 5 + 1) if handbook else None
        chunks.append("synthetic chunk %d about %s" % (i, "policy" if i % 3 else "grants"))
        metadata.append({
            "page": i + 1, "section": "", "chunk_type": "paragraph", "chunk_id": i, "text": chunks[-1],
            "source_id": HANDBOOK if handbook else OTHER, "source_title": "H" if handbook else "O",
            "source_type": "pdf", "source_version": "", "chapter": chapter, "chapter_title": None,
            "section_no": section, "section_title": None, "page_section_nos": [section] if section else [],
        })
    # Page 10 is shared by sections 1.1 and 1.2 of chapter 1, as boundary pages
    # are in the real map; chunk 38 (chapter 4) carries a section the document
    # prints as "12.3", like the handbook's page 180.
    metadata[9]["page_section_nos"] = ["1.1", "1.2"]
    metadata[38]["page_section_nos"] = ["4.2", "12.3"]
    # Identical strong lexical text inside and outside the handbook.
    chunks[3] = chunks[50] = "annual leave entitlement thirty days each academic year"
    metadata[3]["text"] = metadata[50]["text"] = chunks[3]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return vectors, chunks, metadata, index


def _query(vectors, i):
    return vectors[i:i + 1].copy()


def _brute(vectors, query, allowed):
    scores = vectors[allowed] @ query[0]
    order = np.argsort(-scores, kind="stable")
    return [int(allowed[j]) for j in order], [float(scores[j]) for j in order]


# ---------------------------------------------------------------------------
# RetrievalScope: construction and matching
# ---------------------------------------------------------------------------
def test_scope_needs_at_least_one_constraint():
    with pytest.raises(ValueError, match="at least one"):
        RetrievalScope()
    with pytest.raises(ValueError, match="at least one"):
        RetrievalScope(source_ids=[], section_numbers=set())


@pytest.mark.parametrize("kwargs", [
    {"source_ids": "uos_faculty_handbook_2025_26"},      # a string is not a collection
    {"section_numbers": ["1.14", "chapter one"]},
    {"section_prefixes": ["12."]},
    {"page_ranges": [(5, 1)]},
    {"page_ranges": [(0, 3)]},
    {"page_ranges": [7]},
    {"chapters": [0]},
    {"chapters": [True]},
    {"source_ids": [""]},
])
def test_invalid_scopes_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        RetrievalScope(**kwargs)


def test_scope_normalises_inputs_and_is_immutable_and_hashable():
    scope = RetrievalScope(source_ids=[HANDBOOK, HANDBOOK], chapters=[3, 1], section_numbers=("1.14",),
                           section_prefixes=["12"], page_ranges=[(40, 55), (1, 3), (40, 55)])
    assert scope.source_ids == frozenset({HANDBOOK}) and scope.chapters == frozenset({1, 3})
    assert scope.section_numbers == frozenset({"1.14"}) and scope.page_ranges == ((1, 3), (40, 55))
    assert hash(scope) == hash(RetrievalScope(source_ids=[HANDBOOK], chapters=[1, 3], section_numbers=["1.14"],
                                              section_prefixes=["12"], page_ranges=[(1, 3), (40, 55)]))
    with pytest.raises(AttributeError):
        scope.chapters = frozenset({2})
    assert scope.describe() == {"source_ids": [HANDBOOK], "chapters": [1, 3], "section_numbers": ["1.14"],
                                "section_prefixes": ["12"], "page_ranges": [[1, 3], [40, 55]]}


def test_scope_matching_rules():
    meta = {"page": 51, "source_id": HANDBOOK, "chapter": 1, "section_no": "1.15", "page_section_nos": ["1.14", "1.15"]}
    assert RetrievalScope(source_ids=[HANDBOOK]).allows(meta)
    assert not RetrievalScope(source_ids=[OTHER]).allows(meta)
    assert RetrievalScope(chapters=[1]).allows(meta) and not RetrievalScope(chapters=[2]).allows(meta)
    assert RetrievalScope(section_numbers=["1.14"]).allows(meta)          # present on the boundary page
    assert RetrievalScope(section_numbers=["1.15"]).allows(meta)
    assert not RetrievalScope(section_numbers=["1.1"]).allows(meta)      # exact match, not a substring
    assert RetrievalScope(section_prefixes=["1"]).allows(meta)
    assert not RetrievalScope(section_prefixes=["12"]).allows(meta)      # "12" is not a prefix of "1.14"
    assert RetrievalScope(section_prefixes=["1.1"]).allows(meta) is False  # "1.1" is not a prefix of "1.14"
    assert RetrievalScope(page_ranges=[(40, 55)]).allows(meta) and not RetrievalScope(page_ranges=[(1, 50)]).allows(meta)
    # AND across constraint types
    assert RetrievalScope(source_ids=[HANDBOOK], chapters=[1]).allows(meta)
    assert not RetrievalScope(source_ids=[HANDBOOK], chapters=[2]).allows(meta)


def test_chunks_without_the_needed_metadata_never_match():
    legacy = {"page": 49, "section": "Page 49", "chunk_type": "paragraph"}       # a v11 cache entry
    assert not RetrievalScope(source_ids=[HANDBOOK]).allows(legacy)
    assert not RetrievalScope(chapters=[1]).allows(legacy)
    assert not RetrievalScope(section_numbers=["1.14"]).allows(legacy)
    assert not RetrievalScope(section_prefixes=["1"]).allows(legacy)
    assert RetrievalScope(page_ranges=[(40, 55)]).allows(legacy)                  # page exists on every chunk
    assert not RetrievalScope(page_ranges=[(40, 55)]).allows({"section_no": "1.14"})
    single = {"chapter": 1, "section_no": "1.14"}                                 # no page_section_nos list
    assert RetrievalScope(section_numbers=["1.14"]).allows(single)
    assert not RetrievalScope(section_numbers=["1.14"]).allows({"section_no": "1.14"})   # no resolved chapter


# ---------------------------------------------------------------------------
# scope=None: unchanged behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,expected_order,expected_first_dense", [
    ("How much annual leave are faculty members entitled to?", [0, 5, 4, 3, 1, 2], 0.566139),
    ("What is the phone number of the Information Technology Center?", [1, 2, 5, 4, 3, 0], 0.447214),
    ("When do classes begin?", [2, 1, 5, 4, 3, 0], 0.534523),
    ("How many degree programs does UoS offer?", [5, 2, 1, 4, 3, 0], 0.46188),
    ("Who chairs the University Council?", [3, 5, 4, 0, 2, 1], 0.408248),
    ("promotion requirements", [4, 5, 3, 0, 2, 1], 0.223607),
])
def test_unscoped_candidates_match_the_pre_scope_snapshot(question, expected_order, expected_first_dense):
    """Candidate order and dense scores recorded from the unrestricted path
    before the scope parameter existed. A change here is a retrieval change."""
    chunks = [row[0] for row in CORPUS]
    metadata = [{"page": row[1], "section": row[2], "chunk_type": row[3]} for row in CORPUS]
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    query = embedder.encode([question]).astype(np.float32)
    candidates = gather_candidates(question, classify_query(question), query, index, chunks, metadata)
    assert [c["idx"] for c in candidates] == expected_order
    assert round(candidates[0]["dense_score"], 6) == expected_first_dense
    again = gather_candidates(question, classify_query(question), query, index, chunks, metadata, scope=None)
    assert again == candidates


def test_unscoped_path_ignores_missing_metadata(synthetic):
    vectors, chunks, metadata, index = synthetic
    bare = [{"page": m["page"], "chunk_type": "paragraph"} for m in metadata]
    a = gather_candidates("policy", "policy", _query(vectors, 4), index, chunks, metadata)
    b = gather_candidates("policy", "policy", _query(vectors, 4), index, chunks, bare)
    assert [c["idx"] for c in a] == [c["idx"] for c in b]
    assert len(a) == RERANK_CANDIDATES


# ---------------------------------------------------------------------------
# Scoped retrieval
# ---------------------------------------------------------------------------
def test_source_scope_excludes_other_source_chunks(synthetic):
    vectors, chunks, metadata, index = synthetic
    scope = RetrievalScope(source_ids=[HANDBOOK])
    for q in (4, 27, 55):
        candidates = gather_candidates("policy grants", "policy", _query(vectors, q), index, chunks, metadata, scope=scope)
        assert candidates and all(metadata[c["idx"]]["source_id"] == HANDBOOK for c in candidates)
        assert all(scope.allows(metadata[c["idx"]]) for c in candidates)
    unscoped = gather_candidates("policy grants", "policy", _query(vectors, 55), index, chunks, metadata)
    assert any(metadata[c["idx"]]["source_id"] == OTHER for c in unscoped)   # the exclusion is the scope's doing


def test_section_scope_excludes_out_of_section_chunks(synthetic):
    vectors, chunks, metadata, index = synthetic
    scope = RetrievalScope(section_numbers=["2.1"])
    candidates = gather_candidates("policy", "policy", _query(vectors, 12), index, chunks, metadata, scope=scope)
    assert candidates
    assert {c["idx"] for c in candidates} == set(range(10, 15))               # chapter 2, section 2.1: pages 11 to 15
    prefix = RetrievalScope(section_prefixes=["2"])
    candidates = gather_candidates("policy", "policy", _query(vectors, 12), index, chunks, metadata, scope=prefix)
    assert {c["idx"] for c in candidates} == set(range(10, 20))
    assert not RetrievalScope(section_prefixes=["2"]).allows(metadata[25])     # chapter 3 chunk


def test_chapter_and_page_range_scopes(synthetic):
    vectors, chunks, metadata, index = synthetic
    chapter = gather_candidates("policy", "policy", _query(vectors, 33), index, chunks, metadata,
                                scope=RetrievalScope(chapters=[4]))
    assert chapter and {c["idx"] for c in chapter} <= set(range(30, 40))
    pages = gather_candidates("policy", "policy", _query(vectors, 33), index, chunks, metadata,
                              scope=RetrievalScope(page_ranges=[(2, 4), (58, 60)]))
    assert {c["idx"] for c in pages} == {1, 2, 3, 57, 58, 59}


def test_dense_retrieval_reaches_in_scope_chunks_below_the_global_top_k(synthetic):
    """The global top TOP_K_DENSE hits are all outside the scope and every
    in-scope chunk ranks below them. A post-filter of the global search would
    return nothing; the scoped search must return the in-scope chunks with
    their exact scores."""
    vectors, chunks, metadata, index = synthetic
    query = _query(vectors, 59)
    order, _ = _brute(vectors, query, list(range(len(chunks))))
    lowest = order[-5:]                                    # ranked 56th to 60th globally
    assert all(o not in order[:TOP_K_DENSE] for o in lowest)
    scope = RetrievalScope(page_ranges=[(i + 1, i + 1) for i in lowest])
    candidates = gather_candidates("zzz no lexical match", "policy", query, index, chunks, metadata, scope=scope)
    assert sorted(c["idx"] for c in candidates) == sorted(lowest)
    expected_ids, expected_scores = _brute(vectors, query, lowest)
    got = {c["idx"]: c["dense_score"] for c in candidates}
    for i, s in zip(expected_ids, expected_scores):
        assert got[i] == pytest.approx(s, abs=1e-5)


def test_scoped_dense_search_equals_brute_force_on_the_allowed_subset(synthetic):
    vectors, chunks, metadata, index = synthetic
    allowed = [i for i, m in enumerate(metadata) if m["chapter"] in (2, 3)]
    query = _query(vectors, 58)
    D, I = _search_within(index, query, 12, allowed)
    expected_ids, expected_scores = _brute(vectors, query, allowed)
    assert I[0].tolist() == expected_ids[:12]
    assert np.allclose(D[0], expected_scores[:12], atol=1e-5)


def test_fallback_scoring_matches_the_index_level_search(synthetic):
    """An index that rejects search parameters is scored from its vectors."""
    vectors, chunks, metadata, index = synthetic

    class NoParams:
        ntotal = index.ntotal
        metric_type = index.metric_type

        def search(self, x, k, params=None):
            if params is not None:
                raise TypeError("params not supported")
            return index.search(x, k)

        def reconstruct_batch(self, ids):
            return index.reconstruct_batch(ids)

    allowed = list(range(20, 45))
    query = _query(vectors, 2)
    D1, I1 = _search_within(index, query, 8, allowed)
    D2, I2 = _search_within(NoParams(), query, 8, allowed)
    assert I1[0].tolist() == I2[0].tolist()
    assert np.allclose(D1[0], D2[0], atol=1e-5)


def test_lexical_retrieval_respects_the_scope(synthetic):
    vectors, chunks, metadata, index = synthetic
    question = "annual leave entitlement thirty days"
    query = _query(vectors, 59)                          # dense signal unrelated to chunks 3 and 50
    unscoped = gather_candidates(question, "policy", query, index, chunks, metadata)
    assert {3, 50} <= {c["idx"] for c in unscoped if c["lexical_score"] > 0}
    scoped = gather_candidates(question, "policy", query, index, chunks, metadata, scope=RetrievalScope(source_ids=[HANDBOOK]))
    ids = {c["idx"] for c in scoped}
    assert 3 in ids and 50 not in ids
    assert next(c for c in scoped if c["idx"] == 3)["lexical_score"] > 0


def test_boosting_and_reranking_cannot_reintroduce_excluded_chunks(synthetic):
    vectors, chunks, metadata, index = synthetic
    scope = RetrievalScope(chapters=[1])
    for query_type in ("contact", "date", "count", "list", "policy_yesno", "policy"):
        candidates = gather_candidates("phone number date total list", query_type, _query(vectors, 45),
                                       index, chunks, metadata, scope=scope)
        assert candidates and all(scope.allows(metadata[c["idx"]]) for c in candidates)
        reranked = sorted(candidates, key=lambda c: (c["idx"] * 7919) % 97, reverse=True)[:5]   # any reorder
        assert all(scope.allows(metadata[c["idx"]]) for c in reranked)


def test_empty_scope_returns_nothing_and_never_falls_back(synthetic):
    vectors, chunks, metadata, index = synthetic
    query = _query(vectors, 4)
    assert gather_candidates("policy", "policy", query, index, chunks, metadata) != []
    for scope in (RetrievalScope(source_ids=["blackboard_guide"]), RetrievalScope(chapters=[9]),
                  RetrievalScope(section_numbers=["7.7"]), RetrievalScope(page_ranges=[(200, 300)]),
                  RetrievalScope(source_ids=[HANDBOOK], chapters=[1], page_ranges=[(30, 39)])):
        assert gather_candidates("policy", "policy", query, index, chunks, metadata, scope=scope) == []


def test_scoped_retrieval_is_deterministic(synthetic):
    vectors, chunks, metadata, index = synthetic
    scope = RetrievalScope(source_ids=[HANDBOOK], section_prefixes=["1", "2"])
    a = gather_candidates("policy grants", "policy", _query(vectors, 7), index, chunks, metadata, scope=scope)
    b = gather_candidates("policy grants", "policy", _query(vectors, 7), index, chunks, metadata, scope=scope)
    assert a == b


def test_all_inclusive_scope_yields_the_same_candidates_as_no_scope(synthetic):
    vectors, chunks, metadata, index = synthetic
    everything = RetrievalScope(page_ranges=[(1, 60)])
    query = _query(vectors, 21)
    unscoped = gather_candidates("policy grants", "policy", query, index, chunks, metadata)
    scoped = gather_candidates("policy grants", "policy", query, index, chunks, metadata, scope=everything)
    assert {c["idx"] for c in scoped} == {c["idx"] for c in unscoped}
    by_idx = {c["idx"]: c for c in unscoped}
    for c in scoped:
        assert c["dense_score"] == pytest.approx(by_idx[c["idx"]]["dense_score"], abs=1e-6)
        assert c["lexical_score"] == by_idx[c["idx"]]["lexical_score"]
        assert c["routing_boost"] == by_idx[c["idx"]]["routing_boost"]


def test_scope_must_be_a_retrieval_scope(synthetic):
    vectors, chunks, metadata, index = synthetic
    with pytest.raises(TypeError):
        gather_candidates("policy", "policy", _query(vectors, 4), index, chunks, metadata, scope="teaching")
    with pytest.raises(TypeError):
        gather_candidates("policy", "policy", _query(vectors, 4), index, chunks, metadata, scope={"chapters": [1]})


def test_retrieval_module_exposes_the_scope_publicly():
    assert retrieval.RetrievalScope is RetrievalScope


# ---------------------------------------------------------------------------
# Chapter-consistent section matching (review finding F-1)
# ---------------------------------------------------------------------------
QUIRK = {"page": 180, "source_id": HANDBOOK, "chapter": 9, "section_no": "12.3", "page_section_nos": ["9.5", "12.3", "9.7"]}
REAL_12_3 = {"page": 212, "source_id": HANDBOOK, "chapter": 12, "section_no": "12.3", "page_section_nos": ["12.2", "12.3"]}


def test_section_constraints_require_the_printed_chapter_to_match_the_resolved_chapter():
    assert not RetrievalScope(section_prefixes=["12"]).allows(QUIRK)                   # A
    assert not RetrievalScope(section_numbers=["12.3"]).allows(QUIRK)                  # B
    assert RetrievalScope(section_numbers=["12.3"]).allows(REAL_12_3)                  # C
    assert RetrievalScope(chapters=[9]).allows(QUIRK)                                  # D
    assert not RetrievalScope(chapters=[9], section_numbers=["12.3"]).allows(QUIRK)    # E: intentionally no match
    assert RetrievalScope(chapters=[12], section_numbers=["12.3"]).allows(REAL_12_3)   # F
    assert not RetrievalScope(chapters=[12], section_numbers=["12.3"]).allows(QUIRK)
    assert RetrievalScope(section_prefixes=["9"]).allows(QUIRK)            # its real chapter's sections still match
    assert RetrievalScope(page_ranges=[(180, 180)]).allows(QUIRK)          # and a page range reaches it


def test_chunks_without_a_resolved_chapter_never_match_section_constraints():
    for meta in ({"section_no": "1.14", "page_section_nos": ["1.14"]}, {"chapter": None, "page_section_nos": ["1.14"]},
                 {"chapter": "1", "page_section_nos": ["1.14"]}, {"chapter": True, "page_section_nos": ["1.14"]}):
        assert not RetrievalScope(section_numbers=["1.14"]).allows(meta)
        assert not RetrievalScope(section_prefixes=["1"]).allows(meta)


def test_printed_quirk_chunk_never_enters_a_prefix_scope_through_retrieval(synthetic):
    vectors, chunks, metadata, index = synthetic
    query = _query(vectors, 38)                                      # the quirk chunk itself is the query vector
    assert gather_candidates("policy", "policy", query, index, chunks, metadata, scope=RetrievalScope(section_prefixes=["12"])) == []
    assert gather_candidates("policy", "policy", query, index, chunks, metadata, scope=RetrievalScope(section_numbers=["12.3"])) == []
    by_chapter = gather_candidates("policy", "policy", query, index, chunks, metadata, scope=RetrievalScope(chapters=[4]))
    assert 38 in {c["idx"] for c in by_chapter}


# ---------------------------------------------------------------------------
# Global invariant: every candidate of a scoped search satisfies the scope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scope", [
    RetrievalScope(source_ids=[HANDBOOK]), RetrievalScope(source_ids=[OTHER]), RetrievalScope(chapters=[2, 3]),
    RetrievalScope(section_numbers=["1.1", "3.2"]), RetrievalScope(section_prefixes=["4"]),
    RetrievalScope(page_ranges=[(5, 12), (50, 55)]), RetrievalScope(source_ids=[HANDBOOK], chapters=[1], page_ranges=[(1, 8)]),
    RetrievalScope(chapters=[4], section_prefixes=["12"]), RetrievalScope(source_ids=["ghost"]),
])
def test_every_scoped_candidate_satisfies_the_scope(synthetic, scope):
    vectors, chunks, metadata, index = synthetic
    for q in (0, 17, 38, 59):
        candidates = gather_candidates("policy grants leave", "policy", _query(vectors, q), index, chunks, metadata, scope=scope)
        allowed = set(scope.allowed_indices(metadata))
        assert {c["idx"] for c in candidates} <= allowed
        assert all(scope.allows(c["meta"]) for c in candidates)
        if not allowed:
            assert candidates == []


# ---------------------------------------------------------------------------
# Dense fallback safety (review finding F-4)
# ---------------------------------------------------------------------------
class _NoParams:
    """An index object that cannot take FAISS search parameters."""

    def __init__(self, inner, metric_type=faiss.METRIC_INNER_PRODUCT):
        self.inner, self.ntotal, self.metric_type = inner, inner.ntotal, metric_type

    def search(self, x, k, params=None):
        if params is not None:
            raise TypeError("params not supported")
        return self.inner.search(x, k)

    def reconstruct_batch(self, ids):
        return self.inner.reconstruct_batch(ids)


def test_fallback_rejects_a_non_inner_product_index():
    l2 = faiss.IndexFlatL2(4)
    l2.add(np.eye(4, dtype=np.float32))
    query = np.eye(4, dtype=np.float32)[:1]
    with pytest.raises(TypeError, match="inner-product"):
        _search_within(_NoParams(l2, faiss.METRIC_L2), query, 2, [1, 2])
    with pytest.raises(TypeError, match="inner-product"):
        _search_within(_NoParams(l2, None), query, 2, [1, 2])


def test_a_genuine_faiss_runtime_error_propagates(synthetic):
    vectors, chunks, metadata, index = synthetic

    class Broken:
        ntotal = index.ntotal
        metric_type = faiss.METRIC_INNER_PRODUCT

        def search(self, x, k, params=None):
            raise RuntimeError("simulated index failure")

    with pytest.raises(RuntimeError, match="simulated"):
        _search_within(Broken(), _query(vectors, 1), 3, [1, 2, 3])
    with pytest.raises(RuntimeError, match="simulated"):
        gather_candidates("policy", "policy", _query(vectors, 1), Broken(), chunks, metadata, scope=RetrievalScope(chapters=[1]))


@pytest.mark.parametrize("seed,n,dim", [(101, 40, 4), (102, 300, 24), (103, 1200, 64)])
def test_selector_and_fallback_match_brute_force_across_seeds(seed, n, dim):
    rng = np.random.default_rng(seed)
    vectors = _unit(rng.standard_normal((n, dim))).astype(np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    for trial in range(4):
        query = _unit(rng.standard_normal((1, dim))).astype(np.float32)
        allowed = sorted(int(i) for i in rng.choice(n, size=int(rng.integers(1, min(n, 50))), replace=False))
        k = min(TOP_K_DENSE, len(allowed))
        expected_ids, expected_scores = _brute(vectors, query, allowed)
        for probe in (index, _NoParams(index)):
            D, I = _search_within(probe, query, k, allowed)
            ids = I[0].tolist()
            assert -1 not in ids and len(set(ids)) == len(ids)
            assert set(ids) == set(expected_ids[:k]) and set(ids) <= set(allowed)
            assert np.allclose(np.sort(D[0]), np.sort(expected_scores[:k]), atol=1e-5)


def test_single_allowed_chunk_and_empty_corpus(synthetic):
    vectors, chunks, metadata, index = synthetic
    only = RetrievalScope(page_ranges=[(23, 23)])
    result = gather_candidates("policy", "policy", _query(vectors, 5), index, chunks, metadata, scope=only)
    assert [c["idx"] for c in result] == [22] and result[0]["meta"]["page"] == 23
    empty_index = faiss.IndexFlatIP(32)
    assert gather_candidates("policy", "policy", _query(vectors, 5), empty_index, [], [], scope=only) == []
    assert gather_candidates("policy", "policy", _query(vectors, 5), empty_index, [], []) == []
