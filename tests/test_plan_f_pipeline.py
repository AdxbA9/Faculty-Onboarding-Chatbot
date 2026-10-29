"""
Runtime integration phase I-2: the feature-flagged seam in ``ui/pipeline.py`` and the per-knowledge-base runtime
cache (``plan_f_runtime.RuntimeCache``).

LEGACY mode (both switches off, the default) must behave exactly like the direct ``answer_question`` call and must
neither import nor build Plan F. SHADOW mode returns the legacy answer unchanged and records the Plan F attempt beside
it. ACTIVE mode returns a Plan F answer only when the I-1 adapter accepted it and falls back to the legacy answer for
every other outcome. No Plan F failure may raise through ``ask``. Nothing here touches the real environment: the
switches are set on the config module by ``monkeypatch`` and reset after every test.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import faiss
import pytest

import test_plan_f_runtime as T
from conftest import CORPUS, GROUNDED_LEAVE, LEAVE_QUESTION, FakeEmbedder, FakeGroqClient, FakeReranker
from handbook_bot import config
from handbook_bot import plan_f_runtime as PF
from handbook_bot.qa import QAResult, answer_question
from ui import pipeline

REPO = Path(__file__).resolve().parents[1]


class NS(SimpleNamespace):
    """A namespace that supports weak references, as the real ``KnowledgeBase`` does (``types.SimpleNamespace`` itself does not)."""


HR_PHONE = T.HR_PHONE
HR_KNOWN_WRONG = T.HR_KNOWN_WRONG
ANNUAL_LEAVE = T.ANNUAL_LEAVE
USER_FIELDS = ("answer", "pages", "best_section", "query_type", "evidence", "items", "used_llm", "llm_calls", "retried", "sub_questions",
               "num_candidates", "num_reranked")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """Every test starts in LEGACY mode with an empty cache and ends the same way."""
    monkeypatch.setattr(config, "PLAN_F_ENABLED", False)
    monkeypatch.setattr(config, "PLAN_F_SHADOW", False)
    pipeline.reset_plan_f_state()
    yield
    pipeline.reset_plan_f_state()


def set_mode(monkeypatch, mode):
    monkeypatch.setattr(config, "PLAN_F_ENABLED", mode == pipeline.MODE_ACTIVE)
    monkeypatch.setattr(config, "PLAN_F_SHADOW", mode == pipeline.MODE_SHADOW)


@pytest.fixture(scope="module")
def kb():
    """The full real handbook, fake embedder, stand-in reranker, no model client."""
    return T.make_kb()


@pytest.fixture(scope="module")
def other_kb():
    """A second, smaller knowledge base: enough real pages for every specialist to build."""
    wanted = set(range(14, 66)) | set(range(79, 82)) | {96, 107} | set(range(153, 157)) | {165, 224}
    return T.make_kb([p for p in T.IS.handbook_pages() if p["page"] in wanted])


def small_kb(*replies):
    """The six-chunk legacy fixture corpus as a knowledge-base object, with a freshly scripted model client."""
    chunks = [row[0] for row in CORPUS]
    metadata = [{"page": row[1], "section": row[2], "chunk_type": row[3]} for row in CORPUS]
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    pages = [{"page": i + 1, "text": ""} for i in range(50)]
    client = FakeGroqClient(*replies) if replies else None
    return NS(embedder=embedder, reranker=FakeReranker(), index=index, chunks=chunks, metadata=metadata, pages=pages,
                           groq_client=client)


def legacy_direct(kb, question):
    return answer_question(question, embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata,
                           pages=kb.pages, groq_client=kb.groq_client)


def visible(result):
    return {f: getattr(result, f) for f in USER_FIELDS}


def trace_of(result):
    return [(e.agent, e.decision, e.used_llm) for e in result.agent_trace]


def build_spy(monkeypatch):
    calls = []
    real = PF.build_runtime

    def spy(kb_, **kwargs):
        calls.append(id(kb_))
        return real(kb_, **kwargs)

    monkeypatch.setattr(PF, "build_runtime", spy)
    return calls


def stub_runtime(kb, **behaviours):
    """A PlanFRuntime on stub specialists, installed as the cache's builder for ``kb``."""
    runtime = PF.PlanFRuntime(T.stub_registry(**behaviours), kb.chunks, kb.metadata)
    return runtime


def install_runtime(monkeypatch, runtime):
    monkeypatch.setattr(PF, "build_runtime", lambda kb_, **kw: (runtime, PF.RegistryBuild(runtime.registry, {}, 0.0)))


def fresh_python(code, env=None):
    merged = dict(os.environ)
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("PLAN_F_ENABLED", "PLAN_F_SHADOW"):
        merged.pop(name, None)
    if env:
        merged.update(env)
    proc = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=str(REPO), env=merged, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# ---------------------------------------------------------------------------
# Feature flags (sections 3, 26, 27)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enabled,shadow,expected", [
    (False, False, pipeline.MODE_LEGACY),
    (False, True, pipeline.MODE_SHADOW),
    (True, False, pipeline.MODE_ACTIVE),
    (True, True, pipeline.MODE_SHADOW),            # documented precedence: SHADOW wins
])
def test_mode_matrix(monkeypatch, enabled, shadow, expected):
    assert pipeline.plan_f_mode(enabled, shadow) == expected
    monkeypatch.setattr(config, "PLAN_F_ENABLED", enabled)
    monkeypatch.setattr(config, "PLAN_F_SHADOW", shadow)
    assert pipeline.plan_f_mode() == expected


def test_defaults_are_legacy():
    assert config.PLAN_F_ENABLED is False and config.PLAN_F_SHADOW is False
    assert pipeline.plan_f_mode() == pipeline.MODE_LEGACY


def test_environment_parsing_follows_the_existing_convention():
    out = fresh_python(
        "import importlib, os, sys\n"
        "sys.path.insert(0, '.')\n"
        "import handbook_bot.config as config\n"
        "for value in [None, 'true', 'True', 'TRUE', 'false', 'False', '1', '0', 'yes', 'on', 'off', ' 1 ', 'maybe', '']:\n"
        "    for name in ('PLAN_F_ENABLED', 'PLAN_F_SHADOW'):\n"
        "        os.environ.pop(name, None)\n"
        "        if value is not None:\n"
        "            os.environ[name] = value\n"
        "    importlib.reload(config)\n"
        "    print('ROW|%r|%s|%s|%s' % (value, config.PLAN_F_ENABLED, config.PLAN_F_SHADOW, config.MAX_LLM_CALLS))\n"
    )
    rows = {}
    for line in out.strip().splitlines():
        if line.startswith("ROW|"):
            _tag, key, enabled, shadow, budget = line.split("|")
            rows[key] = [enabled, shadow, budget]
    truthy = {"'true'", "'True'", "'TRUE'", "'1'", "'yes'", "'on'", "' 1 '"}
    for key, (enabled, shadow, budget) in rows.items():
        expected = "True" if key in truthy else "False"
        assert enabled == shadow == expected, (key, enabled, shadow)
        assert budget == "2"                                                   # the Milestone 1 budget is untouched
    assert rows["None"] == ["False", "False", "2"] and len(rows) == 14


# ---------------------------------------------------------------------------
# LEGACY mode: identical behaviour, no Plan F work (sections 8, 9)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,replies", [
    ("Hello", ()),
    ("What is the phone number for the Information Technology Center?", ()),
    ("How many degree programs does UoS offer?", ()),
    ("When do classes begin?", ()),
    (LEAVE_QUESTION, (GROUNDED_LEAVE,)),
    ("What is the policy on sabbatical leave?", ("I do not have this information.",)),
    ("Is there a deadline for add/drop and when do classes begin?", (GROUNDED_LEAVE,)),
    ("asdfgh qwerty zxcvb", ()),
    (LEAVE_QUESTION, ()),
])
def test_legacy_mode_is_identical_to_answer_question(question, replies):
    through_pipeline = pipeline.ask(small_kb(*replies), question)
    direct = legacy_direct(small_kb(*replies), question)
    assert isinstance(through_pipeline, QAResult)
    assert visible(through_pipeline) == visible(direct)
    assert trace_of(through_pipeline) == trace_of(direct)
    assert set(through_pipeline.timings) == set(direct.timings)
    assert through_pipeline.to_dict().keys() == direct.to_dict().keys()


def test_legacy_mode_never_attempts_plan_f(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan F must not run in LEGACY mode")

    monkeypatch.setattr(pipeline, "_attempt_plan_f", forbidden)
    monkeypatch.setattr(pipeline, "_plan_f_cache", forbidden)
    calls = build_spy(monkeypatch)
    result, diagnostics = pipeline.ask_with_diagnostics(small_kb(), "What is the phone number for the Information Technology Center?")
    assert result.answer.startswith("The phone number is") and diagnostics == {"mode": "legacy", "served": "legacy", "attempted": False}
    assert calls == [] and pipeline._PLAN_F_CACHE is None and pipeline.recent_plan_f_diagnostics() == []


def test_legacy_mode_imports_and_builds_nothing_of_plan_f_in_a_fresh_interpreter():
    out = fresh_python(
        "import sys\n"
        "sys.path.insert(0, '.'); sys.path.insert(0, 'tests')\n"
        "import ui.pipeline as pipeline\n"
        "import faiss\n"
        "from types import SimpleNamespace\n"
        "from conftest import CORPUS, FakeEmbedder, FakeReranker\n"
        "chunks = [r[0] for r in CORPUS]; metadata = [{'page': r[1], 'section': r[2], 'chunk_type': r[3]} for r in CORPUS]\n"
        "e = FakeEmbedder(); index = faiss.IndexFlatIP(FakeEmbedder.dim); index.add(e.encode(chunks))\n"
        "kb = SimpleNamespace(embedder=e, reranker=FakeReranker(), index=index, chunks=chunks, metadata=metadata,\n"
        "                     pages=[{'page': i + 1, 'text': ''} for i in range(50)], groq_client=None)\n"
        "for q in ('Hello', 'What is the phone number for the Information Technology Center?', 'How many degree programs does UoS offer?'):\n"
        "    pipeline.ask(kb, q)\n"
        "names = ['handbook_bot.plan_f_runtime', 'handbook_bot.agents.coordinator', 'handbook_bot.agents.specialists.teaching',\n"
        "         'handbook_bot.agents.specialists.institutional', 'handbook_bot.agents.specialists.dispatch']\n"
        "print('loaded', [n for n in names if n in sys.modules])\n"
        "print('cache', pipeline._PLAN_F_CACHE is None, 'mode', pipeline.plan_f_mode())\n"
    )
    assert "loaded []" in out and "cache True mode legacy" in out


def test_importing_the_pipeline_does_not_load_plan_f_or_the_orchestrator():
    out = fresh_python(
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "import ui.pipeline\n"
        "print('plan_f', 'handbook_bot.plan_f_runtime' in sys.modules, 'orchestrator', 'handbook_bot.orchestrator' in sys.modules)\n"
    )
    assert "plan_f False orchestrator False" in out


def test_ui_contract_is_unchanged():
    import inspect

    assert list(inspect.signature(pipeline.ask).parameters) == ["kb", "question"]
    source = (REPO / "ui" / "pipeline.py").read_text(encoding="utf-8")
    assert "answer_question(" in source and "from handbook_bot import plan_f_runtime" in source     # lazy, inside a function
    assert "plan_f_runtime" not in (REPO / "ui" / "chat.py").read_text(encoding="utf-8")
    assert "plan_f_runtime" not in (REPO / "app.py").read_text(encoding="utf-8")
    assert [f.name for f in dataclasses.fields(QAResult)] == T.QARESULT_FIELDS


# ---------------------------------------------------------------------------
# SHADOW mode (sections 12 to 15)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,outcome", [
    (HR_PHONE, "PLAN_F_ACCEPTED"),
    (HR_KNOWN_WRONG, "PLAN_F_UNSAFE_RESULT"),
    (ANNUAL_LEAVE, "PLAN_F_UNSAFE_RESULT"),
    ("Who handles faculty visas and where is that office?", "PLAN_F_UNSAFE_RESULT"),
    ("asdfgh qwerty zxcvb", "PLAN_F_NOT_APPLICABLE"),
    ("Hello", "PLAN_F_NOT_APPLICABLE"),
])
def test_shadow_returns_the_legacy_answer_unchanged_and_records_plan_f(kb, monkeypatch, question, outcome):
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    result, diagnostics = pipeline.ask_with_diagnostics(kb, question)
    direct = legacy_direct(kb, question)
    assert visible(result) == visible(direct) and trace_of(result) == trace_of(direct)       # Plan F touched nothing
    assert diagnostics["mode"] == "shadow" and diagnostics["served"] == "legacy" and diagnostics["attempted"] is True
    assert diagnostics["outcome"] == outcome and diagnostics["accepted"] is (outcome == "PLAN_F_ACCEPTED")
    assert diagnostics["registry_ok"] is True and diagnostics["plan_f_llm_calls"] == 0 and diagnostics["plan_f_used_llm"] is False
    assert diagnostics["legacy_pages"] == list(direct.pages) and diagnostics["legacy_llm_calls"] == direct.llm_calls
    if outcome == "PLAN_F_ACCEPTED":
        assert diagnostics["plan_f_pages"] == [54] and diagnostics["page_overlap"] == len({54} & set(direct.pages))
        assert diagnostics["answer_equal"] is False and diagnostics["specialist_id"] == "institutional"
    else:
        assert diagnostics["plan_f_pages"] == [] and diagnostics["fallback_reason"]
    assert pipeline.recent_plan_f_diagnostics()[-1] is diagnostics
    text = json.dumps(diagnostics)
    assert question not in text and result.answer not in text                 # no question or answer text is kept


def test_shadow_diagnostics_name_the_known_rejections(kb, monkeypatch):
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    _, hr = pipeline.ask_with_diagnostics(kb, HR_KNOWN_WRONG)
    assert hr["specialist_id"] == "faculty_services" and hr["status"] == "supported" and "ask shape general" in hr["fallback_reason"]
    _, leave = pipeline.ask_with_diagnostics(kb, ANNUAL_LEAVE)
    assert leave["status"] == "partial" and leave["fallback_reason"] == "status partial"
    _, nobody = pipeline.ask_with_diagnostics(kb, "asdfgh qwerty zxcvb")
    assert nobody["specialist_id"] is None and "no specialist selected" in nobody["fallback_reason"]


def test_shadow_builds_the_runtime_once_and_reuses_it(kb, monkeypatch):
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    calls = build_spy(monkeypatch)
    _, first = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    _, second = pipeline.ask_with_diagnostics(kb, HR_KNOWN_WRONG)
    _, third = pipeline.ask_with_diagnostics(kb, "How many theses can a faculty member supervise at once?")
    assert calls == [id(kb)] and pipeline._PLAN_F_CACHE.builds == 1
    assert first["registry_reused"] is False and second["registry_reused"] is True and third["registry_reused"] is True
    assert first["registry_build_ms"] >= 0.0 and all(d["plan_f_timings"].get("total_ms", 0.0) >= 0.0 for d in (first, second, third))
    assert set(third["plan_f_timings"]) >= {"coordinator_ms", "dispatch_ms", "specialist_ms", "guards_ms", "verify_ms", "total_ms"}


def test_shadow_adds_no_model_call_to_the_users_question(monkeypatch):
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    scripted = FakeGroqClient(GROUNDED_LEAVE)
    kb_llm = T.make_kb(groq_client=scripted)
    result, diagnostics = pipeline.ask_with_diagnostics(kb_llm, LEAVE_QUESTION)
    direct = legacy_direct(T.make_kb(groq_client=FakeGroqClient(GROUNDED_LEAVE)), LEAVE_QUESTION)
    assert visible(result) == visible(direct)
    assert scripted.calls == result.llm_calls == direct.llm_calls >= 1          # every call belongs to the legacy answer
    assert diagnostics["attempted"] is True and diagnostics["plan_f_llm_calls"] == 0 and diagnostics["plan_f_used_llm"] is False
    assert diagnostics["legacy_llm_calls"] == direct.llm_calls and diagnostics["legacy_used_llm"] is True


@pytest.mark.parametrize("label", [
    "registry unavailable", "coordinator selects nobody", "specialist rejects", "partial result", "quote guard failure",
    "subject guard failure", "verifier rejection", "specialist exception", "adapter exception", "seam exception", "import failure",
])
def test_shadow_survives_every_plan_f_failure(kb, monkeypatch, label):
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    question = HR_PHONE
    broken_kb = None
    if label == "registry unavailable":
        broken_kb = NS(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata,
                                    pages=kb.pages, groq_client=None, source=kb.source, stats={"source": {"section_records": 0}})
    elif label == "coordinator selects nobody":
        question = "asdfgh qwerty zxcvb"
    elif label == "specialist rejects":
        question = HR_KNOWN_WRONG
    elif label == "partial result":
        question = ANNUAL_LEAVE
    elif label == "quote guard failure":
        cid = T.chunk_index(kb, "Employment offers are drafted by HR", "paragraph")
        install_runtime(monkeypatch, stub_runtime(kb, faculty_services=lambda task: T.retrieval_result(
            kb, task, cid, "Employment offers are drafted by the Dean upon receiving the selected candidate")))
        question = "Who drafts employment offers?"
    elif label == "subject guard failure":
        cid = T.chunk_index(kb, "may terminate faculty contracts")
        install_runtime(monkeypatch, stub_runtime(kb, faculty_services=lambda task: T.retrieval_result(
            kb, task, cid, kb.chunks[cid], requested=("the asked specifics stated with the answer (hr)",),
            covered=("the asked specifics stated with the answer (hr)",))))
        question = HR_KNOWN_WRONG
    elif label == "verifier rejection":
        from handbook_bot.agents.types import VerifyResult
        monkeypatch.setattr(PF, "verify", lambda *a, **k: VerifyResult(False, [], ["stub"], [], None))
    elif label == "specialist exception":
        install_runtime(monkeypatch, stub_runtime(kb, institutional=T._hr_phone_task_stub(kb, T.FindingStatus.SUPPORTED, raise_error=True)))
    elif label == "adapter exception":
        def boom(*a, **k):
            raise RuntimeError(r"failed reading C:\Users\someone\private\cues.json with key gsk_ABCDEFGHIJKLMNOPQRST")
        monkeypatch.setattr(PF, "coordinate", boom)
    elif label == "seam exception":
        def boom(self, kb_):
            raise RuntimeError(r"cache exploded at C:\Users\someone\private\cache.bin gsk_ABCDEFGHIJKLMNOPQRST")
        monkeypatch.setattr(PF.RuntimeCache, "get_or_build", boom)
    elif label == "import failure":
        def no_import():
            raise ImportError("plan_f_runtime could not be imported gsk_ABCDEFGHIJKLMNOPQRST")
        monkeypatch.setattr(pipeline, "_plan_f_cache", no_import)
    target = broken_kb or kb
    result, diagnostics = pipeline.ask_with_diagnostics(target, question)
    direct = legacy_direct(target, question)
    assert visible(result) == visible(direct) and trace_of(result) == trace_of(direct), label
    assert diagnostics["served"] == "legacy" and diagnostics["accepted"] is False and diagnostics["fallback_reason"], label
    text = json.dumps(diagnostics)
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in text and "someone" not in text, label
    if label == "registry unavailable":
        assert diagnostics["attempted"] is False and diagnostics["registry_ok"] is False and diagnostics["registry_errors"]
    if label in ("seam exception", "import failure"):
        assert diagnostics["error"] and diagnostics["attempted"] is False
    if label == "adapter exception":
        assert diagnostics["outcome"] == "PLAN_F_ERROR_REDACTED"
    if label == "quote guard failure":
        assert diagnostics["outcome"] == "PLAN_F_REJECTED_BY_QUOTE_GUARD"
    if label == "subject guard failure":
        assert diagnostics["outcome"] == "PLAN_F_REJECTED_BY_SUBJECT_GUARD"
    if label == "verifier rejection":
        assert diagnostics["outcome"] == "PLAN_F_REJECTED_BY_VERIFIER"


def test_shadow_registry_failure_is_cached_per_knowledge_base_and_isolated(kb, monkeypatch):
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    calls = build_spy(monkeypatch)
    broken = NS(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata,
                             pages=kb.pages, groq_client=None, source=None, stats=kb.stats)
    _, first = pipeline.ask_with_diagnostics(broken, HR_PHONE)
    _, second = pipeline.ask_with_diagnostics(broken, HR_PHONE)
    assert first["registry_ok"] is False and second["registry_ok"] is False and second["registry_reused"] is True
    assert calls == [id(broken)]                                              # the failure is remembered, not retried per question
    _, healthy = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert healthy["registry_ok"] is True and healthy["outcome"] == "PLAN_F_ACCEPTED" and calls == [id(broken), id(kb)]


# ---------------------------------------------------------------------------
# ACTIVE mode (sections 16 to 19)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,specialist,pages,starts_with", [
    (HR_PHONE, "institutional", [54], "Human Resources +971 6 5050023 +971 6 5585200"),
    ("Does HR arrange travel and hotel accommodations for new faculty members?", "faculty_services", [61],
     "• HR arranges travel and hotel accommodations for new faculty members."),
    ("How many credit hours is a full teaching load?", "teaching", [96], 'o "A" Regular Faculty'),
    ("How many theses can a faculty member supervise at once?", "research", [81],
     "• A faculty member is allowed to supervise a maximum of four theses/dissertations simultaneously"),
])
def test_active_returns_an_accepted_plan_f_answer_from_each_specialist(kb, monkeypatch, question, specialist, pages, starts_with):
    set_mode(monkeypatch, pipeline.MODE_ACTIVE)
    result, diagnostics = pipeline.ask_with_diagnostics(kb, question)
    assert isinstance(result, QAResult) and result.answer.startswith(starts_with) and result.pages == pages
    assert result.used_llm is False and result.llm_calls == 0 and result.retried is False and result.agent_trace[0].agent == "plan_f_adapter"
    assert diagnostics["served"] == "plan_f" and diagnostics["accepted"] is True and diagnostics["specialist_id"] == specialist
    assert diagnostics["plan_f_llm_calls"] == 0 and diagnostics["legacy_llm_calls"] is None             # legacy was not run
    assert pipeline.ask(kb, question).answer == result.answer


@pytest.mark.parametrize("label", [
    "coordinator no-match", "general ask", "hr known wrong-support", "annual-leave partial", "multipart", "requires synthesis",
    "handoff", "registry failure", "specialist exception", "quote failure", "verifier rejection",
])
def test_active_falls_back_to_legacy_for_every_rejected_outcome(kb, monkeypatch, label):
    set_mode(monkeypatch, pipeline.MODE_ACTIVE)
    question = {
        "coordinator no-match": "asdfgh qwerty zxcvb",
        "general ask": "What does the Technology Transfer Office do?",
        "hr known wrong-support": HR_KNOWN_WRONG,
        "annual-leave partial": ANNUAL_LEAVE,
        "multipart": "Who handles faculty visas and where is that office?",
        "requires synthesis": "Where is the Research Institute and what research funding is available?",
    }.get(label, HR_PHONE)
    target = kb
    if label == "handoff":
        install_runtime(monkeypatch, stub_runtime(kb, institutional=T._hr_phone_task_stub(kb, T.FindingStatus.SUPPORTED, handoff=True)))
    elif label == "registry failure":
        target = NS(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata,
                                 pages=kb.pages, groq_client=None, source=None, stats=kb.stats)
    elif label == "specialist exception":
        install_runtime(monkeypatch, stub_runtime(kb, institutional=T._hr_phone_task_stub(kb, T.FindingStatus.SUPPORTED, raise_error=True)))
    elif label == "quote failure":
        cid = T.chunk_index(kb, "Employment offers are drafted by HR", "paragraph")
        install_runtime(monkeypatch, stub_runtime(kb, faculty_services=lambda task: T.retrieval_result(
            kb, task, cid, "Employment offers are drafted by the Dean upon receiving the selected candidate")))
        question = "Who drafts employment offers?"
    elif label == "verifier rejection":
        from handbook_bot.agents.types import VerifyResult
        monkeypatch.setattr(PF, "verify", lambda *a, **k: VerifyResult(False, [], ["stub"], [], None))
    result, diagnostics = pipeline.ask_with_diagnostics(target, question)
    direct = legacy_direct(target, question)
    assert visible(result) == visible(direct) and trace_of(result) == trace_of(direct), label
    assert diagnostics["mode"] == "active" and diagnostics["served"] == "legacy" and diagnostics["accepted"] is False, label
    assert diagnostics["fallback_reason"], label
    assert result.answer != "Human Resources +971 6 5050023 +971 6 5585200", label      # never the Plan F rendering


def test_active_never_serves_a_plan_f_result_without_an_accepted_outcome(kb, monkeypatch):
    set_mode(monkeypatch, pipeline.MODE_ACTIVE)
    unsafe = QAResult(answer="UNSAFE PLAN F ANSWER", pages=[1])

    def tampered(self, question):
        return PF.AdapterResult(outcome=PF.PlanFOutcome.UNSAFE_RESULT, reason="stub", question=question, result=unsafe)

    monkeypatch.setattr(PF.PlanFRuntime, "attempt", tampered)
    result, diagnostics = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert result.answer != unsafe.answer and diagnostics["served"] == "legacy" and diagnostics["accepted"] is False

    def accepted_without_result(self, question):
        return PF.AdapterResult(outcome=PF.PlanFOutcome.ACCEPTED, reason="stub", question=question, result=None)

    monkeypatch.setattr(PF.PlanFRuntime, "attempt", accepted_without_result)
    result, diagnostics = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert isinstance(result, QAResult) and diagnostics["served"] == "legacy"


def test_active_and_shadow_share_one_cache(kb, monkeypatch):
    calls = build_spy(monkeypatch)
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    pipeline.ask(kb, HR_PHONE)
    set_mode(monkeypatch, pipeline.MODE_ACTIVE)
    result, diagnostics = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert calls == [id(kb)] and diagnostics["registry_reused"] is True and diagnostics["served"] == "plan_f"
    set_mode(monkeypatch, pipeline.MODE_LEGACY)
    pipeline.ask(kb, HR_PHONE)
    assert calls == [id(kb)]


def test_a_new_knowledge_base_gets_its_own_runtime(kb, other_kb, monkeypatch):
    set_mode(monkeypatch, pipeline.MODE_ACTIVE)
    calls = build_spy(monkeypatch)
    first, d1 = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    second, d2 = pipeline.ask_with_diagnostics(other_kb, HR_PHONE)
    assert calls == [id(kb), id(other_kb)] and d1["registry_reused"] is False and d2["registry_reused"] is False
    assert first.answer == second.answer == "Human Resources +971 6 5050023 +971 6 5585200"
    cache = pipeline._PLAN_F_CACHE
    assert cache.get_or_build(kb).runtime is not cache.get_or_build(other_kb).runtime
    assert cache.get_or_build(kb).runtime.registry is not cache.get_or_build(other_kb).runtime.registry
    assert cache.get_or_build(kb).runtime.chunks is kb.chunks and cache.get_or_build(other_kb).runtime.chunks is other_kb.chunks


# ---------------------------------------------------------------------------
# RuntimeCache unit tests (sections 10, 11)
# ---------------------------------------------------------------------------
def fake_builder(fail_for=()):
    log = []

    def build(kb_):
        log.append(id(kb_))
        if id(kb_) in fail_for or getattr(kb_, "broken", False):
            return None, PF.RegistryBuild(None, {"teaching": "ValueError: stub failure"}, 1.0)
        return NS(kb_id=id(kb_), registry=object()), PF.RegistryBuild(object(), {}, 1.0)

    return build, log


def test_cache_builds_once_per_object_and_separates_objects():
    build, log = fake_builder()
    cache = PF.RuntimeCache(builder=build)
    a, b = NS(name="a"), NS(name="b")
    first = cache.get_or_build(a)
    again = cache.get_or_build(a)
    other = cache.get_or_build(b)
    assert first.reused is False and again.reused is True and other.reused is False
    assert again.runtime is first.runtime and other.runtime is not first.runtime
    assert log == [id(a), id(b)] and cache.builds == 2 and len(cache) == 2


def test_cache_remembers_a_failed_build_without_poisoning_other_objects():
    build, log = fake_builder()
    cache = PF.RuntimeCache(builder=build)
    broken, healthy = NS(broken=True), NS(broken=False)
    first = cache.get_or_build(broken)
    again = cache.get_or_build(broken)
    assert first.runtime is None and first.ok is False and first.build.errors and again.reused is True and again.runtime is None
    good = cache.get_or_build(healthy)
    assert good.ok and good.runtime is not None and log == [id(broken), id(healthy)]


def test_cache_tolerates_a_raising_builder_and_redacts_it():
    def raising(kb_):
        raise RuntimeError(r"boom C:\Users\someone\private\x.faiss gsk_ABCDEFGHIJKLMNOPQRST")

    cache = PF.RuntimeCache(builder=raising)
    entry = cache.get_or_build(NS())
    assert entry.runtime is None and "runtime" in entry.build.errors
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in entry.build.errors["runtime"] and "someone" not in entry.build.errors["runtime"]


def test_cache_releases_collected_objects_and_is_bounded():
    build, log = fake_builder()
    cache = PF.RuntimeCache(max_entries=2, builder=build)
    keep = NS(name="keep")
    cache.get_or_build(keep)
    gone = NS(name="gone")
    cache.get_or_build(gone)
    assert len(cache) == 2
    del gone
    gc.collect()
    assert len(cache) == 1 and cache.get_or_build(keep).reused is True
    third, fourth = NS(name="3"), NS(name="4")
    cache.get_or_build(third)
    cache.get_or_build(fourth)
    assert len(cache) == 2 and cache.get_or_build(keep).reused is False          # the oldest entry was evicted, rebuilt on demand
    cache.clear()
    assert len(cache) == 0


def test_cache_is_safe_under_concurrent_first_use():
    def slow(kb_):
        time.sleep(0.05)
        return NS(), PF.RegistryBuild(object(), {}, 50.0)

    cache = PF.RuntimeCache(builder=slow)
    kb_ = NS()
    seen = []

    def worker():
        seen.append(cache.get_or_build(kb_).runtime)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.builds == 1 and len(seen) == 8 and all(r is seen[0] for r in seen)


def test_cache_builds_but_does_not_cache_an_object_without_weak_references():
    class Slotted:
        __slots__ = ("chunks", "metadata")

    build, log = fake_builder()
    cache = PF.RuntimeCache(builder=build)
    obj = Slotted()
    first, second = cache.get_or_build(obj), cache.get_or_build(obj)
    assert first.reused is False and second.reused is False and len(cache) == 0 and len(log) == 2


# ---------------------------------------------------------------------------
# Diagnostics buffer and seam isolation (sections 14, 28)
# ---------------------------------------------------------------------------
def test_recent_diagnostics_are_bounded_and_resettable(kb, monkeypatch):
    set_mode(monkeypatch, pipeline.MODE_SHADOW)
    for _ in range(3):
        pipeline.ask(kb, "Hello")
    recent = pipeline.recent_plan_f_diagnostics(limit=2)
    assert len(recent) == 2 and all(d["mode"] == "shadow" for d in recent) and pipeline._RECENT.maxlen == 200
    assert pipeline.recent_plan_f_diagnostics(limit=0) == []
    pipeline.reset_plan_f_state()
    assert pipeline.recent_plan_f_diagnostics() == [] and pipeline._PLAN_F_CACHE is None


def test_legacy_core_modules_stay_free_of_plan_f():
    core = ["handbook_bot/orchestrator.py", "handbook_bot/qa.py", "handbook_bot/retrieval.py", "handbook_bot/agents/router.py",
            "handbook_bot/agents/synthesis.py", "handbook_bot/agents/verifier.py", "handbook_bot/agents/types.py"]
    for name in core:
        text = (REPO / name).read_text(encoding="utf-8")
        assert "PLAN_F" not in text and "specialists" not in text and "plan_f_runtime" not in text, name
    for name in ("ui/chat.py", "app.py"):
        assert "PLAN_F" not in (REPO / name).read_text(encoding="utf-8"), name
