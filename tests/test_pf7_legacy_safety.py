"""Post-freeze PF-7 (decision record D-117): the legacy answer path that every mode falls back to.

* Registered source conflicts are enforced on legacy evidence and on the legacy answer (``conflict_guard``), generically:
  the topics, documents and values below are synthetic, and the real registry is only read, never named.
* An accepted model answer cites only the items its claims rest on, so a page number shared by several documents
  cannot pull in a document that supports nothing; a refusal shows no source chip; inline citation marks are not
  answer text.
* A reasoning model's hidden tokens get an optional allowance, and a reply cut off before any answer is a generation
  failure, not the refusal sentence.
"""
from __future__ import annotations

import json
import subprocess
import sys

import faiss
import pytest

from handbook_bot import conflict_guard as G
from handbook_bot import orchestrator as orch
from handbook_bot import qa
from handbook_bot.qa import REFUSAL, split_answer_and_claimed_pages
from handbook_bot.sources import SourceConflict, load_source_registry
from ui import presentation

from conftest import FakeEmbedder, FakeGroqClient, FakeReranker

# ---------------------------------------------------------------------------
# A synthetic registry: one value conflict, one recommendation conflict
# ---------------------------------------------------------------------------
BOOKING = SourceConflict(topic="room_booking_quota", phrases=("maximum 12", "bookings per month"))
ADVICE = SourceConflict(topic="lab_usage_advice_2021", phrases=("avoid the old lab", "lab"))
CONFLICTS = {"doc_rooms": {BOOKING.topic: BOOKING}, "doc_labs": {ADVICE.topic: ADVICE}}

ROOMS_CHUNK = ("How to book a room 1) Open the booking portal 2) Choose a free room and a time slot 3) Press Confirm. "
               "Each member may hold a specific number of rooms (maximum 12 bookings per month).")
ROWS = [
    # (chunk, page, source_id, title, section, conflict topic or None)
    (ROOMS_CHUNK, 1, "doc_rooms", "Room Booking Guide", "Booking", BOOKING.topic),
    ("Use the old lab for chemistry practicals; avoid the old lab after hours.", 1, "doc_labs", "Lab Guide", "Labs", ADVICE.topic),
    ("Faculty members are entitled to thirty days of annual leave each academic year subject to approval by the dean.",
     3, "doc_book", "Staff Book", "Leave", None),
    ("Mark attendance in the Details and Actions panel and choose Add Attendance to record present late or absent.",
     1, "doc_attend", "Attendance Guide", "Access attendance", None),
    ("Session settings let the moderator allow chat, share audio and record the online session.",
     1, "doc_sessions", "Sessions Guide", "Session settings", None),
]


def build_kb(rows=ROWS):
    chunks = [r[0] for r in rows]
    metadata = [{"page": r[1], "source_id": r[2], "source_title": r[3], "section": r[4], "chunk_type": "paragraph",
                 "source_conflict": r[5]} for r in rows]
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return dict(embedder=embedder, reranker=FakeReranker(), index=index, chunks=chunks, metadata=metadata,
                pages=[{"page": i + 1, "text": ""} for i in range(10)])


@pytest.fixture
def synthetic_registry(monkeypatch):
    monkeypatch.setattr(G, "registered_conflicts", lambda path=None: CONFLICTS)


def run(question, client, **kb):
    return orch.run(question, groq_client=client, **(kb or build_kb()))


def decisions(result, agent):
    return [entry.decision for entry in result.agent_trace if entry.agent == agent]


# ---------------------------------------------------------------------------
# Claim detection and evidence screening
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unit,disputed", [
    ("Each member may hold a specific number of rooms (maximum 12 bookings per month).", True),   # a phrase
    ("You can make up to 12 bookings.", True),                    # a number and a phrase word
    ("The monthly limit for rooms is fixed.", True),              # a limit word
    ("Bookings are counted per member.", True),                   # a count word
    ("2) Choose a free room and a time slot", False),             # a list label is not a number
    ("Open the booking portal and press Confirm.", False),
])
def test_disputed_claim_detection(unit, disputed):
    assert G.states_disputed_claim(BOOKING, unit) is disputed


def test_the_legacy_rule_is_at_least_as_strict_as_the_plan_f_rule():
    from handbook_bot.guide_grounding import disputed_claim
    units = G.split_units(ROOMS_CHUNK) + ["Rooms: maximum 12 per member", "A booking counts once.", "Up to 3 rooms per day.",
                                           "Press Confirm to finish.", "12 rooms are available."]
    for unit in units:
        if disputed_claim(BOOKING, unit):
            assert G.states_disputed_claim(BOOKING, unit), unit


def test_value_conflict_keeps_only_the_undisputed_units():
    items = [{"chunk": ROOMS_CHUNK, "meta": {"page": 1, "source_id": "doc_rooms", "source_conflict": BOOKING.topic}}]
    kept, report = G.screen_items(items, CONFLICTS)
    assert len(kept) == 1 and report[0]["action"] == "units_removed"
    text = kept[0]["chunk"]
    assert "Open the booking portal" in text and "Press Confirm" in text
    assert "12" not in text and not BOOKING.matches(text)
    assert kept[0]["meta"]["conflict_screened"] == BOOKING.topic
    assert items[0]["chunk"] == ROOMS_CHUNK                       # the retrieved item itself is not modified


@pytest.mark.parametrize("sid,topic", [("doc_labs", ADVICE.topic),          # a recommendation conflict
                                       ("doc_rooms", "unregistered_topic"),  # a stamp the registry does not know
                                       ("doc_other", BOOKING.topic)])        # a topic of another source
def test_any_other_stamp_removes_the_whole_chunk(sid, topic):
    items = [{"chunk": "Avoid the old lab. Open the portal.", "meta": {"page": 1, "source_id": sid, "source_conflict": topic}},
             {"chunk": "Unstamped text.", "meta": {"page": 2, "source_id": sid, "source_conflict": None}}]
    kept, report = G.screen_items(items, CONFLICTS)
    assert [k["chunk"] for k in kept] == ["Unstamped text."]
    assert report == [{"topic": topic, "source_id": sid, "page": 1, "action": "chunk_removed"}]


def test_a_chunk_that_is_nothing_but_the_disputed_claim_is_removed():
    items = [{"chunk": "Bookings per month)", "meta": {"page": 1, "source_id": "doc_rooms", "source_conflict": BOOKING.topic}}]
    assert G.screen_items(items, CONFLICTS)[0] == []


def test_answer_check_only_applies_to_value_conflicts_retrieved_for_the_question():
    stamped = [("doc_rooms", BOOKING.topic)]
    assert G.answer_problem("You may make a maximum of 12 bookings per month.", stamped, CONFLICTS)
    assert G.answer_problem("The cap is twelve bookings monthly; the limit applies to everyone.", stamped, CONFLICTS)
    assert G.answer_problem("Open the booking portal and press Confirm.", stamped, CONFLICTS) is None
    assert G.answer_problem("You may make a maximum of 12 bookings per month.", [], CONFLICTS) is None
    assert G.answer_problem("Avoid the old lab.", [("doc_labs", ADVICE.topic)], CONFLICTS) is None


@pytest.mark.parametrize("unit,refused", [
    ("Members may hold up to 12 bookings.", True),                                   # a number that counts the subject
    ("Each member gets twelve room bookings.", True),                                # the number in words
    ("Bookings are limited for every member.", True),                                # a limit word with the subject
    ("A cancelled booking is not counted.", True),                                   # a count word with the subject
    ("New bookings can be made every month.", True),                                 # a rate with the subject
    ("Enter the booking reference, then upload a file under 100 MB.", False),        # a number that measures something else
    ("Faculty members are entitled to thirty days of annual leave each year.", False),  # a count about another subject
    ("The portal is open 24 hours a day.", False),
])
def test_an_answer_is_refused_only_for_a_claim_about_the_conflicts_subject(unit, refused):
    assert G.answer_states_disputed_claim(BOOKING, unit) is refused


def test_every_registered_conflict_is_classified_and_its_phrases_are_disputed():
    registry = load_source_registry()
    found = [c for r in registry.list_sources() for c in (r.conflicts or ())]
    assert found, "the registry records conflicts"
    for conflict in found:
        if G.is_value_conflict(conflict):
            for phrase in conflict.phrases:
                assert G.states_disputed_claim(conflict, phrase)
    by_source = G.registered_conflicts()
    for record in registry.list_sources():
        assert set(by_source[record.source_id]) == {c.topic for c in (record.conflicts or ())}


# ---------------------------------------------------------------------------
# Conflicts on the full legacy path
# ---------------------------------------------------------------------------
def test_a_disputed_value_is_never_served_even_when_the_model_states_it(synthetic_registry):
    client = FakeGroqClient("Each member may make a maximum of 12 bookings per month.\nPages: 1")
    result = run("How many room bookings can a member make per month?", client)
    assert result.answer == REFUSAL and result.pages == []
    assert all("maximum 12" not in p.lower() for p in client.prompts)     # the disputed unit never reached the model


def test_the_answer_check_holds_even_when_a_verifier_would_accept(synthetic_registry):
    from handbook_bot.agents.types import VerifyResult

    def accept_all(question, synthesis, evidence, sub_questions, *, is_retry):
        return VerifyResult(True, [1], [], [], None)

    client = FakeGroqClient("Each member may make a maximum of 12 bookings per month.\nPages: 1")
    result = orch.run("How many room bookings can a member make per month?", groq_client=client, verifier=accept_all, **build_kb())
    assert result.answer == REFUSAL and result.pages == []
    assert "answer_refused" in decisions(result, "conflict_guard")


def test_the_screened_evidence_still_answers_an_undisputed_procedure(synthetic_registry):
    client = FakeGroqClient("Open the booking portal, choose a free room and a time slot, then press Confirm.\nPages: 1")
    result = run("How do I book a room in the booking portal?", client)
    assert result.answer.startswith("Open the booking portal")
    assert result.pages == [1] and "screened" in decisions(result, "conflict_guard")
    assert "answer_refused" not in decisions(result, "conflict_guard")


def test_without_a_model_the_stamped_recommendation_is_not_even_cited(synthetic_registry):
    result = run("Should I use the old lab after hours?", None)
    assert result.answer == orch.NO_KEY_REPLY
    assert all(i["meta"]["source_id"] != "doc_labs" for i in result.items)


def test_an_unrelated_answer_is_served_although_a_stamped_chunk_was_retrieved(synthetic_registry):
    client = FakeGroqClient("Faculty members are entitled to thirty days of annual leave each academic year.\nPages: 3")
    result = run("How much annual leave are faculty members entitled to?", client)
    assert result.answer.startswith("Faculty members are entitled") and result.pages == [3]
    assert "answer_refused" not in decisions(result, "conflict_guard")


def test_evidence_without_any_stamp_is_untouched(synthetic_registry):
    rows = [r for r in ROWS if r[5] is None]
    client = FakeGroqClient("Faculty members are entitled to thirty days of annual leave each academic year.\nPages: 3")
    result = run("How much annual leave are faculty members entitled to?", client, **build_kb(rows))
    assert result.answer.startswith("Faculty members are entitled") and result.pages == [3]
    assert decisions(result, "conflict_guard") == []


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def test_a_shared_page_number_cites_only_the_document_that_supports_the_answer(synthetic_registry):
    client = FakeGroqClient("Mark attendance in the Details and Actions panel and choose Add Attendance to record "
                            "present, late or absent.\nPages: 1")
    result = run("How do I mark attendance in the details and actions panel?", client)
    assert result.pages == [1]
    assert {i["meta"]["source_id"] for i in result.items} == {"doc_attend"}
    chips = presentation.source_chips(result)
    assert ("doc", "Attendance Guide") in chips and not any("Sessions" in label for _c, label in chips)


def test_two_documents_that_each_support_a_claim_are_both_cited(synthetic_registry):
    client = FakeGroqClient("Mark attendance in the Details and Actions panel and choose Add Attendance. "
                            "Session settings let the moderator allow chat and record the online session.\nPages: 1")
    result = run("How do I mark attendance and record the online session settings?", client)
    assert {i["meta"]["source_id"] for i in result.items} == {"doc_attend", "doc_sessions"}


def test_a_refusal_shows_no_source_chip(synthetic_registry):
    client = FakeGroqClient(REFUSAL)
    result = run("How do I mark attendance in the details and actions panel?", client)
    assert result.answer == REFUSAL and result.pages == []
    assert presentation.source_chips(result) == []


def test_inline_citation_marks_are_not_answer_text():
    items = [{"meta": {"page": p}} for p in (2, 9, 109)]
    body, pages = split_answer_and_claimed_pages(
        "Faculty teach courses 【Source 3 | Page 109】. They do research [Source 1, Page 2].\nPages: 9", items)
    assert body == "Faculty teach courses. They do research." and pages == [9, 109, 2]
    assert split_answer_and_claimed_pages("A [Source 2, Page 999] claim.", items) == ("A claim.", [])   # never a page outside the context
    assert split_answer_and_claimed_pages("Brackets [like this] stay.\nPages: 2", items) == ("Brackets [like this] stay.", [2])


@pytest.mark.parametrize("text,numbers", [
    ("Files must be less than 100MB.", ["100"]),        # a data-size unit against the number does not hide it
    ("under 100 MB", ["100"]),
    ("no more than 2GB or 512kb", ["2", "512"]),
    ("1st semester", []), ("B2 level", []), ("COVID19", []), ("24h", []),   # any other letter still does
])
def test_numbers_written_with_a_data_size_unit_are_the_same_number(text, numbers):
    from handbook_bot.agents import verifier as V
    assert V._numbers(text, []) == numbers


def test_the_verifier_still_rejects_a_different_size():
    from handbook_bot.agents import verifier as V
    item = V._Item(0, {"chunk": "Choose the file and click Upload. Files must be less than 100MB.", "meta": {"page": 1},
                       "rerank_score": 1.0})
    ok, _ = V._support(V._Claim("Choose the file (it must be under 100 MB) and click Upload."), item, [item])
    wrong, _ = V._support(V._Claim("Choose the file (it must be under 300 MB) and click Upload."), item, [item])
    assert ok and not wrong


# ---------------------------------------------------------------------------
# Reasoning models
# ---------------------------------------------------------------------------
class _Choice:
    def __init__(self, content, finish_reason):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish_reason


class RecordingClient:
    """A client that records the request and answers with a given finish reason."""

    def __init__(self, content, finish_reason="stop"):
        self.kwargs = []
        self.reply = (content, finish_reason)
        self.chat = type("C", (), {"completions": self})()

    def with_options(self, **_):
        return self

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return type("R", (), {"choices": [_Choice(*self.reply)]})()


def test_the_reasoning_allowance_is_added_to_the_cap_and_is_zero_by_default(monkeypatch):
    client = RecordingClient("An answer.")
    qa.ask_groq(client, "prompt")
    assert client.kwargs[-1]["max_tokens"] == qa.LLM_MAX_TOKENS
    monkeypatch.setattr(qa, "LLM_REASONING_TOKENS", 2048)
    qa.ask_groq(client, "prompt")
    assert client.kwargs[-1]["max_tokens"] == qa.LLM_MAX_TOKENS + 2048


def test_a_reply_cut_off_before_any_answer_is_a_generation_failure_not_a_refusal(synthetic_registry):
    result = run("How much annual leave are faculty members entitled to?", RecordingClient("", "length"))
    assert result.answer == orch.LLM_UNAVAILABLE_REPLY and result.answer != REFUSAL
    with pytest.raises(qa.LLMOutputTruncated):
        qa.ask_groq(RecordingClient("", "length"), "prompt")
    assert qa.ask_groq(RecordingClient("Partial text", "length"), "prompt") == "Partial text"   # text is still judged


def test_the_guard_keeps_legacy_free_of_plan_f_modules():
    code = ("import sys, json; import handbook_bot.conflict_guard, handbook_bot.orchestrator; "
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('handbook_bot.') and "
            "any(k in m for k in ('plan_f', 'guide_grounding', 'guide_structure', 'guide_inference', 'source_scopes')))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).parents[1]))
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
