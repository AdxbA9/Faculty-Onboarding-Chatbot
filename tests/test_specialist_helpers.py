"""Neutral specialist helpers (Milestone 4, Step 4.4): scoping, evidence units, detail primitives and redaction, tested
on their own so that every specialist can rely on them without importing another specialist."""
from __future__ import annotations

import re

import pytest

from handbook_bot.agents.specialists import base, details, evidence, scoping
from handbook_bot.agents.specialists.details import Detail, authority_check, requested_details, responsibility_check
from handbook_bot.agents.specialists.evidence import detail_covered, is_table_row, quote_problem, select_quote, unit_coverage
from handbook_bot.agents.specialists.scoping import SectionGuard, compile_scope, heading_marker
from handbook_bot.sources import load_section_map

SECTION_MAP = load_section_map("knowledge/handbook_sections.json")
SOURCE = "uos_faculty_handbook_2025_26"


# ---------------------------------------------------------------------------
# scoping
# ---------------------------------------------------------------------------
def test_compile_scope_reads_owned_sections_as_page_ranges():
    scope = compile_scope(SECTION_MAP, SOURCE, [(3, "3.5", "Responsibility to Research"), (8, "8.1", "Research Institutes")])
    assert set(scope.source_ids) == {SOURCE} and list(scope.page_ranges) == [(80, 82), (153, 154)]


def test_compile_scope_merges_adjacent_ranges():
    scope = compile_scope(SECTION_MAP, SOURCE, [(8, "8.1", "a"), (8, "8.2", "b")])
    assert list(scope.page_ranges) == [(153, 156)]


@pytest.mark.parametrize("owned", [[(8, "8.99", "missing")], [(9, "9.6", "missing")]])
def test_compile_scope_fails_loudly_for_an_absent_section(owned):
    with pytest.raises(ValueError):
        compile_scope(SECTION_MAP, SOURCE, owned)


def test_compile_scope_rejects_a_foreign_map():
    with pytest.raises(ValueError):
        compile_scope(SECTION_MAP, "other_source", [(8, "8.1", "a")])


def test_extra_pages_admit_a_chapter_preamble():
    scope = compile_scope(SECTION_MAP, SOURCE, [(9, "9.1", "IP ownership")], extra_pages=[(175, 175)])
    assert list(scope.page_ranges) == [(175, 177)]                             # 175 merges with 9.1's 176-177


@pytest.mark.parametrize("pages", [[(0, 1)], [(272, 273)], [(120, 119)], [("175", 175)]])
def test_extra_pages_must_lie_inside_the_document(pages):
    with pytest.raises(ValueError):
        compile_scope(SECTION_MAP, SOURCE, [(9, "9.1", "a")], extra_pages=pages)


def test_extra_pages_never_widen_into_a_section_the_specialist_does_not_own():
    with pytest.raises(ValueError, match="not owned"):
        compile_scope(SECTION_MAP, SOURCE, [(9, "9.1", "a")], extra_pages=[(174, 174)])       # 8.11 lies on 174
    with pytest.raises(ValueError, match="not owned"):
        compile_scope(SECTION_MAP, SOURCE, [(3, "3.5", "a")], extra_pages=[(80, 80)])         # 3.4 shares page 80
    scope = compile_scope(SECTION_MAP, SOURCE, [(8, "8.11", "a")], extra_pages=[(174, 174)])  # owned: allowed
    assert list(scope.page_ranges) == [(163, 174)]


def test_heading_marker_uses_number_and_first_two_words():
    record = next(r for r in SECTION_MAP.sections() if r.chapter == 3 and r.section_no == "3.5")
    marker = heading_marker(record)
    assert marker.match("3.5 Responsibility to Research") and not marker.match("13.5 Responsibility to Research")


def test_guard_refuses_unordered_metadata():
    with pytest.raises(ValueError):
        SectionGuard(["a", "b"], [{"chunk_id": 1, "page": 80}, {"chunk_id": 0, "page": 80}], SECTION_MAP, [(3, "3.5", "a")],
                     compile_scope(SECTION_MAP, SOURCE, [(3, "3.5", "a")]))


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def test_is_table_row_generic_rule_and_shapes():
    assert is_table_row("Laboratory 20 10 Lecture 70-120 30")
    assert not is_table_row("Mon 25 Aug 02 Rabi I Classes begin")                      # a domain shape, not the generic rule
    weekday = lambda t: bool(re.match(r"^(?:mon|tue|wed|thu|fri|sat|sun)\b", t, re.I))
    assert is_table_row("Mon 25 Aug 02 Rabi I Classes begin", shapes=(weekday,))
    assert not is_table_row("x", shapes=(weekday,))


@pytest.mark.parametrize("quote,kind,problem", [
    ("8.9 Scientific Publishing at the University of Sharjah Journal", "prose", "heading"),
    ("the following conditions:", "prose", "lead-in"),
    ("subject to the following guidelines", "prose", "dangling"),
    ("b. These syllabi must be uploaded on Blackboard.", "prose", None),
    ("Too short.", "prose", "too short"),
    ("Special Assignment 1-9", "line", "line fragment"),
    ("Laboratory 20 10 Lecture 70-120 30", "line", None),
])
def test_quote_problem_default(quote, kind, problem):
    assert quote_problem(quote, kind) == problem


def test_quote_problem_uses_the_supplied_table_rule():
    assert quote_problem("Mon 25 Aug 02 Rabi I Classes begin", "line") == "line fragment"
    assert quote_problem("Mon 25 Aug 02 Rabi I Classes begin", "line", table_row=lambda t: True) is None


def relevance_for(words):
    """A stand-in relevance: anchors are words in ``words``, vocabulary too."""
    def rel(text):
        hits = {w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text, re.I)}
        return len(hits), len(hits)
    return rel


def test_select_quote_takes_a_relevance_callable():
    text = "Instructors must prepare course syllabi: a. These syllabi must be uploaded on Blackboard. Unrelated sentence here."
    got = select_quote(text, "paragraph", relevance_for(["syllabi"]))
    assert got and got[0] == "Instructors must prepare course syllabi: a. These syllabi must be uploaded on Blackboard."
    assert select_quote(text, "paragraph", relevance_for(["nothing"])) is None


def test_lead_in_join_is_opt_in_and_never_joins_a_sibling():
    text = ("7.2 Research and Scholarship For promotion purposes, research considered admissible includes: "
            "A. Research published in refereed journals. B. Books published by academic publishers.")
    rel = relevance_for(["promotion", "research"])
    assert select_quote(text, "paragraph", rel)[0] == "A. Research published in refereed journals."          # default: unchanged
    joined = select_quote(text, "paragraph", rel, join_lead_in=True)[0]
    assert joined.startswith("7.2 Research and Scholarship For promotion purposes") and joined.endswith("refereed journals.")
    assert "B. Books" not in joined and joined in text
    bare = "12.12 Examinations Policy: A. Research published in refereed journals for promotion."
    assert select_quote(bare, "paragraph", rel, join_lead_in=True)[0].startswith("A. Research")               # a bare heading is not a lead-in


def test_unit_coverage_never_unions_across_units():
    d1 = Detail("amount", "amount", lambda u: "AED" in u)
    d2 = Detail("deadline", "deadline", lambda u: "January" in u)
    rel = relevance_for(["grant"])
    assert unit_coverage([d1, d2], rel, "The grant pays AED 5000. The grant closes in January.") == frozenset({"amount"})
    assert unit_coverage([d1, d2], rel, "The grant pays AED 5000 and closes in January.") == frozenset({"amount", "deadline"})
    assert detail_covered(d2, rel, "The grant pays AED 5000. The grant closes in January.")


def test_lettered_items_are_units():
    units = evidence._local_units("Rules: A. Breach of Exam Rules: If a student cheats, penalties apply. B. Attempted Cheating: The exam is void.")
    assert [u[:12] for u in units] == ["Rules:", "A. Breach of", "B. Attempted"]


# ---------------------------------------------------------------------------
# details
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unit,expected", [
    ("a maximum of four days", True), ("not exceed 40.", True), ("90 percent maximum", True), ("a maximum of 75,000 Dirhams", True),
    ("maximum reached in 2024", False), ("The maximum is on page 12", False), ("maximum described in section 3.4", False),
])
def test_max_ok_needs_a_value_shaped_number(unit, expected):
    assert details._max_ok(unit) is expected


def test_limit_ok_accepts_table_rows_only_through_the_supplied_rule():
    row = "Class Type Maximum Enrollment Lecture 70-120 30 Tutorial 30-40 15"
    assert details._max_ok(row, table_row=lambda t: True)
    assert details._limit_ok("Maximum values 12 18 24", details._LIMIT_MAX) is False


@pytest.mark.parametrize("unit,expected", [
    ("fee of AED 100", True), ("cost is 100 AED", True), ("registration fees up to $750", True), ("free of charge", True),
    ("fee applies to examination 101", False), ("payment of a fixed fee", False),
])
def test_fee_ok(unit, expected):
    assert details._fee_ok(unit) is expected


def test_authority_check_is_configurable_and_needs_a_named_body():
    default = authority_check()
    assert default("The Dean approves the request.") and not default("Approval is required for the request.")
    research = authority_check(nouns=re.compile(r"\b(?:vcrgs|irb)\b", re.I))
    assert research("subject to approval by the VCRGS") and not research("The Dean approves the request.")


def test_responsibility_check_needs_a_verb_form_and_a_party():
    check = responsibility_check(re.compile(r"\bIT department\b", re.I))
    assert check("The IT department administers the LMS.")
    assert not check("Learning Management System policy of the IT department")


class _C:
    def __init__(self, vocab):
        self.vocab = frozenset(vocab)


def _concepts(text):
    return _C({w for w in ("office_hours", "students", "theses") if w.replace("_", " ") in text.lower()})


def test_requested_details_builds_quantity_per_clause():
    families = (("quantity", "number or amount", re.compile(r"\bhow\s+many\b", re.I), lambda u: False),
                ("maximum", "maximum or limit", re.compile(r"\bmaximum\b", re.I), details._max_ok))
    got = requested_details("How many office hours must I hold?", families, (), _concepts)
    assert [d.family for d in got] == ["quantity"]
    assert got[0].check("five office hours per week") and not got[0].check("3 credit hours")
    assert got[0].check("48 hours of office hours") and not got[0].check("48 hours of training")


def test_requested_details_categories_use_the_supplied_label():
    got = requested_details("What is the seed grant ceiling?", (), (("seed", re.compile(r"\bseed\b", re.I)),), _concepts,
                            category_label="the requested research category (%s)")
    assert got[0].family == "category" and got[0].label == "the requested research category (seed)"


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("message,hidden", [
    (r"index failure at C:\Users\PC\secret\path", ("Users", "secret")),
    ("cannot open /home/user/private/file", ("home", "private")),
    ("cannot open /Users/name/private/file", ("private", "name")),
    ("failure gsk_ABCDEFGHIJKLMNOP while reading /var/lib/app/index.faiss", ("gsk_ABC", "/var", "index.faiss")),
])
def test_redact_error_hides_paths_and_keys(message, hidden):
    text = base.redact_error(RuntimeError(message))
    assert text.startswith("RuntimeError:") and ("<path>" in text or "gsk_***" in text)
    for h in hidden:
        assert h not in text


# Step 4.5A (SAF-4): a directory name with spaces, a UNC share and an escaped repr are redacted whole
@pytest.mark.parametrize("exc,hidden", [
    (RuntimeError(r"failed loading C:\Users\John Smith\AppData\Local\groq\key.txt with key gsk_abc123"), ("John", "Smith", "AppData", "key.txt", "gsk_abc")),
    (RuntimeError(r"failed loading C:\Users\PC\Desktop\Senior Project\Senior Project\Faculty-Onboarding-Chatbot\knowledge\index.faiss"),
     ("Senior", "Project", "Faculty-Onboarding", "index.faiss")),
    (RuntimeError(r"failed loading \\fileserver\share\secret\keys.txt"), ("fileserver", "share", "secret", "keys.txt")),
    (FileNotFoundError(2, "No such file or directory", r"\\fileserver\share\secret\keys.txt"), ("fileserver", "secret", "keys.txt", "\\")),
    (FileNotFoundError(2, "No such file or directory", r"C:\Users\John Smith\keys.txt"), ("John", "Smith", "keys.txt")),
    (RuntimeError("failed loading /home/john smith/keys.txt now"), ("john", "smith", "keys.txt")),
])
def test_redact_error_hides_paths_with_spaces_and_unc_shares(exc, hidden):
    text = base.redact_error(exc)
    assert "<path>" in text, text
    for h in hidden:
        assert h not in text, (h, text)


# Step 4.5A re-check: forward-slash UNC shares, a final name with spaces and spaced Unix directories
@pytest.mark.parametrize("exc,hidden", [
    (RuntimeError("failed loading //fileserver/share/secret/keys.txt with key gsk_abc"), ("fileserver", "secret", "keys.txt", "gsk_abc")),
    (PermissionError(13, "Permission denied", "//fileserver/share/secret/keys.txt"), ("fileserver", "secret", "keys.txt")),
    (FileNotFoundError(2, "No such file or directory", r"C:\Users\John Smith"), ("John", "Smith")),
    (RuntimeError(r"cannot open C:\Users\PC\Documents\my secret keys.txt"), ("Documents", "secret", "keys.txt")),
    (RuntimeError(r"cannot open C:\Users\John  Smith\key.txt"), ("John", "Smith", "key.txt")),
    (RuntimeError("cannot open /Volumes/My Drive/secrets/groq.txt"), ("Volumes", "Drive", "secrets", "groq.txt")),
    (RuntimeError("cannot open ~/Library/Application Support/groq/key.txt"), ("Library", "Support", "groq", "key.txt")),
])
def test_redact_error_hides_forward_slash_unc_and_spaced_final_names(exc, hidden):
    text = base.redact_error(exc)
    assert "<path>" in text, text
    for h in hidden:
        assert h not in text, (h, text)


@pytest.mark.parametrize("message", ["reranker returned 3 scores for 4 candidates", "ratio 3/4 and 1/2", "plain message with no path"])
def test_redact_error_keeps_messages_without_a_path(message):
    assert base.redact_error(RuntimeError(message)) == "RuntimeError: " + message


def test_redaction_is_shared_by_dispatch_and_teaching():
    from handbook_bot.agents.specialists import dispatch, teaching

    assert dispatch.redact_error is base.redact_error and teaching._safe_error is base.redact_error
    assert scoping.compile_scope is not teaching.compile_scope                    # Teaching keeps its own defaults
    assert teaching.SectionGuard is scoping.SectionGuard
