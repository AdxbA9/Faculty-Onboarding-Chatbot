"""PF-7 (D-117) evidence summaries from the live runs (the raw run files stay outside the repository).

    python -X utf8 qa/post_freeze_pf7/summarize.py <run dir> <out dir>

<run dir> holds the BEFORE runs (``repro1.json``: the reproduction; ``probe1.json`` and ``probe_long.json``: fresh
paraphrases with the response metadata) and the AFTER runs (``val_conflict.json``, ``val_nl.json`` from
``validate_live.py``; ``val_cite.json`` from ``citation_live.py``). Answers are kept to 400 characters. Every answer is
checked against the REGISTERED value conflicts with ``conflict_guard.answer_states_disputed_claim``.
Writes ``reproduction_before.json``, ``conflict_validation.json``, ``natural_language_validation.json`` and
``citation_validation.json``.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from handbook_bot import conflict_guard as G                        # noqa: E402

RUN, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
VALUE_CONFLICTS = [c for by_topic in G.registered_conflicts().values() for c in by_topic.values() if G.is_value_conflict(c)]
REFUSAL = "I do not have this information."


def load(name):
    return json.load(open(os.path.join(RUN, name), encoding="utf-8"))


def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)


def disputed(answer):
    return sorted({c.topic for c in VALUE_CONFLICTS for u in G.split_units(answer or "") if G.answer_states_disputed_claim(c, u)})


def row(r):
    llm = r.get("llm") or []
    return {"question": r["question"], "served": r.get("served"), "plan_f_outcome": r.get("outcome"),
            "answer": (r.get("answer") or "")[:400], "refused": (r.get("answer") or "").strip() == REFUSAL,
            "states_disputed_claim": disputed(r.get("answer")), "chips": r.get("chips"),
            "llm_calls": [{k: c.get(k) for k in ("finish", "finish_reason", "tokens", "completion_tokens", "reasoning", "reasoning_tokens",
                                                 "content_chars", "max_tokens") if k in c} for c in llm],
            "trace": [t[:2] for t in (r.get("trace") or [])]}


before = [row(r) for name in ("repro1.json", "probe1.json", "probe_long.json") for r in load(name)]
dump("reproduction_before.json", {
    "method": "PF-6 code (commit 962976e) in the acceptance configuration: ACTIVE, guide scopes on, openai/gpt-oss-120b, no reasoning allowance.",
    "summary": {"questions": len(before),
                "disputed_claim_served": [r["question"] for r in before if r["states_disputed_claim"] and not r["refused"]],
                "empty_drafts_cut_at_the_cap": [r["question"] for r in before
                                                if any(c.get("finish_reason") == "length" and not c.get("content_chars") for c in r["llm_calls"])]},
    "questions": before})

conflict = [row(r) for r in load("val_conflict.json")]
dump("conflict_validation.json", {
    "method": "D-117 code, acceptance configuration plus LLM_REASONING_TOKENS=2048 (validate_live.py, questions_conflict.txt).",
    "summary": {"questions": len(conflict),
                "disputed_claim_served": [r["question"] for r in conflict if r["states_disputed_claim"] and not r["refused"]],
                "answered": [r["question"] for r in conflict if not r["refused"]],
                "refused": [r["question"] for r in conflict if r["refused"]],
                "answer_check_fired": [r["question"] for r in conflict if ["conflict_guard", "answer_refused"] in r["trace"]]},
    "questions": conflict})

nl = [row(r) for r in load("val_nl.json")]
dump("natural_language_validation.json", {
    "method": "D-117 code, acceptance configuration plus LLM_REASONING_TOKENS=2048 (validate_live.py, questions_natural_language.txt: "
              "fresh paraphrases written for PF-7, none from the 60-question acceptance set).",
    "summary": {"questions": len(nl), "answered": [r["question"] for r in nl if not r["refused"]],
                "refused": [r["question"] for r in nl if r["refused"]],
                "empty_drafts_cut_at_the_cap": [r["question"] for r in nl if any(c.get("finish") == "length" and not c.get("tokens") for c in r["llm_calls"])],
                "drafts_over_the_old_512_cap": [r["question"] for r in nl if any((c.get("tokens") or 0) > 512 for c in r["llm_calls"])],
                "disputed_claim_served": [r["question"] for r in nl if r["states_disputed_claim"] and not r["refused"]]},
    "questions": nl})

cite = load("val_cite.json")
legacy = [r for r in cite if "before_chips" in r]
dump("citation_validation.json", {
    "method": ("The same model answer and evidence give the PF-6 chips (every evidence item, the Verifier's page-number citation) and the "
               "D-117 chips (one best supporting item per claim); a displayed document is supported when one of its items on a displayed "
               "page supports at least one answer claim (the Verifier's item-level test). citation_live.py, questions_citation.txt."),
    "summary": {"questions": len(cite), "legacy_model_answers": len(legacy),
                "documents_shown_before": sum(1 for r in legacy for c in r["before_chips"] if c[0] == "doc"),
                "documents_shown_after": sum(1 for r in legacy for c in r["after_chips"] if c[0] == "doc"),
                "page_chips_before": sum(1 for r in legacy for c in r["before_chips"] if c[0] == "page"),
                "page_chips_after": sum(1 for r in legacy for c in r["after_chips"] if c[0] == "page"),
                "unsupported_documents_before": sum(len(r["unsupported_doc_chips_before"]) for r in legacy),
                "unsupported_documents_after": sum(len(r["unsupported_doc_chips_after"]) for r in legacy),
                "every_displayed_document_supports_a_claim": all(r["every_displayed_doc_supports_a_claim"] for r in legacy)},
    "questions": cite})
print("written")
