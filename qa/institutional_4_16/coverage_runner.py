"""
Step 4.16-B5: answer correctness, coverage and conservative behaviour of the Institutional Navigation specialist on
independent questions.

The 62 questions of ``coverage_cases.jsonl`` and their expected answers were written by blind authors from the handbook
pages alone (no access to the specialist, its tests or its cases); every expected quote was checked verbatim on its cited
page before any run. Each question runs twice:

* system path: the unchanged Coordinator and the dispatcher with all four specialists, Institutional Navigation added only
  through ``executable=`` (how phase B6 would run it);
* direct path: Institutional Navigation alone on the whole question as one focus clause (the component).

Each Institutional result is judged with fixed rules (``judge``):

  correct           supported, and the findings state the expected answer
  correct_partial   partial, and the findings state the expected answer (e.g. both entries of an ambiguous name)
  incomplete        findings that are on target but do not state the answer (the office name without its number, one of
                    several units), never counted as an answer
  wrong             findings that state another answer (another office, number, place or fact); "unsafe" when supported
  unexpected_answer findings for a question the handbook does not answer or that is not Institutional's
  abstained         no finding (not_found or out_of_scope); a correct refusal is never counted as an answer
  not_selected      (system path) the Coordinator did not select Institutional Navigation

Usage: python qa/institutional_4_16/coverage_runner.py --repo . --out <dir> [--default-dispatch]

Since Step 4.16-B6 (D-105) the default dispatch executes all four specialists; ``--default-dispatch`` runs the system
path without the ``executable=`` override (the results are the same; the option shows it).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter


def _setup(repo):
    for p in (os.path.join(repo, ".venv", "Lib", "site-packages"), os.path.join(repo, "tests"), repo):
        if p not in sys.path:
            sys.path.insert(0, p)


def norm(text):
    text = (text or "").replace("’", "'").replace("‘", "'").replace("&", "and")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


def digits(number):
    return re.sub(r"\D", "", number or "")


NUMBER = re.compile(r"(?:\+971\s?\d\s?\d{3}\s?\d{4}|0\d-\s?\d{7})")


def expected_names(case):
    """Each expected unit as a set of normalised forms: its name without a parenthesis, and an acronym in parentheses."""
    out = []
    for part in (case.get("expected_office") or "").split("|"):
        part = part.strip()
        if not part:
            continue
        forms = {norm(re.sub(r"\([^)]*\)", "", part))}
        forms |= {norm(a) for a in re.findall(r"\(([^)]*)\)", part) if not re.search(r"\d", a)}
        out.append({f for f in forms if f})
    return out


def office_forms(N, office_id):
    if office_id in (None, N.UNIVERSITY):
        return set()
    o = N.OFFICES_BY_ID[office_id]
    return {norm(o.name)} | {norm(a) for a, _p in o.aliases + o.subject_aliases}


def answer_check(N, case, findings):
    """match | partial_match | mismatch, for the expected answer of an answerable question."""
    kind = case["expected_answer_type"]
    if kind in ("telephone", "fax", "telephone_and_fax"):
        want = {digits(n) for n in NUMBER.findall(case.get("expected_value") or "")}
        got = {digits(v) for f in findings for v in (f.metadata.get("contact_values") or {}).values()}
        names = expected_names(case)
        on_target = [f for f in findings if office_forms(N, f.metadata.get("office_id")) & set().union(*names)] if names else []
        if got - want:
            return "mismatch"
        if want and want <= got:
            return "match"
        if got:
            return "partial_match"
        return "partial_match" if on_target and len(on_target) == len(findings) else "mismatch"
    if kind in ("responsible_office", "office_role"):
        names = expected_names(case)
        all_forms = set().union(*names) if names else set()
        hit = [False] * len(names)
        for f in findings:
            oid = f.metadata.get("office_id")
            forms = office_forms(N, oid)
            if oid not in (None, N.UNIVERSITY):
                if not forms & all_forms:
                    return "mismatch"                      # an office the handbook does not state for this duty
            else:
                forms = {x for x in all_forms if re.search(r"\b%s\b" % re.escape(x), norm(f.evidence_quote))}
            for i, n in enumerate(names):
                hit[i] = hit[i] or bool(forms & n)
        if names and all(hit):
            return "match"
        return "partial_match" if any(hit) else "mismatch"
    if kind == "location":
        value = norm(case.get("expected_value") or "")
        entries = [p for p in (case.get("expected_value") or "").split("|") if p.strip()]
        found = 0
        for f in findings:
            code, place = f.metadata.get("building_code"), norm(f.metadata.get("place") or "")
            if (code and norm(code) in value) or (place and place in value):
                found += 1
            else:
                return "mismatch"
        return "match" if found >= len(entries) else ("partial_match" if found else "mismatch")
    # chair, composition, count, colleges_list, organisation: the stated fact on the expected page
    pages = set(case.get("expected_pages") or [])
    key = norm(re.sub(r"\([^)]*\)", "", (case.get("expected_value") or "").split("|")[0]))
    key_words = [w for w in key.split() if w not in ("the", "of", "and", "a", "an", "in", "total")]
    on_page = [f for f in findings if f.page in pages]
    if not on_page:
        return "mismatch"
    text = " ".join(norm(f.evidence_quote) for f in on_page)
    if kind in ("count", "chair"):
        return "match" if key_words and all(w in text.split() for w in key_words[:4]) else "mismatch"
    ev = set(norm(case.get("evidence_quote")).split())
    overlap = len(ev & set(text.split())) / max(1, len(ev))
    return "match" if overlap >= 0.5 else "partial_match"


def judge(N, case, result, texts):
    if result is None:
        return {"outcome": "not_selected", "status": None, "citations_ok": None, "findings": []}
    findings = result.findings
    info = {"status": result.status.value, "findings": [{"page": f.page, "office_id": f.metadata.get("office_id"),
                                                         "contact_values": f.metadata.get("contact_values"),
                                                         "building_code": f.metadata.get("building_code"),
                                                         "quote": f.evidence_quote[:200]} for f in findings],
            "handoffs": [h.requested_specialist.value for h in result.handoff_requests]}
    if not findings:
        return dict(info, outcome="abstained", citations_ok=None)
    verbatim = all(f.evidence_quote in texts.get(f.metadata.get("chunk_id"), "") and f.source_id == "uos_faculty_handbook_2025_26"
                   for f in findings)
    if not case["answerable_from_handbook"] or "institutional" not in case["expected_owner"]:
        return dict(info, outcome="unexpected_answer", citations_ok=verbatim)
    check = answer_check(N, case, findings)
    pages = set(case.get("expected_pages") or [])
    cites = verbatim and (not pages or all(f.page in pages for f in findings))
    supported = result.status.value == "supported"
    if check == "match":
        outcome = "correct" if supported else "correct_partial"
    elif check == "partial_match":
        outcome = "wrong" if (supported and case.get("ambiguous")) else "incomplete"
    else:
        outcome = "wrong"
    return dict(info, outcome=outcome, citations_ok=cites)


def metrics(rows, path):
    ans = [r for r in rows if r["answerable_from_handbook"] and "institutional" in r["expected_owner"]]
    other = [r for r in rows if r not in ans]
    oc = Counter(r[path]["outcome"] for r in rows)
    answered = [r for r in rows if r[path]["outcome"] in ("correct", "correct_partial", "wrong", "unexpected_answer")]
    right = [r for r in answered if r[path]["outcome"] in ("correct", "correct_partial")]
    unsafe = [r["id"] for r in rows if r[path]["status"] == "supported" and r[path]["outcome"] in ("wrong", "unexpected_answer")]
    cited = [r for r in rows if r[path]["citations_ok"] is not None and r[path]["outcome"] in ("correct", "correct_partial", "incomplete")]
    return {
        "outcomes": dict(oc),
        "answer_correctness_on_answered": "%d/%d" % (len(right), len(answered)),
        "unsafe_supported": unsafe,
        "wrong_any_status": [r["id"] for r in rows if r[path]["outcome"] in ("wrong", "unexpected_answer")],
        "coverage_strict_supported": "%d/%d" % (sum(r[path]["outcome"] == "correct" for r in ans), len(ans)),
        "coverage_correct_supported_or_partial": "%d/%d" % (sum(r[path]["outcome"] in ("correct", "correct_partial") for r in ans), len(ans)),
        "answerable_not_answered": [r["id"] for r in ans if r[path]["outcome"] not in ("correct", "correct_partial")],
        "abstention_rate_all": "%d/%d" % (sum(r[path]["outcome"] in ("abstained", "not_selected") for r in rows), len(rows)),
        "abstention_rate_answerable": "%d/%d" % (sum(r[path]["outcome"] in ("abstained", "not_selected") for r in ans), len(ans)),
        "correct_refusals_on_unanswerable_or_foreign": "%d/%d" % (sum(r[path]["outcome"] in ("abstained", "not_selected") for r in other), len(other)),
        "citation_accuracy": "%d/%d" % (sum(bool(r[path]["citations_ok"]) for r in cited), len(cited)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--default-dispatch", action="store_true", help="no executable= override (Step 4.16-B6)")
    ap.add_argument("--cases", default="coverage_cases.jsonl", help="case file in qa/institutional_4_16 (Step 4.16-C1: holdout_cases.jsonl)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    _setup(repo)
    os.chdir(repo)
    from handbook_bot.agents.specialists import dispatch as D
    from handbook_bot.agents.specialists import navigation_cues as N
    from handbook_bot.agents.specialists.contracts import SpecialistId

    import institutional_support as IS

    cases = [json.loads(l) for l in open(os.path.join(repo, "qa", "institutional_4_16", args.cases), encoding="utf-8")]
    registry = IS.four_specialist_registry()
    sp = IS.make_institutional()
    texts = {m.get("chunk_id"): t for t, m in zip(sp.resources.chunks, sp.resources.metadata)}
    override = D.EXECUTABLE_SPECIALISTS + (SpecialistId.INSTITUTIONAL,)
    rows = []
    for case in cases:
        direct = IS.run(sp, case["question"])
        out = (D.run_question(case["question"], registry) if args.default_dispatch
               else D.run_question(case["question"], registry, executable=override))
        inst = next((r.findings for r in out.runs if r.specialist_id is SpecialistId.INSTITUTIONAL), None)
        selected = sorted(s.value for s in out.decision.selected_specialists)
        expected = sorted(o for o in case["expected_owner"] if o != "none")
        others = {r.specialist_id.value: {"status": r.findings.status.value, "findings": len(r.findings.findings),
                                          "handoffs": [h.requested_specialist.value for h in r.findings.handoff_requests]}
                  for r in out.runs if r.specialist_id is not SpecialistId.INSTITUTIONAL}
        rows.append(dict({k: case[k] for k in ("id", "area", "question", "expected_owner", "answerable_from_handbook",
                                                 "expected_answer_type", "expected_office", "expected_value", "expected_pages")},
                         selected=selected, routing_exact=selected == expected,
                         institutional_routing_ok=("institutional" in selected) == ("institutional" in expected),
                         system=judge(N, case, inst, texts), direct=judge(N, case, direct, texts), other_specialists=others))
    summary = {"cases": len(rows),
               "answerable_institutional": sum(r["answerable_from_handbook"] and "institutional" in r["expected_owner"] for r in rows),
               "routing_exact": "%d/%d" % (sum(r["routing_exact"] for r in rows), len(rows)),
               "routing_institutional_inclusion": "%d/%d" % (sum(r["institutional_routing_ok"] for r in rows), len(rows)),
               "system": metrics(rows, "system"), "direct": metrics(rows, "direct")}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "coverage_results.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "coverage_summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
