"""PF-3D adversarial results: runs the adversarial cases of ``tests/test_guide_structure.py`` (synthetic pages built by the
PF-2 chunker, the real corpus without a model, and the adapter end to end) and reads the safety groups of the PF-3D
evaluation (``metrics.json``). Writes ``adversarial_results.json``.

    python -X utf8 qa/post_freeze_pf3d/adversarial.py <repo> <metrics.json> <out.json>
"""
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

REPO = os.path.abspath(sys.argv[1])
METRICS = json.load(open(sys.argv[2], encoding="utf-8"))
OUT = sys.argv[3]

CASES = {
    "incomplete procedure: first or middle step missing, two procedures mixed": "test_incomplete_or_mixed_numbered_procedures_are_refused",
    "incomplete procedure: last step announces content that does not follow": "test_a_last_step_that_announces_missing_content_is_refused",
    "incomplete procedure: a figure reference is not a missing step": "test_a_last_step_that_points_at_a_figure_is_not_a_missing_step",
    "incomplete procedure: page continuation not indexed": "test_a_page_continuation_that_is_not_indexed_leaves_the_end_unproven",
    "incomplete procedure: heading from another procedure": "test_a_heading_that_belongs_to_another_procedure_is_refused",
    "verification: a dropped last unit, reordered or repeated units, an unapproved specialist": "test_the_verification_refuses_a_structure_that_is_not_exactly_complete",
    "verification: a forged section answer through the adapter": "test_forged_section_answers_are_refused_by_the_adapter",
    "exact quotes: a block no chunk holds verbatim": "test_a_block_that_no_chunk_holds_verbatim_makes_the_structure_unusable",
    "conflict: a registered conflict inside a structure": "test_a_registered_conflict_inside_a_structure_refuses_it",
    "unordered list: never presented as Step 1/2/3": "test_an_unordered_list_is_never_presented_as_steps",
    "unordered list: lettered titled callouts are options": "test_lettered_callouts_with_titles_are_options_not_a_sequence",
    "list types: classification": "test_list_type_classification",
    "mixed list: lettered sub-steps keep their printed labels": "test_steps_with_lettered_substeps_keep_their_hierarchy",
    "enumeration: a list must name the asked set": "test_a_list_must_name_the_asked_set_to_answer_an_enumeration",
    "real corpus: structures never serve what they do not prove": "test_structures_never_serve_what_they_do_not_prove",
    "real corpus: a system name in a heading never names the asked action": "test_a_system_name_in_a_heading_never_names_the_asked_action",
}

with tempfile.TemporaryDirectory() as tmp:
    xml_path = os.path.join(tmp, "junit.xml")
    subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_guide_structure.py",
                    "-k", " or ".join(CASES.values()), "--junitxml", xml_path], cwd=REPO, capture_output=True, text=True)
    cases = ET.parse(xml_path).getroot().iter("testcase")
    outcome = {}
    for case in cases:
        name = case.get("name")
        failed = any(child.tag in ("failure", "error") for child in case)
        skipped = any(child.tag == "skipped" for child in case)
        outcome[name] = "failed" if failed else "skipped" if skipped else "passed"

tests = {}
for label, func in CASES.items():
    runs = {name: result for name, result in outcome.items() if name == func or name.startswith(func + "[")}
    tests[label] = {"test": func, "cases": len(runs), "passed": sum(1 for r in runs.values() if r == "passed"),
                    "failed": sorted(n for n, r in runs.items() if r != "passed")}

# guide answers served inside a safety-named group, each reviewed by hand (``review.json``)
NOTES = {
    "P11": "guide_contamination: an interface question that must be answered from the gradebook guide and never from a "
           "handbook policy sentence; served from the guide (the Post option, page 6) and reviewed CORRECT, unchanged since PF-3C",
    "C29": "incomplete_procedure: labelled incomplete in PF-3C because the PF-2 reading took the section label 'D.' for a fifth "
           "step; the printed page holds the complete four-step keyboard list under 'C. Reorder content:' (reviewed CORRECT); "
           "C30 still falls back",
}
groups = METRICS.get("groups") or {}
SAFETY = ("conflict", "guide_contamination", "handbook_contamination", "wrong_source", "appraisal", "known_hr", "annual_leave",
          "incomplete_procedure", "authority", "ambiguous", "routing", "cross_document")
evaluation = {g: {k: groups[g][k] for k in ("questions", "served", "guide_served", "ids_guide_served")} for g in SAFETY if g in groups}
summary = {"synthetic_and_corpus_cases": sum(t["cases"] for t in tests.values()), "passed": sum(t["passed"] for t in tests.values()),
           "failed": sorted(n for t in tests.values() for n in t["failed"]),
           "evaluation_guide_served_in_blocked_groups": {g: v["ids_guide_served"] for g, v in evaluation.items()
                                                         if g not in ("cross_document",) and v["ids_guide_served"]},
           "notes": {qid: note for qid, note in NOTES.items() if any(qid in v["ids_guide_served"] for v in evaluation.values())},
           "unexplained": sorted(q for g, v in evaluation.items() if g != "cross_document" for q in v["ids_guide_served"] if q not in NOTES)}
json.dump({"method": __doc__.strip(), "summary": summary, "tests": tests, "evaluation_safety_groups": evaluation},
          open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
