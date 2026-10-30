"""Step 4.16-B2: build the contact-accuracy cases of the Institutional Navigation directory lookups.

The expected numbers are taken from the printed directory as the PDF draws it (``institutional_support.printed_directory``:
the ruled cells of the office column and the words inside the telephone and fax columns), never from the extracted lines
the specialist reads. Which offices may be answered (the "clean" entries) is a reviewed decision listed below: an entry
whose printed cell holds exactly one value in each column AND whose extracted line holds the whole row.

Usage (from anywhere; writes only the --out file):
  python -X utf8 build_contact_cases.py --repo "<repo root>" --out contact_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

#: Entries answered with both numbers: one printed value per column, the whole row on one extracted line.
CLEAN = ("registration", "disability_resource_center", "passport_unit", "college_sharia_islamic_studies", "college_fine_arts_design",
         "college_communication", "college_medicine", "college_dental_medicine", "college_pharmacy", "information_technology_center",
         "physical_plant", "human_resources", "public_relations", "media_center", "finance_department", "alumni_association",
         "central_laboratories")
#: How a question names each clean office (the lexicon's printed name; "Registration" needs an office noun).
ASKED_NAME = {"registration": "the Registration office"}
#: Entries whose line starts with the office but whose numbers cannot be bound (partial: the office name only).
SINGLE = ("college_sciences", "college_computing_informatics", "college_public_policy")
FRAGMENTED = ("student_housing", "college_business_administration", "college_engineering", "college_health_sciences", "college_law",
              "deanship_student_affairs", "medical_service_department", "university_city_security", "mens_library", "womens_library",
              "medical_colleges_library")
#: Directory entries whose printed name is on no single extracted line (not found).
SPLIT = ("college_arts_humanities_social_sciences", "college_graduate_studies_research", "continuing_education_professional_development",
         "business_services_purchases_department")
#: Lexicon offices the directory does not list (not found).
ABSENT = ("main_library", "health_and_safety_office", "facilities_management_planning_department", "business_service_department",
          "board_of_trustees", "office_international_relations")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo, out_path = os.path.abspath(args.repo), os.path.abspath(args.out)
    for p in (os.path.join(repo, ".venv", "Lib", "site-packages"), os.path.join(repo, "tests"), repo):
        sys.path.insert(0, p)
    os.chdir(repo)
    from institutional_support import printed_directory  # noqa: E402
    from handbook_bot.agents.specialists import navigation_cues as N  # noqa: E402

    printed = {e["office_id"]: e for e in printed_directory() if e["office_id"]}
    def name(office_id):
        return ASKED_NAME.get(office_id, "the " + N.OFFICES_BY_ID[office_id].name)

    def one(office_id, column):
        lines = printed[office_id][column]
        assert len(lines) == 1, (office_id, column, lines)
        return lines[0]

    cases = []

    def add(case_id, category, question, status, values=None, rationale="", **extra):
        case = {"case_id": case_id, "category": category, "question": question, "expected_status": status,
                "expected_values": values or {}, "rationale": rationale}
        case.update(extra)
        cases.append(case)

    # ---- C1 / C2: clean entries, telephone and fax ------------------------------------------------------------------
    for k, o in enumerate(CLEAN, 1):
        add("C1-%02d" % k, "clean_telephone", "What is the phone number of %s?" % name(o), "supported", {o: {"telephone": one(o, "telephone")}},
            "one printed value per column; the extracted line holds the whole row (page %d)" % printed[o]["page"])
        add("C2-%02d" % k, "clean_fax", "What is the fax number of %s?" % name(o), "supported", {o: {"fax": one(o, "fax")}},
            "one printed value per column; the extracted line holds the whole row (page %d)" % printed[o]["page"])
    # ---- C3: generic contact asks: both columns ---------------------------------------------------------------------
    for k, (o, q) in enumerate((("human_resources", "How can I contact Human Resources?"),
                                ("finance_department", "What are the contact details of the Finance Department?"),
                                ("college_medicine", "How do I reach the College of Medicine?"),
                                ("passport_unit", "What are the contact numbers of the Passport Unit?")), 1):
        add("C3-%02d" % k, "generic_contact", q, "supported", {o: {"telephone": one(o, "telephone"), "fax": one(o, "fax")}},
            "a contact ask naming no field asks for both directory columns")
    add("C3-05", "both_fields", "What are the telephone and fax numbers of the College of Pharmacy?", "supported",
        {"college_pharmacy": {"telephone": one("college_pharmacy", "telephone"), "fax": one("college_pharmacy", "fax")}})
    # ---- C4: attested aliases ---------------------------------------------------------------------------------------
    add("C4-01", "alias", "What is HR's phone number?", "supported", {"human_resources": {"telephone": one("human_resources", "telephone")}},
        "HR is printed for Human Resources (pp. 57, 72)")
    add("C4-02", "alias", "What is the fax number of the ITC?", "supported",
        {"information_technology_center": {"fax": one("information_technology_center", "fax")}}, "ITC is printed for the Information Technology Center (p. 183)")
    add("C4-03", "alias", "What is the Registration Department phone number?", "supported", {"registration": {"telephone": one("registration", "telephone")}},
        "Registration beside an office noun")
    # ---- C5: neighbouring look-alikes keep their own numbers --------------------------------------------------------
    add("C5-01", "look_alike", "What is the phone number of the College of Dental Medicine?", "supported",
        {"college_dental_medicine": {"telephone": one("college_dental_medicine", "telephone")}}, "not the College of Medicine row")
    add("C5-02", "look_alike", "What is the fax number of the Media Center?", "supported", {"media_center": {"fax": one("media_center", "fax")}},
        "Media Center and Public Relations print the same numbers; each is answered from its own line", expected_limitation="also printed")
    add("C5-03", "look_alike", "What is the phone number of Physical Plant?", "supported", {"physical_plant": {"telephone": one("physical_plant", "telephone")}},
        "Physical Plant is not the Facilities Management and Planning Department")
    add("C5-04", "look_alike", "What is the phone number of the FMPD?", "not_found", {}, "FMPD is not in the contact directory")
    # ---- C6: one-number lines and the labelled line ----------------------------------------------------------------
    k = 0
    for o in SINGLE:
        for field in ("phone", "fax"):
            k += 1
            add("C6-%02d" % k, "single_number_line", "What is the %s number of %s?" % (field, name(o)), "partial", {},
                "the extracted line holds one number; one line cannot show its column (printed: telephone %r, fax %r)"
                % (printed[o]["telephone"], printed[o]["fax"]), expected_entry="unbound_row")
    add("C6-07", "labelled_line", "What is the phone number of the Dental Hospital Call Center?", "partial", {},
        "the line labels its values External / Internal; quoted with the labels, no telephone field assigned", expected_entry="labelled_row")
    add("C6-08", "labelled_line", "What is the extension of the Dental Hospital Call Center?", "partial", {},
        "the extension is quoted with its printed label only", expected_entry="labelled_row")
    add("C6-09", "labelled_line", "What is the fax number of the Dental Hospital Call Center?", "not_found", {},
        "the fax column holds an internal extension, not a fax number")
    # ---- C7: fragmented entries (Men / Women, multi-line, sub-entries) --------------------------------------------
    for k, o in enumerate(FRAGMENTED, 1):
        add("C7-%02d" % k, "fragmented_entry", "What is the phone number of %s?" % name(o), "partial", {},
            "the entry runs over several extracted lines, which are never joined", expected_entry="unbound_row")
    # ---- C8: entries split over lines (name on no single line) -----------------------------------------------------
    for k, o in enumerate(SPLIT, 1):
        add("C8-%02d" % k, "split_entry", "What is the phone number of %s?" % name(o), "not_found", {},
            "the printed name is split over extracted lines; nothing is stitched")
    add("C8-05", "split_entry", "What is the fax number of the CCE?", "not_found", {}, "CCE alias of a split entry")
    # ---- C9: offices the directory does not list -------------------------------------------------------------------
    for k, o in enumerate(ABSENT, 1):
        add("C9-%02d" % k, "not_in_directory", "What is the phone number of %s?" % name(o), "not_found", {}, "not listed in section 1.16")
    # ---- C10: names the handbook does not print as an office --------------------------------------------------------
    for k, q in enumerate(("What is the phone number of the IT Center?", "What is the phone number of the IT Department?",
                           "what is the hr phone number?", "What is the phone number of the medical college?",
                           "What is the phone number of business services?", "What is the phone number for registration?",
                           "What is the phone number of the Admissions office?", "What is the phone number of the Chancellor's Office?"), 1):
        add("C10-%02d" % k, "unattested_name", q, "not_found", {}, "no lexicon office is named; nothing is guessed")
    # ---- C11: e-mail, website, extension --------------------------------------------------------------------------
    for k, q in enumerate(("What is the email of Human Resources?", "What is the e-mail address of the ITC?",
                           "What is the email of the IT service desk?", "What is the website of the Information Technology Center?",
                           "What is the URL of the Finance Department web page?", "What is the extension of Human Resources?"), 1):
        add("C11-%02d" % k, "email_website_extension", q, "not_found", {}, "no e-mail, website or extension is printed; none is composed")
    add("C11-07", "email_website_extension", "What is the phone number and email of Human Resources?", "partial",
        {"human_resources": {"telephone": one("human_resources", "telephone")}}, "the telephone is bound; the e-mail is not printed")
    # ---- C12: the office named is not the unit asked about ---------------------------------------------------------
    for k, q in enumerate(("What is the phone number of the Vice Chancellor for Public Relations?",
                           "What is the phone number of the College of Medicine library?",
                           "What is the phone number of the dean of the College of Law?",
                           "What is the phone number of the women's section of the College of Medicine?",
                           "What is the phone number of the Human Resources manager?",
                           "What is the phone number of the Finance Department of the College of Engineering?",
                           "What is the phone number of the College of Dental Medicine hospital?",
                           "What is the phone number of the Central Laboratories pharmacy?"), 1):
        add("C12-%02d" % k, "referent_trap", q, "not_found", {}, "the named office is not the unit asked about; its line is not used")
    # ---- C13: numbers printed for more than one entry --------------------------------------------------------------
    add("C13-01", "shared_number", "What is the phone number of the Information Technology Center?", "supported",
        {"information_technology_center": {"telephone": one("information_technology_center", "telephone")}},
        "the switchboard number is printed for Registration, Admissions and the ITC", expected_limitation="also printed")
    add("C13-02", "shared_number", "What is the phone number of Public Relations?", "supported",
        {"public_relations": {"telephone": one("public_relations", "telephone")}}, "shared with the Media Center", expected_limitation="Media Center")
    add("C13-03", "shared_number", "What is the fax number of the Alumni Association?", "supported",
        {"alumni_association": {"fax": one("alumni_association", "fax")}}, "printed again on another directory line", expected_limitation="also printed")
    # ---- C14: several offices in one clause -----------------------------------------------------------------------
    add("C14-01", "several_offices", "What are the phone numbers of HR and the Finance Department?", "supported",
        {"human_resources": {"telephone": one("human_resources", "telephone")}, "finance_department": {"telephone": one("finance_department", "telephone")}})
    add("C14-02", "several_offices", "What are the fax numbers of Physical Plant and the College of Sciences?", "partial",
        {"physical_plant": {"fax": one("physical_plant", "fax")}}, "Physical Plant bound; the College of Sciences line holds one number")
    # ---- C15: mixed focus clauses ---------------------------------------------------------------------------------
    add("C15-01", "mixed_clauses", "What is the phone number of HR and where is the Main Library located?", "partial",
        {"human_resources": {"telephone": one("human_resources", "telephone")}},
        "the location clause gives two printed Main Library entries (phase B3), so the whole stays partial; no number for it",
        clauses=["What is the phone number of HR", "where is the Main Library located"])
    add("C15-02", "mixed_clauses", "What is the grading policy?", "out_of_scope", {},
        "not a channel ask (Step 4.16-B3: the former question, on health and safety, is now answered from section 13); a "
        "Teaching subject, out_of_scope with no handoff since Step 4.16-B4")
    # ---- SYN: perturbed directory lines (negative controls of the binding protections) -----------------------------
    add("SYN-01", "synthetic_neighbour", "What is the phone number of the Information Technology Center?", "partial", {},
        "the line above now runs on with a trailing slash", replace_lines={"2099": "+971 6 5585160 /"}, expected_entry="unbound_row")
    add("SYN-02", "synthetic_neighbour", "What is the phone number of Public Relations?", "partial", {},
        "the line below now starts with the rest of a broken number", replace_lines={"2104": "5585209"}, expected_entry="unbound_row")
    add("SYN-03", "synthetic_row_start", "What is the phone number of Human Resources?", "not_found", {},
        "the only line naming Human Resources now names another unit first", replace_lines={"2102": "Directorate of Human Resources +971 6 5051111 +971 6 5052222"})
    add("SYN-04", "synthetic_header", "What is the phone number of Human Resources?", "supported", {"human_resources": {"telephone": "+971 6 5585200"}},
        "the page header now prints the fax column first, so the second value is the telephone",
        replace_lines={"2093": "Office / College / Department Fax Number Telephone Number"})
    add("SYN-05", "synthetic_header", "What is the phone number of Human Resources?", "partial", {},
        "the page header no longer names the columns", replace_lines={"2093": "Office / College / Department"}, expected_entry="unbound_row")
    add("SYN-06", "synthetic_single_entry", "What is the phone number of Human Resources?", "partial", {},
        "a second line starting with Human Resources now holds other numbers",
        replace_lines={"2112": "Human Resources +971 6 5051111 +971 6 5052222"}, expected_entry="unbound_row")

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("%d cases written to %s" % (len(cases), out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
