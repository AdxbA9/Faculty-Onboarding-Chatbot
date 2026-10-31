"""
Shared domain cue source (multi-agent architecture, Milestone 4, Step 4.5A-R).

One lexicon of Research ownership terms, read by the Faculty Onboarding
Coordinator (to select the Research & Innovation Specialist), by the Research
& Innovation Specialist (to accept a clause as its own) and by the Teaching &
Learning Specialist (to hand a research intent off). Before this module the
three kept separate tables that drifted apart: the Coordinator routed "Who
owns software I develop?" to Research, which then rejected the clause as
carrying no research cue.

Step 4.11 adds the Faculty Services & HR lexicon on the same footing: strong
ownership cues, weak context cues that count only beside another cue, and the
collision words (classes, attendance, travel, evaluation, workload, chair,
dean, semester, credit hours, a bare "leave") that never route a question to
HR on their own. The Coordinator and the Faculty Services & HR Specialist read
the same tables, so a question routed to HR is never rejected there for a
phrase this module lacks.

Routing vocabulary only: no policy content, no model, no network. It imports
only the standard library.
"""
from __future__ import annotations

import re
from typing import Tuple

#: Hyphen-like separators between word characters are read as spaces, same
#: length as the original (the Coordinator's ``cue_text``).
_CUE_SEPARATORS = re.compile(r"(?<=\w)[-‐‑‒–—/](?=\w)")


def cue_text(text: str) -> str:
    return _CUE_SEPARATORS.sub(" ", text)


#: WLAM faculty categories. They are Teaching vocabulary (the workload model
#: names them), so the Research cues score a view of the text in which these
#: phrases are blanked (same length); "research" elsewhere in the clause
#: still counts. Every other domain scores the normal cue text.
WLAM_CATEGORY = re.compile(r"\b(?:research[- ]intensive|active[- ]research|research[- ]active)\s+faculty\b", re.I)
#: Funding for professional, faculty or teaching development is Teaching's
#: (faculty development), never a research-funding cue.
DEVELOPMENT_FUNDING = re.compile(r"\b(?:fund(?:s|ed|ing)?|grants?)\b(?=[^.?]{0,70}\b(?:(?:professional|faculty|teaching|staff)\s+development|teaching\s+(?:workshops?|"
        r"awards?|excellence|innovation|conferences?)|course\s+(?:re)?design|curriculum\s+(?:re)?design|pedagog\w*)\b)|"
        r"\b(?:professional|faculty|teaching|staff)\s+development\s+(?:fund(?:s|ing)?|grants?)\b", re.I)


def research_view(text: str) -> str:
    text = WLAM_CATEGORY.sub(lambda m: " " * len(m.group(0)), text)
    return DEVELOPMENT_FUNDING.sub(lambda m: " " * len(m.group(0)), text)


#: Research ownership patterns (strong cues). The Coordinator scores each once per question on the research view.
RESEARCH_OWNERSHIP_PATTERNS: Tuple[str, ...] = (
    r"\bresearch(?:er|ers)?\b",
    # post-freeze PF-3B (D-110): iThenticate is the university's originality checker for researchers' manuscripts (the
    # official ITC guideline says it "is used for university researchers"); course-assignment checkers (SafeAssign,
    # Turnitin) stay Teaching cues
    r"\bi\s?thenticate\b",
    # "grant" as a verb ("who grants tenure", "grant a student an extension") and a grant of leave are not research grants
    r"(?<!who )(?<!to )(?<!\bi )(?<!we )\bgrants?\b(?!\s+(?:tenure|leave|access|permission|approval|exemptions?|exceptions?|extensions?|"
        r"an?\s|me\b|him\b|her\b|them\b|students?\b|of\s+(?:leave|absence|tenure)))|\b(?:fund(?:ed|ing)|seed fund|principal investigator|pi)\b",
    r"\b(?:research ethics|(?<!professional )ethics (?:committee|approval|application|review|clearance)|\brec\b(?!\s+(?:cent(?:er|re)|room|hall))|\birb\b|\bacuc\b|"
        r"human subjects|animal (?:use|care))\b",
    # "publication of course materials" and library journals are not scientific publishing
    r"\b(?:publications?(?!\s+of\s+(?:my\s+|the\s+)?(?:course|lecture|teaching|class)\s+materials?)|publishing (?:support|unit)|"
        r"publish(?:ing|ed)? (?:a |an |my |the |our )?(?:papers?|articles?|research|books?|journal|work)|"
        r"(?<!library )(?<!ieee )journals?(?!\s+(?:in\s+the\s+library|subscriptions?|databases?))|scopus|h[- ]?index|citations?)\b",
    r"\b(?:intellectual property|\bip\b(?!\s+address)|patents?|tech(?:nology)? transfer|\btto\b|commerciali[sz]ation|licens(?:e|ing) (?:an? )?(?:invention|patent|technology))\b",
    r"\bresearch (?:institutes?|groups?|centers?|centres?|projects?|committees?|board)\b",
    r"\b(?:iacuc|institutional review board|animal care and use committee)\b",
    r"\b(?:royalt(?:y|ies)|inventors?|invention disclosures?|disclos\w*\s+(?:an\s+|the\s+|my\s+)?inventions?|commerciali[sz]\w*|"
        r"licens(?:e|es|ing)\b[^.?]{0,30}\b(?:revenue|royalt\w*|inventions?|patents?|technolog\w*|ip)|technology transfer office)\b",
    r"\b(?:research (?:priority areas?|misconduct|assistants?|advance|budget|portal|office|support units?|funding department|"
        r"publishing|regulations?|agreements?|leave)|university research board|vcrgs|ovcrgs|rfd|rimhs|rihss|scientific (?:publishing|research)|"
        r"uos journal)\b",
    # consultancy; "consulting students / a colleague" is advising, not consultancy
    r"\bconsultanc(?:y|ies)\b|\bconsulting\b(?!\s+(?:with\s+)?(?:the\s+|my\s+|a\s+|an\s+)?(?:students?|colleagues?|chair|dean|advisors?|supervisors?|"
        r"parents?|peers?)\b)|\bpaid consultant\b|\bconsultants?\b(?=[^.?]{0,40}\b(?:compan(?:y|ies)|firms?|industry|private|clients?|paid)\b)",
    # Step 4.5A reachability: conference policy (5.4) in context; a bare "conference" stays a hint, and video
    # conferences, conference rooms, halls and calls are never research
        r"\bper diem\b|\bsymposi(?:um|a)\b|\bcongress(?:es)?\b|"
        r"\b(?<!video )(?<!web )conferences? (?:registration|fees?|funding|funds?|travel|trips?|support|attendance|allowances?|expenses?|costs?|request(?:s)?|"
        r"applications?|documents?|documentation|deadlines?|budget|participation|papers?|proceedings|policy|abroad|overseas|approvals?)\b|"
        r"\b(?:attend(?:ing|ed)?|attendance at|participat\w* in|take part in|taking part in|join(?:ing)?|present(?:ing)? (?:a paper )?at|travel(?:l?ing)? (?:to|for)|"
        r"register(?:ing)? for|registration (?:fees? )?for|fund(?:ing)? for|support for|reimburs\w* for|go(?:ing)? to)\s+(?:an? |the |my )?(?:local |international |"
        r"external |regional |scientific |academic |overseas |university supported )?conferences?\b(?!\s+(?:calls?|rooms?|halls?|cent(?:er|re)s?))|"
        r"\b(?:local|international|external|regional|overseas|scientific|academic|university supported) conferences?\b(?!\s+(?:calls?|rooms?|halls?))|"
        r"\bconferences? (?:abroad|overseas)\b|\bpresent(?:ing)? a paper\b|\bhow many conferences\b|"
        r"\bconferences? per (?:year|semester|academic year)\b|\b(?:before|after|number of|limit on) (?:the |a |each |my )?conferences?\b(?!\s+(?:calls?|rooms?|halls?|cent(?:er|re)s?))|"
        r"\b(?<!video )conferences?\b(?!\s+(?:calls?|rooms?|halls?|cent(?:er|re)s?))[^.?]{0,40}\b(?:registration|documents?|documentation|deadline|per diem|funding|"
        r"reimburs\w*|covered|allowances?|approv\w*|expenses?|papers?|abstracts?|fliers?|flyers?)\b|"
        r"\b(?:registration fees?|documents?|documentation|reimburs\w*|approv\w*)\b[^.?]{0,40}\bconferences?\b(?!\s+(?:calls?|rooms?|halls?))",
    # scientific publishing roles, petty cash, seed projects, inventions and copyright, MTAs, licence agreements,
    # conflict of interest, final or progress project reports, publishing costs, informed consent, co-investigators,
    # book awards, tangible research property, disclosure evaluation and project closure
        r"\b(?:chief editors?|editors? in chief|editorial (?:board|secretary)|petty cash|seed (?:projects?|grants?|research|funding)|inventions?|copyright\w*|"
        r"materials? transfer agreements?|mtas?|licen[cs](?:e|es|ing) agreements?|conflicts? of (?:interest|commitment)|informed consent|co investigators?|"
        r"(?<!student )(?<!capstone )(?<!course )(?:final|progress|annual) (?:research|project) reports?|research (?:project )?(?:final |progress |annual )?reports?|"
        r"project (?:final|progress|annual) reports?|"
        r"publish\w* (?:costs?|fees?|charges?|expenses?)|(?:costs?|fees?|charges?|expenses?) (?:of|for) publish\w*|"
        r"open access (?:fees?|charges?|publication|publishing|journals?|articles?)|article processing charges?|page charges?|"
        r"(?:book|author|authors|authored book) awards?|awards? for (?:authored )?books|authored books?|tangible research property|trp|"
        r"disclosure evaluation|project closure|closure of (?:a |the |my )?(?:research )?project)\b|"
        r"\bwho signs\b[^.?]{0,30}\blicen[cs]\w*|\bsigns?\b[^.?]{0,20}\blicen[cs](?:e|ing) agreements?|"
        r"\bwho owns\b[^.?]{0,40}\b(?:software|inventions?|intellectual property|patents?|copyright\w*|creative works?|books?|data|designs?)\b",
    # supervision by detail cue: a cap or count of theses, dissertations, students or candidates supervised, or the
    # supervision of research projects and postgraduate research; "thesis supervision credits/load" stays Teaching
    r"\b(?:how many|maximum|max|limit|cap|simultaneous\w*|at once|at a time)\b[^.?]{0,40}\b(?:thes[ie]s|dissertations?)\b(?!\s+supervision\s+(?:credits?|load|hours|workload))|"
        r"\b(?:thes[ie]s|dissertations?)\b(?!\s+supervision\s+(?:credits?|load|hours|workload))[^.?]{0,40}\b(?:how many|maximum|limit|cap|simultaneous\w*|at once|at a time)|"
        r"\b(?:how many|maximum)\b[^.?]{0,30}\b(?:students?|candidates?|supervisees?)\b[^.?]{0,30}\bsupervis\w*|"
        r"\bsupervis\w*\s+(?:of\s+)?(?:a\s+|the\s+|my\s+)?(?:research (?:projects?|students?|work)|postgraduate research|graduate research|phd (?:students?|research|candidates?))\b",
)
RESEARCH_OWNERSHIP = tuple(re.compile(p, re.I) for p in RESEARCH_OWNERSHIP_PATTERNS)


def research_owned(text: str) -> bool:
    """A Research ownership cue fires on the research view of ``text``."""
    view = research_view(cue_text(text))
    return any(p.search(view) for p in RESEARCH_OWNERSHIP)


# ---------------------------------------------------------------------------
# Faculty Services & HR (Milestone 4, Step 4.11; D-085)
# ---------------------------------------------------------------------------
#: Strong HR ownership cues. The Coordinator scores each once per question; the HR specialist accepts a clause when one
#: fires. Employment words, the leave types, pay and benefits, the appointment life cycle, the evaluation and promotion
#: PROCESS, discipline and grievance, the joining formalities and the faculty services the handbook describes.
HR_OWNERSHIP_PATTERNS: Tuple[str, ...] = (
    r"\b(?:hr|human resources?|personnel|employee (?:services?|information|records?|portal|self[- ]service|details|profile|file|handbook|id(?: cards?)?)|"
    r"employment|employer|employees?)\b",
    # the leave types and leave as an entitlement; a bare "leave" (the verb, "leave the campus") is a context cue only
    r"\b(?:annual|sick|maternity|paternity|emergency|unpaid|special|hajj|vacation|study|compassionate|casual|medical|marriage|bereavement)\s+leaves?\b|"
    r"\bleaves?\s+(?:of\s+absence|entitlements?|balances?|requests?|applications?|days|policy|policies|and\s+absences?|approval|period)\b|\bdays?\s+of\s+leave\b|"
    r"\b(?:take|taking|took|taken|request|requesting|requested|apply\s+for|applying\s+for|applied\s+for|grant|granted|be\s+on|am\s+on|go\s+on|going\s+on)\s+"
    r"(?:a\s+|my\s+|some\s+|the\s+|an?\s+\w+\s+)?leaves?\b|"
    r"\b(?:sabbatical|vacations?|sick\s+days?|maternity|paternity|hajj|paid\s+leave|unpaid\s+leave|leave\s+with(?:out)?\s+pay)\b",
    r"\b(?:contracts?|contractual|resign(?:ation|ing|ed|s)?|probation(?:ary)?|appointments?|appointed|hiring|hired|recruitment|recruited|"
    r"termination|terminated|non[- ]renewal|renewal|renewed|confirmation(?:\s+of\s+(?:appointment|employment|contract))?|notice\s+period|"
    r"(?:employment|job)\s+offers?|offer\s+(?:letter|acceptance)|dual\s+career|nepotism|employment\s+of\s+relatives)\b",
    r"\b(?:benefits?|salary|salaries|pay\s+scale|payroll|payslips?|pay\s+slips?|pay|allowances?|gratuity|end[- ]of[- ]service|pension|bonus(?:es)?|incentives?|"
    r"compensation|remuneration|monetary\s+benefits?|duplication\s+of\s+benefits?)\b",
    r"\b(?:housing|accommodation|insurance|medical\s+cover(?:age)?|(?:entry|residence|residency)\s+visas?|visas?|residenc[ey]|emirates\s+id|passport(?:\s+unit)?|"
    r"relocation|shipment|air\s?tickets?|airfare|return\s+tickets?|children'?s?\s+education|education\s+allowance|school\s+fees|tuition\s+(?:waiver|scholarship|discount))\b",
    r"\b(?:promotion|promoted|rank|tenure|performance\s+evaluation|annual\s+evaluation|faculty\s+(?:performance\s+)?evaluation|evaluation\s+(?:process|criteria|weights?|rubric|form|period)|"
    r"contract\s+renewal|renewal\s+rubric|faculty\s+information\s+system|\bfis\b|dossier|appeals?\s+(?:of|against)\s+(?:a\s+|the\s+)?promotion|"
    r"(?:academic\s+)?promotions?\s+committees?|teaching\s+load\s+reductions?|reduction\s+in\s+teaching\s+load)\b",
    r"\b(?:demise|emirati|emeritus|clinical\s+(?:faculty|appointment|allowance|professor)|administrative\s+(?:appointment|positions?|compensation|faculty)|"
    r"compensation\s+for\s+administrative|grievances?|disciplinary|faculty\s+discipline|discipline\s+and\s+grievance|"
    r"(?<!research\s)(?<!academic\s)(?<!student\s)misconduct|professional\s+ethics|(?:sexual\s+)?harassment|"
    r"child\s*care(?:\s+cent(?:er|re))?|nursery|medical\s+clinics?|university\s+medical\s+clinics?|\bumc\b)\b",
)
#: Weak HR context cues: they count only beside another cue (the Coordinator scores them as hints). Library, parking,
#: facilities and clinics name a service; "evaluation", "travel", "training" and "dependents" need an HR word next to
#: them; a bare "leave" is the verb as often as the entitlement.
HR_CONTEXT_PATTERNS: Tuple[str, ...] = (
    r"\b(?:librar(?:y|ies)|parking|child\s?care|clinic|health\s+services?|facilities|id\s+cards?|onboarding|joining|induction)\b",
    r"\b(?:evaluation|appraisal|travel|training|dependents?|spouse|family|administrators?|chairpersons?|directors?)\b",
    r"\bleaves?\b",
    r"\b(?:aed|dirhams?|dhs?)\b",
    r"\bmonthly\b|\bper\s+month\b|\bone[- ]time\b",
)
#: Words that never route a question to HR on their own (documented and tested): each belongs to another domain or to
#: ordinary English until an HR cue stands beside it.
HR_COLLISION_PATTERNS: Tuple[str, ...] = (
    r"\bclass(?:es)?\b", r"\battendance\b", r"\btravel\b", r"\bevaluation\b", r"\bworkloads?\b", r"\bchairs?\b", r"\bdeans?\b",
    r"\bsemesters?\b", r"\bcredit\s+hours?\b", r"\bleaves?\b", r"\bgrants?\b",
)
HR_OWNERSHIP = tuple(re.compile(p, re.I) for p in HR_OWNERSHIP_PATTERNS)
HR_CONTEXT = tuple(re.compile(p, re.I) for p in HR_CONTEXT_PATTERNS)
HR_COLLISION = tuple(re.compile(p, re.I) for p in HR_COLLISION_PATTERNS)


def hr_owned(text: str) -> bool:
    """An HR ownership cue fires on the cue text, or two distinct HR context cues do."""
    view = cue_text(text)
    if any(p.search(view) for p in HR_OWNERSHIP):
        return True
    return sum(1 for p in HR_CONTEXT if p.search(view)) >= 2


__all__ = ["DEVELOPMENT_FUNDING", "HR_COLLISION", "HR_COLLISION_PATTERNS", "HR_CONTEXT", "HR_CONTEXT_PATTERNS", "HR_OWNERSHIP",
           "HR_OWNERSHIP_PATTERNS", "RESEARCH_OWNERSHIP", "RESEARCH_OWNERSHIP_PATTERNS", "WLAM_CATEGORY", "cue_text", "hr_owned",
           "research_owned", "research_view"]
