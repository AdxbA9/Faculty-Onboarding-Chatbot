"""
Guide-intent inference (post-freeze PF-5C, D-115).

WHY
    D-110 and D-111 let a system guide answer only a clause that NAMES the system the guide documents ("... in
    Blackboard?"). PF-5B measured the cost on a fresh held-out set: most answerable guide questions never name the system
    ("How do I make my course visible to students?", "Where is the similarity report?"), so the guide registry was never
    selected and nothing was served.

WHAT
    ``infer(clause)``: a deterministic, explainable choice of the system a clause is about when it names none. A fixed
    table of SIGNALS, each a word or phrase PRINTED in one enabled guide (a heading, a tool, a control or a printed task),
    scores every system family (Blackboard Ultra, Grade Transfer, iThenticate). An object alone scores its weight; a task
    verb that governs one of the family's objects ("send ... final grades", "take attendance") adds one bonus. A multi-word signal matches the
    printed words themselves, adjacent and in order (plural -s aside; never a possessive, never another inflection:
    "learn the modules" is not "learning module"). The clause is inferred only when the best family scores at least
    ``ACCEPT_SCORE``, leads the runner-up by at least ``MARGIN`` and rests on at least one distinctive object (weight >= 1).
    The candidate guides are the guides of that family whose own signals matched.

    Refused before any scoring: a clause that names a system (the D-111 path applies), names another platform (Zoom,
    Teams, Turnitin ...), names an organizational unit, place, role or contact (``ORGANIZATIONAL``) or asks for help or
    support, asks a policy, entitlement, obligation, permission, advice, misconduct or appeal question or a handbook
    subject, asks what someone other than the asker does (students, a TA, a secretary, a colleague), asks about the asker's
    or a colleague's own attendance or accommodation (or a task done for a colleague or scoped to one student), or asks
    anything but a procedure (a list ask names a set no heading names; a place, control, value, yes/no or definition ask
    needs a single statement, and a statement never answers an inferred clause). Single
    ambiguous words ("grades", "session", "report", "messages", "submit", "online", "collaborate", "record") carry no or
    too little weight to infer anything alone, and a signal is blocked in a context that gives its word another meaning
    (a research journal, a faculty meeting, a course file, housing, grades sent to students, a student's own work).

WHAT IT IS NOT
    Inference is never evidence. It only lets a clause be BOUND to a candidate guide (``guide_grounding.bind`` with
    ``inferred=True`` skips the system-name check): every word the clause asks still has to be stated by the guide text,
    and retrieval, procedure completeness, conflicts, source authority, the policy boundary, the specialist's approved
    sources and the adapter's independent re-verification decide the answer as for a clause that names its system. An
    inferred binding is held to a STRICTER evidence profile (``guide_grounding.inferred_problem``): only a complete
    procedure, section or labelled list item whose printed title names the asked action and every asked object answers
    it, never a single statement. The adapter recomputes the inference from the clause; nothing a specialist reports
    is trusted. No model call. The diagnostics hold signal keys, families, scores and source ids, never question text.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .guide_grounding import ANSWERABLE_INTENTS, ORGANIZATIONAL, GuideIntent, _HANDBOOK_SUBJECT, _STOP, _TOKEN, read_intent, stem
from .source_scopes import SYSTEM_TERMS, _POLICY_ASK

VERSION = "guide-inference-1"
PHASE = "PF-5C"

#: The minimum score of the inferred family.
ACCEPT_SCORE = 2.0
#: The minimum lead of the inferred family over the runner-up.
MARGIN = 1.5
#: Added once to a family when a task verb governs one of its matched objects.
ACTION_BONUS = 1.0
#: A task verb governs an object it precedes by at most this many content words ("send my final grades").
ACTION_WINDOW = 3
#: The words of a phrase are adjacent content words (stop words and the asker aside): "final exam grades" is not "final grades".
PHRASE_GAP = 0
#: A matched signal of at least this weight is a distinctive object; a family is never inferred from weaker context alone.
DISTINCTIVE = 1.0
#: The only ask inferred: a procedure ("how do I ..."), the only ask the strict evidence profile of an inferred binding
#: (``guide_grounding.inferred_problem``) can check against a printed heading; a list ask ("What can I add ...?") names a set no
#: heading names, and a place, control, value, yes/no or definition ask would need a single statement.
INFERABLE_INTENTS = frozenset({GuideIntent.PROCEDURE})

SHORT = "uos_blackboard_faculty_short_guide"
GRADEBOOK = "uos_blackboard_gradebook_guide"
ATTENDANCE = "uos_blackboard_attendance_guide"
COLLABORATE = "uos_blackboard_collaborate_guide"
COMMUNICATION = "uos_blackboard_communication_guide"
GRADE_TRANSFER = "uos_grade_transfer_dashboard_guide"
ITHENTICATE = "uos_ithenticate_guidelines"

#: System families and their guides (registry order). A family is what a clause is inferred to be about.
FAMILIES: Dict[str, Tuple[str, ...]] = {
    "blackboard": (SHORT, GRADEBOOK, ATTENDANCE, COLLABORATE, COMMUNICATION),
    "grade_transfer": (GRADE_TRANSFER,),
    "ithenticate": (ITHENTICATE,),
}
_FAMILY_OF = {source_id: family for family, ids in FAMILIES.items() for source_id in ids}

#: Words that never take part in a signal match: the D-111 stop words, the first-person asker and discourse adverbs
#: ("Finally, how do I send grades?" is not about final grades).
_FILLER = frozenset(_STOP) | frozenset("""
i me my mine myself we us our ours you your yours yourself please
finally lastly actually basically eventually really just simply also now currently
""".split())


def _surface(token: str) -> Optional[str]:
    """The printed form of a word for a phrase match: lower case, plural -s/-es/-ies removed; None for a possessive (a
    possessive never takes part in a phrase: "my child's courses" is not "child courses")."""
    w = token.lower().replace("’", "'")
    if w.endswith("'s") or w.endswith("s'"):
        return None
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[:-2].endswith(("s", "x", "ch", "sh")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


@dataclass(frozen=True)
class Signal:
    """One word or phrase printed in one or more guides of one family (``provenance`` quotes where it is printed)."""

    key: str                       # logged: "<family>.<name>"
    family: str
    guides: Tuple[str, ...]
    words: Tuple[str, ...]         # one word: its stem; a phrase: the printed (surface) words, in order
    weight: float
    actions: FrozenSet[str]        # stems of the task verbs that govern it
    provenance: str
    gap: int = PHRASE_GAP          # other words a phrase may span (a verb and its object: one modifier)

    @property
    def phrase(self) -> bool:
        return len(self.words) > 1


def _signal(key: str, guides: Sequence[str], phrase: str, weight: float, actions: str = "", provenance: str = "",
            gap: int = PHRASE_GAP) -> Signal:
    families = {_FAMILY_OF[g] for g in guides}
    assert len(families) == 1, key
    words = phrase.split()
    printed = tuple(_surface(w) or w for w in words) if len(words) > 1 else (stem(words[0]),)
    return Signal(key, families.pop(), tuple(guides), printed, weight, frozenset(stem(w) for w in actions.split()), provenance, gap)


_CONTENT_ACTIONS = "add upload create copy import reorder move hide show delete edit release organize"
_VIEW_ACTIONS = "view see access check open find"

#: The signal table. Every phrase is printed in the guide(s) it points to (a test checks each one against the guide text).
SIGNALS: Tuple[Signal, ...] = (
    # ---- Blackboard Ultra faculty short guide --------------------------------------------------------------------
    _signal("blackboard.course_availability", [SHORT], "course availability", 2.0, provenance="Course Availability: o Open ... o Private"),
    _signal("blackboard.course_private", [SHORT], "course private", 2.0, provenance="Make a course private while you add ..."),
    _signal("blackboard.course_image", [SHORT], "course image", 2.0, provenance="Course Image / Customize Your Course"),
    _signal("blackboard.banner_image", [SHORT], "banner image", 2.0, provenance="upload a course banner image"),
    _signal("blackboard.course_card", [SHORT], "course card", 2.0, provenance="the course card thumbnail"),
    _signal("blackboard.student_preview", [SHORT], "student preview", 2.0, provenance="With student preview, you can review ..."),
    _signal("blackboard.learning_module", [SHORT], "learning module", 2.0, provenance="Add a Learning Module or Folder"),
    _signal("blackboard.content_collection", [SHORT, GRADEBOOK], "content collection", 2.0, provenance="Content Collection"),
    _signal("blackboard.cloud_storage", [SHORT], "cloud storage", 2.0, provenance="Cloud Storage: Instantly connect ..."),
    _signal("blackboard.content_visibility", [SHORT], "content visibility", 2.0, provenance="Content Visibility"),
    _signal("blackboard.content", [SHORT], "content", 1.0, _CONTENT_ACTIONS, "Adding Contents / Create Content / Copy Content"),
    _signal("blackboard.visible", [SHORT, COMMUNICATION], "visible", 1.0, "make", "make it visible / make them visible"),
    _signal("blackboard.announcement", [SHORT], "announcement", 1.0, "send post create schedule edit copy delete make",
            "Send an Announcement"),
    _signal("blackboard.assignment", [SHORT], "assignment", 1.0, "create add make hide show randomize", "Create an Assignment"),
    _signal("blackboard.time_limit", [SHORT], "time limit", 1.0, "add set", "Add a time limit."),
    _signal("blackboard.roster", [SHORT], "roster", 1.0, _VIEW_ACTIONS, "Roster: View your class list."),
    _signal("blackboard.class_list", [SHORT], "class list", 2.0, provenance="Roster: View your class list."),
    _signal("blackboard.syllabus", [SHORT], "syllabus", 1.0, "add upload post", "To add a course content like syllabus ..."),
    _signal("blackboard.favorite", [SHORT], "favorite", 1.0, "mark", "mark one of your courses as a favorite"),
    _signal("blackboard.course", [SHORT], "course", 0.5, _CONTENT_ACTIONS, "context: the course a content action acts on"),
    _signal("blackboard.document", [SHORT], "document", 0.5, provenance="Add a Document (also an iThenticate word)"),
    # ---- Gradebook ---------------------------------------------------------------------------------------------------
    _signal("blackboard.gradable_item", [GRADEBOOK], "gradable item", 2.0, provenance="Gradable Items list"),
    _signal("blackboard.student_grid", [GRADEBOOK], "student grid", 2.0, provenance="Student Grid"),
    _signal("blackboard.global_grades", [GRADEBOOK], "global grades", 2.0, provenance="Global Grades Page"),
    _signal("blackboard.automatic_zero", [GRADEBOOK], "automatic zero", 2.0, provenance="Assign automatic zeros"),
    _signal("blackboard.zeros", [GRADEBOOK], "zero", 1.0, "assign", "assign zeros to missing work"),
    _signal("blackboard.grading_schema", [GRADEBOOK, ATTENDANCE], "grading schema", 2.0, provenance="the default grading schema"),
    _signal("blackboard.grade_schema", [GRADEBOOK, ATTENDANCE], "grade schema", 2.0, provenance="attendance Grade Schema"),
    _signal("blackboard.child_course", [GRADEBOOK], "child course", 2.0, provenance="Filter by child courses"),
    _signal("blackboard.merged_course", [GRADEBOOK], "merged course", 2.0, provenance="based on merged courses"),
    _signal("blackboard.accommodation", [GRADEBOOK, SHORT], "accommodation", 1.0, "add set create view",
            "Add accommodations from the Student list view"),
    _signal("blackboard.calculation", [GRADEBOOK], "calculation", 1.0, "add", "Add Calculation"),
    _signal("blackboard.overall_grade", [GRADEBOOK], "overall grade", 1.0, _VIEW_ACTIONS, "Overall Grade column"),
    _signal("blackboard.post_grades", [GRADEBOOK], "post grade", 2.0, provenance="Post all grades / grades you need to post", gap=1),
    _signal("blackboard.release_grades", [GRADEBOOK], "release grade", 2.0, provenance="ready to release grades to students", gap=1),
    _signal("blackboard.download_grades", [GRADEBOOK], "download grade", 2.0, provenance="Download and upload grades", gap=1),
    _signal("blackboard.upload_grades", [GRADEBOOK], "upload grade", 2.0, provenance="Upload & Download grades", gap=1),
    _signal("blackboard.insert_grades", [GRADEBOOK], "insert grade", 2.0, provenance="Insert grades & edit them", gap=1),
    # ---- Attendance --------------------------------------------------------------------------------------------------
    _signal("blackboard.attendance", [ATTENDANCE, COLLABORATE], "attendance", 1.0,
            "take mark record add track edit export share set change insert access check", "Mark attendance / Add Attendance"),
    _signal("blackboard.attendance_report", [ATTENDANCE, COLLABORATE], "attendance report", 2.0, provenance="Attendance Reporting"),
    _signal("blackboard.attendance_record", [ATTENDANCE, COLLABORATE], "attendance record", 2.0, provenance="access detailed records"),
    _signal("blackboard.attendance_sheet", [COLLABORATE], "attendance sheet", 2.0, provenance="export the attendance sheet"),
    _signal("blackboard.lateness", [ATTENDANCE], "lateness", 1.0, "set choose", "Choose the maximum allowed lateness"),
    # ---- Collaborate -------------------------------------------------------------------------------------------------
    _signal("blackboard.online_session", [COLLABORATE], "online session", 2.0, provenance="early entry for the online sessions"),
    _signal("blackboard.online_class", [COLLABORATE], "online class", 2.0, provenance="The online classes in UOS are weekly."),
    _signal("blackboard.virtual_session", [COLLABORATE], "virtual session", 2.0, provenance="online sessions (online/virtual, D-114)"),
    _signal("blackboard.virtual_class", [COLLABORATE], "virtual class", 2.0, provenance="online classes (online/virtual, D-114)"),
    _signal("blackboard.session", [COLLABORATE], "session", 0.5,
            "create schedule start join record end leave access manage set share edit", "Create New Scheduled Sessions"),
    # the stem of "recording" is the verb "record" ("Where do I record this?"): only printed phrases
    _signal("blackboard.start_recording", [COLLABORATE], "start recording", 2.0, provenance="start recording the session"),
    _signal("blackboard.stop_recording", [COLLABORATE], "stop recording", 2.0, provenance="click on stop recording"),
    _signal("blackboard.recording_download", [COLLABORATE], "recording download", 2.0, provenance="Allow recording downloads"),
    _signal("blackboard.downloading_recording", [COLLABORATE], "downloading recordings", 2.0, provenance="allow downloading the recordings"),
    _signal("blackboard.download_recording", [COLLABORATE], "download recordings", 2.0, provenance="allow downloading the recordings"),
    _signal("blackboard.record_session", [COLLABORATE], "record session", 2.0, provenance="start recording the session"),
    _signal("blackboard.breakout_group", [COLLABORATE], "breakout group", 2.0, provenance="breakout groups"),
    _signal("blackboard.whiteboard", [COLLABORATE], "whiteboard", 1.0, "open share use", "open a new whiteboard"),
    _signal("blackboard.poll", [COLLABORATE], "poll", 1.0, "start create", "Start a poll."),
    _signal("blackboard.moderator", [COLLABORATE], "moderator", 1.0, "make change add", "change any participant to moderator"),
    _signal("blackboard.early_entry", [COLLABORATE], "early entry", 2.0, provenance="allow or prevent the early entry"),
    _signal("blackboard.profanity_filter", [COLLABORATE], "profanity filter", 2.0, provenance="Profanity filter option"),
    _signal("blackboard.session_attendance", [COLLABORATE, ATTENDANCE], "session attendance", 2.0, provenance="Checking Session Attendance"),
    # ---- Communication tools -----------------------------------------------------------------------------------------
    _signal("blackboard.discussion_board", [COMMUNICATION], "discussion board", 2.0, provenance="How to create a discussion board?"),
    _signal("blackboard.discussion", [COMMUNICATION], "discussion", 1.0,
            "create add post delete grade hide show edit allow enable start", "Discussions / New Discussion page"),
    _signal("blackboard.class_conversation", [COMMUNICATION], "class conversation", 2.0, provenance="Allow class conversations."),
    _signal("blackboard.conversation", [COMMUNICATION], "conversation", 1.0, "enable allow access open turn", "Enable conversations"),
    _signal("blackboard.journal", [COMMUNICATION], "journal", 1.0, "create add grade show hide", "Create a journal"),
    _signal("blackboard.post_first", [COMMUNICATION], "post first", 2.0, provenance="Select Post first"),
    # ---- Grade Transfer ----------------------------------------------------------------------------------------------
    # "approve" is the Chair's and Dean's step, never the instructor's: it is no governing verb here
    _signal("grade_transfer.final_grades", [GRADE_TRANSFER], "final grade", 1.0, "send submit transfer upload move export push",
            "How to Send Grades to Banner as instructor? (the registry declares final grades)"),
    _signal("grade_transfer.grades", [GRADE_TRANSFER], "grade", 0.0, "send submit transfer", "How to Send Grades to Banner"),
    # ---- iThenticate -------------------------------------------------------------------------------------------------
    _signal("ithenticate.similarity", [ITHENTICATE], "similarity", 1.0, "check view see find access read get run",
            "Checking Similarity Report"),
    _signal("ithenticate.similarity_report", [ITHENTICATE], "similarity report", 2.0, provenance="view the similarity report"),
    _signal("ithenticate.similarity_index", [ITHENTICATE], "similarity index", 2.0, provenance="The similarity index"),
    _signal("ithenticate.plagiarism", [ITHENTICATE], "plagiarism", 1.0, "check detect scan test", "check the plagiarism"),
    _signal("ithenticate.originality_report", [ITHENTICATE], "originality report", 2.0, provenance="originality report"),
    _signal("ithenticate.originality", [ITHENTICATE], "originality", 1.0, "check ensure", "ensure the originality of written work"),
    _signal("ithenticate.text_matching", [ITHENTICATE], "text matching", 2.0, provenance="a text-matching tool"),
    _signal("ithenticate.exclude_resource", [ITHENTICATE], "exclude resource", 2.0, provenance="Exclude resource"),
    _signal("ithenticate.new_revision", [ITHENTICATE], "new revision", 2.0, provenance="Upload a New Revision"),
    _signal("ithenticate.document", [ITHENTICATE], "document", 0.5, provenance="How to upload a document (also a Blackboard word)"),
)

#: Another platform for the same tasks: a clause that names one is never inferred to be about an enabled guide.
OTHER_PLATFORM = re.compile(
    r"\b(?:zoom|webex|moodle|canvas|d2l|brightspace|turnitin|safe\s?assign|grammarly|copyleaks|urkund|ouriginal|plagscan|unicheck|"
    r"quetext|scribbr|crossref|kahoot|mentimeter|panopto|youtube|whatsapp|slack|respondus|proctorio|gradescope|piazza|edmodo|"
    r"schoology|padlet|nearpod|socrative|examsoft|outlook|gmail|google\s+(?:meet|classroom|forms?|docs|sheets|hangouts)|"
    r"(?:microsoft|ms)\s+(?:teams|forms))\b", re.I)
#: A request for help, support, training or troubleshooting: a person or an office is asked for, not a software step.
_SUPPORT_ASK = re.compile(
    r"\b(?:get|find|need|seek|ask\s+for|go\s+for|request)\s+(?:some\s+|more\s+|technical\s+)?(?:help|assistance|support|training|guidance)\b|"
    r"\bhelp\s+(?:with|on|for)\b|\bassist(?:ance)?\b|\btrainings?\b|\btutorials?\b|\bworkshops?\b|\btroubleshoot\w*|\bnot\s+working\b|"
    r"\bbroken\b|\berrors?\b|\bproblems?\b|\bissues?\b|\bcomplain\w*|"
    # a request for information about a thing ("Where can I find out more about ...?") asks for no step, place or control
    r"\bfind\s+out\b|\blearn\s+(?:more\s+)?about\b|\bknow\s+more\b|\bmore\s+(?:information|info|details)\b|\b(?:information|info|details)\s+(?:about|on)\b",
    re.I)
#: Permission, obligation, advice, appeal and misconduct asks: the handbook's questions, never a manual's.
_PERMISSION = re.compile(
    r"^\s*(?:must|shall|ought)\b|^\s*may\s+(?:i|we)\b|\bshould\b|\bought\s+to\b|"
    r"\bis\s+it\s+(?:ok|okay|alright|acceptable|fine|allowed|permitted|necessary|required|mandatory|compulsory|advisable|recommended|legal)\b|"
    r"\b(?:supposed|expected|obliged|responsible)\s+(?:to|for)\b|\bneed\s+to\b|\brequired\b|\bnecessary\b|\bacceptable\b|(?<!maximum\s)\ballowed\b|"
    r"\bpermitted\b|\brecommended\b|\bbest\s+practices?\b|\bappeal\w*|\bgrade\s+(?:changes?|disputes?|complaints?)\b|\bcopyright\w*|"
    r"\b(?:minimum|maximum)\s+(?:attendance|absences?|similarity|percentage)\b|"
    r"\b(?:attendance|absence|grading|similarity|plagiarism)\s+(?:requirements?|thresholds?|percentages?|limits?|rules?|criteria)\b|"
    r"\bcheat\w*|\bmisconduct\b|\bdishonest\w*|\bpenalt\w*|\bsanction\w*|"
    r"\breport\s+(?:a\s+|an\s+|the\s+)?(?:plagiarism|misconduct|cheating|case|incident|student)\b", re.I)
#: Someone other than the asker does the asked thing ("How do students submit ...?", "Can a TA take attendance?").
_OTHER_ACTOR = re.compile(
    r"\b(?:do|does|did|can|could|will|would|should|may|might|must|have|has)\s+(?:the\s+|my\s+|our\s+|a\s+|an\s+|all\s+|any\s+)?"
    r"(?:students?|learners?|tas?|teaching\s+assistants?|secretar(?:y|ies)|assistants?|colleagues?|parents?|guests?|participants?|"
    r"(?:another|other)\s+(?:instructors?|faculty|teachers?|professors?|lecturers?|members?)|guest\s+lecturers?)\b", re.I)
#: The asker's or a colleague's own attendance, absence or accommodation (HR, housing), not a course's.
_ROLE_OBJECT = re.compile(
    r"\b(?:my|our|faculty(?:\s+members?)?|instructors?|teachers?|professors?|lecturers?|employees?|staff|work|colleagues?)(?:'s|’s|s')?\s+"
    r"(?:own\s+)?(?:attendance|absences?|accommodations?)\b|"
    r"\b(?:attendance|absences?|accommodations?)\s+(?:records?\s+)?(?:of|for)\s+(?:the\s+)?(?:faculty|instructors?|teachers?|professors?|"
    r"lecturers?|employees?|staff|me|myself|us)\b|"
    # a task done for or to a colleague ("check similarity for a professor"), or scoped to one person ("to one student")
    r"\b(?:for|to|on\s+behalf\s+of)\s+(?:a\s+|an\s+|the\s+|my\s+|another\s+|other\s+|our\s+|all\s+|course\s+)?(?:professors?|instructors?|"
    r"colleagues?|faculty(?:\s+members?)?|teachers?|lecturers?|researchers?|co-?authors?)\b|"
    r"\b(?:one|a\s+single|single|specific|particular|individual|just\s+one|only\s+one|each)\s+(?:students?|learners?|persons?|members?|"
    r"participants?)\b|\b(?:to|for)\s+(?:a|one|this|that|a\s+specific|a\s+particular|each|certain)\s+(?:groups?|sections?)\b", re.I)
#: iThenticate is "used for university researchers": a clause about students' work is not inferred to be about it.
_STUDENT_WORK = re.compile(r"\b(?:students?|learners?|assignments?|homework|exams?|quiz(?:zes)?|class(?:es)?|courses?)\b", re.I)
#: A research journal is not the Blackboard journal tool.
_PUBLICATION = re.compile(r"\b(?:publish\w*|publications?|papers?|articles?|manuscripts?|research\w*|impact|indexed|scopus|"
                          r"submissions?|submit\w*|editors?|editorial|peer|rank\w*|q[1-4]|approved|lists?|club|predatory|cv|citations?|"
                          r"quartiles?|promotion|wos|web\s+of\s+science)\b", re.I)
#: Grades sent TO a person are not transferred to Banner (students see them in the Gradebook; instructors are not recipients).
_TO_STUDENTS = re.compile(r"\bto\s+(?:the\s+|my\s+|our\s+|all\s+|course\s+|the\s+course\s+)?(?:students?|learners?|class|instructors?|teachers?|"
                          r"professors?|lecturers?|faculty)\b|"
                          r"\b(?:send|give|show|release|email|e-mail)\s+(?:the\s+|my\s+|our\s+|all\s+)?(?:students?|learners?|class)\b", re.I)
#: A teaching assignment (a workload term) or a student's submission is not the instructor's Blackboard assignment.
_NOT_AN_ASSIGNMENT = re.compile(r"\bteaching\s+assignments?\b|\bassignment\s+submissions?\b|\bsubmit\w*\s+(?:an?\s+|the\s+|my\s+|their\s+)?assignments?\b",
                                re.I)
#: An event (a meeting, a committee, a training or an exam) is not a course session, its attendance or its discussion.
_EVENT = re.compile(r"\b(?:meetings?|committees?|councils?|conferences?|workshops?|seminars?|symposi\w*|webinars?|trainings?|orientations?|"
                    r"panel\s+discussions?|discussion\s+panels?|round\s?tables?|debates?|events?|ceremon\w*|defen[cs]es?|vivas?|exams?|"
                    r"examinations?|interviews?|elections?)\b", re.I)
_EVENT_SENSITIVE = frozenset("""
attendance attendance_report attendance_record attendance_sheet lateness session online_session online_class virtual_session virtual_class
session_attendance start_recording stop_recording record_session recording_download downloading_recording download_recording
breakout_group whiteboard poll moderator early_entry profanity_filter discussion discussion_board post_first
""".split())
#: Course administration (a course file, a proposal, an evaluation) is not Blackboard course content.
_COURSE_ADMIN = re.compile(r"\bcourse\s+(?:files?|portfolios?|reports?|specifications?|specs?|descriptions?|proposals?|codes?|equivalen\w*|"
                           r"evaluations?|registration|loads?)\b|\bnew\s+courses?\b|\bpropos\w+|\bcurricul\w*|\bprogram(?:me)?s?\b|\bstudy\s+plans?\b",
                           re.I)
#: An accommodation is a student's (exempt from due dates, time limits) only with a student or course word, never housing.
_ACCOMMODATION_CONTEXT = re.compile(r"\b(?:students?|learners?|course|class|assignments?|tests?|exams?|due\s+dates?|time\s+limits?)\b", re.I)
_HOUSING = re.compile(r"\b(?:housing|apartments?|hotels?|family|wife|husband|children|dependents?|rent|residence|living|flat)\b", re.I)


def _blocked(signal: Signal, text: str) -> bool:
    name = signal.key.split(".", 1)[1]
    if signal.family == "ithenticate" and _STUDENT_WORK.search(text):
        return True
    if name == "journal" and _PUBLICATION.search(text):
        return True
    if signal.family == "grade_transfer" and _TO_STUDENTS.search(text):
        return True
    if name == "assignment" and _NOT_AN_ASSIGNMENT.search(text):
        return True
    if name in _EVENT_SENSITIVE and _EVENT.search(text):
        return True
    if name in ("course", "content", "syllabus", "learning_module") and _COURSE_ADMIN.search(text):
        return True
    if name == "accommodation" and (not _ACCOMMODATION_CONTEXT.search(text) or _HOUSING.search(text)):
        return True
    return False


@dataclass(frozen=True)
class Inference:
    """The outcome of ``infer`` for one clause. ``accepted`` is True only when ``candidates`` may be bound."""

    accepted: bool
    reason: str                                 # a fixed code: accepted, explicit, other_platform, organizational, support, policy,
                                                # other_actor, role_object, intent:<intent>, no_signal, weak, ambiguous, no_object
    family: Optional[str] = None                # the best family (also when refused for weakness or ambiguity)
    candidates: Tuple[str, ...] = ()            # source ids, registry order; empty unless accepted
    scores: Tuple[Tuple[str, float], ...] = ()  # family -> score, best first
    runner_up: Optional[str] = None
    margin: Optional[float] = None
    signals: Tuple[str, ...] = ()               # signal keys matched for the best family
    governed: bool = False                      # a task verb governs one of them
    intent: Optional[str] = None

    @property
    def score(self) -> float:
        return self.scores[0][1] if self.scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable; signal keys, families, scores and source ids only (no question text)."""
        return {"version": VERSION, "accepted": self.accepted, "reason": self.reason, "family": self.family,
                "candidates": list(self.candidates), "score": self.score, "runner_up": self.runner_up,
                "runner_up_score": self.scores[1][1] if len(self.scores) > 1 else 0.0, "margin": self.margin,
                "signals": list(self.signals), "governed": self.governed, "intent": self.intent,
                "threshold": ACCEPT_SCORE, "required_margin": MARGIN}


def _content(text: str) -> List[Tuple[str, Optional[str]]]:
    """The content words of ``text``: (stem, printed form) pairs, stop words, the asker and discourse adverbs left out."""
    return [(stem(tok), _surface(tok)) for tok in _TOKEN.findall(text) if tok.lower() not in _FILLER]


def _match(signal: Signal, tokens: Sequence[Tuple[str, Optional[str]]]) -> Optional[int]:
    """The position of the first word of the first occurrence of ``signal`` in ``tokens``, or None: a word by its stem, a
    phrase by its printed words, in order, at most ``signal.gap`` other words apart (a possessive never matches)."""
    if not signal.phrase:
        return next((i for i, (s, _p) in enumerate(tokens) if s == signal.words[0]), None)
    printed = [p for _s, p in tokens]
    for start, word in enumerate(printed):
        if word != signal.words[0]:
            continue
        pos, ok = start, True
        for nxt_word in signal.words[1:]:
            nxt = next((j for j in range(pos + 1, min(len(printed), pos + 2 + signal.gap)) if printed[j] == nxt_word), None)
            if nxt is None:
                ok = False
                break
            pos = nxt
        if ok:
            return start
    return None


@functools.lru_cache(maxsize=4096)
def infer(clause: str) -> Inference:
    """The system ``clause`` is about, when it names none and the official guide vocabulary says so with confidence.
    Deterministic; never raises for ordinary input."""
    text = (clause or "").strip()
    if not text:
        return Inference(False, "no_signal")
    if any(pattern.search(text) for pattern in SYSTEM_TERMS.values()):
        return Inference(False, "explicit")
    if OTHER_PLATFORM.search(text):
        return Inference(False, "other_platform")
    if ORGANIZATIONAL.search(text):
        return Inference(False, "organizational")
    if _SUPPORT_ASK.search(text):
        return Inference(False, "support")
    if _POLICY_ASK.search(text) or _HANDBOOK_SUBJECT.search(text) or _PERMISSION.search(text):
        return Inference(False, "policy")
    if _OTHER_ACTOR.search(text):
        return Inference(False, "other_actor")
    if _ROLE_OBJECT.search(text):
        return Inference(False, "role_object")
    intent, _info = read_intent(text)
    if intent not in INFERABLE_INTENTS:
        return Inference(False, "intent:%s" % intent.value, intent=intent.value)
    tokens = _content(text)
    matched: Dict[str, List[Tuple[Signal, bool]]] = {}
    for signal in SIGNALS:
        if _blocked(signal, text):
            continue
        start = _match(signal, tokens)
        if start is None:
            continue
        governed = any(s in signal.actions for s, _p in tokens[max(0, start - ACTION_WINDOW):start])    # a task verb before it
        matched.setdefault(signal.family, []).append((signal, governed))
    if not matched:
        return Inference(False, "no_signal", intent=intent.value)
    scores = {family: sum(s.weight for s, _g in hits) + (ACTION_BONUS if any(g for _s, g in hits) else 0.0)
              for family, hits in matched.items()}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], list(FAMILIES).index(kv[0])))
    best, best_score = ranked[0]
    runner_up, second = (ranked[1] if len(ranked) > 1 else (None, 0.0))
    hits = matched[best]
    common = dict(family=best, scores=tuple((f, round(s, 2)) for f, s in ranked), runner_up=runner_up,
                  margin=round(best_score - second, 2), signals=tuple(s.key for s, _g in hits), governed=any(g for _s, g in hits),
                  intent=intent.value)
    if best_score < ACCEPT_SCORE:
        return Inference(False, "weak", **common)
    if best_score - second < MARGIN:
        return Inference(False, "ambiguous", **common)
    if not any(s.weight >= DISTINCTIVE for s, _g in hits):
        return Inference(False, "no_object", **common)
    guides = {g for s, _g in hits if s.weight > 0 for g in s.guides}
    candidates = tuple(g for g in FAMILIES[best] if g in guides)
    return Inference(True, "accepted", candidates=candidates, **common)


def inferred_sources(clause: str) -> Tuple[str, ...]:
    """The candidate guide ids of ``clause`` (empty unless the inference is accepted)."""
    return infer(clause or "").candidates


@dataclass(frozen=True)
class QuestionInference:
    """The inference of a whole question: the question itself and each of its clauses (``clauses`` as the Coordinator
    splits them). ``candidates`` is the union of every accepted clause's candidates."""

    accepted: bool
    candidates: Tuple[str, ...]
    parts: Tuple[Inference, ...]

    @property
    def lead(self) -> Inference:
        """The first accepted inference, else the whole question's."""
        return next((p for p in self.parts if p.accepted), self.parts[0])

    def to_dict(self) -> Dict[str, Any]:
        return {"accepted": self.accepted, "candidates": list(self.candidates), "lead": self.lead.to_dict(),
                "parts": [{"accepted": p.accepted, "reason": p.reason, "family": p.family, "candidates": list(p.candidates)}
                          for p in self.parts]}


def infer_question(text: str, clauses: Iterable[str] = ()) -> QuestionInference:
    """``infer`` on ``text`` and on each of ``clauses``."""
    parts = [infer(text or "")] + [infer(c) for c in clauses if isinstance(c, str) and c.strip() and c.strip() != (text or "").strip()]
    candidates: List[str] = []
    for part in parts:
        for source_id in part.candidates:
            if source_id not in candidates:
                candidates.append(source_id)
    order = [g for ids in FAMILIES.values() for g in ids]
    return QuestionInference(bool(candidates), tuple(sorted(candidates, key=order.index)), tuple(parts))


__all__ = ["ACCEPT_SCORE", "ACTION_BONUS", "FAMILIES", "Inference", "MARGIN", "OTHER_PLATFORM", "PHASE", "QuestionInference", "SIGNALS",
           "Signal", "VERSION", "infer", "infer_question", "inferred_sources"]
