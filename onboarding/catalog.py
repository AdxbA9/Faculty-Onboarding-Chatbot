"""
The curated onboarding catalog (post-freeze PF-14, D-125). Deterministic data, no model.

Every task, procedure step and resource carries its provenance: the approved source id, the physical page and one or
more EXACT quotes of indexed chunks of that page (``tests/test_onboarding_catalog.py`` re-checks every quote against the
corpus cache and the frozen evidence fixture). Nothing here is inferred: a task exists only where the approved documents
state the activity, a procedure only where the deterministic guide recovery (``guide_grounding.GuideIndex``) reads an
ordered procedure, a contact only where the Faculty Handbook prints it, an official link only after it was opened and
checked on the date recorded.

Kinds (what the source says about the activity):
    required     stated as a faculty responsibility or a condition the handbook imposes on new faculty
    recommended  a documented activity the guides or handbook describe for faculty, not stated as mandatory
    optional     explicitly optional or conditional in the source
    info         guidance to read; "completed" means "reviewed"

The eight-versus-nine question (Blackboard announcements): the original PDF page prints nine bulleted rows: the heading
"Send an Announcement:" and eight items. The procedure therefore has eight items; items 4 and 5 are conditional ("If you
want to ...") and item 8 describes editing, copying or deleting after posting. This module keeps all eight, flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CATALOG_VERSION = 1
CATEGORIES = ("teaching", "research", "faculty_services", "institutional")
KINDS = ("required", "recommended", "optional", "info")
STATUSES = ("not_started", "in_progress", "completed", "not_applicable")

HANDBOOK = "uos_faculty_handbook_2025_26"
SHORT_GUIDE = "uos_blackboard_faculty_short_guide"
ATTENDANCE = "uos_blackboard_attendance_guide"
GRADE_TRANSFER = "uos_grade_transfer_dashboard_guide"
ITHENTICATE = "uos_ithenticate_guidelines"
GRADEBOOK = "uos_blackboard_gradebook_guide"
COLLABORATE = "uos_blackboard_collaborate_guide"
COMMUNICATION = "uos_blackboard_communication_guide"
#: Registered but on ingestion hold: never listed, never cited, never linked (tests assert its absence).
HELD_SOURCES = ("uos_performance_appraisal_phase2_guide",)

SOURCE_TITLES = {
    HANDBOOK: "UOS Faculty Handbook 2025/26", SHORT_GUIDE: "Blackboard Ultra Faculty Short Guide", ATTENDANCE: "Attendance Tool in Blackboard Ultra",
    GRADE_TRANSFER: "Grade Transfer Dashboard Guide", ITHENTICATE: "iThenticate Guidelines", GRADEBOOK: "Blackboard Ultra Guide: Gradebook",
    COLLABORATE: "Blackboard Collaborate Ultra Guide", COMMUNICATION: "Blackboard Learn Ultra: Communication Tools",
}


@dataclass(frozen=True)
class Evidence:
    source_id: str
    page: int
    quote: str                      # exact substring of an indexed chunk of that page
    section: str = ""


@dataclass(frozen=True)
class Step:
    text: str                       # exact quote of the printed item (the step as the guide states it)
    page: int
    chunk_id: int                   # index of the chunk in the corpus cache that holds the quote
    label: str = ""                 # the printed label ("1.", "A)", "-" for a bullet)
    flag: str = ""                  # "" | "conditional" (the item opens with a condition) | "after" (what follows the procedure)


@dataclass(frozen=True)
class Procedure:
    id: str
    title_en: str
    title_ar: str
    source_id: str
    pages: Tuple[int, ...]
    heading: str                    # the heading as printed in the guide
    structure_key: str              # the key the deterministic recovery gives this list (or the keys, "+"-joined)
    steps: Tuple[Step, ...]
    specialist: str                 # the specialist that answers questions about this system (D-110 allow-list)
    conditions_en: str = ""
    conditions_ar: str = ""
    version: str = ""               # the guide's publication date from the source registry


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    kind: str
    title_en: str
    title_ar: str
    description_en: str
    description_ar: str
    evidence: Tuple[Evidence, ...]
    procedure_id: Optional[str] = None
    resources: Tuple[str, ...] = ()
    timing_en: str = ""
    timing_ar: str = ""
    order: int = 0


@dataclass(frozen=True)
class Contact:
    label: str                      # exactly as printed in the handbook contact table
    value: str                      # exactly as printed
    evidence: Evidence


@dataclass(frozen=True)
class Link:
    url: str
    kind: str                       # "official_website" | "document_link"
    label_en: str
    label_ar: str
    verified_on: str                # ISO date the URL was opened and its title read
    observed_title: str             # the page title observed on that date
    note_en: str = ""
    note_ar: str = ""


@dataclass(frozen=True)
class Resource:
    id: str
    type: str                       # "document" | "department" | "website"
    category: str
    title_en: str
    title_ar: str
    description_en: str
    description_ar: str
    keywords_en: Tuple[str, ...]
    keywords_ar: Tuple[str, ...]
    source_id: Optional[str] = None
    pages: Tuple[int, ...] = ()
    section: str = ""
    evidence: Tuple[Evidence, ...] = ()
    contacts: Tuple[Contact, ...] = ()
    links: Tuple[Link, ...] = ()
    related_tasks: Tuple[str, ...] = ()
    contact_note_en: str = ""
    contact_note_ar: str = ""


# =====================================================================================================================
# Procedures (steps copied verbatim from the deterministic recovery of the approved guides)
# =====================================================================================================================
PROCEDURES: Dict[str, Procedure] = {p.id: p for p in (
    Procedure(
        id="bb_access", title_en="Access Blackboard", title_ar="الدخول إلى بلاكبورد", source_id=SHORT_GUIDE, pages=(1,),
        heading="How to access Blackboard", structure_key="uos_blackboard_faculty_short_guide:L:1:1:0", specialist="teaching", version="2022-08-22",
        steps=(
            Step("Open the University website http://www.sharjah.ac.ae", 1, 13711, "-"),
            Step("Click Quick Links “MyUOS” , If you are connected to the University Network Domain, you will be directly logged in automatically.", 1, 13712, "-"),
            Step("If you are singing from outside university campus, click on Login from the main page.", 1, 13713, "-", "conditional"),
            Step("Click on Login.", 1, 13714, "-"),
            Step("Then sign in using your University of Sharjah email address and password.", 1, 13715, "-"),
            Step("You must verify your identity using the multifactor factor authentication service by receiving an SMS on the registered phone number or using Microsoft authenticator application.", 1, 13716, "-"),
            Step("Then you will be redirected to the main page of the blackboard.", 1, 13717, "-"),
        ),
        conditions_en="Step 3 applies only when you sign in from outside the university campus.",
        conditions_ar="تنطبق الخطوة 3 فقط عند تسجيل الدخول من خارج الحرم الجامعي.",
    ),
    Procedure(
        id="bb_announcement", title_en="Send an Announcement", title_ar="إرسال إعلان", source_id=SHORT_GUIDE, pages=(9,),
        heading="Send an Announcement", structure_key="uos_blackboard_faculty_short_guide:L:9:67:0", specialist="teaching", version="2022-08-22",
        steps=(
            Step("From details and actions click on create Announcements.", 9, 13978, "-"),
            Step("Click on the Plus sign in the top right of the page.", 9, 13978, "-"),
            Step("Add the Title, recipients and message of the announcement.", 9, 13978, "-"),
            Step("If you want to send a copy of this announcement via Email, check “Send an email copy to recipients” option.", 9, 13978, "-", "conditional"),
            Step("If you want to schedule the announce and control the visibility, check “schedule announcement” option and choose the start and end dates and time.", 9, 13978, "-", "conditional"),
            Step("Click on save – a draft of the announcement will be saved.", 9, 13978, "-"),
            Step("Click on Post so the announcement will reach students.", 9, 13978, "-"),
            Step("You can edit copy or delete announcement by going to the announcement page, then click on the “Three dots” icon, then you will find Edit, Copy and delete.", 9, 13978, "-", "after"),
        ),
        conditions_en="Items 4 and 5 are optional settings (an email copy; scheduling and visibility). Item 8 describes editing, copying or deleting an announcement after it exists.",
        conditions_ar="البندان 4 و5 إعدادان اختياريان (نسخة بالبريد الإلكتروني؛ الجدولة والظهور). يصف البند 8 تعديل الإعلان أو نسخه أو حذفه بعد إنشائه.",
    ),
    Procedure(
        id="bb_attendance_reports", title_en="Set your attendance reports", title_ar="ضبط تقارير الحضور", source_id=ATTENDANCE, pages=(2,),
        heading="Follow these steps to set your attendance reports", structure_key="uos_blackboard_attendance_guide:L:2:24:0+uos_blackboard_attendance_guide:L:2:26:2",
        specialist="teaching", version="2022-07-20",
        steps=(
            Step("1. Click on “Share Attendance information with LMS”", 2, 14632, "1."),
            Step("2. Choose the maximum allowed lateness", 2, 14633, "2."),
            Step("3. Insert the number of minutes after which student will be considered absent", 2, 14634, "3."),
            Step("4. Set your required time in the session (percentage)", 2, 14635, "4."),
        ),
        conditions_en="The guide's General Tips add: share attendance only for required sessions, not for sessions that are not for all students or office hours.",
        conditions_ar="تضيف النصائح العامة في الدليل: شارك الحضور للجلسات المطلوبة فقط، لا للجلسات غير الموجهة لجميع الطلبة ولا للساعات المكتبية.",
    ),
    Procedure(
        id="grade_transfer_access", title_en="Open the Grade Transfer Dashboard", title_ar="فتح لوحة نقل الدرجات", source_id=GRADE_TRANSFER, pages=(1,),
        heading="To access the Grade Transfer link follow these steps", structure_key="uos_grade_transfer_dashboard_guide:L:1:3:0", specialist="teaching", version="2024-11-21",
        steps=(
            Step("1. Log in to Blackboard and use your credentials to log in to Blackboard securely.", 1, 14116, "1."),
            Step("2. After logging in, select the Institution Page tab from the main navigation menu.", 1, 14117, "2."),
            Step("3. Look for the announcement containing the Grade Transfer link and click on it.", 1, 14118, "3."),
            Step("4. Once in the Grade Transfer Dashboard, select the option corresponding to your role and follow the on-screen instructions to complete the process.", 1, 14119, "4."),
            Step("5. Once logged in, navigate to the Grade Transfer section, where you’ll find the dashboard and its associated tools for managing the grade transfer process.", 1, 14120, "5."),
        ),
        conditions_en="The guide also prints a direct link to the dashboard (see the Grade Transfer Dashboard Guide resource).",
        conditions_ar="يطبع الدليل أيضًا رابطًا مباشرًا إلى اللوحة (انظر مورد دليل لوحة نقل الدرجات).",
    ),
    Procedure(
        id="grade_transfer_send", title_en="Send grades to Banner as instructor", title_ar="إرسال الدرجات إلى بانر بصفة مدرّس", source_id=GRADE_TRANSFER, pages=(2,),
        heading="How to Send Grades to Banner as instructor?", structure_key="uos_grade_transfer_dashboard_guide:L:2:9:0", specialist="teaching", version="2024-11-21",
        steps=(
            Step("1. Search for the course by using the search bar at the top of the dashboard to locate the specific course you wish to transfer. You can search by Course ID and Term.", 2, 14135, "1."),
            Step("2. Click on View Details on the dashboard", 2, 14136, "2."),
            Step("3. Click on Transfer:", 2, 14137, "3."),
            Step("4. You can click again in View Detail to see the grade under the Banner Column after the approval from Chair and Dean as below:", 2, 14123, "4.", "after"),
        ),
        conditions_en="Grades appear under the Banner column only after the Chair and the Dean approve the transfer (step 4 is a check, not an action).",
        conditions_ar="تظهر الدرجات تحت عمود بانر فقط بعد موافقة رئيس القسم والعميد على النقل (الخطوة 4 تحقق لا إجراء).",
    ),
    Procedure(
        id="ithenticate_access", title_en="Access iThenticate", title_ar="الدخول إلى iThenticate", source_id=ITHENTICATE, pages=(1,),
        heading="How to access IThenticate", structure_key="uos_ithenticate_guidelines:L:1:2:0", specialist="research", version="2022-04-25",
        steps=(
            Step("1) Open www.ithenticate.com", 1, 14468, "1)"),
            Step("2) Click on login", 1, 14468, "2)"),
            Step("3) Type your university email and your password (activation password is sent to you through ithenticate.com )", 1, 14468, "3)"),
        ),
        conditions_en="The guide names a support email for access problems; the finder shows it exactly as printed.",
        conditions_ar="يذكر الدليل بريدًا للدعم عند تعذّر الدخول؛ يعرضه الباحث كما هو مطبوع.",
    ),
    Procedure(
        id="ithenticate_upload", title_en="Upload a document to iThenticate", title_ar="رفع مستند إلى iThenticate", source_id=ITHENTICATE, pages=(1,),
        heading="How to upload a document", structure_key="uos_ithenticate_guidelines:L:1:6:0", specialist="research", version="2022-04-25",
        steps=(
            Step("1) Select the “Upload a File” link under the “Submit a Document” section on the home screen", 1, 14468, "1)"),
            Step("2) Enter in the author’s name and the document title, then click the “Choose a File” button in order to open your file", 1, 14468, "2)"),
            Step("3) Once you have opened your file, click “Upload” to upload your file", 1, 14468, "3)"),
            Step("4) If you will upload the same document but with an updated version use the following button and upload the updated document", 1, 14468, "4)", "conditional"),
        ),
        conditions_en="Step 4 applies only when you re-upload an updated version of the same document. The guide notes that the analysis may take a few hours.",
        conditions_ar="تنطبق الخطوة 4 فقط عند إعادة رفع نسخة محدّثة من المستند نفسه. يشير الدليل إلى أن التحليل قد يستغرق بضع ساعات.",
    ),
)}


# =====================================================================================================================
# Tasks
# =====================================================================================================================
TASKS: Dict[str, Task] = {t.id: t for t in (
    # ---- Teaching & Learning ---------------------------------------------------------------------------------------
    Task(
        id="bb_access", category="teaching", kind="recommended", order=10,
        title_en="Sign in to Blackboard Ultra", title_ar="تسجيل الدخول إلى بلاكبورد ألترا",
        description_en="Faculty members have LMS accounts with Instructor privileges; the Blackboard short guide explains how to reach Blackboard through MyUOS with your university email and multifactor authentication.",
        description_ar="لأعضاء هيئة التدريس حسابات في نظام إدارة التعلم بصلاحيات المدرّس؛ يشرح دليل بلاكبورد المختصر كيفية الوصول إلى بلاكبورد عبر MyUOS ببريدك الجامعي والمصادقة متعددة العوامل.",
        evidence=(Evidence(HANDBOOK, 193, "Faculty members have LMS accounts with Instructor privileges as long as they are signed up with the University services.", "10.6.1 LMS Account Management"),
                  Evidence(SHORT_GUIDE, 1, "Then sign in using your University of Sharjah email address and password.", "How to access Blackboard")),
        procedure_id="bb_access", resources=("doc_bb_short_guide", "web_elearning", "dept_itc"),
    ),
    Task(
        id="bb_announcement", category="teaching", kind="recommended", order=20,
        title_en="Send your first announcement to students", title_ar="إرسال أول إعلان إلى الطلبة",
        description_en="The Blackboard short guide gives the steps to create, save and post an announcement from a course's Details and Actions; an email copy and scheduling are optional settings.",
        description_ar="يقدّم دليل بلاكبورد المختصر خطوات إنشاء الإعلان وحفظه ونشره من «التفاصيل والإجراءات» في المقرر؛ النسخة بالبريد الإلكتروني والجدولة إعدادان اختياريان.",
        evidence=(Evidence(SHORT_GUIDE, 9, "Click on Post so the announcement will reach students.", "Send an Announcement"),),
        procedure_id="bb_announcement", resources=("doc_bb_short_guide", "doc_bb_communication"),
    ),
    Task(
        id="bb_attendance", category="teaching", kind="optional", order=30,
        title_en="Set up the Attendance tool for your sessions", title_ar="إعداد أداة الحضور لجلساتك",
        description_en="The Attendance guide describes sharing attendance with the LMS and setting lateness, absence and required-time thresholds; its tips say to share attendance only for required sessions.",
        description_ar="يصف دليل الحضور مشاركة الحضور مع نظام إدارة التعلم وضبط حدود التأخر والغياب والوقت المطلوب؛ وتنصح إرشاداته بمشاركة الحضور للجلسات المطلوبة فقط.",
        evidence=(Evidence(ATTENDANCE, 2, "Remember to share attendance only for required sessions.", "General Tips"),),
        procedure_id="bb_attendance_reports", resources=("doc_bb_attendance", "doc_bb_collaborate"),
    ),
    Task(
        id="grade_transfer", category="teaching", kind="recommended", order=40,
        title_en="Transfer final grades from Blackboard to Banner", title_ar="نقل الدرجات النهائية من بلاكبورد إلى بانر",
        description_en="The Grade Transfer Dashboard Guide gives the instructor's steps to open the dashboard and send course grades to Banner; the transfer is approved by the Chair and the Dean.",
        description_ar="يقدّم دليل لوحة نقل الدرجات خطوات المدرّس لفتح اللوحة وإرسال درجات المقرر إلى بانر؛ ويعتمد النقل رئيس القسم والعميد.",
        evidence=(Evidence(GRADE_TRANSFER, 1, "This guide provides simple steps for transferring grades from Blackboard to Banner.", "Grade Transfer Dashboard Guide"),),
        procedure_id="grade_transfer_send", resources=("doc_grade_transfer", "doc_bb_gradebook"),
        timing_en="At the end of a semester, when final grades are ready.", timing_ar="في نهاية الفصل الدراسي عندما تكون الدرجات النهائية جاهزة.",
    ),
    # ---- Research & Innovation -------------------------------------------------------------------------------------
    Task(
        id="ithenticate", category="research", kind="optional", order=50,
        title_en="Check a manuscript with iThenticate", title_ar="فحص مخطوطة باستخدام iThenticate",
        description_en="The iThenticate Guidelines explain how to sign in with your university email and upload a document for a similarity report.",
        description_ar="توضح إرشادات iThenticate كيفية تسجيل الدخول ببريدك الجامعي ورفع مستند للحصول على تقرير التشابه.",
        evidence=(Evidence(ITHENTICATE, 1, "Select the “Upload a File” link under the “Submit a Document” section on the home screen", "How to upload a document"),),
        procedure_id="ithenticate_upload", resources=("doc_ithenticate",),
    ),
    Task(
        id="seed_grant", category="research", kind="optional", order=60,
        title_en="Learn about the Seed Research Grant for newly appointed faculty", title_ar="التعرف على منحة البحث التأسيسية لأعضاء هيئة التدريس الجدد",
        description_en="The handbook lists a Seed Research Grant allocated to newly appointed faculty among the internal funding opportunities, and states when research grant applications can be submitted.",
        description_ar="يذكر الدليل منحة بحث تأسيسية مخصصة لأعضاء هيئة التدريس المعيّنين حديثًا ضمن فرص التمويل الداخلي، ويحدد متى يمكن تقديم طلبات المنح البحثية.",
        evidence=(Evidence(HANDBOOK, 154, "Seed Research Grant: Allocated to newly appointed faculty, this grant aims to kickstart their research endeavors at the University.", "8.2.1.1 Internal Funding"),
                  Evidence(HANDBOOK, 165, "Applications for all types of research grants can be submitted from the beginning of the first semester until the end of April of every academic year.", "Scientific Research Procedures and Regulations")),
        resources=("dept_research_funding", "doc_handbook", "web_research"),
        timing_en="Seed projects are announced at the beginning of each academic year (handbook, p. 165).", timing_ar="تُعلن المشاريع التأسيسية في بداية كل عام أكاديمي (الدليل، ص 165).",
    ),
    # ---- Faculty Services / HR -------------------------------------------------------------------------------------
    Task(
        id="hr_documents", category="faculty_services", kind="required", order=70,
        title_en="Complete the pre-joining document formalities with HR", title_ar="إتمام إجراءات المستندات مع الموارد البشرية قبل الالتحاق",
        description_en="HR compiles the documents for your employee file; documents must be attested in your home country for the residence visa, and new faculty must notify HR promptly of changes in personal data and documents.",
        description_ar="تجمع الموارد البشرية مستندات ملف الموظف؛ ويلزم تصديق المستندات في بلدك للحصول على إقامة، وعلى أعضاء هيئة التدريس الجدد إبلاغ الموارد البشرية فورًا بأي تغيير في البيانات الشخصية والمستندات.",
        evidence=(Evidence(HANDBOOK, 61, "Document authentication is required for obtaining residence visas in the UAE, and HR informs new faculty members of the documents that should be attested in their home country.", "2.2.3.8 Pre-joining Formalities"),
                  Evidence(HANDBOOK, 61, "New faculty members are informed to promptly notify HR of any changes in personal data and documents.", "2.2.3.8 Pre-joining Formalities")),
        resources=("dept_hr", "doc_handbook"),
    ),
    Task(
        id="probation_rules", category="faculty_services", kind="info", order=80,
        title_en="Review the probation and confirmation rules", title_ar="مراجعة قواعد فترة التجربة والتثبيت",
        description_en="The faculty probation period is one year; performance is evaluated at the end of the first contract year, and written notification of confirmation or termination is due six months before the contract ends.",
        description_ar="فترة التجربة لعضو هيئة التدريس سنة واحدة؛ يُقيَّم الأداء في نهاية السنة الأولى من العقد، ويُشترط إشعار كتابي بالتثبيت أو الإنهاء قبل ستة أشهر من انتهاء العقد.",
        evidence=(Evidence(HANDBOOK, 63, "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval.", "2.4 Probation and Resignation Notice"),
                  Evidence(HANDBOOK, 63, "New faculty members must receive written notification of employment confirmation or termination six months before the end of the contract.", "2.5 Confirmation")),
        resources=("dept_hr", "doc_handbook"),
    ),
    Task(
        id="leave_entitlements", category="faculty_services", kind="info", order=90,
        title_en="Review your leave entitlements", title_ar="مراجعة استحقاقات الإجازات",
        description_en="Chapter 4 of the handbook states the vacation, sick, maternity, unpaid special and Hajj leave rules for faculty.",
        description_ar="يحدد الفصل الرابع من الدليل قواعد الإجازة السنوية والمرضية وإجازة الأمومة والإجازة الخاصة بدون راتب وإجازة الحج لأعضاء هيئة التدريس.",
        evidence=(Evidence(HANDBOOK, 107, "Faculty are entitled to sick leave with full pay for up to 15 working days and half pay for an additional 15 working days per contract year.", "4.3.2 Sick Leave"),),
        resources=("dept_hr", "doc_handbook"),
    ),
    # ---- Institutional Navigation ----------------------------------------------------------------------------------
    Task(
        id="ilhe_training", category="institutional", kind="required", order=100,
        title_en="Attend the New Faculty Development Training (ILHE)", title_ar="حضور برنامج تدريب أعضاء هيئة التدريس الجدد (ILHE)",
        description_en="Attending the orientation and training programmes of the Institute of Leadership in Higher Education is listed among the development activities faculty are responsible for engaging in.",
        description_ar="يُدرج حضور برامج التهيئة والتدريب التي يقدمها معهد القيادة في التعليم العالي ضمن أنشطة التطوير التي يتحمل أعضاء هيئة التدريس مسؤولية المشاركة فيها.",
        evidence=(Evidence(HANDBOOK, 83, "Attend orientation and training programs covering pedagogy, technology, research skills, and institutional policies.", "3.7 Professional Development Responsibilities"),
                  Evidence(HANDBOOK, 113, "the Institute of Leadership in Higher Education (ILHE) offers a comprehensive training program tailored for new faculty.", "5.2 New Faculty Development Training")),
        resources=("dept_ilhe", "doc_handbook"),
    ),
    Task(
        id="contacts", category="institutional", kind="info", order=110,
        title_en="Save the key university contact numbers", title_ar="حفظ أرقام الاتصال الرئيسية في الجامعة",
        description_en="The handbook's contact table lists the telephone numbers of offices such as Human Resources, the Information Technology Center and the Finance Department.",
        description_ar="يسرد جدول الاتصال في الدليل أرقام هواتف مكاتب مثل الموارد البشرية ومركز تقنية المعلومات وإدارة المالية.",
        evidence=(Evidence(HANDBOOK, 54, "Human Resources +971 6 5050023 +971 6 5585200", "Office / College / Department Telephone Number"),),
        resources=("dept_hr", "dept_itc", "dept_finance", "dept_libraries"),
    ),
    Task(
        id="lms_policy", category="institutional", kind="info", order=120,
        title_en="Review the LMS account and course management policy", title_ar="مراجعة سياسة حسابات نظام إدارة التعلم وإدارة المقررات",
        description_en="The handbook states that LMS courses are generated from the Student Information System schedules and that instructors may nominate teaching assistants while remaining the primary course instructor.",
        description_ar="ينص الدليل على أن مقررات نظام إدارة التعلم تُنشأ من جداول نظام معلومات الطلبة، وأن المدرّسين يمكنهم ترشيح مساعدي تدريس مع بقائهم المدرّس الرئيسي للمقرر.",
        evidence=(Evidence(HANDBOOK, 193, "Courses in LMS will be generated based on the course schedules provided by the Student Information System.", "10.6.3 Course Management"),
                  Evidence(HANDBOOK, 193, "Instructors may nominate additional teaching assistants for their LMS course upon request, but they retain responsibility as the primary course instructor.", "10.6.1 LMS Account Management")),
        resources=("doc_handbook", "doc_bb_short_guide"),
    ),
)}


# =====================================================================================================================
# Resources: approved documents, departments (contacts only as printed), official links (each checked individually)
# =====================================================================================================================
def _contact(label: str, value: str, page: int = 54) -> Contact:
    return Contact(label, value, Evidence(HANDBOOK, page, "%s %s" % (label, value), "Office / College / Department Telephone Number"))


VERIFIED_ON = "2026-11-09"
WEB_UOS = Link("https://www.sharjah.ac.ae/en/", "official_website", "University of Sharjah website", "موقع جامعة الشارقة", VERIFIED_ON, "University of Sharjah",
               "Official website; outside the approved document corpus.", "الموقع الرسمي؛ خارج مجموعة المستندات المعتمدة.")
WEB_RESEARCH = Link("https://www.sharjah.ac.ae/en/Research", "official_website", "Research at the University of Sharjah", "البحث العلمي في جامعة الشارقة", VERIFIED_ON, "Research",
                    "Official website; outside the approved document corpus.", "الموقع الرسمي؛ خارج مجموعة المستندات المعتمدة.")
WEB_ELEARNING = Link("https://elearning.sharjah.ac.ae/", "official_website", "Blackboard (eLearning) sign-in", "تسجيل الدخول إلى بلاكبورد (التعلم الإلكتروني)", VERIFIED_ON, "Sign in to your account",
                     "Redirects to the university's Microsoft sign-in; outside the approved document corpus.", "يحوّل إلى صفحة تسجيل دخول مايكروسوفت الخاصة بالجامعة؛ خارج مجموعة المستندات المعتمدة.")
WEB_LIBRARY = Link("https://library.sharjah.ac.ae/", "official_website", "Library catalogue", "فهرس المكتبة", VERIFIED_ON, "Millennium Web Catalog",
                   "Official library catalogue; outside the approved document corpus.", "فهرس المكتبة الرسمي؛ خارج مجموعة المستندات المعتمدة.")
#: Printed inside an approved guide (Grade Transfer Dashboard Guide, p. 3): a document link, not an independently verified website.
LINK_GRADE_TRANSFER = Link("https://op.sharjah.ac.ae/GradeTransfer", "document_link", "Grade Transfer Portal (link printed in the guide)", "بوابة نقل الدرجات (رابط مطبوع في الدليل)", VERIFIED_ON,
                           "printed in the Grade Transfer Dashboard Guide, page 3", "As printed in the approved guide; reachability from outside the campus was not confirmed.",
                           "كما هو مطبوع في الدليل المعتمد؛ لم يُتحقق من إمكانية الوصول من خارج الحرم الجامعي.")

RESOURCES: Dict[str, Resource] = {r.id: r for r in (
    # ---- documents (the approved corpus) -----------------------------------------------------------------------------
    Resource("doc_handbook", "document", "institutional", "UOS Faculty Handbook 2025/26", "دليل أعضاء هيئة التدريس 2025/2026",
             "The primary approved document: the university, appointment, responsibilities, benefits, development, evaluation, promotion, research, IT, library, admission and student policies.",
             "المستند المعتمد الرئيسي: الجامعة والتعيين والمسؤوليات والمزايا والتطوير والتقييم والترقية والبحث العلمي وتقنية المعلومات والمكتبة وسياسات القبول والطلبة.",
             ("faculty handbook", "handbook", "policy", "policies", "appointment", "benefits", "leave", "probation", "promotion", "research", "orientation"),
             ("دليل", "دليل أعضاء هيئة التدريس", "سياسة", "سياسات", "تعيين", "مزايا", "إجازة", "ترقية", "بحث", "تهيئة"),
             source_id=HANDBOOK, pages=(1, 272), related_tasks=("hr_documents", "probation_rules", "leave_entitlements", "ilhe_training", "seed_grant", "lms_policy", "contacts")),
    Resource("doc_bb_short_guide", "document", "teaching", "Blackboard Ultra Faculty Short Guide", "دليل بلاكبورد ألترا المختصر لأعضاء هيئة التدريس",
             "ITC Academic Computing guide: access, courses, course view and availability, adding content, content visibility and announcements.",
             "دليل الحوسبة الأكاديمية في مركز تقنية المعلومات: الدخول، المقررات، عرض المقرر وإتاحته، إضافة المحتوى، ظهور المحتوى، والإعلانات.",
             ("blackboard", "blackboard help", "ultra", "lms", "announcement", "course", "content", "access", "login", "myuos"),
             ("بلاكبورد", "بلاك بورد", "إعلان", "مقرر", "محتوى", "دخول", "تسجيل الدخول"),
             source_id=SHORT_GUIDE, pages=(1, 11), related_tasks=("bb_access", "bb_announcement", "lms_policy")),
    Resource("doc_bb_communication", "document", "teaching", "Blackboard Learn Ultra: Communication Tools", "أدوات التواصل في بلاكبورد ليرن ألترا",
             "Discussions, conversations, messages and journals in Blackboard Learn Ultra.", "المناقشات والمحادثات والرسائل واليوميات في بلاكبورد ليرن ألترا.",
             ("blackboard", "communication", "discussion", "messages", "journal", "conversations"), ("بلاكبورد", "تواصل", "مناقشة", "رسائل", "يوميات"),
             source_id=COMMUNICATION, pages=(1, 5), related_tasks=("bb_announcement",)),
    Resource("doc_bb_attendance", "document", "teaching", "Attendance Tool in Blackboard Ultra", "أداة الحضور في بلاكبورد ألترا",
             "Attendance settings, lateness and absence thresholds, attendance reports and tips.", "إعدادات الحضور وحدود التأخر والغياب وتقارير الحضور ونصائح.",
             ("attendance", "blackboard", "absence", "late", "reports"), ("حضور", "غياب", "تأخر", "تقارير", "بلاكبورد"),
             source_id=ATTENDANCE, pages=(1, 2), related_tasks=("bb_attendance",)),
    Resource("doc_bb_collaborate", "document", "teaching", "Blackboard Collaborate Ultra Guide", "دليل بلاكبورد كولابوريت ألترا",
             "Scheduled sessions, sharing content and checking session attendance in Collaborate Ultra.", "الجلسات المجدولة ومشاركة المحتوى والتحقق من حضور الجلسات في كولابوريت ألترا.",
             ("collaborate", "virtual class", "online session", "blackboard", "session attendance"), ("كولابوريت", "جلسة", "فصل افتراضي", "بلاكبورد"),
             source_id=COLLABORATE, pages=(1, 4), related_tasks=("bb_attendance",)),
    Resource("doc_bb_gradebook", "document", "teaching", "Blackboard Ultra Guide: Gradebook", "دليل بلاكبورد ألترا: سجل الدرجات",
             "Gradebook views, grading items, posting grades, settings, rows and columns, download and upload of grades.", "عروض سجل الدرجات وعناصر التقييم ونشر الدرجات والإعدادات والصفوف والأعمدة وتنزيل الدرجات ورفعها.",
             ("gradebook", "grades", "grading", "blackboard", "post grades"), ("سجل الدرجات", "درجات", "تقييم", "بلاكبورد"),
             source_id=GRADEBOOK, pages=(1, 10), related_tasks=("grade_transfer",)),
    Resource("doc_grade_transfer", "document", "teaching", "Grade Transfer Dashboard Guide", "دليل لوحة نقل الدرجات",
             "Transferring grades from Blackboard to Banner: access, sending grades as instructor, approving as Chair or Dean.",
             "نقل الدرجات من بلاكبورد إلى بانر: الوصول، إرسال الدرجات بصفة مدرّس، والاعتماد بصفة رئيس قسم أو عميد.",
             ("grade transfer", "banner", "final grades", "grades", "dashboard"), ("نقل الدرجات", "بانر", "الدرجات النهائية", "لوحة"),
             source_id=GRADE_TRANSFER, pages=(1, 3), links=(LINK_GRADE_TRANSFER,), related_tasks=("grade_transfer",)),
    Resource("doc_ithenticate", "document", "research", "iThenticate Guidelines", "إرشادات iThenticate",
             "Accessing iThenticate, uploading a document, reading the similarity report and file requirements. (The yearly submission limit is a registered unresolved conflict between the English and Arabic pages and is not stated here.)",
             "الدخول إلى iThenticate ورفع مستند وقراءة تقرير التشابه ومتطلبات الملفات. (الحد السنوي للإرسال تعارض مسجَّل غير محسوم بين الصفحتين الإنجليزية والعربية ولا يُذكر هنا.)",
             ("ithenticate", "plagiarism", "similarity", "manuscript", "research"), ("انتحال", "تشابه", "مخطوطة", "بحث"),
             source_id=ITHENTICATE, pages=(1, 1), related_tasks=("ithenticate",)),
    # ---- departments and units (contacts exactly as printed in the handbook; none invented) -----------------------
    Resource("dept_hr", "department", "faculty_services", "Human Resources", "الموارد البشرية",
             "Recruitment, pre-joining formalities, contracts, probation and confirmation, leaves and benefits (handbook chapters 2 and 4).",
             "التوظيف وإجراءات ما قبل الالتحاق والعقود وفترة التجربة والتثبيت والإجازات والمزايا (الفصلان 2 و4 من الدليل).",
             ("human resources", "hr", "hr services", "contract", "visa", "probation", "leave", "salary"), ("الموارد البشرية", "خدمات الموارد البشرية", "عقد", "إقامة", "إجازة", "راتب"),
             source_id=HANDBOOK, pages=(54, 61, 63, 107), contacts=(_contact("Human Resources", "+971 6 5050023 +971 6 5585200"),),
             evidence=(Evidence(HANDBOOK, 61, "HR compiles all required documents and information to maintain in the Employee File.", "2.2.3.8 Pre-joining Formalities"),),
             related_tasks=("hr_documents", "probation_rules", "leave_entitlements", "contacts")),
    Resource("dept_itc", "department", "institutional", "Information Technology Center", "مركز تقنية المعلومات",
             "IT services; its Academic Computing unit publishes the Blackboard, Grade Transfer and iThenticate guides.",
             "خدمات تقنية المعلومات؛ تنشر وحدة الحوسبة الأكاديمية فيه أدلة بلاكبورد ونقل الدرجات وiThenticate.",
             ("information technology", "itc", "it", "it support", "helpdesk", "blackboard support", "email"), ("تقنية المعلومات", "مركز تقنية المعلومات", "دعم فني", "بريد"),
             source_id=HANDBOOK, pages=(54,), contacts=(_contact("Information Technology Center", "+971 6 5050000 +971 6 5585101"),), related_tasks=("bb_access", "contacts")),
    Resource("dept_finance", "department", "faculty_services", "Finance Department", "إدارة المالية",
             "Telephone contact as printed in the handbook's contact table.", "رقم الهاتف كما هو مطبوع في جدول الاتصال في الدليل.",
             ("finance", "payroll", "finance department"), ("المالية", "إدارة المالية", "رواتب"),
             source_id=HANDBOOK, pages=(54,), contacts=(_contact("Finance Department", "+971 6 5050040 +971 6 5585183"),), related_tasks=("contacts",)),
    Resource("dept_cgsr", "department", "research", "College of Graduate Studies and Research", "كلية الدراسات العليا والبحث العلمي",
             "Telephone contact as printed in the handbook's contact table.", "رقم الهاتف كما هو مطبوع في جدول الاتصال في الدليل.",
             ("graduate studies", "research college", "cgsr", "research"), ("الدراسات العليا", "البحث العلمي", "كلية الدراسات العليا"),
             source_id=HANDBOOK, pages=(54,), contacts=(_contact("College of Graduate Studies and Research", "+971 6 5050550 +971 6 5050552"),), related_tasks=("seed_grant",)),
    Resource("dept_libraries", "department", "institutional", "University Libraries", "مكتبات الجامعة",
             "Library telephone numbers as printed in the handbook's contact table; the online catalogue is linked below.",
             "أرقام هواتف المكتبات كما هي مطبوعة في جدول الاتصال في الدليل؛ الفهرس الإلكتروني مرتبط أدناه.",
             ("library", "libraries", "books", "journals", "databases"), ("مكتبة", "مكتبات", "كتب", "دوريات", "قواعد بيانات"),
             source_id=HANDBOOK, pages=(54, 246), contacts=(_contact("Men’s Library:", "+971 6 5050245"), _contact("Women’s Library:", "+971 6 5050673")),
             links=(WEB_LIBRARY,), related_tasks=("contacts",)),
    Resource("dept_ilhe", "department", "institutional", "Institute of Leadership in Higher Education (ILHE)", "معهد القيادة في التعليم العالي (ILHE)",
             "Offers the New Faculty Development Training programme, mentoring and peer learning initiatives (handbook 5.2).",
             "يقدم برنامج تدريب أعضاء هيئة التدريس الجدد ومبادرات الإرشاد والتعلم بين الأقران (الدليل 5.2).",
             ("ilhe", "orientation", "orientation information", "training", "new faculty", "development", "mentoring", "induction"), ("تهيئة", "تدريب", "أعضاء هيئة التدريس الجدد", "تطوير", "إرشاد", "معهد القيادة"),
             source_id=HANDBOOK, pages=(113, 114, 83), evidence=(Evidence(HANDBOOK, 113, "the Institute of Leadership in Higher Education (ILHE) offers a comprehensive training program tailored for new faculty.", "5.2 New Faculty Development Training"),),
             related_tasks=("ilhe_training",), contact_note_en="The approved documents print no telephone number or email for ILHE.", contact_note_ar="لا تطبع المستندات المعتمدة رقم هاتف أو بريدًا إلكترونيًا لمعهد ILHE."),
    Resource("dept_research_funding", "department", "research", "Research Funding Department", "إدارة تمويل البحث العلمي",
             "Pre- and post-award administration of research funds, grant-writing assistance and the internal funding schemes (handbook 8.2.1).",
             "إدارة تمويل البحث قبل المنح وبعدها والمساعدة في كتابة المنح وبرامج التمويل الداخلي (الدليل 8.2.1).",
             ("research funding", "funding", "grant", "grants", "seed grant", "rfd"), ("تمويل البحث", "تمويل", "منحة", "منح", "منحة تأسيسية"),
             source_id=HANDBOOK, pages=(154, 155, 165), evidence=(Evidence(HANDBOOK, 154, "The Research Funding Department (RFD) is responsible for pre- and post-awards administration of research project's funds.", "8.2.1 Research Funding Department"),),
             links=(WEB_RESEARCH,), related_tasks=("seed_grant",), contact_note_en="The approved documents print no telephone number or email for this department.", contact_note_ar="لا تطبع المستندات المعتمدة رقم هاتف أو بريدًا إلكترونيًا لهذه الإدارة."),
    Resource("dept_tto", "department", "research", "Technology Transfer Office", "مكتب نقل التكنولوجيا",
             "Manages intellectual property matters related to research activities and outputs (handbook 8.2.3).", "يدير شؤون الملكية الفكرية المتعلقة بالأنشطة والمخرجات البحثية (الدليل 8.2.3).",
             ("technology transfer", "intellectual property", "patent", "licensing", "tto"), ("نقل التكنولوجيا", "الملكية الفكرية", "براءة", "ترخيص"),
             source_id=HANDBOOK, pages=(155,), evidence=(Evidence(HANDBOOK, 155, "The Technology Transfer Office (TTO) is entrusted with managing all intellectual property matters related to research activities and outputs at UoS.", "8.2.3 Technology Transfer Office"),),
             related_tasks=(), contact_note_en="The approved documents print no telephone number or email for this office.", contact_note_ar="لا تطبع المستندات المعتمدة رقم هاتف أو بريدًا إلكترونيًا لهذا المكتب."),
    # ---- official websites (checked individually on the date recorded; outside the RAG corpus) ---------------------
    Resource("web_uos", "website", "institutional", "University of Sharjah website", "موقع جامعة الشارقة",
             "The university's official public website.", "الموقع العام الرسمي للجامعة.",
             ("website", "university website", "sharjah", "uos", "homepage"), ("موقع", "موقع الجامعة", "جامعة الشارقة"),
             links=(WEB_UOS,), related_tasks=("bb_access",)),
    Resource("web_elearning", "website", "teaching", "Blackboard (eLearning) sign-in", "تسجيل الدخول إلى بلاكبورد",
             "The eLearning address redirects to the university's Microsoft sign-in page.", "يحوّل عنوان التعلم الإلكتروني إلى صفحة تسجيل دخول مايكروسوفت الخاصة بالجامعة.",
             ("blackboard login", "elearning", "sign in", "login", "lms"), ("تسجيل الدخول", "التعلم الإلكتروني", "بلاكبورد"),
             links=(WEB_ELEARNING,), related_tasks=("bb_access",)),
    Resource("web_research", "website", "research", "Research at the University of Sharjah", "البحث العلمي في جامعة الشارقة",
             "The research section of the official website.", "قسم البحث العلمي في الموقع الرسمي.",
             ("research", "research website", "research office"), ("بحث", "البحث العلمي", "موقع البحث"),
             links=(WEB_RESEARCH,), related_tasks=("seed_grant",)),
)}


# =====================================================================================================================
# Lookups
# =====================================================================================================================
def task(task_id: str) -> Optional[Task]:
    return TASKS.get(str(task_id or ""))


def procedure(procedure_id: str) -> Optional[Procedure]:
    return PROCEDURES.get(str(procedure_id or ""))


def procedure_for_task(task_id: str) -> Optional[Procedure]:
    t = task(task_id)
    return PROCEDURES.get(t.procedure_id) if t and t.procedure_id else None


def resource(resource_id: str) -> Optional[Resource]:
    return RESOURCES.get(str(resource_id or ""))


def tasks_by_category(category: Optional[str] = None) -> List[Task]:
    items = [t for t in TASKS.values() if category in (None, "", "all") or t.category == category]
    return sorted(items, key=lambda t: t.order)


def resources_for_task(task_id: str) -> List[Resource]:
    t = task(task_id)
    if t is None:
        return []
    chosen = [RESOURCES[r] for r in t.resources if r in RESOURCES]
    for r in RESOURCES.values():
        if task_id in r.related_tasks and r not in chosen:
            chosen.append(r)
    return chosen


__all__ = ["CATALOG_VERSION", "CATEGORIES", "Contact", "Evidence", "HELD_SOURCES", "KINDS", "Link", "PROCEDURES", "Procedure", "RESOURCES", "Resource",
           "SOURCE_TITLES", "STATUSES", "Step", "TASKS", "Task", "procedure", "procedure_for_task", "resource", "resources_for_task", "task", "tasks_by_category"]
