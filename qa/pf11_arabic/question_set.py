"""The PF-11 multilingual evaluation set: Arabic questions paired with the English questions of the frozen I-8 master set
(``qa/plan_f_i8/question_set.py``) whose expected facts, pages and behaviour were established there. Every entry records
the information need in both languages, the specialist domain it belongs to, the expected source pages (handbook unless a
``source`` says otherwise), the expected behaviour of a correct runtime (``serve`` / ``fallback`` / ``either``) and the
evaluation split.

Splits
    dev      used while the bilingual lexicon was written and corrected
    holdout  written AFTER the lexicon was frozen for the dev run, with different wording; run once and reported apart

Expected pages are printed pages of the UOS Faculty Handbook 2025/26 unless ``source`` names another registered source.
``pages: None`` marks a case whose exact page was not established by an earlier evaluation; such cases are scored on
behaviour and source only and are marked "for review" in the reports. Nothing here is ground truth invented for this
phase: facts come from the frozen set and from the PF-9 live acceptance records.
"""

HANDBOOK = "uos_faculty_handbook_2025_26"
SHORT_GUIDE = "uos_blackboard_faculty_short_guide"


def _q(qid, pair, domain, category, ar, en, expect, pages, fact, split="dev", source=HANDBOOK, specialists=None, mixed=False):
    return {"id": qid, "pair": pair, "domain": domain, "category": category, "ar": ar, "en": en, "expect": expect,
            "pages": pages, "fact": fact, "split": split, "source": source,
            "specialists": specialists if specialists is not None else ([domain] if domain in ("teaching", "research", "faculty_services", "institutional") else []),
            "mixed": mixed}


QUESTIONS = [
    # ---- Teaching & Learning -----------------------------------------------------------------------------------------
    _q("AR-T01", "T01", "teaching", "numeric", "كم عدد الساعات المعتمدة للعبء التدريسي الكامل؟", "How many credit hours is a full teaching load?", "serve", [91, 96], "12 credit hours per regular semester"),
    _q("AR-T02", "T02", "teaching", "numeric", "كم عدد الساعات المكتبية التي يجب على أعضاء هيئة التدريس الالتزام بها؟", "How many office hours must faculty hold?", "serve", [79], "minimum five office hours per week over two days"),
    _q("AR-T03", "T03", "teaching", "direct", "ماذا يجب على المدرسين فعله بأوراق إجابة الامتحان النهائي؟", "What must instructors do with final examination answer sheets?", "serve", [224], "correct and submit results to the Department Chair within 48 hours"),
    _q("AR-T04", "T04", "teaching", "numeric", "ما هو العبء التدريسي للمحاضرين؟", "What is the teaching load for lecturers?", "serve", [96], "15 credit hours per semester"),
    _q("AR-T05", "T05", "teaching", "numeric", "ما هي نسبة الغياب التي تحرم الطالب من دخول الامتحان النهائي؟", "What percentage of absences bars a student from the final examination?", "serve", [228], "20%"),
    _q("AR-T07", "T07", "teaching", "conditional", "هل الحضور إلزامي للطلبة؟", "Is attendance mandatory for students?", "either", [228], "absences reaching 20% bar the final examination"),
    _q("AR-T08", "T08", "teaching", "direct", "ما هي سياسة الامتحانات التعويضية؟", "What is the policy on make-up examinations?", "fallback", [], "no make-up examination policy statement in the handbook"),
    _q("AR-T09", "T09", "teaching", "numeric", "ما هو العبء التدريسي للمدرسين؟", "What is the teaching load for instructors?", "serve", [96], "20 credit hours per semester"),
    _q("AR-T13", "T13", "teaching", "numeric", "ما هو الحد الأدنى للعبء التدريسي المباشر لأعضاء هيئة التدريس ذوي المهام غير الإدارية؟", "What is the minimum direct teaching load for faculty with non-administrative duties?", "serve", [94], "6 credit hours per semester"),
    _q("AR-T18", "T18", "teaching", "conditional", "كم عدد الساعات المكتبية المسائية المطلوبة من أعضاء هيئة التدريس الذين يدرّسون مقررات مسائية؟", "How many evening office hours must faculty teaching evening courses schedule?", "serve", [80], "at least one office hour in the evening"),
    _q("AR-T19", "T19", "teaching", "numeric", "خلال كم ساعة يجب على المدرسين تسليم نتائج الامتحان النهائي إلى رئيس القسم؟", "Within how many hours must instructors submit final examination results to the Department Chair?", "serve", [224], "forty-eight hours"),
    _q("AR-T20", "T20", "teaching", "authority", "من يوافق على خطة العبء السنوية؟", "Who approves the annual workload plan?", "either", [96], "department chair reviews and approves; final approval rests with the dean"),
    # ---- Research & Innovation ---------------------------------------------------------------------------------------
    _q("AR-R01", "R01", "research", "numeric", "كم عدد الأطروحات التي يمكن لعضو هيئة التدريس الإشراف عليها في الوقت نفسه؟", "How many theses can a faculty member supervise at once?", "serve", [81], "a maximum of four simultaneously"),
    _q("AR-R03", "R03", "research", "numeric", "ما مقدار السلفة النقدية التي يمكن للباحث الرئيسي الحصول عليها؟", "How much petty cash advance can a principal investigator receive?", "serve", [166], "up to AED 5000, subject to VCRGS approval"),
    _q("AR-R04", "R04", "research", "authority", "أي لجنة تراجع البحوث التي تُجرى على البشر؟", "Which committee reviews research on human subjects?", "serve", [265], "the Institutional Review Board (IRB)"),
    _q("AR-R05", "R05", "research", "numeric", "كم يوماً في الأسبوع يمكن لأعضاء هيئة التدريس قضاؤه في الاستشارات؟", "How many days per week can faculty spend on consultancy?", "serve", [260], "up to 20% of their time or one working day per week"),
    _q("AR-R07", "R07", "research", "authority", "من يوافق على السفر لحضور المؤتمرات؟", "Who approves conference travel?", "serve", [117], "the Vice Chancellor for Academic Affairs"),
    _q("AR-R16", "R16", "research", "conditional", "هل يمكن لعضو هيئة التدريس أن يكون باحثاً رئيسياً في مشروعين في الوقت نفسه؟", "Can a faculty member be principal investigator for two projects simultaneously?", "either", [165], "no, except for seed projects"),
    _q("AR-R21", "R21", "research", "numeric", "ما هي المدة القصوى لإجازة التفرغ العلمي للباحثين الزائرين؟", "How long can a sabbatical leave for visiting researchers last?", "either", [174], "should not exceed two months"),
    _q("AR-R22", "R22", "research", "adversarial", "كم تبلغ مدة إجازة التفرغ العلمي لأعضاء هيئة التدريس؟", "How long is sabbatical leave for faculty members?", "fallback", [], "p174 states two months for VISITING researchers only"),
    _q("AR-R23", "R23", "research", "numeric", "ما هو البدل اليومي لسفر المؤتمرات؟", "What is the per diem rate for conference travel?", "either", [119], "AED 1000-1500 for a maximum of four days"),
    _q("AR-R27", "R27", "research", "paraphrase", "من يوافق على طلبات السفر لحضور المؤتمرات؟", "Who approves conference travel requests?", "serve", [117], "the VCAA"),
    # ---- Faculty Services & HR ---------------------------------------------------------------------------------------
    _q("AR-H01", "H01", "faculty_services", "conditional", "هل ترتب الموارد البشرية السفر والإقامة الفندقية لأعضاء هيئة التدريس الجدد؟", "Does HR arrange travel and hotel accommodations for new faculty members?", "serve", [61], "yes"),
    _q("AR-H03", "H03", "faculty_services", "numeric", "كم يوماً من إجازة الأمومة تستحقها عضوات هيئة التدريس؟", "How many days of maternity leave are female faculty entitled to?", "serve", [107], "sixty calendar days with full pay"),
    _q("AR-H04", "H04", "faculty_services", "conditional", "هل تساعد الموارد البشرية أعضاء هيئة التدريس الجدد في الحصول على تأشيرات الدخول؟", "Does HR assist new faculty members in obtaining entry visas?", "serve", [61], "yes, through the Passport Unit"),
    _q("AR-H05", "H05", "faculty_services", "numeric", "كم يوماً من الإجازة المرضية براتب كامل يستحق أعضاء هيئة التدريس؟", "How many days of sick leave with full pay are faculty entitled to?", "serve", [107], "up to 15 working days"),
    _q("AR-H13", "H13", "faculty_services", "authority", "من يوافق على إجازة الأمومة الإضافية بدون راتب؟", "Who approves additional maternity leave without pay?", "serve", [107], "the Vice Chancellor for Academic Affairs"),
    _q("AR-H15", "H15", "faculty_services", "numeric", "كم عدد أيام إجازة الأمومة بدون راتب التي يمكن الموافقة عليها؟", "How many days of unpaid maternity leave may be approved?", "either", [107], "up to thirty days, approved by the VCAA"),
    _q("AR-H16", "H16", "faculty_services", "numeric", "كم يوماً تقويمياً من الإجازة المدفوعة يمكن أن يحصل عليها أعضاء هيئة التدريس المسلمون لأداء فريضة الحج لأول مرة؟", "How many calendar days of paid leave can Muslim faculty receive for their first Hajj?", "either", [107], "up to twenty calendar days"),
    _q("AR-H21", "H21", "faculty_services", "adversarial", "كم يوماً من إجازة الأبوة يستحق أعضاء هيئة التدريس الذكور؟", "How many days of paternity leave are male faculty entitled to?", "fallback", [], "no paternity leave in the handbook; the maternity sentence must not be served"),
    _q("AR-H25", "H25", "faculty_services", "numeric", "كم يبلغ التعويض الشهري الذي يحصل عليه العميد المشارك؟", "How much monthly compensation does an Associate Dean receive?", "either", [108], "AED 5,000"),
    _q("AR-H42", "H42", "faculty_services", "adversarial", "كم يوماً من إجازة الأمومة يحصل عليها أعضاء هيئة التدريس الذكور؟", "How many days of maternity leave do male faculty get?", "fallback", [], "female faculty only"),
    # ---- Institutional Navigation ------------------------------------------------------------------------------------
    _q("AR-I01", "I01", "institutional", "contact", "ما هو رقم هاتف قسم الموارد البشرية؟", "What is the phone number of the Human Resources Department?", "serve", [54], "+971 6 5050023"),
    _q("AR-I02", "I02", "institutional", "contact", "ما هو رقم فاكس وحدة الجوازات؟", "What is the fax number of the Passport Unit?", "serve", [53], "+971 6 5050026"),
    _q("AR-I03", "I03", "institutional", "contact", "ما هو رقم هاتف مكتب التسجيل؟", "What is the telephone number of the Registration office?", "serve", [53], "+971 6 5050000"),
    _q("AR-I05", "I05", "institutional", "contact", "ما هو رقم هاتف مركز تكنولوجيا المعلومات؟", "What is the phone number of the Information Technology Center?", "serve", [54], "+971 6 5050000"),
    _q("AR-I06", "I06", "institutional", "numeric", "كم عدد الكليات في جامعة الشارقة؟", "How many colleges does the University of Sharjah have?", "serve", [24], "fifteen academic colleges"),
    _q("AR-I12", "I12", "institutional", "contact", "ما هو رقم هاتف وحدة الجوازات؟", "What is the phone number of the Passport Unit?", "serve", [53], "+971 6 5050066"),
    _q("AR-I25", "I25", "institutional", "contact", "ما هو رقم هاتف كلية الطب؟", "What is the phone number of the College of Medicine?", "serve", [53], "+971 6 5057201"),
    _q("AR-I37", "I37", "institutional", "adversarial", "ما هو عنوان البريد الإلكتروني لقسم الموارد البشرية؟", "What is the email address of Human Resources?", "fallback", [], "the directory has no email addresses"),
    _q("AR-I45", "I45", "institutional", "contact", "ما هو رقم هاتف كلية الدراسات العليا والبحث العلمي؟", "What is the phone number of the College of Graduate Studies and Research?", "serve", [54], "+971 6 5050550"),
    _q("AR-I07", "I07", "institutional", "location", "أين تقع المكتبة الرئيسية؟", "Where is the Main Library?", "fallback", [], "no location statement"),
    # ---- Cross-domain and multi-part -----------------------------------------------------------------------------------
    _q("AR-X01", "X01", "cross", "multipart", "كم عدد الساعات المعتمدة للعبء التدريسي الكامل، وكم عدد الساعات المكتبية التي يجب على أعضاء هيئة التدريس الالتزام بها؟", "How many credit hours is a full teaching load, and how many office hours must faculty hold?", "serve", [91, 96, 79], "12 credit hours + five office hours", specialists=["teaching"]),
    _q("AR-X03", "X03", "cross", "multipart", "ما هو رقم هاتف قسم الموارد البشرية وما هو رقم فاكس وحدة الجوازات؟", "What is the phone number of the Human Resources Department and what is the fax number of the Passport Unit?", "serve", [54, 53], "+971 6 5050023 + +971 6 5050026", specialists=["institutional"]),
    _q("AR-X04", "X04", "cross", "cross_domain", "كم عدد الساعات المعتمدة للعبء التدريسي الكامل وكم يوماً من الإجازة المرضية براتب كامل يستحق أعضاء هيئة التدريس؟", "How many credit hours is a full teaching load and how many days of sick leave with full pay are faculty entitled to?", "serve", [91, 96, 107], "12 credit hours + 15 working days", specialists=["teaching", "faculty_services"]),
    _q("AR-X06", "X06", "cross", "cross_domain", "هل تساعد الموارد البشرية أعضاء هيئة التدريس الجدد في الحصول على تأشيرات الدخول، وما هو رقم هاتف وحدة الجوازات؟", "Does HR assist new faculty members in obtaining entry visas, and what is the phone number of the Passport Unit?", "serve", [61, 53], "yes via the Passport Unit + +971 6 5050066", specialists=["faculty_services", "institutional"]),
    _q("AR-X08", "X08", "cross", "cross_domain", "ما هو العبء التدريسي للمحاضرين وما هو رقم هاتف مكتب التسجيل؟", "What is the teaching load for lecturers and what is the phone number of the Registration office?", "serve", [96, 53], "15 credit hours + +971 6 5050000", specialists=["teaching", "institutional"]),
    _q("AR-X19", "X19", "cross", "cross_domain", "من يوافق على سفر المؤتمرات وما هو رقم هاتف وحدة الجوازات؟", "Who approves conference travel and what is the phone number of the Passport Unit?", "serve", [117, 53], "VCAA + +971 6 5050066", specialists=["research", "institutional"]),
    # ---- Adversarial, unanswerable, ambiguous, mixed-language ----------------------------------------------------------
    _q("AR-A02", "A02", "none", "unsupported", "ما هي عاصمة فرنسا؟", "What is the capital of France?", "fallback", [], "off-topic", specialists=[]),
    _q("AR-A14", "A14", "teaching", "adversarial", "هل العبء التدريسي لهيئة التدريس البحثية النشطة 12 ساعة معتمدة؟", "Is the teaching load for Active Research Faculty 12 credit hours?", "either", [96], "it is 9; a served answer must state 9, never confirm 12"),
    _q("AR-A15", "A15", "research", "adversarial", "هل يقول الدليل إن عضو هيئة التدريس يمكنه الإشراف على عدد غير محدود من الأطروحات؟", "Does the handbook say faculty may supervise an unlimited number of theses?", "either", [81], "maximum of four"),
    _q("AR-A30", "A30", "faculty_services", "negation", "هل صحيح أن أعضاء هيئة التدريس لا يحصلون على إجازة مرضية؟", "Is it true that faculty get no sick leave?", "either", [107], "15 working days with full pay; the negation must not be confirmed"),
    _q("AR-A16", "A16", "research", "adversarial", "من يوافق على سفر المؤتمرات: العميد أم نائب مدير الجامعة للشؤون الأكاديمية؟", "Who approves conference travel: the Dean or the Vice Chancellor?", "either", [117], "the VCAA"),
    _q("AR-G01", "PF10-EX", "teaching", "procedure", "كيف أرسل إعلانًا للطلبة عبر Blackboard؟", "How do I send an announcement to my students in Blackboard?", "serve", [9], "Blackboard Ultra Faculty Short Guide p.9 (PF-9 T3 / PF-10 example)", source=SHORT_GUIDE, mixed=True),
    _q("AR-M01", "MIX", "teaching", "mixed", "كيف أستخدم Blackboard لإرسال Announcement؟", "How do I use Blackboard to send an Announcement?", "either", None, "Blackboard guides (page for review)", source=SHORT_GUIDE, mixed=True),
    _q("AR-M02", "MIX", "none", "mixed", "ما هي إجراءات Faculty Appraisal؟", "What are the procedures of the Faculty Appraisal?", "fallback", [], "the appraisal guide is not an enabled source; a general ask", specialists=[], mixed=True),
]

#: Written after the dev run froze the lexicon (see MULTILINGUAL_EVALUATION_REPORT.md); appended by the holdout step.
HOLDOUT = [
    _q("HO-T06", "T06", "teaching", "direct", "ماذا يُقصد بنموذج توزيع عبء العمل؟", "What is the Workload Allocation Model?", "either", [90, 96], "WLAM defines faculty workload", split="holdout"),
    _q("HO-T10", "T10", "teaching", "numeric", "كم ساعة معتمدة يُدرّس أعضاء هيئة التدريس النشطون بحثياً في الفصل الواحد؟", "What is the teaching load for Active Research Faculty?", "serve", [96], "9 credit hours per semester", split="holdout"),
    _q("HO-T14", "T14", "teaching", "paraphrase", "كم ساعة مكتبية ينبغي أن أخصص أسبوعياً؟", "How many office hours per week should I keep?", "serve", [79], "five per week over two days", split="holdout"),
    _q("HO-T29", "T29", "teaching", "adversarial", "ما هو النصاب التدريسي لمساعدي التدريس؟", "What is the teaching load for teaching assistants?", "fallback", [], "no teaching-assistant profile in the WLAM table", split="holdout"),
    _q("HO-R12", "R12", "research", "authority", "من الذي يعتمد صرف السلفة النقدية للباحث الرئيسي؟", "Who approves the petty cash advance for a principal investigator?", "serve", [166], "the VCRGS", split="holdout"),
    _q("HO-R25", "R25", "research", "numeric", "كم مؤتمراً مدعوماً من الجامعة يمكن لعضو هيئة التدريس حضوره سنوياً؟", "How many university-supported conferences can a faculty member attend per year?", "either", [117], "one international and one local", split="holdout"),
    _q("HO-R35", "R35", "research", "paraphrase", "ما الحد الأقصى لعدد الرسائل العلمية التي يجوز لعضو هيئة التدريس الإشراف عليها بالتزامن؟", "What is the maximum number of theses a faculty member may supervise simultaneously?", "serve", [81], "four", split="holdout"),
    _q("HO-H02", "H02", "faculty_services", "authority", "من الجهة التي تتولى إعداد عروض العمل؟", "Who drafts employment offers?", "serve", [61], "HR, upon receiving the candidate's Security Clearance", split="holdout"),
    _q("HO-H18", "H18", "faculty_services", "numeric", "كم يوم عمل من الإجازة المرضية بنصف الأجر يستحقه عضو هيئة التدريس؟", "How many days of sick leave at half pay are faculty entitled to?", "either", [107], "an additional 15 working days", split="holdout"),
    _q("HO-H34", "H34", "faculty_services", "adversarial", "هل تبلغ الإجازة السنوية ثلاثين يوماً؟", "Is annual leave 30 days?", "fallback", [], "p107 gives no number of vacation days", split="holdout"),
    _q("HO-I04", "I04", "institutional", "contact", "أريد رقم تلفون الإدارة المالية من فضلك", "What is the phone number of the Finance Department?", "serve", [54], "+971 6 5050040", split="holdout"),
    _q("HO-I20", "I20", "institutional", "contact", "كيف أتصل بجمعية الخريجين هاتفياً؟", "What is the phone number of the Alumni Association?", "serve", [54], "+971 6 5050527", split="holdout"),
    _q("HO-X16", "X16", "cross", "cross_domain", "كم ساعة مكتبية يجب أن يلتزم بها عضو هيئة التدريس، وكم يوماً من الإجازة المرضية بأجر كامل يستحق؟", "How many office hours must faculty hold and how many days of sick leave with full pay are faculty entitled to?", "serve", [79, 107], "five office hours + 15 working days", split="holdout", specialists=["teaching", "faculty_services"]),
    _q("HO-A21", "A21", "faculty_services", "conditional", "هل توفر الجامعة تأميناً صحياً لأعضاء هيئة التدريس؟", "Does the University provide health insurance for faculty?", "either", None, "chapter 4 benefits (page for review)", split="holdout"),
]

ALL = QUESTIONS + HOLDOUT
BY_ID = {q["id"]: q for q in ALL}
