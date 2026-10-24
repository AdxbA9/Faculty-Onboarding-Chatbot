"""
Shared fixtures for the Faculty Services & HR specialist tests (Milestone 4, Step 4.11).

The fixture pages mirror the printed handbook lines of the HR sections (chapter 2, chapter 4, the evaluation and
promotion process pages, the services pages and the shared pages p122, p124, p255, p257, p263, p266 and p268). No
model is loaded: the embedder is the hashed bag of words of ``conftest.FakeEmbedder`` and the rerankers are
deterministic stand-ins.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import faiss

from handbook_bot.agents.specialists.base import check_findings
from handbook_bot.agents.specialists.contracts import SpecialistTask
from handbook_bot.agents.specialists.faculty_services import APPROVED_SOURCE_ID, FacultyServicesResources, FacultyServicesSpecialist
from handbook_bot.chunking import build_chunks
from handbook_bot.sources import annotate_metadata, load_section_map, load_source_registry
from handbook_bot.text_utils import normalize_text, tokenize

from conftest import FakeEmbedder

REPO = Path(__file__).resolve().parents[1]
TITLE = "UOS Faculty Handbook 2025/26"


def page(number, *lines):
    lines = list(lines)
    return {"page": number, "lines": ["Page %d" % number] + lines, "rows": lines,
            "text": normalize_text(" ".join(["Page %d" % number] + lines)), "section_hint": "", "ocr_snippets": []}


# ---------------------------------------------------------------------------
# Verbatim handbook lines (printed rows joined into their sentences)
# ---------------------------------------------------------------------------
P56 = page(56,
           "Chapter 2: Faculty Appointment",
           "The University of Sharjah aims to attract and retain top talent to achieve its vision. The Faculty Employment Policy is designed to rigorously select and appoint the best candidates across all academic units, ensuring the recruitment of highly qualified individuals.",
           "Abbreviations and Definitions",
           "• Visiting: Involves a faculty appointment for a limited period, not exceeding three years.",
           "• Emerita/Emeritus: Represents an honorary title conferred on a faculty member in recognition of their distinguished service to the University.",
           "• Regular: Identifies a faculty appointment that signifies continuous employment.",
           "Appointment Policy Authority and Responsibility",
           "• The Vice Chancellor for Academic Affairs, in consultation with the Chancellor, exercises authority over the Faculty Employment Policy.")
P61 = page(61,
           "2.2.3.7 Candidate Communication and Offer",
           "• HR manages all communication with candidates and potential recruits, ensuring effective and consistent messaging.",
           "• Employment offers are drafted by HR upon receiving the selected candidate's Security Clearance.",
           "• HR informs the candidate to review the Employment Offer and submit an offer acceptance within 10 days.",
           "• Employment Offers are considered final upon submission of the signed copy by the candidate to HR.",
           "2.2.3.8 Pre-joining Formalities",
           "• HR compiles all required documents and information to maintain in the Employee File.",
           "• New faculty members are informed to promptly notify HR of any changes in personal data and documents.",
           "• Document authentication is required for obtaining residence visas in the UAE, and HR informs new faculty members of the documents that should be attested in their home country.",
           "• HR provides information to new faculty members through an Employee Handbook, (covering various topics such as UoS overview, HR policies, employment policies, living in the UAE, and settling in the UAE).",
           "2.2.3.9 Entry Visas, Travel, and Accommodation",
           "• HR assists new faculty members in obtaining entry visas through the Passport Unit.",
           "• HR arranges travel and hotel accommodations for new faculty members.",
           "• HR coordinates with the Services Department to provide on-campus accommodation or housing allowances.")
P62 = page(62,
           "2.2.4 Hiring Exceptions",
           "The Chancellor holds the jurisdiction to appoint national faculty who fulfil required criteria and professors with a high Scopus H-index or rare specializations without undergoing the regular review and appointment process.",
           "2.3 Contract Types",
           "The University offers various contract types for regular and non-regular faculty:",
           "• Definite Contract: Initial appointment for regular faculty is a two-year offer, including a probation year. The contract may be renewed for three years upon successful annual evaluations as per UoS policies. Subsequently, the contract may be renewed for four years after successful annual evaluations. Regular faculty members must apply for promotion to associate professor during their appointments and within ten years of obtaining the assistant professor rank.",
           "• Joint Appointment: Allows faculty members to perform responsibilities across two or more units within UoS. The primary unit, responsible for decisions regarding performance review, promotion, and other conditions, must account for at least 60% of the appointment.")
P63 = page(63,
           "• Visiting Appointment: Involves a definite term contract of up to two years, including a probation year. Renewal for the second year is subject to department needs and successful annual evaluations as per UoS policies. In exceptional cases, and with Chancellor approval, the visiting contract may be extended for an additional year.",
           "• Lecturers/Senior Lecturers/Instructors and Senior Instructors: Appointed on a definite contract for two years, including a probation year. The contract is renewable every two years based on satisfactory performance as per UoS policies.",
           "2.4 Probation and Resignation Notice",
           "• The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval. Termination during the probation period requires a recommendation by the relevant academic administrators and approval from the Dean, Vice Chancellor, and Chancellor. Faculty members intending to resign during probation must provide a six-month notice.",
           "• Faculty members intending to resign from UoS must submit their resignation to their respective college deans and through the HR portal at least six months before the effective date and must coincide with the end of an academic semester (typically January or August). Failure to comply may result in a 50% deduction from the end of service compensation unless exceptions are granted by the Chancellor. New faculty members who are terminated during probation are entitled to return tickets, while new faculty members who resign during probation are not entitled to benefits such as annual leave and ticket allowances.",
           "2.5 Confirmation",
           "• At the end of the first year of the contract, the performance of new faculty members is evaluated by the department chair, dean, central hiring and evaluation committee, and student evaluations. The evaluation results determine whether the faculty member's contract is renewed, probation is extended, or contract renewal occurs with specific teaching and research terms.",
           "• New faculty members must receive written notification of employment confirmation or termination six months before the end of the contract. If neither occurs, the new faculty member's employment is considered confirmed.",
           "2.6 Dual Career",
           "Faculty candidates with spouses also seeking employment at UoS may be provided with opportunities for both individuals if feasible.")
P64 = page(64,
           "2.7 Special Appointments for Academic Administrators",
           "• All University-level academic administrators receive regular academic appointments in the appropriate college and rank based on their credentials upon assuming an administrative appointment.",
           "2.8 Termination for Cause of a Contract",
           "• UoS may terminate faculty contracts in the event of discontinued academic colleges/divisions/departments or financial emergencies declared by the Board of Trustees (BOT), necessitating faculty reduction according to UoS HR policies. Terminated faculty members receive a minimum six-month notice.",
           "• Termination of faculty contracts for cause requires the approval of the Chancellor, and recommendations for termination must be made with justifiable reasons.",
           "2.9 Hiring and Transitioning of Emirati Faculty Members",
           "The University of Sharjah strategically promotes the recruitment of Emirati talent as faculty members to contribute effectively to national human development.",
           "2.9.1.1 Eligibility for Assistant Professor-Emirati Track Appointment (New Graduate):",
           "a. Applicants must be Emirati Nationals. b. Applicants must have obtained their terminal degree within two years of the application. c. Applicants must be below 40 years of age. Exceptions are subject to Chancellor's approval.")
P65 = page(65,
           "2.9.1.3 Teaching Load Reduction: In the first year of appointment, faculty on the Assistant Professor Emirati Track are entitled to a teaching load reduction of 6 credit hours.",
           "2.9.1.4 Performance Evaluation: Faculty members must maintain above-average performance in their annual evaluations, including teaching, research, and service. Unsatisfactory performance during the first three years may lead to contract termination.",
           "2.9.1.5 Mandatory Training: During the first year of appointment, faculty members must attend a mandatory training program designed to enhance their effectiveness in teaching and advising.")
P106 = page(106,
            "Chapter 4: Appointment Benefits",
            "4.1 Basic Salary",
            "The University of Sharjah offers a competitive basic salary based on faculty rank and specialization, following the salary scale below:",
            "Academic Rank Basic Monthly Salary (AED)",
            "Professor 30,576 – 43,860",
            "Associate Professor 25,318 – 37,926",
            "Assistant Professor 20,142 – 31,674",
            "Lecturer 14,242 – 23,022",
            "Instructor 18,744 – 22,744",
            "• At the end of their service, faculty members will receive an end-of-service salary equivalent to one month's salary for each year served.",
            "• Faculty engaged in clinical activities will be eligible for a clinical allowance ranging from 10% to 20% of the basic salary.",
            "4.2 Monetary Benefits",
            "Faculty members are entitled to various benefits as part of their employment package:",
            "• Annual Return Air Ticket: Faculty members, their spouse, and three dependent children are eligible for an annual return air ticket to their home country, as per university regulations.",
            "• Housing: Suitable free housing with electricity and water is provided in UoS housing compounds. In case suitable accommodation is unavailable, a housing allowance shall be given according to university regulations.",
            "• Furniture Allowance: A one-time furniture allowance of 30,000 Dirhams is provided, which can be utilized over four years.",
            "• Health Insurance: Comprehensive health insurance is offered for faculty members, their spouse, and up to three dependent children under 18 years old.",
            "• Shipment Allowance: A shipment allowance is provided at the beginning and end of the contract, following university regulations.",
            "• Children's Education: An annual children's education allowance of up to 25,000 Dirhams per child (inclusive of textbook fees) is provided, with a maximum of 75,000 Dirhams for all qualifying children from KG1 to the end of their secondary education.")
P107 = page(107,
            "4.3 Leaves and Absences",
            "4.3.1 Vacation Leave",
            "• Vacation days are provided as per the Faculty and Staff Pay Grade and Benefits Chart, including summer vacation, religious and national holidays, and other specified vacations in the academic calendar.",
            "• During mid-semester and final exams, faculty involved in the exam process are expected to conduct exams, even during weekends and regular holidays.",
            "4.3.2 Sick Leave",
            "• Faculty are entitled to sick leave with full pay for up to 15 working days and half pay for an additional 15 working days per contract year. This applies to personal illness or attendance upon an ill immediate family member, either in or out of the UAE.",
            "• If all sick leave days have been utilized, annual vacation leave may be used. In special circumstances, up to an additional 30 days of sick leave without pay may be granted.",
            "4.3.3 Maternity Leave",
            "• Female faculty employed on a full-time basis are entitled to sixty calendar days of maternity leave with full pay. Additional maternity leave without pay, up to a maximum of thirty days, may be approved by the Vice Chancellor for Academic Affairs.",
            "4.3.4 Unpaid Special Leave",
            "Unpaid special leave, not exceeding thirty days, may be granted upon approval from appropriate University authorities if the employee has no unused vacation leave. The unpaid leave will be deducted from the next month's salary.",
            "4.3.5 Hajj Leave",
            "Muslim faculty attending their first Hajj may receive up to twenty calendar days of paid leave.",
            "4.4 Duplication of Benefits",
            "• If a university employee's spouse also receives benefits from another UAE government or partially owned UAE government entity, duplication of benefits may occur.",
            "• The University will adjust benefits accordingly to avoid duplication.")
P108 = page(108,
            "4.5 Compensation for Administrative Positions",
            "• Compensation for administrative positions includes a monthly amount and a reduction in teaching load for certain faculty roles: o Dean: Monthly compensation of AED 10,000 and a reduction in teaching load of nine credit hours per semester. o Associate Dean: Monthly compensation of AED 5,000 and a reduction in teaching load of three to six credit hours per semester. o Assistant Dean: Monthly compensation of AED 4,000 and a reduction in teaching load of three credit hours per semester. o Chairperson: Monthly compensation of AED 4,000 and a reduction in teaching load of three to six credit hours. o Director of a Unit: Monthly compensation of AED 2,000 to AED 3,000 and a reduction in teaching load of three to six credit hours. o Program Coordinator: Monthly compensation of AED 2,000 and a reduction in teaching load of three credit hours.",
            "4.6 Annual Incentives for Distinguished Faculty Members",
            "The Annual Bonus Award for Outstanding Faculty is a prestigious recognition bestowed upon distinguished academic faculty at the University of Sharjah who have exhibited exceptional academic achievements. It is important to note that the award shall not impact the monthly salary of faculty members.")
P122 = page(122,
            "Chapter 6: Performance Evaluation",
            "To ensure a comprehensive assessment of faculty performance, an annual evaluation is conducted annually, taking into account their contributions in the areas of teaching and learning, research and scholarly activities, and community service.",
            "6.1 General Evaluation Considerations",
            "• New faculty must undergo a probationary review within the first year of appointment.",
            "• All faculty including those with administrative assignment shall undergo an annual evaluation.",
            "• Although relative weights may vary, all faculty are expected to maintain productivity in each of the three areas over time.",
            "6.2 Teaching and Learning Evaluation",
            "Through their teaching, faculty play a crucial role in transmitting knowledge, fostering critical thinking, and engaging students in the learning process. The evaluation of teaching effectiveness includes considerations such as:",
            "• Utilization of student evaluations and other mechanisms to assess teaching performance and enhance educational outcomes.",
            "• Clearly stated course objectives and evidence of employing appropriate teaching and learning methods to achieve these objectives.",
            "• Faculty on probation are evaluated on their teaching through student evaluations each semester.")
P124 = page(124,
            "• Recognition through awards, grants, contracts, patents, and peer-reviewed accolades.",
            "• Supporting graduate students including thesis and dissertation supervision.",
            "6.5 Community Service Evaluation:",
            "Faculty members play an essential role in the University's development and the wider community. The evaluation of service activities includes:",
            "• Involvement in community projects, consulting, and collaborations with external stakeholders.",
            "• Faculty with administrative appointments receive a service weight of 30 percent in the evaluation.",
            "6.6 Evaluation Process and Criteria:",
            "• Faculty members shall submit an annual report detailing their teaching and advising activities, research output, professional development, and service contributions.",
            "• The Department Chair and College Dean shall conduct the faculty's evaluation based on their professional responsibilities and performance in the evaluation areas.",
            "• The evaluation process and evaluation ratings should be consistent with and benchmarked against average department and college performance.",
            "6.7 Special Issues and Considerations",
            "• Number of years in formal administrative appointments after the rank.")
P133 = page(133,
            "6.10 Rubric for Contract Renewal",
            "Contract renewal is not based on a single year's performance but rather on the faculty member's cumulative contributions over the duration of the current contract. In the domain of research, faculty are expected to demonstrate a minimum average of two Scopus-indexed publications per year, calculated across the contract term.",
            "In accordance with university policy and national regulations, faculty members over the age of 65 will be offered renewable one-year contracts, subject to annual performance evaluation, until the age of 70.")
P148 = page(148,
            "3. For papers written in languages other than English, the University Promotion Committee classifies the corresponding journals (not in the SCOPUS Database with SJR metric) submitted by the applicant. Applicants planning to submit papers in the future can request journal classification from the University Promotion Committee before submission.",
            "7.8 Promotion Procedures at the Department Level",
            "The promotion procedures at the department level involve the following steps:",
            "A. Applications for promotion can be submitted during the first week of January, April, or October of each year to the department chair. Applicants should attach a dossier with their application, including their CV, research submitted for promotion, evidence of teaching contributions, academic advising, university, and community service, and a summary of scientific research contributions. The department chair shall present these applications to the Department Academic Promotions Committee.")
P149 = page(149,
            "The Committee is chaired by the department chair if a full professor or by another full professor within the department if the department chair is not a full professor.",
            "C. The Department Academic Promotions Committee shall review the promotion applications, ensuring they meet the promotion criteria. If recommended, the department chair shall forward the Committee's report, application dossier, and report from the Department Academic Promotions Committee to the college dean within one month of receiving the application.",
            "7.8 Promotion Procedures at the College Level",
            "A. In September of each year, the College Council shall form the College Academic Promotions Committee, chaired by the college dean if a full professor or by another professor within the college if the dean is not a full professor.")
P64_APPT = page(64,
            "2.9.1.2 Initial Faculty Appointment: The first faculty appointment on the Assistant Professor Emirati Track shall be for a duration of three years.")
P149_HEAD = page(149,
            "7.8 Promotion Procedures at the College Level The promotion procedures at the college level involve the following steps: A.")
P150 = page(150,
            "7.9 Promotion Procedures at the University Level",
            "Applications for promotion shall be processed at the University level as follows:",
            "A. At the beginning of each academic year, the University Central Committee for the Promotion of Faculty Members shall be formed and chaired by the Chancellor or a delegate. Its membership shall include at least six full professors from various University colleges.",
            "B. The University Central Committee for the Promotion of Faculty Members shall review promotion applications from various colleges to determine if they meet the promotion criteria. The Committee may ask applicants for additional information and supporting documentation.")
P255 = page(255,
            "The Facilities Management and Planning Department remains committed to supporting the University's infrastructure needs and contributing to its mission of excellence in education and research.",
            "15.4 Child Care Center",
            "The University of Sharjah Nursery is dedicated to providing the highest quality care and services for children, prioritizing their psychological and physical health and wellbeing.",
            "15.5 Disability Resource Center",
            "The Disability Resource Center (DRC) provides support services to students with disabilities, including academic accommodations and assistive technology.")
P257 = page(257,
            "DRC also organize and participate in conferences, seminars, and workshops related to disability issues and promote volunteerism within the Center and the broader community.",
            "15.6 Medical Clinics",
            "The University Medical Clinics (UMC) is dedicated administrative, medical, and nursing staff are committed to ensuring your health and safety while on campus, whether you are in classrooms, university buildings, student dormitories, or faculty housing.",
            "15.7 Sport Facilities",
            "The main campus of the University of Sharjah boasts two state-of-the-art sports complexes, one for men and one for women. These facilities include Olympic-sized swimming pools, several multipurpose indoor sports halls, and fully-equipped gymnasiums.",
            "15.8 Food Courts",
            "The campus food courts offer a variety of dining options for students, faculty and staff throughout the working day.")
P263 = page(263,
            "• Teaching second-degree relatives by blood. • Supervising or participating in decisions that affect second-degree relatives by blood.",
            "University's Authority: The University of Sharjah has the authority to require disclosures, investigate conflicts of interest, manage or eliminate conflicts, impose appropriate sanctions for policy violations, and release information about conflicts of interest.",
            "16.3 Animal Use in Research",
            "The University of Sharjah (UoS) is committed to conducting high-quality research to advance knowledge and improve human and animal health. UoS researchers must ensure the well-being and humane treatment of animals throughout all stages of the research process.")
P266 = page(266,
            "Training and Education: All researchers, faculty, staff, and students involved in human research must undergo appropriate training on research ethics, informed consent, and human subjects' protection.",
            "16.5 Policy Prohibiting Sexual Harassment",
            "The University of Sharjah is dedicated to fostering a safe and respectful environment characterized by mutual trust, dignity, and respect for all members of its community. As such, the University strictly prohibits any form of sexual harassment.")
P268 = page(268,
            "16.5.4 Scope of Application: This policy applies to all members of the University community, including students, faculty, academic staff, administrators, temporary and part-time employees, as well as contractors, suppliers, volunteers, visitors, and campus patrons.",
            "16.5.6 Procedures: The University provides multiple channels for reporting sexual harassment, both formal and informal. Individuals seeking help, advice, or wishing to report sexual harassment incidents are encouraged to contact the designated Sexual Harassment Proceedings Coordinator or Officer.",
            "16.6 University of Sharjah Guidelines on Artificial Intelligence and Large Language Models",
            "The rapid advancements in Artificial Intelligence (AI) and Large Language Models (LLMs) have transformed teaching and learning. Faculty members may use AI tools to prepare course materials provided that academic integrity is preserved.")

PAGES = [P56, P61, P62, P63, P64, P65, P106, P107, P108, P122, P124, P133, P148, P149, P150, P255, P257, P263, P266, P268]

# Step 4.12-C5 (D-092): verbatim lines behind the Step 4.12-R4 sameness and personal-status cases. Kept out of PAGES so
# that the corpora of the earlier tests are unchanged.
P69_CLINICAL = page(69,
                    "2.10.4 Performance Evaluation",
                    "• Clinical faculty/academic non-faculty members' evaluation follows the same timeline and process as that of regular faculty.")
P148_B = page(148,
              "B. The Department Academic Promotions Committee should comprise at least three department faculty members holding an academic rank equivalent to or higher than the rank applied for. Members of this committee should not be part of higher-level committees for the same purpose.")
P149_D = page(149,
              "D. If the number of departmental faculty holding a rank equal to or higher than the desired rank for promotion is less than three, the college dean, in consultation with the department chair, shall select faculty members from outside the department to form a committee of at least three members. These external members should have the same field of specialization as the applicant and hold the same academic rank being applied for or higher.")
P150_D = page(150,
              "D. If the number of College Academic Promotions Committee members holding a rank equal to or higher than that being sought for promotion is less than three, the dean shall follow the same procedure mentioned above at the departmental level.")
P63_CONFIRMATION = page(63,
                        "2.5 Confirmation",
                        "• New faculty members must receive written notification of employment confirmation or termination six months before the end of the contract. If neither occurs, the new faculty member's employment is considered confirmed. Employment confirmation includes the probation period when calculating the employee's End of Service Benefit.")
P152 = page(152,
            "2. The Chancellor shall form a committee to investigate the case. The Committee shall make recommendations to the Chancellor within three weeks. The Chancellor may uphold or overturn the original decision based on the Committee recommendations. The decision of the Chancellor is final and is communicated to the College Dean and concerned faculty member. Should the application for promotion be approved, the HR Department is notified.")

# Step 4.12-C9 (D-096): verbatim lines behind the Step 4.12-R8 semantic cases (negated rules outside the D-094 predicate
# list, narrower negations, conditional facts). Kept out of PAGES so that the corpora of the earlier tests are unchanged.
P66_TRANSFER = page(66,
                    "2.9.3 Transfer Between Branches/Campuses",
                    "a. Transfer between branches/campuses is not applicable for Assistant Professor-Emirati Track.")
P67_STANDARDS = page(67,
                     "2.9.6.9 Unsatisfactory Performance and Termination: Newly appointed faculty members not meeting UoS standards should be notified as early as possible.")
P69_CLINICAL_RULES = page(69,
                          "• A clinical faculty member shall not be permitted to transfer to a regular faculty member (teaching and research) while employed at UoS.",
                          "• Clinical faculty members with no prior appointment in an academic setting shall be appointed at the rank of an assistant clinical professor.")
P72_RELATIVES = page(72,
                     "1. Direct or Indirect Line Management: Employment of a relative in a position where they directly or indirectly manage another relative is not allowed.")
P73_RESORT = page(73,
                  "If not convinced, termination can be decided by management. Termination should not be the first resort.")
P99_MISCONDUCT = page(99,
                      "• Misconduct does not require intent and shall be investigated according to institutional procedures.",
                      "• UoS has no statute of limitations for professional misconduct and may take action for violations during academic employment.")
P102_SUSPENSION = page(102,
                       "If the case is not referred to the Disciplinary Council within one month after the suspension decision, the suspension shall be "
                       "revoked, and the Disciplinary Council's decision shall be enforced.")
P103_RESIGNATION = page(103,
                        "8. Resignation Does Not Void the Case",
                        "The investigation case is not rendered void if the faculty member under investigation submits their resignation.")
P110_BONUS = page(110,
                  "Faculty members may apply for the bonus award in consecutive years. Parttime and visiting faculty members are not eligible for the bonus award.")
P267_PROSECUTION = page(267,
                        "The accused individual will be subject to internal disciplinary procedures as per the University's bylaws. Additionally, this "
                        "disciplinary action will not impede or hinder any criminal prosecution pursued by the accuser before the UAE judicial authorities.")
P151_DEANS = page(151,
                  "G. If the Deans Council approves a promotion, the Chancellor shall forward the decision to the Board of Trustees for approval. The "
                  "Chancellor shall issue the Board of Trustee decisions that approve promotion cases.")
C9_PAGES = [P66_TRANSFER, P67_STANDARDS, P69_CLINICAL_RULES, P72_RELATIVES, P73_RESORT, P99_MISCONDUCT, P102_SUSPENSION, P103_RESIGNATION,
            P110_BONUS, P267_PROSECUTION, P151_DEANS]
# Step 4.12-C10 (D-097): the modality, scope and exclusivity evidence of the known failures and their controls (verbatim,
# kept out of PAGES; P110_ELIGIBILITY replaces P110_BONUS in the C10 corpus).
P69_PERMISSION = page(69,
                      "• Based on the recommendation of the Vice Chancellor for Academic Affairs, the Chancellor may grant special permission for a "
                      "clinical faculty appointment in disciplines other than those listed in herein.")
P71_EMERITUS = page(71,
                    "Although it is an honorary title without employment rights or obligations, emeritus faculty may serve ex officio on committees and "
                    "teach courses as needed, with compensation provided according to University policies.")
P85_DEAN = page(85,
                "3.9 Role and Responsibilities of Administrative Faculty",
                "3.9.1 College Dean",
                "Individuals appointed as Deans must hold the rank of full professor. In exceptional cases, a faculty member from a different rank may "
                "be assigned as the acting Dean of the college. The appointment of the Dean is subject to the decision of the Board of Trustees, based "
                "on the Chancellor’s recommendation, for a period of three years, renewable only once.")
P102_EXTENSION = page(102,
                      "The suspension period may only be extended by a decision from the Disciplinary Council.")
P104_HEARING = page(104,
                    "c) Informal Resolution",
                    "Upon filing a grievance, the Committee Chair may suspend any applicable time frames to permit informal discussions or mediation to "
                    "resolve the grievance. If all parties involved sign a confidential, written statement describing the resolution, the grievance shall "
                    "be considered informally resolved.",
                    "d) The Grievance Hearing",
                    "The Faculty Grievance Committee shall hold a hearing within thirty days of receiving the written grievance, providing written notice "
                    "of the date and time to all parties. The hearing may be open to the public unless one or both parties request a closed hearing.")
P105_WITNESSES = page(105,
                      "During the hearing, both the faculty member and the University administrator responsible for the action or inaction being contested "
                      "have the opportunity to make opening and closing statements, examine and cross-examine witnesses, and present documentary evidence. "
                      "The Faculty Grievance Committee may contact witnesses upon request, ensuring their appearance at the hearing, unless the Committee "
                      "Chair determines it to be unreasonable or irrelevant to the issues at hand.")
P110_ELIGIBILITY = page(110,
                        "Eligibility: To be eligible for the award, a faculty member must be full-time and have completed at least two academic years at "
                        "UoS by the time of submitting the Annual Activity Report. Faculty members may apply for the bonus award in consecutive years. "
                        "Parttime and visiting faculty members are not eligible for the bonus award.")
C10_PAGES = [P69_PERMISSION, P71_EMERITUS, P85_DEAN, P102_EXTENSION, P104_HEARING, P105_WITNESSES, P110_ELIGIBILITY]
# Step 4.12-C11 (D-098): the evidence of the R10 and DR2 composite cases and of the avoidable-loss guards (verbatim, kept out
# of PAGES and C10_PAGES; in the C11 corpus P69_C11, P71_C11 and P102_C11 replace the C10 pages of the same number).
P58_HIRING = page(58,
                  "Each college forms a Department and College Hiring Committee at the beginning of each year.")
P69_C11 = page(69,
               "2.10.1 Hiring Limitation",
               "• The appointment of clinical faculty/academic non-faculty is limited to Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, "
               "and Law.",
               "• Based on the recommendation of the Vice Chancellor for Academic Affairs, the Chancellor may grant special permission for a "
               "clinical faculty appointment in disciplines other than those listed in herein.")
P71_C11 = page(71,
               "Although it is an honorary title without employment rights or obligations, emeritus faculty may serve ex officio on committees and "
               "teach courses as needed, with compensation provided according to University policies.",
               "2.12 Administrative Appointment of Regular Faculty",
               "A faculty member's primary and continuous appointment remains their faculty role, even when holding an administrative title. Senior "
               "administrative appointments require Chancellor approval, except for Deans, Vice Chancellors, and the Chancellor, whose appointments "
               "are endorsed by the Chancellor and approved by the Board of Trustees and the President.")
P86_CHAIR = page(86,
                 "3.9.2 Chair of Academic Department",
                 "The Chancellor appoints a faculty member preferably holding the rank of professor as department chair upon the recommendation of the "
                 "Dean, for a period of three years, renewable normally only once. A department chair may also be appointed from among faculty "
                 "members of a lower rank.")
P102_C11 = page(102,
                "2. Disciplinary Process",
                "As a precautionary measure, the Chancellor may suspend the faculty member under investigation from their duties for a maximum of "
                "three months, if deemed necessary for the investigation. The suspension period may only be extended by a decision from the "
                "Disciplinary Council. Unless otherwise specified by the Disciplinary Council, suspension from duty also results in the suspension of "
                "salaries from the date of the suspension decision.")
P111_GRATUITY = page(111,
                     "Gratuity: The family of the deceased faculty member shall also be entitled to an end-of-service gratuity. This gratuity shall be "
                     "equivalent to one month’s salary for each year of service completed by the faculty member.")
C11_PAGES = [P58_HIRING, P69_C11, P71_C11, P86_CHAIR, P102_C11, P111_GRATUITY]
C11_CORPUS_PAGES = PAGES + [p for p in C10_PAGES if p not in (P69_PERMISSION, P71_EMERITUS, P102_EXTENSION)] + C11_PAGES + [P99_MISCONDUCT]
# Step 4.12-C12 (D-099): the Faculty Grievance Committee's composition (verbatim, page 104, printed before the C10 hearing lines of
# the same page); the C12 corpus replaces P104_HEARING with it.
P104_C12 = page(104,
                "b) The Faculty Grievance Committee",
                "The Faculty Grievance Committee, composed of seven faculty members elected annually from the faculty at-large, shall be "
                "responsible for addressing grievances. The Committee shall elect a Chair from its members.",
                *P104_HEARING["lines"][1:])
C12_CORPUS_PAGES = [P104_C12 if p is P104_HEARING else p for p in C11_CORPUS_PAGES]


# ---------------------------------------------------------------------------
# Rerankers, corpus and specialist builders
# ---------------------------------------------------------------------------
class RawOverlapReranker:
    """Raw token overlap from the shared tokenizer; strongly negative when there is none."""

    def predict(self, pairs):
        return [float(len(set(tokenize(q)) & set(tokenize(t)))) or -5.0 for q, t in pairs]


class ScriptedReranker:
    """Predetermined scores: the first rule whose substring occurs in the chunk text wins; other chunks get ``default``."""

    def __init__(self, rules, default=-5.0):
        self.rules = list(rules)
        self.default = default

    def predict(self, pairs):
        out = []
        for _query, text in pairs:
            score = next((s for needle, s in self.rules if needle in text), self.default)
            out.append(float(score))
        return out


def registry_and_map():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    return registry.get_source(APPROVED_SOURCE_ID), load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))


def corpus_from(registry_and_map_value, pages):
    source, section_map = registry_and_map_value
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


def make_specialist(corpus, reranker=None, **kwargs):
    res = FacultyServicesResources(embed=lambda t: corpus.embedder.encode([t]), rerank=(reranker or RawOverlapReranker()).predict, index=corpus.index,
                                   chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source, section_map=corpus.section_map)
    return FacultyServicesSpecialist(res, **kwargs)


def task(question, *, clauses=None, system=None, intent="policy", task_id="task-1", specialist_id="faculty_services", **extra):
    context = {"full_question": question, "focus_clauses": list(clauses) if clauses else [question]}
    return SpecialistTask(task_id=task_id, question=question, specialist_id=specialist_id, domain="faculty_services", intent=intent, system=system,
                          context=context, requested_by="coordinator", **extra)


def run(sp, question, **kw):
    t = task(question, **kw)
    return check_findings(sp, t, sp.run(t))


def assert_traceable(result, corpus):
    for f in result.findings:
        chunk, meta = corpus.chunks[f.metadata["chunk_id"]], corpus.metadata[f.metadata["chunk_id"]]
        assert f.evidence_quote and f.evidence_quote in chunk
        assert f.claim == f.evidence_quote
        assert f.source_id == meta["source_id"] == APPROVED_SOURCE_ID and f.source_title == meta["source_title"]
        assert f.page == meta["page"]


__all__ = ["PAGES", "P56", "P61", "P62", "P63", "P64", "P65", "P106", "P107", "P108", "P122", "P124", "P133", "P148", "P149", "P255", "P257", "P263", "P266",
           "P268", "P69_CLINICAL", "P148_B", "P149_D", "P150_D", "P152", "P63_CONFIRMATION", "REPO", "TITLE", "RawOverlapReranker", "ScriptedReranker", "assert_traceable", "corpus_from", "make_specialist", "page",
           "registry_and_map", "run", "task", "P104_C12", "C12_CORPUS_PAGES"]
