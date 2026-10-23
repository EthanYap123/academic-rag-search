"""Generate the sample handbooks in data/samples/.

The documents describe a fictional institution (Northbridge Polytechnic) and are
written for this project, so they are free to publish. Run:

    python scripts/make_samples.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "samples"
SCHOOL = "Northbridge Polytechnic"

# Each document: (filename, title, [pages]); each page is a list of sections;
# each section is (heading, [paragraph or ["bullet", ...]]).
DOCUMENTS = [
    (
        "academic-policy-handbook.pdf",
        "Academic Policy Handbook 2026",
        [
            [
                ("1 Purpose and Scope", [
                    "This handbook sets out the academic rules that apply to every full-time diploma student at Northbridge Polytechnic for Academic Year 2026. Where a module guide gives a stricter rule, the module guide applies to that module only.",
                    "Students are expected to read this handbook at the start of each semester. Not knowing a rule is not accepted as a reason for breaking it.",
                ]),
                ("2 Attendance Requirements", [
                    "Students must attend at least 85 percent of the scheduled lessons in each module, including lectures, tutorials and practical sessions. Attendance is taken in the first 15 minutes of every lesson through the NorthbridgeOne app.",
                    "A student who arrives more than 15 minutes after the lesson starts is marked late. Three late arrivals in the same module count as one absence.",
                    "A student whose attendance falls below 85 percent in a module is barred from the final assessment of that module and receives a grade of F (Fail, debarred). Absences covered by a valid medical certificate or an approved leave of absence are excluded from the calculation.",
                ]),
                ("2.1 Leave of Absence", [
                    "Students who expect to miss lessons for official representation, compassionate reasons or national service obligations must apply for a leave of absence through the Student Portal at least five working days in advance. Compassionate leave for the death of an immediate family member may be applied for within three working days after the event.",
                ]),
            ],
            [
                ("3 Assessment and Grading", [
                    "Each module is assessed through a combination of coursework and a final examination or final project. The weighting of each component is stated in the module guide published in Week 1.",
                    "Grades are awarded as follows:",
                    ["A: 80 marks and above (grade point 4.0)",
                     "B+: 75 to 79 marks (grade point 3.5)",
                     "B: 70 to 74 marks (grade point 3.0)",
                     "C+: 65 to 69 marks (grade point 2.5)",
                     "C: 60 to 64 marks (grade point 2.0)",
                     "D+: 55 to 59 marks (grade point 1.5)",
                     "D: 50 to 54 marks (grade point 1.0)",
                     "F: below 50 marks (grade point 0.0)"],
                    "The Grade Point Average (GPA) is the credit-weighted average of grade points for all modules taken in a semester. The Cumulative GPA (CGPA) covers every module taken since the start of the diploma.",
                ]),
                ("3.1 Late Submission of Coursework", [
                    "Coursework submitted after the deadline without an approved extension is penalised by 10 percent of the marks obtained for each calendar day it is late, up to a maximum of three days. Work submitted more than three days late receives zero marks.",
                    "An extension of up to five working days may be granted by the module coordinator if the request is made before the deadline and is supported by evidence, such as a medical certificate. Requests made after the deadline are only considered in exceptional circumstances.",
                ]),
                ("3.2 Academic Probation", [
                    "A student whose semester GPA falls below 1.5 is placed on academic probation for the following semester and must meet an academic advisor at least twice during that semester. A student who is placed on probation for two consecutive semesters may be withdrawn from the diploma.",
                ]),
            ],
            [
                ("4 Academic Integrity", [
                    "All submitted work must be the student's own. Plagiarism, collusion, contract cheating and the fabrication of data are serious academic offences.",
                    "Students may use generative AI tools only where the module guide explicitly allows it, and must declare how the tool was used in an appendix to the submission. Undeclared use of generative AI is treated as plagiarism.",
                    "Penalties depend on the severity of the offence and whether it is a first offence:",
                    ["First minor offence: a written warning and the work is re-marked with the copied portions excluded.",
                     "First major offence: zero marks for the assessment component.",
                     "Repeat offence: zero marks for the module and referral to the Disciplinary Committee, which may suspend or expel the student."],
                ]),
                ("5 Appeals", [
                    "A student may appeal against a final module grade within five working days after results are released. Appeals are submitted through the Student Portal and cost an administrative fee of 20 dollars per module, which is refunded if the appeal results in a grade change.",
                    "An appeal is a review of whether marks were totalled and recorded correctly and whether the assessment followed the published rules. It is not a re-mark of the student's answers against a different standard. The outcome is released within 15 working days.",
                ]),
            ],
        ],
    ),
    (
        "examination-regulations.pdf",
        "Examination Regulations 2026",
        [
            [
                ("1 Examination Schedule", [
                    "Final examinations take place in Weeks 17 and 18 of each semester. The examination timetable is published on the Student Portal at least four weeks before the first paper.",
                    "Students must check their own timetable, seat number and venue. Examinations are not rescheduled because of personal travel plans or clashes with part-time work.",
                ]),
                ("2 Admission to the Examination Hall", [
                    "Candidates must arrive at the venue at least 15 minutes before the paper begins and must bring their student card. A candidate without a student card must obtain a temporary examination pass from the Examination Office before the paper starts.",
                    "Candidates who arrive more than 30 minutes after the start of the paper are not admitted. No candidate may leave the hall during the first 30 minutes or the last 15 minutes of a paper.",
                ]),
                ("3 Permitted and Prohibited Items", [
                    "Only the following items may be brought to the candidate's desk:",
                    ["Pens, pencils, erasers, rulers and a clear pencil case.",
                     "A calculator from the approved list, with its cover removed. Programmable and graphing calculators are not allowed unless the paper states otherwise.",
                     "A clear bottle of water without a label."],
                    "Mobile phones, smart watches, earbuds and other electronic devices must be switched off and placed in the candidate's bag at the front of the hall. Having a device on one's person during a paper is treated as attempted cheating, whether or not the device was used.",
                ]),
            ],
            [
                ("4 Absence from an Examination", [
                    "A candidate who is absent from a final examination receives zero marks for that paper unless the absence is excused.",
                    "To be excused on medical grounds, the candidate must submit a medical certificate through the Student Portal within two working days after the paper. Medical certificates must be issued by a registered medical practitioner and must cover the date of the paper.",
                    "A candidate whose absence is excused may sit a make-up examination, normally scheduled within four weeks after the end of the examination period. The make-up paper is graded without any cap on marks.",
                ]),
                ("5 Supplementary Examinations", [
                    "A student who fails a module with a final mark between 40 and 49 may be offered one supplementary examination for that module. The grade for a module passed through a supplementary examination is capped at D (50 marks).",
                    "Supplementary examinations are held in the week after the semester results are released. A student may take supplementary examinations in at most two modules in a semester.",
                ]),
                ("6 Examination Misconduct", [
                    "Examination misconduct includes copying from another candidate, communicating with anyone other than an invigilator, bringing unauthorised materials and removing an answer script from the hall.",
                    "Invigilators will record any suspected misconduct and allow the candidate to finish the paper. The case is referred to the Examination Board, which may award zero marks for the paper or the module and refer the matter to the Disciplinary Committee.",
                ]),
                ("7 Release of Results", [
                    "Semester results are released on the Student Portal about three weeks after the last paper. Results are withheld from students who have outstanding fees or library fines above 50 dollars until the amount is paid.",
                ]),
            ],
        ],
    ),
    (
        "student-conduct-code.pdf",
        "Student Code of Conduct",
        [
            [
                ("1 Principles", [
                    "Northbridge Polytechnic expects students to act with honesty, respect and responsibility, on campus, during internships and overseas programmes, and online when they can be identified as Northbridge students.",
                ]),
                ("2 Dress and Behaviour on Campus", [
                    "Students should dress neatly and appropriately for a learning environment. Laboratory and workshop sessions require covered shoes and any protective equipment listed in the safety briefing; students without them will not be allowed to take part.",
                    "Smoking and vaping are prohibited everywhere on campus, including car parks. Eating is not allowed in lecture theatres, laboratories or the library, except in designated cafe areas.",
                ]),
                ("3 Harassment and Bullying", [
                    "Harassment, bullying and discrimination of any kind, in person or online, will not be tolerated. This includes sharing images of another person without consent, repeated unwanted messages and offensive remarks about race, religion, gender or disability.",
                    "Students who experience or witness harassment can report it to any lecturer, to Student Services, or anonymously through the Speak Up form on the Student Portal. Reports are handled confidentially and the person making a report in good faith will not be penalised.",
                ]),
            ],
            [
                ("4 Disciplinary Process", [
                    "Minor breaches of this code, such as eating in the library, are dealt with by the staff member on the spot and may lead to a verbal or written warning.",
                    "Serious or repeated breaches are referred to the Disciplinary Committee. The student is told of the allegation in writing and is given at least five working days to prepare a written response before the hearing. The student may bring one support person, who may not speak on the student's behalf.",
                    "The Committee may impose one or more of the following sanctions:",
                    ["A written reprimand placed on the student's record.",
                     "Community service of up to 40 hours.",
                     "Suspension from the polytechnic for up to one semester.",
                     "Expulsion, for the most serious cases."],
                ]),
                ("5 Appeal against a Disciplinary Decision", [
                    "A student may appeal against a decision of the Disciplinary Committee to the Deputy Principal within ten working days after receiving the decision in writing. The appeal must state the grounds, such as new evidence or a failure to follow the process. The decision of the Deputy Principal is final.",
                ]),
            ],
        ],
    ),
    (
        "it-and-library-services-guide.pdf",
        "IT and Library Services Guide",
        [
            [
                ("1 Your Student Account", [
                    "Every student receives a Northbridge account that gives access to email, the Student Portal, the learning management system and campus Wi-Fi. The account is activated on the first day of orientation.",
                    "Passwords must be at least 12 characters long and are changed every 180 days. Multi-factor authentication through the NorthbridgeOne app is compulsory. If you forget your password, reset it at the self-service reset page or visit the IT Help Desk at Block B, Level 2, with your student card.",
                    "Never share your account. Activity on your account is treated as your own, including any breach of the Acceptable Use rules.",
                ]),
                ("2 Campus Wi-Fi", [
                    "Connect to the NB-Secure network and sign in with your Northbridge account. The NB-Guest network is for visitors only, is limited to 2 hours per day and does not give access to internal systems.",
                    "Each student may register up to three devices on NB-Secure at a time. Devices can be removed from the Student Portal under My Devices.",
                ]),
                ("3 Acceptable Use", [
                    "Campus IT resources are provided for learning. Students must not attempt to access systems they are not authorised to use, run network scanning tools, mine cryptocurrency or host file-sharing services on the campus network. Breaches may lead to the account being suspended and to action under the Student Code of Conduct.",
                ]),
            ],
            [
                ("4 Library Opening Hours", [
                    "During the semester the library is open from 8.00 am to 9.00 pm on weekdays and from 9.00 am to 1.00 pm on Saturdays. It is closed on Sundays and public holidays. During examination weeks, the library stays open until 11.00 pm on weekdays.",
                ]),
                ("5 Borrowing", [
                    "Students may borrow up to eight items at a time. Books are loaned for 14 days and may be renewed twice online, as long as no other user has reserved them. Reserved high-demand textbooks are loaned for 2 hours and must be used inside the library.",
                    "Overdue books are fined 20 cents per day per item. Overdue 2-hour loans are fined 50 cents per hour. Borrowing rights are suspended while fines exceed 10 dollars.",
                    "Lost or damaged items must be reported at the loans counter. The student pays the replacement cost of the item plus a processing fee of 15 dollars.",
                ]),
                ("6 Discussion Rooms and Printing", [
                    "Discussion rooms can be booked through the Student Portal for up to 2 hours per day, for groups of three to eight students. A booking is released if nobody checks in within 15 minutes of the start time.",
                    "Each student receives a printing quota of 30 dollars per semester. Black-and-white printing costs 5 cents per page and colour printing costs 30 cents per page. Unused quota does not carry over to the next semester.",
                ]),
            ],
        ],
    ),
]


def build(filename: str, title: str, pages: list) -> None:
    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("t", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, spaceAfter=14)
    h_section = ParagraphStyle("h", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=6)

    total = len(pages)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillGray(0.4)
        canvas.drawString(20 * mm, A4[1] - 12 * mm, f"{SCHOOL} | {title}")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page} of {total}")
        canvas.restoreState()

    story = [Paragraph(title, h_title)]
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        for heading, blocks in page:
            story.append(Paragraph(heading, h_section))
            for block in blocks:
                if isinstance(block, list):
                    story.append(ListFlowable(
                        [ListItem(Paragraph(item, body), leftIndent=12) for item in block],
                        bulletType="bullet", start="•", leftIndent=14,
                    ))
                    story.append(Spacer(1, 4))
                else:
                    story.append(Paragraph(block, body))

    doc = SimpleDocTemplate(
        str(OUT / filename), pagesize=A4, title=title, author=SCHOOL,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=22 * mm, bottomMargin=20 * mm,
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, title, pages in DOCUMENTS:
        build(filename, title, pages)
        print("wrote", OUT / filename)
