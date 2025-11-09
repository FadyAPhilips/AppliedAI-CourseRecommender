"""
Synthetic dataset generator for the University of Windsor MAC program.

The script materializes the CSV files described in `docs/data_schema.md`.
It purposely avoids external dependencies to keep execution portable.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "synthetic"
RNG = random.Random(8760)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def join_tags(values: Iterable[str]) -> str:
    cleaned = [v for v in values if v]
    return "|".join(sorted(set(cleaned)))


def build_courses() -> List[Dict[str, str]]:
    courses: List[Dict[str, str]] = [
        {
            "course_id": "MAC-COMP-8110",
            "course_code": "COMP-8110",
            "title": "Advanced Computing Concepts",
            "credits": 3.0,
            "category": "core",
            "delivery_mode": "in-person",
            "skills": ["software-architecture", "design-patterns", "team-collaboration"],
            "prerequisites": [],
            "term_patterns": ["Fall", "Winter"],
            "difficulty_level": 3,
            "description": "Advanced topics in applied computing including design patterns, "
            "scalable architectures, and professional practice.",
        },
        {
            "course_id": "MAC-COMP-8150",
            "course_code": "COMP-8150",
            "title": "Advanced Software Engineering",
            "credits": 3.0,
            "category": "core",
            "delivery_mode": "in-person",
            "skills": ["software-engineering", "agile-methods", "requirements"],
            "prerequisites": ["MAC-COMP-8110"],
            "term_patterns": ["Fall"],
            "difficulty_level": 4,
            "description": "Software lifecycle management, agile processes, testing strategies, "
            "and delivery pipelines for enterprise-scale systems.",
        },
        {
            "course_id": "MAC-COMP-8220",
            "course_code": "COMP-8220",
            "title": "Internet Applications and Distributed Systems",
            "credits": 3.0,
            "category": "core",
            "delivery_mode": "hybrid",
            "skills": ["distributed-systems", "web-services", "cloud-computing"],
            "prerequisites": ["MAC-COMP-8110"],
            "term_patterns": ["Winter"],
            "difficulty_level": 4,
            "description": "Design and evaluation of scalable, distributed applications for the modern "
            "internet, covering microservices and event-driven architectures.",
        },
        {
            "course_id": "MAC-COMP-8250",
            "course_code": "COMP-8250",
            "title": "Advanced Systems Programming",
            "credits": 3.0,
            "category": "core",
            "delivery_mode": "in-person",
            "skills": ["systems-programming", "performance-engineering", "operating-systems"],
            "prerequisites": ["MAC-COMP-8110"],
            "term_patterns": ["Fall"],
            "difficulty_level": 5,
            "description": "Low-level programming, concurrency, and performance engineering "
            "for high-reliability systems.",
        },
        {
            "course_id": "MAC-COMP-8340",
            "course_code": "COMP-8340",
            "title": "Advanced Database Topics",
            "credits": 3.0,
            "category": "core",
            "delivery_mode": "in-person",
            "skills": ["data-modeling", "database-admin", "sql-optimization"],
            "prerequisites": ["MAC-COMP-8110"],
            "term_patterns": ["Winter"],
            "difficulty_level": 3,
            "description": "Advanced topics in relational, NoSQL, and distributed databases with "
            "emphasis on optimization and data governance.",
        },
        {
            "course_id": "MAC-COMP-8470",
            "course_code": "COMP-8470",
            "title": "Networking and Data Security",
            "credits": 3.0,
            "category": "core",
            "delivery_mode": "hybrid",
            "skills": ["cybersecurity", "network-engineering", "risk-assessment"],
            "prerequisites": ["MAC-COMP-8110"],
            "term_patterns": ["Fall", "Winter"],
            "difficulty_level": 4,
            "description": "Network protocols, secure architecture design, threat modeling, and "
            "compliance considerations for enterprise environments.",
        },
        {
            "course_id": "MAC-COMP-8650",
            "course_code": "COMP-8650",
            "title": "Applied Machine Learning",
            "credits": 3.0,
            "category": "technical-elective",
            "delivery_mode": "hybrid",
            "skills": ["machine-learning", "model-deployment", "python"],
            "prerequisites": ["MAC-COMP-8340"],
            "term_patterns": ["Winter"],
            "difficulty_level": 4,
            "description": "Hands-on machine learning with emphasis on applied modeling, "
            "evaluation, and deployment in cloud environments.",
        },
        {
            "course_id": "MAC-COMP-8720",
            "course_code": "COMP-8720",
            "title": "Cloud and DevOps Engineering",
            "credits": 3.0,
            "category": "technical-elective",
            "delivery_mode": "online",
            "skills": ["cloud-computing", "devops", "automation"],
            "prerequisites": ["MAC-COMP-8220"],
            "term_patterns": ["Fall", "Summer"],
            "difficulty_level": 3,
            "description": "Infrastructure-as-code, container orchestration, and continuous delivery "
            "practices tailored to applied computing projects.",
        },
        {
            "course_id": "MAC-COMP-8780",
            "course_code": "COMP-8780",
            "title": "Data Analytics for Business",
            "credits": 3.0,
            "category": "technical-elective",
            "delivery_mode": "in-person",
            "skills": ["data-analytics", "business-intelligence", "storytelling"],
            "prerequisites": ["MAC-COMP-8340"],
            "term_patterns": ["Fall"],
            "difficulty_level": 3,
            "description": "Analytics lifecycle, dashboarding, and translating technical insight into "
            "business strategy for stakeholders.",
        },
        {
            "course_id": "MAC-COMP-8830",
            "course_code": "COMP-8830",
            "title": "Human-Centered Computing",
            "credits": 3.0,
            "category": "technical-elective",
            "delivery_mode": "in-person",
            "skills": ["ux-design", "user-research", "accessibility"],
            "prerequisites": ["MAC-COMP-8110"],
            "term_patterns": ["Winter"],
            "difficulty_level": 2,
            "description": "User-centered design, accessibility, and evaluation methods for applied "
            "software products.",
        },
        {
            "course_id": "MAC-COMP-8890",
            "course_code": "COMP-8890",
            "title": "Applied Computing Project",
            "credits": 6.0,
            "category": "project",
            "delivery_mode": "in-person",
            "skills": ["project-delivery", "stakeholder-management", "written-communication"],
            "prerequisites": [
                "MAC-COMP-8110",
                "MAC-COMP-8150",
                "MAC-COMP-8340",
            ],
            "term_patterns": ["Fall", "Winter", "Summer"],
            "difficulty_level": 5,
            "description": "Capstone project or co-op placement applying MAC competencies to an "
            "industry problem.",
        },
        {
            "course_id": "MAC-BU-7500",
            "course_code": "BUSI-7500",
            "title": "Finance in a Global Perspective",
            "credits": 3.0,
            "category": "business",
            "delivery_mode": "in-person",
            "skills": ["finance", "quant-analysis", "decision-making"],
            "prerequisites": [],
            "term_patterns": ["Fall"],
            "difficulty_level": 2,
            "description": "Financial principles, valuation, and budgeting for technology initiatives in "
            "global markets.",
        },
        {
            "course_id": "MAC-BU-7600",
            "course_code": "BUSI-7600",
            "title": "Marketing Strategy for Technology Ventures",
            "credits": 3.0,
            "category": "business",
            "delivery_mode": "online",
            "skills": ["marketing", "product-strategy", "communication"],
            "prerequisites": [],
            "term_patterns": ["Winter"],
            "difficulty_level": 2,
            "description": "Market analysis, positioning, and go-to-market planning for software "
            "products and services.",
        },
        {
            "course_id": "MAC-BU-7700",
            "course_code": "BUSI-7700",
            "title": "Managing for Organizational Effectiveness",
            "credits": 3.0,
            "category": "business",
            "delivery_mode": "in-person",
            "skills": ["leadership", "change-management", "team-dynamics"],
            "prerequisites": [],
            "term_patterns": ["Fall", "Winter"],
            "difficulty_level": 2,
            "description": "People management, leadership styles, and organizational behaviour for "
            "technical leaders.",
        },
    ]

    for course in courses:
        course["skills"] = join_tags(course["skills"])
        course["prerequisites"] = join_tags(course["prerequisites"])
        course["term_patterns"] = join_tags(course["term_patterns"])
        course["credits"] = f'{course["credits"]:.1f}'
        course["difficulty_level"] = str(course["difficulty_level"])
    return courses


def build_course_offerings(courses: List[Dict[str, str]]) -> List[Dict[str, str]]:
    term_sequence = [
        ("2024F", {"Mon|Wed": ("10:00", "11:20"), "Tue|Thu": ("14:30", "15:50")}),
        ("2025W", {"Mon|Wed": ("13:00", "14:20"), "Tue|Thu": ("09:00", "10:20")}),
        ("2025S", {"Tue": ("18:00", "21:00"), "Thu": ("18:00", "21:00")}),
        ("2025F", {"Mon|Wed": ("10:00", "11:20"), "Tue|Thu": ("14:30", "15:50")}),
        ("2026W", {"Mon|Wed": ("13:00", "14:20"), "Tue|Thu": ("09:00", "10:20")}),
    ]
    rooms = ["Erie-1012", "Erie-2014", "Lambton-120", "Odette-2104", "Odette-140"]

    offerings: List[Dict[str, str]] = []
    for course in courses:
        meetings = list(term_sequence)
        RNG.shuffle(meetings)
        for term_code, slot_options in meetings[:3]:
            meeting_days, (start_time, end_time) = random.choice(list(slot_options.items()))
            offerings.append(
                {
                    "offering_id": f"{term_code}-{course['course_code']}-A",
                    "course_id": course["course_id"],
                    "term_code": term_code,
                    "instructor": f"Faculty-{RNG.randint(1, 20)}",
                    "meeting_days": meeting_days,
                    "start_time": start_time,
                    "end_time": end_time,
                    "location": random.choice(rooms),
                    "delivery_mode": course["delivery_mode"],
                }
            )
    return offerings


def build_degree_requirements(courses: List[Dict[str, str]]) -> List[Dict[str, str]]:
    course_ids_by_category: Dict[str, List[str]] = defaultdict(list)
    for course in courses:
        course_ids_by_category[course["category"]].append(course["course_id"])

    requirements = [
        {
            "requirement_id": "MAC_CORE",
            "label": "Complete all MAC core courses",
            "category": "core",
            "credit_min": "18.0",
            "credit_max": "",
            "eligible_courses": join_tags(course_ids_by_category["core"]),
            "notes": "Core courses build foundational applied computing competencies.",
        },
        {
            "requirement_id": "MAC_TECH_ELECTIVE",
            "label": "Choose at least two technical electives",
            "category": "technical-elective",
            "credit_min": "6.0",
            "credit_max": "12.0",
            "eligible_courses": join_tags(course_ids_by_category["technical-elective"]),
            "notes": "Technical electives can also satisfy AI stream depth areas.",
        },
        {
            "requirement_id": "MAC_BUSINESS",
            "label": "Complete two business/management courses",
            "category": "business",
            "credit_min": "6.0",
            "credit_max": "6.0",
            "eligible_courses": join_tags(course_ids_by_category["business"]),
            "notes": "Students typically select from Odette School of Business offerings.",
        },
        {
            "requirement_id": "MAC_PROJECT",
            "label": "Complete applied project or co-op placement",
            "category": "project",
            "credit_min": "6.0",
            "credit_max": "6.0",
            "eligible_courses": "MAC-COMP-8890",
            "notes": "Prerequisites must be satisfied before enrolling.",
        },
        {
            "requirement_id": "MAC_GPA",
            "label": "Maintain minimum cumulative GPA of 3.0",
            "category": "minimum-gpa",
            "credit_min": "",
            "credit_max": "",
            "eligible_courses": "",
            "notes": "Advisors review academic progress each term.",
        },
    ]
    return requirements


def random_choice_weighted(choices: List[Tuple[str, float]]) -> str:
    total = sum(weight for _, weight in choices)
    r = RNG.random() * total
    upto = 0.0
    for value, weight in choices:
        upto += weight
        if upto >= r:
            return value
    return choices[-1][0]


def build_student_population(n_students: int = 120) -> List[Dict[str, str]]:
    admit_terms = ["2024F", "2025W", "2025F"]
    majors = [
        "Computer Science",
        "Software Engineering",
        "Information Technology",
        "Electrical Engineering",
        "Data Science",
        "Mathematics",
    ]
    interests_pool = [
        "cloud-computing",
        "cybersecurity",
        "data-analytics",
        "machine-learning",
        "project-management",
        "software-engineering",
        "human-centered-design",
        "entrepreneurship",
    ]
    learning_styles = ["project-based", "lecture", "self-paced"]
    citizenship_choices = [("domestic", 0.35), ("international", 0.65)]
    demographic_clusters = ["Group-A", "Group-B", "Group-C", "Group-D"]

    students: List[Dict[str, str]] = []
    for idx in range(1, n_students + 1):
        student_id = f"STU-{idx:04d}"
        admit_term = random.choice(admit_terms)
        program_stream = "MAC-AI" if RNG.random() < 0.35 else "MAC-General"
        undergrad_major = random.choice(majors)
        gpa_entry = round(RNG.normalvariate(3.25, 0.25), 2)
        gpa_entry = max(2.7, min(gpa_entry, 3.9))
        co_op_status = random.choice(["seeking", "placed", "not-applicable"])
        demographic_group = random.choice(demographic_clusters)
        citizenship_status = random_choice_weighted(citizenship_choices)
        interests = RNG.sample(interests_pool, k=random.randint(2, 4))
        learning_style = random.choice(learning_styles)
        work_exp = max(0.0, round(random.random() * 8, 1))

        students.append(
            {
                "student_id": student_id,
                "admit_term": admit_term,
                "program_stream": program_stream,
                "undergrad_major": undergrad_major,
                "gpa_entry": f"{gpa_entry:.2f}",
                "co_op_status": co_op_status,
                "demographic_group": demographic_group,
                "citizenship_status": citizenship_status,
                "interests": join_tags(interests),
                "learning_style": learning_style,
                "work_experience_years": f"{work_exp:.1f}",
            }
        )
    return students


def next_term(term_code: str) -> str:
    year = int(term_code[:4])
    season = term_code[4]
    if season == "F":
        return f"{year + 1}W"
    if season == "W":
        return f"{year}S"
    if season == "S":
        return f"{year}F"
    raise ValueError(f"Invalid term code: {term_code}")


def build_term_sequence(start_term: str, length: int) -> List[str]:
    terms = [start_term]
    while len(terms) < length:
        terms.append(next_term(terms[-1]))
    return terms


def choose_courses_for_term(
    program_stream: str,
    completed_course_ids: set,
    all_courses: Dict[str, Dict[str, str]],
    term_idx: int,
) -> List[str]:
    """Assign 3–4 courses per term following basic prerequisite logic."""
    required_core = [
        "MAC-COMP-8110",
        "MAC-COMP-8150",
        "MAC-COMP-8220",
        "MAC-COMP-8250",
        "MAC-COMP-8340",
        "MAC-COMP-8470",
    ]
    business_courses = [
        "MAC-BU-7500",
        "MAC-BU-7600",
        "MAC-BU-7700",
    ]
    technical_electives = [
        "MAC-COMP-8650",
        "MAC-COMP-8720",
        "MAC-COMP-8780",
        "MAC-COMP-8830",
    ]
    course_load = 4 if term_idx < 2 else 3
    planned: List[str] = []

    # Prioritize outstanding core courses.
    outstanding_core = [c for c in required_core if c not in completed_course_ids]
    RNG.shuffle(outstanding_core)
    while outstanding_core and len(planned) < course_load:
        candidate = outstanding_core.pop()
        prereqs = all_courses[candidate]["prerequisites"].split("|") if all_courses[candidate]["prerequisites"] else []
        if all(pr in completed_course_ids for pr in prereqs):
            planned.append(candidate)

    # Add business courses after first term.
    if len(planned) < course_load and term_idx >= 1:
        available_business = [c for c in business_courses if c not in completed_course_ids and c not in planned]
        RNG.shuffle(available_business)
        while available_business and len(planned) < course_load:
            planned.append(available_business.pop())

    # Technical electives for AI stream or later terms.
    if len(planned) < course_load:
        available_tech = [c for c in technical_electives if c not in planned]
        if program_stream == "MAC-AI":
            RNG.shuffle(available_tech)
        else:
            RNG.shuffle(available_tech)
        while available_tech and len(planned) < course_load:
            candidate = available_tech.pop()
            prereqs = all_courses[candidate]["prerequisites"].split("|") if all_courses[candidate]["prerequisites"] else []
            if all(pr in completed_course_ids for pr in prereqs):
                planned.append(candidate)

    return planned


def grade_from_point(point: float) -> str:
    if point >= 3.9:
        return "A+"
    if point >= 3.7:
        return "A"
    if point >= 3.3:
        return "A-"
    if point >= 3.0:
        return "B+"
    if point >= 2.7:
        return "B"
    if point >= 2.3:
        return "B-"
    if point >= 2.0:
        return "C+"
    if point >= 1.7:
        return "C"
    if point >= 1.3:
        return "C-"
    if point >= 1.0:
        return "D"
    return "F"


def build_enrollments(
    students: List[Dict[str, str]], courses: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    course_map = {course["course_id"]: course for course in courses}
    enrollments: List[Dict[str, str]] = []
    enrollments_by_student: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    enrollment_counter = 1

    for student in students:
        start_term = student["admit_term"]
        num_terms = random.randint(2, 4)
        terms = build_term_sequence(start_term, num_terms)
        completed_courses: set = set()
        for term_idx, term_code in enumerate(terms):
            planned_courses = choose_courses_for_term(
                student["program_stream"], completed_courses, course_map, term_idx
            )
            if term_idx == len(terms) - 1 and "MAC-COMP-8890" not in completed_courses:
                # Reserve project for final term if prerequisites mostly met.
                if "MAC-COMP-8890" not in planned_courses and len(planned_courses) < 4:
                    planned_courses.append("MAC-COMP-8890")

            for course_id in planned_courses:
                course = course_map[course_id]
                completion_status = "completed"
                grade_point = random.gauss(3.3, 0.4)
                if term_idx == len(terms) - 1 and RNG.random() < 0.45:
                    completion_status = "in-progress"
                    grade_point = None
                elif RNG.random() < 0.05:
                    completion_status = "withdrawn"
                    grade_point = None

                grade_point_str = "" if grade_point is None else f"{max(1.0, min(4.0, grade_point)):.2f}"
                grade_letter = "" if grade_point is None else grade_from_point(float(grade_point_str))
                feedback_rating = "" if completion_status != "completed" else str(random.randint(3, 5))
                hours_per_week = random.uniform(6, 14)
                engagement_score = random.uniform(0.45, 0.95)

                enrollment = {
                    "enrollment_id": f"ENR-{enrollment_counter:05d}",
                    "student_id": student["student_id"],
                    "course_id": course_id,
                    "term_code": term_code,
                    "grade_point": grade_point_str,
                    "grade_letter": grade_letter,
                    "completion_status": completion_status,
                    "feedback_rating": feedback_rating,
                    "hours_per_week": f"{hours_per_week:.1f}",
                    "engagement_score": f"{engagement_score:.2f}",
                }
                enrollments.append(enrollment)
                enrollments_by_student[student["student_id"]].append(enrollment)
                enrollment_counter += 1

                if completion_status == "completed":
                    completed_courses.add(course_id)

    return enrollments, enrollments_by_student


def term_sort_key(term_code: str) -> Tuple[int, int]:
    season_rank = {"W": 1, "S": 2, "F": 3}
    return int(term_code[:4]), season_rank.get(term_code[4], 0)


def compute_performance_profiles(
    students: List[Dict[str, str]],
    enrollments_by_student: Dict[str, List[Dict[str, str]]],
    courses: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    course_lookup = {course["course_id"]: course for course in courses}
    results: List[Dict[str, str]] = []

    for student in students:
        student_id = student["student_id"]
        interactions = enrollments_by_student.get(student_id, [])
        completed = [
            enr
            for enr in interactions
            if enr["completion_status"] == "completed" and enr["grade_point"]
        ]

        if completed:
            cumulative = sum(float(enr["grade_point"]) for enr in completed) / len(completed)
            # Determine last term GPA
            completed.sort(key=lambda e: term_sort_key(e["term_code"]))
            last_term = completed[-1]["term_code"]
            last_term_grades = [
                float(enr["grade_point"])
                for enr in completed
                if enr["term_code"] == last_term
            ]
            last_term_gpa = sum(last_term_grades) / len(last_term_grades)
        else:
            cumulative = student.get("gpa_entry", 3.0)
            last_term_gpa = float(student.get("gpa_entry", 3.0))

        def strength_metric(filter_fn) -> int:
            relevant = [
                float(enr["grade_point"])
                for enr in completed
                if filter_fn(course_lookup[enr["course_id"]])
            ]
            if not relevant:
                relevant = [float(student.get("gpa_entry", 3.0))]
            avg = sum(relevant) / len(relevant)
            if avg >= 3.8:
                return 5
            if avg >= 3.5:
                return 4
            if avg >= 3.0:
                return 3
            if avg >= 2.5:
                return 2
            return 1

        technical_strength = strength_metric(
            lambda course: course["category"] in {"core", "technical-elective", "project"}
        )
        analytical_strength = strength_metric(
            lambda course: "data-analytics" in course["skills"]
            or "machine-learning" in course["skills"]
        )
        communication_strength = strength_metric(
            lambda course: course["category"] == "business"
            or "project-delivery" in course["skills"]
        )

        drop = cumulative - last_term_gpa
        risk_flag = "none"
        if last_term_gpa < 2.7 or drop >= 0.6:
            risk_flag = "performance-drop"
        elif student["co_op_status"] == "seeking" and cumulative < 3.0:
            risk_flag = "co-op-risk"

        results.append(
            {
                "student_id": student_id,
                "cumulative_gpa": f"{cumulative:.2f}",
                "last_term_gpa": f"{last_term_gpa:.2f}",
                "technical_strength": str(technical_strength),
                "analytical_strength": str(analytical_strength),
                "communication_strength": str(communication_strength),
                "risk_flag": risk_flag,
            }
        )
    return results


def build_student_preferences(
    students: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    career_options = [
        "ai-specialist",
        "cloud-engineer",
        "cybersecurity-analyst",
        "product-manager",
        "data-analyst",
        "software-architect",
    ]
    delivery_pref = ["in-person", "hybrid", "online"]
    time_of_day_options = ["morning", "afternoon", "evening"]
    skill_focus_options = [
        "machine-learning|model-deployment",
        "cloud-computing|devops",
        "cybersecurity|threat-modeling",
        "data-analytics|visualization",
        "leadership|communication",
    ]

    preferences: List[Dict[str, str]] = []
    for student in students:
        student_id = student["student_id"]
        career_goal = random.choice(career_options)
        delivery = random.choice(delivery_pref)
        time_pref = random.choice(time_of_day_options)
        skill_focus = random.choice(skill_focus_options)

        preferences.extend(
            [
                {
                    "student_id": student_id,
                    "preference_type": "career_goal",
                    "preference_value": career_goal,
                    "weight": f"{random.uniform(0.6, 1.0):.2f}",
                    "source": "survey",
                },
                {
                    "student_id": student_id,
                    "preference_type": "delivery_mode",
                    "preference_value": delivery,
                    "weight": f"{random.uniform(0.3, 0.8):.2f}",
                    "source": "survey",
                },
                {
                    "student_id": student_id,
                    "preference_type": "time_of_day",
                    "preference_value": time_pref,
                    "weight": f"{random.uniform(0.2, 0.6):.2f}",
                    "source": "survey",
                },
                {
                    "student_id": student_id,
                    "preference_type": "skills_to_build",
                    "preference_value": skill_focus,
                    "weight": f"{random.uniform(0.4, 0.9):.2f}",
                    "source": "advisor_note",
                },
            ]
        )
    return preferences


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_data_dir()

    courses = build_courses()
    course_offerings = build_course_offerings(courses)
    degree_requirements = build_degree_requirements(courses)
    students = build_student_population()
    enrollments, enrollments_by_student = build_enrollments(students, courses)
    performance = compute_performance_profiles(students, enrollments_by_student, courses)
    preferences = build_student_preferences(students)

    write_csv(DATA_DIR / "courses.csv", courses)
    write_csv(DATA_DIR / "course_offerings.csv", course_offerings)
    write_csv(DATA_DIR / "degree_requirements.csv", degree_requirements)
    write_csv(DATA_DIR / "students.csv", students)
    write_csv(DATA_DIR / "student_performance.csv", performance)
    write_csv(DATA_DIR / "enrollments.csv", enrollments)
    write_csv(DATA_DIR / "student_preferences.csv", preferences)

    print(f"Wrote synthetic dataset to {DATA_DIR}")


if __name__ == "__main__":
    main()

