from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "synthetic"


def _read_csv(filename: str) -> List[Dict[str, str]]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset file missing: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


@dataclass(frozen=True)
class SyntheticDataset:
    courses: Dict[str, Dict[str, str]]
    students: Dict[str, Dict[str, str]]
    enrollments: List[Dict[str, str]]
    preferences: Dict[str, List[Dict[str, str]]]
    performance: Dict[str, Dict[str, str]]
    student_completed_courses: Dict[str, Set[str]]
    student_interest_tags: Dict[str, Set[str]]
    course_skill_tags: Dict[str, Set[str]]
    collaborative_matrix: Dict[str, Set[str]]
    interest_catalog: Tuple[str, ...]


def _split_tags(value: str) -> Set[str]:
    return {item.strip() for item in value.split("|") if item.strip()} if value else set()


def _build_dataset() -> SyntheticDataset:
    courses_rows = _read_csv("courses.csv")
    students_rows = _read_csv("students.csv")
    enrollments_rows = _read_csv("enrollments.csv")
    preferences_rows = _read_csv("student_preferences.csv")
    performance_rows = _read_csv("student_performance.csv")

    courses = {row["course_id"]: row for row in courses_rows}
    students = {row["student_id"]: row for row in students_rows}
    performance = {row["student_id"]: row for row in performance_rows}

    preferences: Dict[str, List[Dict[str, str]]] = {}
    for row in preferences_rows:
        preferences.setdefault(row["student_id"], []).append(row)

    course_skill_tags = {
        course_id: _split_tags(row.get("skills", ""))
        for course_id, row in courses.items()
    }

    student_completed_courses: Dict[str, Set[str]] = {}
    collaborative_matrix: Dict[str, Set[str]] = {}
    for row in enrollments_rows:
        if row.get("completion_status") != "completed":
            continue
        student_id = row["student_id"]
        course_id = row["course_id"]
        student_completed_courses.setdefault(student_id, set()).add(course_id)
        collaborative_matrix.setdefault(course_id, set()).add(student_id)

    student_interest_tags: Dict[str, Set[str]] = {}
    for student_id, student in students.items():
        tags = _split_tags(student.get("interests", ""))
        pref_tags = {
            pref_tag
            for pref in preferences.get(student_id, [])
            if pref.get("preference_type") in {"skills_to_build", "career_goal"}
            for pref_tag in _split_tags(pref.get("preference_value", ""))
        }
        student_interest_tags[student_id] = tags | pref_tags

    interest_catalog = tuple(
        sorted(
            {
                tag
                for tags in student_interest_tags.values()
                for tag in tags
            }
        )
    )

    return SyntheticDataset(
        courses=courses,
        students=students,
        enrollments=enrollments_rows,
        preferences=preferences,
        performance=performance,
        student_completed_courses=student_completed_courses,
        student_interest_tags=student_interest_tags,
        course_skill_tags=course_skill_tags,
        collaborative_matrix=collaborative_matrix,
        interest_catalog=interest_catalog,
    )


@lru_cache(maxsize=1)
def get_dataset() -> SyntheticDataset:
    return _build_dataset()



