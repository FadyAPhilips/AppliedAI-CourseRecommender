from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Set

from .data_loader import SyntheticDataset, get_dataset


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    spread = max_score - min_score
    return {key: (value - min_score) / spread for key, value in scores.items()}


def _prerequisites_met(
    course_id: str, completed_courses: Set[str], dataset: SyntheticDataset
) -> bool:
    prereqs = dataset.courses[course_id].get("prerequisites", "")
    if not prereqs:
        return True
    return _split_to_set(prereqs).issubset(completed_courses)


def _split_to_set(value: str) -> Set[str]:
    return {item.strip() for item in value.split("|") if item.strip()}


def _content_profile_from_courses(
    course_ids: Iterable[str], dataset: SyntheticDataset
) -> Counter:
    profile: Counter[str] = Counter()
    for course_id in course_ids:
        skills = dataset.course_skill_tags.get(course_id, set())
        for skill in skills:
            profile[skill] += 1.0
        course = dataset.courses.get(course_id)
        if course:
            profile[f"category::{course['category']}"] += 0.6
            for term in _split_to_set(course.get("term_patterns", "")):
                profile[f"term::{term.lower()}"] += 0.2
            profile[f"delivery::{course.get('delivery_mode', '')}"] += 0.4
    return profile


def _content_profile_from_interests(
    interests: Sequence[str],
) -> Counter:
    profile: Counter[str] = Counter()
    for interest in interests:
        profile[interest] += 1.0
    return profile


def _score_content(
    candidate_courses: Set[str],
    profile: Counter,
    dataset: SyntheticDataset,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for course_id in candidate_courses:
        course = dataset.courses[course_id]
        score = 0.0
        for skill in dataset.course_skill_tags.get(course_id, set()):
            score += profile.get(skill, 0.0)
        score += profile.get(f"category::{course['category']}", 0.0)
        score += profile.get(f"delivery::{course.get('delivery_mode', '')}", 0.0)
        for term in _split_to_set(course.get("term_patterns", "")):
            score += 0.1 * profile.get(f"term::{term.lower()}", 0.0)
        scores[course_id] = score
    return scores


def _score_collaborative_history(
    student_id: str,
    candidate_courses: Set[str],
    completed_courses: Set[str],
    dataset: SyntheticDataset,
) -> Dict[str, float]:
    collab_scores: Dict[str, float] = defaultdict(float)
    for other_student, other_courses in dataset.student_completed_courses.items():
        if other_student == student_id:
            continue
        intersection = len(completed_courses & other_courses)
        if not intersection:
            continue
        union = len(completed_courses | other_courses)
        if not union:
            continue
        similarity = intersection / union
        if similarity <= 0.0:
            continue

        for course_id in other_courses:
            if course_id in completed_courses or course_id not in candidate_courses:
                continue
            collab_scores[course_id] += similarity
    return dict(collab_scores)


def _score_collaborative_interests(
    candidate_courses: Set[str],
    interest_tags: Set[str],
    dataset: SyntheticDataset,
) -> Dict[str, float]:
    collab_scores: Dict[str, float] = defaultdict(float)
    if not interest_tags:
        return collab_scores

    for student_id, tags in dataset.student_interest_tags.items():
        overlap = len(tags & interest_tags)
        if not overlap:
            continue
        similarity = overlap / len(interest_tags)
        if similarity <= 0.0:
            continue

        for course_id in dataset.student_completed_courses.get(student_id, set()):
            if course_id not in candidate_courses:
                continue
            collab_scores[course_id] += similarity
    return dict(collab_scores)


def _candidate_courses(
    completed_courses: Set[str],
    dataset: SyntheticDataset,
) -> Set[str]:
    return {
        course_id
        for course_id in dataset.courses
        if course_id not in completed_courses
    }


def recommend_for_student(
    student_id: str,
    top_n: int = 6,
) -> List[Dict[str, object]]:
    dataset = get_dataset()
    completed_courses = dataset.student_completed_courses.get(student_id, set())
    candidate = {
        course_id
        for course_id in _candidate_courses(completed_courses, dataset)
        if _prerequisites_met(course_id, completed_courses, dataset)
    }
    if not candidate:
        return []

    content_profile = _content_profile_from_courses(completed_courses, dataset)
    content_scores = _normalize(
        _score_content(candidate, content_profile, dataset)
    )

    collab_scores = _normalize(
        _score_collaborative_history(student_id, candidate, completed_courses, dataset)
    )

    combined_scores: Dict[str, float] = {}
    for course_id in candidate:
        content = content_scores.get(course_id, 0.0)
        collab = collab_scores.get(course_id, 0.0)
        combined_scores[course_id] = 0.6 * content + 0.4 * collab

    return _build_recommendation_payload(
        combined_scores,
        content_scores,
        collab_scores,
        dataset,
        top_n=top_n,
    )


def recommend_for_interests(
    interest_tags: Sequence[str],
    top_n: int = 6,
) -> List[Dict[str, object]]:
    dataset = get_dataset()
    cleaned_interests = {tag for tag in interest_tags if tag}
    candidate = _candidate_courses(set(), dataset)
    if not candidate:
        return []

    content_profile = _content_profile_from_interests(tuple(cleaned_interests))
    content_scores = _normalize(
        _score_content(candidate, content_profile, dataset)
    )
    collab_scores = _normalize(
        _score_collaborative_interests(candidate, cleaned_interests, dataset)
    )

    combined_scores: Dict[str, float] = {}
    for course_id in candidate:
        content = content_scores.get(course_id, 0.0)
        collab = collab_scores.get(course_id, 0.0)
        combined_scores[course_id] = 0.7 * content + 0.3 * collab

    return _build_recommendation_payload(
        combined_scores,
        content_scores,
        collab_scores,
        dataset,
        top_n=top_n,
    )


def _build_recommendation_payload(
    combined_scores: Dict[str, float],
    content_scores: Dict[str, float],
    collab_scores: Dict[str, float],
    dataset: SyntheticDataset,
    top_n: int,
) -> List[Dict[str, object]]:
    ranked = sorted(
        combined_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]

    results: List[Dict[str, object]] = []
    for course_id, score in ranked:
        course = dataset.courses[course_id]
        skills = sorted(dataset.course_skill_tags.get(course_id, []))
        explanation = _build_explanation(
            course_id,
            content_scores.get(course_id, 0.0),
            collab_scores.get(course_id, 0.0),
            skills,
            dataset,
        )
        results.append(
            {
                "course_id": course_id,
                "course_code": course.get("course_code"),
                "title": course.get("title"),
                "category": course.get("category"),
                "delivery_mode": course.get("delivery_mode"),
                "combined_score": round(score, 3),
                "content_score": round(content_scores.get(course_id, 0.0), 3),
                "collab_score": round(collab_scores.get(course_id, 0.0), 3),
                "skills": skills,
                "description": course.get("description"),
                "explanation": explanation,
            }
        )
    return results


def _build_explanation(
    course_id: str,
    content_score: float,
    collab_score: float,
    skills: Sequence[str],
    dataset: SyntheticDataset,
) -> str:
    fragments: List[str] = []
    if content_score:
        highlighted_skills = ", ".join(skills[:3]) if skills else "related skills"
        fragments.append(f"matches your focus on {highlighted_skills}")
    if collab_score:
        peers = len(dataset.collaborative_matrix.get(course_id, set()))
        fragments.append(f"popular with {peers} similar students")
    if not fragments:
        fragments.append("broadens your MAC coursework")
    return " and ".join(fragments)



