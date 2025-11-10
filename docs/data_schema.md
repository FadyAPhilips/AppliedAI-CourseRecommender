# Synthetic Dataset Schema (University of Windsor MAC Program)

## Overview
The synthetic dataset mirrors key entities required by the hybrid recommendation methodology: students, courses, historical enrollments, requirements, and schedule slots. Each table is exported as a CSV file (UTF-8, comma-delimited) under `data/synthetic/`. List-valued attributes are pipe-delimited within a single cell (e.g., `machine-learning|cloud-computing`).

| File | Purpose | Primary Keys |
| --- | --- | --- |
| `courses.csv` | Canonical catalogue of MAC courses and supporting business electives | `course_id` |
| `course_offerings.csv` | Term-specific schedule to support conflict checks and cold-start handling | composite (`course_id`, `term_code`) |
| `degree_requirements.csv` | Program constraints and credit requirements for recommendation filtering | `requirement_id` |
| `students.csv` | Synthetic student cohort with demographics, academic profile, and interests | `student_id` |
| `student_performance.csv` | GPA trajectory and skill proficiency features derived from grades | `student_id` |
| `enrollments.csv` | Historical student-course interactions with grades, feedback, and engagement | `enrollment_id` |
| `student_preferences.csv` | Explicit preference signals gathered from survey-style responses | composite (`student_id`, `preference_type`) |

## `courses.csv`
| Column | Type | Description |
| --- | --- | --- |
| `course_id` | string | Unique identifier (e.g., `MAC-COMP-8150`). |
| `course_code` | string | Official or representative code (e.g., `COMP-8150`). |
| `title` | string | Course name. |
| `credits` | float | Credit weight (MAC courses use 3.0). |
| `category` | string | One of `core`, `technical-elective`, `business`, `project`. |
| `delivery_mode` | string | `in-person`, `hybrid`, or `online`. |
| `skills` | string | Pipe-delimited list of skill tags. |
| `prerequisites` | string | Pipe-delimited list of prerequisite course_ids. |
| `term_patterns` | string | Typical terms offered (`Fall|Winter`, etc.). |
| `difficulty_level` | integer | Ordinal difficulty (1–5) derived from historical pass rates. |
| `description` | string | Short synopsis for content-based filtering. |

## `course_offerings.csv`
| Column | Type | Description |
| --- | --- | --- |
| `offering_id` | string | Unique schedule identifier (e.g., `2025F-COMP-8150-A`). |
| `course_id` | string | Foreign key to `courses.csv`. |
| `term_code` | string | Academic term code (`2025F`, `2026W`, etc.). |
| `instructor` | string | Placeholder instructor label (`Faculty-7`). |
| `meeting_days` | string | Pipe-delimited weekday abbreviations (`Mon|Wed`). |
| `start_time` | string | 24h start (`14:30`). |
| `end_time` | string | 24h end (`15:50`). |
| `location` | string | Building-room placeholder. |
| `delivery_mode` | string | Mirrors course-level delivery, allows ad-hoc overrides. |

## `degree_requirements.csv`
| Column | Type | Description |
| --- | --- | --- |
| `requirement_id` | string | Identifier (e.g., `MAC_CORE`). |
| `label` | string | Human-readable requirement. |
| `category` | string | `core`, `technical-elective`, `business`, `project`, `minimum-gpa`. |
| `credit_min` | float | Minimum credits for the requirement scope. |
| `credit_max` | float | Optional maximum credits (empty if not applicable). |
| `eligible_courses` | string | Pipe-delimited list of course_ids that satisfy the requirement. |
| `notes` | string | Constraints or advising text. |

## `students.csv`
| Column | Type | Description |
| --- | --- | --- |
| `student_id` | string | Unique synthetic learner identifier. |
| `admit_term` | string | First term in program. |
| `program_stream` | string | `MAC-General` or `MAC-AI`. |
| `undergrad_major` | string | Prior discipline bucket. |
| `gpa_entry` | float | Entry GPA on 4.0 scale. |
| `co_op_status` | string | `seeking`, `placed`, or `not-applicable`. |
| `demographic_group` | string | Synthetic demographic cluster for fairness metrics. |
| `citizenship_status` | string | `domestic` or `international`. |
| `interests` | string | Pipe-delimited topical interests. |
| `learning_style` | string | `project-based`, `lecture`, or `self-paced`. |
| `work_experience_years` | float | Years of relevant experience. |

## `student_performance.csv`
| Column | Type | Description |
| --- | --- | --- |
| `student_id` | string | Foreign key. |
| `cumulative_gpa` | float | Current GPA (4.0 scale). |
| `last_term_gpa` | float | GPA for most recent completed term. |
| `technical_strength` | integer | Ordinal 1–5 derived from technical course averages. |
| `analytical_strength` | integer | Ordinal 1–5 derived from stats/data courses. |
| `communication_strength` | integer | Ordinal 1–5 based on business/communications courses. |
| `risk_flag` | string | `none`, `performance-drop`, or `co-op-risk`. |

## `enrollments.csv`
| Column | Type | Description |
| --- | --- | --- |
| `enrollment_id` | string | Unique interaction identifier. |
| `student_id` | string | Foreign key to `students.csv`. |
| `course_id` | string | Foreign key to `courses.csv`. |
| `term_code` | string | Term of enrollment. |
| `grade_point` | float | Converted grade on 4.0 scale (supports performance features). |
| `grade_letter` | string | Letter grade placeholder. |
| `completion_status` | string | `completed`, `withdrawn`, `in-progress`. |
| `feedback_rating` | integer | 1–5 satisfaction rating. |
| `hours_per_week` | float | Self-reported workload to support explainability. |
| `engagement_score` | float | Synthetic 0–1 metric from LMS interactions. |

## `student_preferences.csv`
| Column | Type | Description |
| --- | --- | --- |
| `student_id` | string | Foreign key. |
| `preference_type` | string | `career_goal`, `delivery_mode`, `time_of_day`, `skills_to_build`. |
| `preference_value` | string | Value or tag (pipe-delimited when multi-valued). |
| `weight` | float | Relative importance (0–1). |
| `source` | string | `survey`, `advisor_note`, `inferred`. |

### Notes
- Term codes follow format `YYYYT` where `T` ∈ {`F`, `W`, `S`} for Fall, Winter, Summer.
- All IDs are synthetic; no real student information is used.
- The schema supports cold-start handling (content features), fairness metrics (demographic grouping), and explainability (preferences, skill tags), aligning with the outlined methodology.

