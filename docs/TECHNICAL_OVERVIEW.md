# Technical Overview – MAC Course Pathfinder

## Stack & Entry Points

- **Runtime:** Python 3.12+
- **Framework:** Flask 3 (server-rendered templates, no JS build step)
- **Key modules:**
  - `app.py` – route definitions and request handling
  - `app/data_loader.py` – cached access layer for the synthetic CSV dataset
  - `app/recommender.py` – hybrid recommendation engine
  - `templates/` – Jinja2 templates rendered by Flask

## Data Layer

Source CSVs live under `data/synthetic/` (described in `docs/data_schema.md`):

| File | Main usage |
| --- | --- |
| `courses.csv` | Course metadata (skills, prerequisites, delivery mode, term patterns) |
| `students.csv` | Student profile features, including interests |
| `enrollments.csv` | Historical course completions with grades |
| `student_preferences.csv` | Explicit signals such as desired skills and delivery modes |
| `student_performance.csv` | GPA summary used for UI context |
| `course_offerings.csv`, `degree_requirements.csv` | Currently unused but available for future scheduling/constraint work |

The loader builds cached indices:

- `course_skill_tags` – maps course IDs → set of skills
- `student_completed_courses` – student IDs → completed course set (filters recommendations and powers collaborative similarity)
- `student_interest_tags` – fused explicit interests + preference tags
- `collaborative_matrix` – course IDs → students who completed them (for explanation counts)
- `interest_catalog` – master list of unique tags used by the cold-start UI

## Recommendation Flow

### 1. Candidate Generation

Candidates are all catalogue courses the student has not yet completed. We remove any entry that violates prerequisites relative to completed courses.

### 2. Content-Based Scoring

Two profile variants:

- **History mode:** aggregates skills, categories, delivery modes, and term tokens from a student’s completed courses (`_content_profile_from_courses`).
- **Interest mode:** aggregates the tags selected in the cold-start form (`_content_profile_from_interests`).

Each candidate course earns points for overlapping skills, matching category/delivery mode, and (lightly weighted) term availability.

### 3. Collaborative Scoring

- **History mode:** Jaccard similarity between the student’s completed set and every other student. Similar peers contribute their unseen courses with weight equal to similarity (`_score_collaborative_history`).
- **Interest mode:** overlap of interest tags with other students’ interest sets. Matching peers contribute courses they have completed (`_score_collaborative_interests`).

Scores are normalized to `[0, 1]` before blending to keep proportions stable if the candidate set changes.

### 4. Hybrid Blend & Explanation

- **History mode weight:** `0.6 * content + 0.4 * collaborative`
- **Interest mode weight:** `0.7 * content + 0.3 * collaborative`

Recommendations are sorted by blended score (desc) with deterministic tie-breakers on course ID.

Each result includes:

- rounded score components
- sorted skill tags
- short explanation combining top skills and peer count

## Flask Views

| Route | Method(s) | Purpose |
| --- | --- | --- |
| `/` | GET/POST | Home. POST lookup of student ID. If history exists → recommendations; otherwise → interest capture form. |
| `/interests` | POST | Accept selected interest tags, run cold-start recommendations, and render results. |

Templates (`home.html`, `collect_interests.html`, `recommendations.html`) extend the base layout and provide the UI flows.

## Configuration & Deployment

- No environment variables required.
- `app.py` enables Flask’s debug mode by default for local iteration; flip `debug=False` (or use a WSGI server) for production.
- Dependencies are listed in `requirements.txt`.

## Testing & Validation

- `python -m py_compile app.py app/data_loader.py app/recommender.py` ensures syntax validity.
- Manual QA: run `python app.py`, exercise both history and cold-start flows, and verify course explanations.
- Synthetic dataset can be regenerated via `scripts/generate_mac_synthetic_data.py` if you wish to tweak parameters.

## Extension Ideas

- Integrate course scheduling constraints using `course_offerings.csv`.
- Add feedback logging to adjust recommendation weights dynamically.
- Expose a JSON API alongside the server-rendered UI.
- Replace similarity heuristics with cosine similarity or matrix factorization trained from the synthetic enrollments.


