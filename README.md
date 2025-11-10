# MAC Course Pathfinder

A tiny web app that suggests University of Windsor MAC courses using a hybrid (content + collaborative) recommender. Enter a synthetic student ID to see recommendations driven by past coursework, or pick interests to handle brand-new students.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app**
   ```bash
   python app.py
   ```
3. **Visit the UI** – open http://127.0.0.1:5000/ in your browser.
4. **Test it out** – try IDs like `STU-0001`, `STU-0007`, or `STU-0036`. If we cannot find history, you will be prompted to select interests.

## What You Get

- Hybrid scoring that blends course content signals and student similarity.
- Explanations beside every course so you can see why it was recommended.
- Cold-start flow for students with no enrollment history.

## Project Layout

- `app.py` – Flask entry point and routes.
- `app/` – data ingestion and recommendation logic.
- `templates/` – HTML templates for the UI.
- `data/synthetic/` – ready-to-use CSV dataset (no setup required).
- `docs/` – extra background, including the full data schema and technical notes.
- `scripts/generate_mac_synthetic_data.py` – optional script if you want to rebuild the synthetic dataset.

Need the deep dive? Check `docs/TECHNICAL_OVERVIEW.md` for architecture, algorithms, and extension ideas.


