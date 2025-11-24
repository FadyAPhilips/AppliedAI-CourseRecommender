from __future__ import annotations

from flask import Flask, render_template, request, make_response, send_file
from collections import Counter

from app.data_loader import get_dataset
from app.recommender import recommend_for_interests, recommend_for_student
from app.pdf_export import generate_recommendations_pdf


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    dataset = get_dataset()

    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip().upper()
        if not student_id:
            return render_template(
                "home.html",
                error="Please enter a student ID.",
            )

        student = dataset.students.get(student_id)
        completed_history = dataset.student_completed_courses.get(student_id, set())

        if student and completed_history:
            recommendations = recommend_for_student(student_id)
            return render_template(
                "recommendations.html",
                student=student,
                performance=dataset.performance.get(student_id),
                recommendations=recommendations,
                context="history",
            )

        preset_interests = sorted(dataset.student_interest_tags.get(student_id, set()))
        return render_template(
            "collect_interests.html",
            student=student,
            student_id=student_id,
            interest_catalog=dataset.interest_catalog,
            preset_interests=preset_interests,
            missing_student=student is None,
            missing_history=not completed_history,
        )

    return render_template("home.html")


@app.post("/interests")
def collect_interests():
    dataset = get_dataset()
    student_id = request.form.get("student_id", "").strip().upper()
    selected_interests = request.form.getlist("interests")

    recommendations = recommend_for_interests(selected_interests)
    student = dataset.students.get(student_id)

    return render_template(
        "recommendations.html",
        student=student,
        performance=dataset.performance.get(student_id),
        recommendations=recommendations,
        context="interests",
        selected_interests=selected_interests,
    )


@app.route("/browse", methods=["GET"])
def browse_courses():
    dataset = get_dataset()
    
    # Get search and filter parameters
    query = request.args.get("q", "").strip().lower()
    category_filter = request.args.get("category", "")
    delivery_filter = request.args.get("delivery", "")
    difficulty_filter = request.args.get("difficulty", "")
    credits_filter = request.args.get("credits", "")
    sort_by = request.args.get("sort", "relevance")
    
    # Get all courses
    all_courses = []
    for course_id, course_data in dataset.courses.items():
        skills = list(dataset.course_skill_tags.get(course_id, []))
        popularity = len(dataset.collaborative_matrix.get(course_id, set()))
        
        all_courses.append({
            "course_id": course_id,
            "course_code": course_data.get("course_code", ""),
            "title": course_data.get("title", ""),
            "description": course_data.get("description", ""),
            "category": course_data.get("category", ""),
            "delivery_mode": course_data.get("delivery_mode", ""),
            "difficulty_level": course_data.get("difficulty_level", ""),
            "credits": course_data.get("credits", ""),
            "skills": skills,
            "popularity": popularity,
            "prerequisites": course_data.get("prerequisites", ""),
        })
    
    # Apply filters
    results = all_courses
    
    # Search filter
    if query:
        results = [
            course for course in results
            if query in course["course_code"].lower()
            or query in course["title"].lower()
            or query in course["description"].lower()
            or any(query in skill.lower() for skill in course["skills"])
        ]
    
    # Category filter
    if category_filter:
        results = [course for course in results if course["category"] == category_filter]
    
    # Delivery mode filter
    if delivery_filter:
        results = [course for course in results if course["delivery_mode"] == delivery_filter]
    
    # Difficulty filter
    if difficulty_filter:
        results = [course for course in results if course["difficulty_level"] == difficulty_filter]
    
    # Credits filter
    if credits_filter:
        results = [course for course in results if course["credits"] == credits_filter]
    
    # Sorting
    if sort_by == "code":
        results.sort(key=lambda x: x["course_code"])
    elif sort_by == "title":
        results.sort(key=lambda x: x["title"])
    elif sort_by == "difficulty":
        results.sort(key=lambda x: int(x["difficulty_level"]) if x["difficulty_level"] else 0, reverse=True)
    elif sort_by == "popularity":
        results.sort(key=lambda x: x["popularity"], reverse=True)
    elif sort_by == "relevance" and query:
        # Simple relevance scoring based on query matches
        def relevance_score(course):
            score = 0
            if query in course["course_code"].lower():
                score += 10
            if query in course["title"].lower():
                score += 5
            score += sum(1 for skill in course["skills"] if query in skill.lower())
            return score
        
        results.sort(key=relevance_score, reverse=True)
    
    return render_template(
        "browse.html",
        results=results,
        query=query,
        filters={
            "category": category_filter,
            "delivery": delivery_filter,
            "difficulty": difficulty_filter,
            "credits": credits_filter,
        },
        sort_by=sort_by,
    )


@app.post("/export-pdf")
def export_pdf():
    """Export recommendations as PDF."""
    dataset = get_dataset()
    
    student_id = request.form.get("student_id", "").strip().upper()
    context = request.form.get("context", "interests")
    selected_interests_str = request.form.get("selected_interests", "")
    
    # Get student and performance data
    student = dataset.students.get(student_id) if student_id else None
    performance = dataset.performance.get(student_id) if student_id else None
    
    # Get recommendations based on context
    if context == "history" and student_id:
        recommendations = recommend_for_student(student_id)
        selected_interests = None
    else:
        selected_interests = [i.strip() for i in selected_interests_str.split(",") if i.strip()]
        recommendations = recommend_for_interests(selected_interests)
    
    # Generate PDF
    pdf_buffer = generate_recommendations_pdf(
        student,
        performance,
        recommendations,
        context,
        selected_interests
    )
    
    # Create filename
    filename = f"MAC_Recommendations_{student_id if student_id else 'Guest'}_{context}.pdf"
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )


if __name__ == "__main__":
    app.run(debug=True)



