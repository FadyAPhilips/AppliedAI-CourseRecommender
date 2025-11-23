from app.recommender import recommend_for_interests, recommend_for_student
import pandas as pd
rec_data = []
for i in range(1,121):
    student_id = f"STU_{i:03}"
    recommendations = recommend_for_student(student_id);
    rec_course_ids = [rec['course_id'] for rec in recommendations]
    rec_data.append({'student_id': student_id, 'rec_courses': rec_course_ids})


# Load CSV
df = pd.read_csv("data/synthetic/test_enrollments.csv")

# Group by student_id and collect course_id into a list
test_data= []
for student_id, group in df.groupby('student_id'):
    courses = group['course_id'].tolist()
    test_data.append({
        'student_id': student_id,
        'test_courses': courses
    })


# Store precision and recall for each student
results = []

for rec, test in zip(rec_data, test_data):
    student_id = rec['student_id']
    rec_courses = set(rec['rec_courses'])
    test_courses = set(test['test_courses'])

    true_positives = rec_courses & test_courses  # intersection
    precision = len(true_positives) / len(rec_courses) if rec_courses else 0
    recall = len(true_positives) / len(test_courses) if test_courses else 0

    results.append({
        'student_id': student_id,
        'precision': round(precision, 3),
        'recall': round(recall, 3)
    })

# Extract all precision and recall values
all_precisions = [r['precision'] for r in results]
all_recalls = [r['recall'] for r in results]

# Compute overall average
overall_precision = sum(all_precisions) / len(all_precisions)
overall_recall = sum(all_recalls) / len(all_recalls)

print(f"Overall Precision: {overall_precision:.3f}")
print(f"Overall Recall: {overall_recall:.3f}")



