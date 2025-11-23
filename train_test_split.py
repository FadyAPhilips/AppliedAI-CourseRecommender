import pandas as pd
from collections import defaultdict

# Load enrollment data
df = pd.read_csv("data/synthetic/enrollments.csv")

# Keep only completed courses for splitting
completed_df = df[df['completion_status'] == 'completed']

# Sort courses chronologically per student
completed_df = completed_df.sort_values(['student_id', 'term_code'])

# --- Split courses per student into train/test ---
train_courses = []
test_courses = []

split_ratio = 0.7  # 70% train, 30% test

for student_id, group in completed_df.groupby('student_id'):
    split_idx = max(1, int(len(group) * split_ratio))
    train_courses.append(group.iloc[:split_idx])
    test_courses.append(group.iloc[split_idx:])

# --- Combine all students into single DataFrame ---
train_df = pd.concat(train_courses).reset_index(drop=True)
test_df = pd.concat(test_courses).reset_index(drop=True)

# --- Save to CSV ---
train_df.to_csv("data/synthetic/train_enrollments.csv", index=False)
test_df.to_csv("data/synthetic/test_enrollments.csv", index=False)

print("Train and test enrollments saved successfully!")
