import numpy as np
import pandas as pd

raw_df = pd.read_csv('student_performance_prediction.csv')
cleaned_df = pd.read_csv('student_performance_prediction_cleaned.csv')

# normalize numeric features
grades_norm = cleaned_df['Previous Grades'] / 100.0
att_norm = cleaned_df['Attendance Rate'] / 100.0
study_norm = np.clip(cleaned_df['Study Hours per Week'] / 25.0, 0, 1)
extra_norm = cleaned_df['Participation in Extracurricular Activities'].map(
    {'Yes': 1.0, 'No': 0.0}
)
parent_map = {
    'High School': 0.2,
    'Associate': 0.4,
    'Bachelor': 0.6,
    'Master': 0.8,
    'Doctorate': 1.0,
}
parent_norm = cleaned_df['Parent Education Level'].map(parent_map)

# calc composite score
composite_score = (
    0.4 * grades_norm
    + 0.25 * att_norm
    + 0.20 * study_norm
    + 0.05 * extra_norm
    + 0.10 * parent_norm
)

# map through sigmoid function to get probability
np.random.seed(67)
noise = np.random.normal(0, 0.3, size=len(raw_df))
z = 13.0 * (composite_score - 0.55) + noise
prob_passed = 1.0 / (1.0 + np.exp(-z))

# generate new passed column
new_passed = np.where(np.random.rand(len(raw_df)) < prob_passed, 'Yes', 'No')

# get each index of the null values in the set
labeled_mask = raw_df['Passed'].notna()
raw_df.loc[labeled_mask, 'Passed'] = new_passed[labeled_mask]

raw_df.to_csv('student_performance_prediction.csv', index=False)

print(
    'Đã cập nhật thành công cột Passed vào file dữ liệu thô! Tỉ lệ đỗ mới:',
    (raw_df['Passed'] == 'Yes').mean(),
)