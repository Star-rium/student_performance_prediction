import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. ĐỌC VÀ MÃ HÓA DỮ LIỆU
df = pd.read_csv('student_performance_prediction_cleaned.csv')

# Mã hóa Yes/No thành 1/0
df['Participation in Extracurricular Activities'] = df[
    'Participation in Extracurricular Activities'
].map({'Yes': 1, 'No': 0})
df['Passed'] = df['Passed'].map({'Yes': 1, 'No': 0})

# Mã hóa One-Hot cho Parent Education Level
df['is_HighSchool'] = (df['Parent Education Level'] == 'High School').astype(
    int
)
df['is_Associate'] = (df['Parent Education Level'] == 'Associate').astype(int)
df['is_Bachelor'] = (df['Parent Education Level'] == 'Bachelor').astype(int)
df['is_Master'] = (df['Parent Education Level'] == 'Master').astype(int)
df['is_Doctorate'] = (df['Parent Education Level'] == 'Doctorate').astype(int)

# Xóa cột chuỗi gốc
df = df.drop(columns=['Parent Education Level'])

# Danh sách tên đặc trưng và chuyển thành ma trận X, vector y
feature_names = [
    'Study Hours per Week',
    'Attendance Rate',
    'Previous Grades',
    'Participation in Extracurricular Activities',
    'is_HighSchool',
    'is_Associate',
    'is_Bachelor',
    'is_Master',
    'is_Doctorate',
]

X = df[feature_names].values
y = df['Passed'].values

# 2. CHIA TRAIN/TEST (CÓ STRATIFY)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. CHUẨN HÓA VÀ HUẤN LUYỆN LOGISTIC REGRESSION
# fit scaler trêntrain->transform cho cả train & test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Dự đoán nhãn và xác suất
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# 4. IN KẾT QUẢ VÀ CÁC CHỈ SỐ ĐÁNH GIÁ
print('=== BỘ Trọng SỐW ===')
print(f"{'Intercept (Bias w0)':<45}: {model.intercept_[0]:+.6f}")
for name, weight in zip(feature_names, model.coef_[0]):
  print(f'{name:<45}: {weight:+.6f}')

print('\n=== QUÁ KHỚP (TRAIN & TEST ACCURACY) ===')
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy trên tập Train : {train_acc:.4f} ({train_acc*100:.2f}%)')
print(f'Accuracy trên tập Test  : {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'Độ lệch (Train - Test)  : {abs(train_acc - test_acc):.4f}')

print('\n=== CÁC CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP TEST ===')
print(f'Accuracy  : {test_acc:.4f}')
print(f'Precision : {precision_score(y_test, y_test_pred):.4f}')
print(f'Recall    : {recall_score(y_test, y_test_pred):.4f}')
print(f'F1-Score  : {f1_score(y_test, y_test_pred):.4f}')
print(f'ROC-AUC   : {roc_auc_score(y_test, y_test_prob):.4f}')

print('\n=== CÁC CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP TEST ===')
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Thực tế: 0 (No)', 'Thực tế: 1 (Yes)'],
    columns=['Dự đoán: 0 (No)', 'Dự đoán: 1 (Yes)'],
)
print(cm_df)