# ================================================
# IMAGE DAMAGE CLASSIFICATION USING GRIDSEARCHCV
# ================================================
# This pipeline:
# 1. Loads grayscale images
# 2. Extracts features using HOG
# 3. Uses GridSearchCV to choose the best model
# 4. Trains, validates, tests
# 5. Calculates damage count and damage percentage

# ----------------------------
# 1. IMPORT LIBRARIES
# ----------------------------
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from skimage.feature import hog
from joblib import dump, load  # put this with your imports


# ----------------------------
# 2. LOAD CSV LABELS
# ----------------------------
# CSV format assumed:
# filename, damage
# cell0001.png, 1
# cell0002.png, 0

CSV_PATH = r"C:\Users\asus\problem 3\elpv-dataset\src\elpv_dataset\data\labels.csv"
IMAGE_ROOT = r"C:\Users\asus\problem 3\elpv-dataset\src\elpv_dataset\data"

# 1) load with names
df = pd.read_csv(
    CSV_PATH,
    sep=r"\s+",
    header=None,
    names=["rel_path", "damage", "material"],
    engine="python"
)

# 2) binary label: 0 = good, 1 = damaged (any non‑zero treated as damaged)
df["label"] = (df["damage"] > 0).astype(int)

print(df.head())


# ----------------------------
# 3. FEATURE EXTRACTION FUNCTION
# ----------------------------
def extract_hog(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

# ----------------------------
# 4. BUILD FEATURE MATRIX
# ----------------------------
X = []
y = []

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_ROOT, row["rel_path"])  # e.g. data + images/cell0001.png
    features = extract_hog(img_path)
    X.append(features)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)
print("Samples:", X.shape[0], "Features:", X.shape[1])


# ----------------------------
# 5. TRAIN / VALIDATION / TEST SPLIT
# ----------------------------
# 70% Train, 15% Validation, 15% Test

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ----------------------------
# ----------------------------
# 6. MODEL SELECTION + GRIDSEARCHCV
# ----------------------------
# We test MULTIPLE models and let GridSearchCV choose the best one

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())  # placeholder, will be replaced by GridSearch
])

param_grid = [
    # ---- SVM ----
    {
        'clf': [SVC()],
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 'auto']
    },
    # ---- RANDOM FOREST ----
    {
        'clf': [__import__('sklearn.ensemble').ensemble.RandomForestClassifier()],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    },
    # ---- LOGISTIC REGRESSION ----
    {
        'clf': [__import__('sklearn.linear_model').linear_model.LogisticRegression(max_iter=500)],
        'clf__C': [0.1, 1, 10]
    }
]

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ----------------------------
# 7. VALIDATION PERFORMANCE

# ----------------------------
y_val_pred = grid.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

# ----------------------------
# 8. TEST PERFORMANCE
# ----------------------------
y_test_pred = grid.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# ----------------------------
# 9. DAMAGE STATISTICS
# ----------------------------
total_samples = len(y)
damaged_samples = np.sum(y == 1)
damage_percentage = (damaged_samples / total_samples) * 100

# ----------------------------
# 10. OUTPUT RESULTS
# ----------------------------
print("Best Parameters:", grid.best_params_)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

print(f"\nTotal Images: {total_samples}")
print(f"Damaged Images: {damaged_samples}")
print(f"Damage Percentage: {damage_percentage:.2f}%")

# ----------------------------
# 11. SAVE BEST MODEL TO PKL
# ----------------------------
MODEL_PATH = r"C:\Users\asus\problem 3\best_damage_model.pkl"

dump(best_model, MODEL_PATH)
print(f"Saved best model to: {MODEL_PATH}")


# ================================================
# END OF PIPELINE
# ================================================
