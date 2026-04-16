"""
Retrains the Random Forest classifier and saves all artifacts to artifacts/.

Improvements over the original:
  - SMOTE targets for Type 1 and Gestational raised from 10,000 to 20,000
  - class_weight="balanced" added to RandomForestClassifier

Run:
    python retrain.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

os.makedirs("artifacts", exist_ok=True)

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_excel("DataSet.xlsx")

# ── 2. CLEAN ───────────────────────────────────────────────────────────────────
print(f"Original shape: {df.shape}")
df = df[df["diastolic_bp"] < df["systolic_bp"]]
print(f"After BP cleaning: {df.shape}")

# ── 3. ENCODE ──────────────────────────────────────────────────────────────────
cat_cols = ["gender", "ethnicity", "education_level",
            "income_level", "employment_status", "smoking_status"]

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

le = LabelEncoder()
df_encoded["diabetes_stage"] = le.fit_transform(df_encoded["diabetes_stage"])

print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print("Shape after encoding:", df_encoded.shape)

# ── 4. DROP UNUSED TARGET COLUMNS ─────────────────────────────────────────────
df_encoded = df_encoded.drop(columns=["diagnosed_diabetes"])

# ── 5. SPLIT ───────────────────────────────────────────────────────────────────
X = df_encoded.drop(columns=["diabetes_stage"])
y = df_encoded["diabetes_stage"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")
print("Class distribution (train):")
print(y_train.value_counts().sort_index())

# ── 6. SMOTE ───────────────────────────────────────────────────────────────────
# Only oversample the majority classes — oversampling Type 1/Gestational from
# too few real samples creates synthetic noise that hurts rather than helps.
sampling_strategy = {
    1: 46461,  # No Diabetes
    2: 46461,  # Pre-Diabetes
    4: 46461,  # Type 2
}
print("\nApplying SMOTE (majority classes only)...")
smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Train: {X_train.shape}")

# ── 7. SCALE ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("Scaling complete")

# ── 8. TRAIN ───────────────────────────────────────────────────────────────────
# Manual class weights: heavily penalise missing Type 1 and Gestational
# 0=Gestational, 1=No Diabetes, 2=Pre-Diabetes, 3=Type 1, 4=Type 2
class_weight = {0: 300, 1: 5, 2: 1, 3: 700, 4: 1}

print("\nTraining Random Forest (this may take a few minutes)...")
print(f"Class weights: {class_weight}")
rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    n_estimators=200,
    class_weight=class_weight
)
rf_model.fit(X_train, y_train)

# ── 9. EVALUATE ────────────────────────────────────────────────────────────────
rf_predictions = rf_model.predict(X_test)
acc = accuracy_score(y_test, rf_predictions)
print(f"\nRandom Forest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions, target_names=le.classes_))

# ── 10. SAVE ARTIFACTS ─────────────────────────────────────────────────────────
feature_names = pd.DataFrame({"feature": X.columns.tolist()})
feature_names.to_csv("artifacts/feature_names.csv", index=False)

joblib.dump(rf_model, "artifacts/random_forest_model.pkl")
joblib.dump(scaler,   "artifacts/scaler.pkl")
joblib.dump(le,       "artifacts/label_encoder.pkl")

print("\nArtifacts saved to artifacts/")
print("  random_forest_model.pkl")
print("  scaler.pkl")
print("  label_encoder.pkl")
print("  feature_names.csv")
print("\nDone. Restart app.py to use the new model.")
