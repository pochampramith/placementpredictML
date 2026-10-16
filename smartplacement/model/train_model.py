"""
Trains:
1) A classifier to predict Placed / Not Placed
2) A regressor to predict expected Salary (LPA) for placed students
Saves both models + feature list as .pkl files for the Flask app to load.
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

df = pd.read_csv("/home/claude/smartplacement/data/students_data.csv")

FEATURES = [
    "CGPA", "Internships", "Projects", "Backlogs",
    "Aptitude_Score", "Soft_Skills_Rating",
    "Communication_Skills", "Extra_Curricular_Activities",
]

X = df[FEATURES]
y_class = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Placement classifier accuracy: {acc:.4f}")

# Salary regressor trained only on placed students
placed_df = df[df["Placed"] == 1]
Xs = placed_df[FEATURES]
ys = placed_df["Salary_LPA"]
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys, test_size=0.2, random_state=42
)
reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
reg.fit(Xs_train, ys_train)
mae = mean_absolute_error(ys_test, reg.predict(Xs_test))
print(f"Salary regressor MAE: {mae:.4f} LPA")

joblib.dump(clf, "/home/claude/smartplacement/model/placement_model.pkl")
joblib.dump(reg, "/home/claude/smartplacement/model/salary_model.pkl")
joblib.dump(FEATURES, "/home/claude/smartplacement/model/features.pkl")
print("Models saved.")
