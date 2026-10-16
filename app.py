"""
SmartPlacement - Student Placement Prediction System
Flask web application with Dashboard, Predict, Students, and Analytics pages.
"""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ---------- Load data & models once at startup ----------
DATA_PATH = os.path.join(BASE_DIR, "data", "students_data.csv")
df = pd.read_csv(DATA_PATH)

clf = joblib.load(os.path.join(BASE_DIR, "model", "placement_model.pkl"))
reg = joblib.load(os.path.join(BASE_DIR, "model", "salary_model.pkl"))
FEATURES = joblib.load(os.path.join(BASE_DIR, "model", "features.pkl"))


def get_dashboard_stats():
    total = len(df)
    placed = int((df["Placed"] == 1).sum())
    not_placed = total - placed
    avg_salary = round(df.loc[df["Placed"] == 1, "Salary_LPA"].mean(), 2)
    placed_pct = round(placed / total * 100, 2)
    not_placed_pct = round(100 - placed_pct, 2)

    # salary distribution buckets for the bar chart
    bins = [0, 2, 4, 6, 8, 10, 12, 100]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12+"]
    placed_df = df[df["Placed"] == 1].copy()
    placed_df["bucket"] = pd.cut(placed_df["Salary_LPA"], bins=bins, labels=labels, right=False)
    salary_dist = placed_df["bucket"].value_counts().reindex(labels).fillna(0).astype(int).tolist()

    return {
        "total": total,
        "placed": placed,
        "not_placed": not_placed,
        "placed_pct": placed_pct,
        "not_placed_pct": not_placed_pct,
        "avg_salary": avg_salary,
        "salary_labels": labels,
        "salary_dist": salary_dist,
    }


def get_analytics_stats():
    avg_cgpa = round(df.loc[df["Placed"] == 1, "CGPA"].mean(), 2)
    placement_rate = round((df["Placed"] == 1).mean() * 100, 2)
    avg_aptitude = round(df["Aptitude_Score"].mean(), 2)

    # placement rate by CGPA range
    cgpa_bins = [0, 6, 7, 8, 9, 10]
    cgpa_labels = ["0-6", "6-7", "7-8", "8-9", "9-10"]
    tmp = df.copy()
    tmp["cgpa_bucket"] = pd.cut(tmp["CGPA"], bins=cgpa_bins, labels=cgpa_labels, right=True)
    rate_by_cgpa = tmp.groupby("cgpa_bucket", observed=True)["Placed"].mean() * 100
    rate_by_cgpa = rate_by_cgpa.reindex(cgpa_labels).fillna(0).round(1).tolist()

    # placement by branch
    branch_counts = df[df["Placed"] == 1]["Branch"].value_counts()
    branch_labels = branch_counts.index.tolist()
    branch_values = branch_counts.values.tolist()

    return {
        "avg_cgpa": avg_cgpa,
        "placement_rate": placement_rate,
        "avg_aptitude": avg_aptitude,
        "cgpa_labels": cgpa_labels,
        "rate_by_cgpa": rate_by_cgpa,
        "branch_labels": branch_labels,
        "branch_values": branch_values,
    }


@app.route("/")
def dashboard():
    stats = get_dashboard_stats()
    return render_template("dashboard.html", active="dashboard", stats=stats)


@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html", active="predict")


@app.route("/predict", methods=["POST"])
def predict_result():
    form = request.form
    values = {
        "CGPA": float(form.get("cgpa", 0)),
        "Internships": int(form.get("internships", 0)),
        "Projects": int(form.get("projects", 0)),
        "Backlogs": int(form.get("backlogs", 0)),
        "Aptitude_Score": float(form.get("aptitude", 0)),
        "Soft_Skills_Rating": float(form.get("soft_skills", 0)),
        "Communication_Skills": float(form.get("communication", 0)),
        "Extra_Curricular_Activities": float(form.get("extra_curricular", 0)),
    }
    X = pd.DataFrame([values])[FEATURES]

    proba = clf.predict_proba(X)[0]  # [prob_not_placed, prob_placed]
    prediction = clf.predict(X)[0]
    placed_prob = round(proba[1] * 100, 1)
    not_placed_prob = round(proba[0] * 100, 1)

    salary = None
    if prediction == 1:
        salary = round(float(reg.predict(X)[0]), 2)

    return render_template(
        "result.html",
        active="predict",
        placed=bool(prediction == 1),
        salary=salary,
        placed_prob=placed_prob,
        not_placed_prob=not_placed_prob,
        inputs=values,
    )


@app.route("/students")
def students():
    page = int(request.args.get("page", 1))
    search = request.args.get("q", "").strip()
    per_page = 10

    view = df.copy()
    view.insert(0, "ID", view.index + 1)
    if search:
        view = view[view["Name"].str.contains(search, case=False, na=False)]

    total_rows = len(view)
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page = min(max(1, page), total_pages)
    start = (page - 1) * per_page
    page_rows = view.iloc[start:start + per_page].to_dict("records")

    return render_template(
        "students.html",
        active="students",
        rows=page_rows,
        page=page,
        total_pages=total_pages,
        search=search,
    )


@app.route("/analytics")
def analytics():
    stats = get_analytics_stats()
    return render_template("analytics.html", active="analytics", stats=stats)


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/help")
def help_page():
    return render_template("help.html", active="help")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
