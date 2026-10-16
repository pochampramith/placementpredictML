"""
Generates a synthetic dataset of 50,000 students with academic/skill
attributes and a Placed/Not Placed label + Salary (for placed students).
This mimics the dataset used in the SmartPlacement UI.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 50000

names_first = ["Rohan", "Priya", "Aditya", "Neha", "Karan", "Sneha", "Vikram",
               "Ananya", "Arjun", "Divya", "Rahul", "Pooja", "Suresh", "Kavya",
               "Manoj", "Isha", "Varun", "Riya", "Sandeep", "Meera"]
names_last = ["Verma", "Sharma", "Singh", "Gupta", "Patel", "Reddy", "Nair",
              "Iyer", "Kumar", "Joshi", "Mehta", "Rao", "Das", "Chopra"]

branches = ["CSE", "ECE", "EEE", "MECH", "Others"]
branch_probs = [0.41, 0.20, 0.15, 0.10, 0.14]

cgpa = np.round(np.random.normal(7.2, 1.1, N).clip(4, 10), 2)
internships = np.random.poisson(1.3, N).clip(0, 5)
projects = np.random.poisson(2.5, N).clip(0, 8)
backlogs = np.random.poisson(0.6, N).clip(0, 6)
aptitude = np.round(np.random.normal(65, 15, N).clip(20, 100), 0)
soft_skills = np.round(np.random.normal(6.5, 1.8, N).clip(1, 10), 1)
comm_skills = np.round(np.random.normal(6.5, 1.7, N).clip(1, 10), 1)
extra_curricular = np.round(np.random.normal(5.5, 2.2, N).clip(0, 10), 1)
branch = np.random.choice(branches, N, p=branch_probs)

# Placement score drives probability of placement
score = (
    cgpa * 1.4
    + internships * 1.8
    + projects * 0.9
    - backlogs * 2.2
    + aptitude * 0.06
    + soft_skills * 0.7
    + comm_skills * 0.7
    + extra_curricular * 0.2
    + np.random.normal(0, 3, N)
)
threshold = np.percentile(score, 42.54)  # calibrate to ~57.46% placed
placed = (score > threshold).astype(int)

salary = np.where(
    placed == 1,
    np.round((3 + (score - threshold) * 0.35 + np.random.normal(0, 1.4, N)).clip(2.5, 24), 2),
    np.nan
)

df = pd.DataFrame({
    "Name": [f"{np.random.choice(names_first)} {np.random.choice(names_last)}" for _ in range(N)],
    "Branch": branch,
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects,
    "Backlogs": backlogs,
    "Aptitude_Score": aptitude.astype(int),
    "Soft_Skills_Rating": soft_skills,
    "Communication_Skills": comm_skills,
    "Extra_Curricular_Activities": extra_curricular,
    "Placed": placed,
    "Salary_LPA": salary,
})

df.to_csv("/home/claude/smartplacement/data/students_data.csv", index=False)
print("Generated dataset:", df.shape)
print(df["Placed"].value_counts(normalize=True))
