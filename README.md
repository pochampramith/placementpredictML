# SmartPlacement — Student Placement Prediction System

A Flask + Machine Learning web app that predicts student placement status
and estimated salary based on academic/skill parameters.

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Data & models are already included, but to regenerate them):
   ```
   python model/generate_data.py
   python model/train_model.py
   ```

4. Run the app:
   ```
   python run.py
   ```

5. Open your browser at:
   ```
   http://127.0.0.1:5000
   ```

## Pages
- `/`          Dashboard — key stats and charts
- `/predict`   Predict — enter student details
- `/students`  Students — searchable, paginated list
- `/analytics` Analytics — placement insights
- `/about`, `/help`

## Project Structure
```
smartplacement/
├── app.py
├── run.py
├── requirements.txt
├── data/students_data.csv
├── model/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── placement_model.pkl
│   ├── salary_model.pkl
│   └── features.pkl
├── templates/  (dashboard, predict, result, students, analytics, about, help)
└── static/css/style.css
```
"# placementpredictML" 
"# placementpredictML" 
