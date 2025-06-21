import joblib
import numpy as np

# Map user-friendly field names to model feature names (if needed)
field_alias = {
    "Fasting Blood Glucose (mg/dL)": "glucose",
    "Postprandial Blood Sugar (mg/dL)": "pp_glucose",
    "HbA1c (%)": "hba1c",
    "BMI (kg/m²)": "bmi",
    "Age": "age",
    "Blood Pressure (mmHg)": "bp",
    "Insulin Level (μU/mL)": "insulin",
    "Skin Thickness (mm)": "skin",

    "Resting Blood Pressure (mmHg)": "resting_bp",
    "Cholesterol (mg/dL)": "cholesterol",
    "Fasting Blood Sugar (>120 mg/dL)": "fbs",
    "Maximum Heart Rate Achieved": "max_hr",
    "ST Depression (Oldpeak)": "oldpeak",
    "Chest Pain Type (categorical)": "chest_pain",
    "Exercise-induced Angina (Yes/No)": "angina",
    "Gender": "gender",

    "Serum Creatinine (mg/dL)": "creatinine",
    "Blood Urea (mg/dL)": "urea",
    "Albumin (g/dL)": "albumin",
    "Hemoglobin (g/dL)": "hemoglobin",
    "Sodium (mEq/L)": "sodium",
    "Potassium (mEq/L)": "potassium",
    "Red Blood Cell Count": "rbc",
    "Urine Specific Gravity": "gravity",
    "Hypertension Status": "htn",
    "Diabetes Status": "diabetes",

    "Tumor Size (mm or cm)": "tumor_size",
    "Clump Thickness": "clump",
    "Cell Size Uniformity": "size_uniformity",
    "Cell Shape Uniformity": "shape_uniformity",
    "Nuclear Size": "nuclei",
    "Mitoses Count": "mitoses",
    "Biopsy Results (Malignant/Benign)": "biopsy",

    "Work Interference (Yes/No)": "interfere",
    "Treatment History (Yes/No)": "treatment",
    "Family History of Mental Illness": "family_history",
    "Workplace Support": "support",
    "Stress Level (Scale)": "stress",

    "FEV1 (L)": "fev1",
    "FVC (L)": "fvc",
    "FEV1/FVC Ratio": "ratio",
    "Smoking Status (Yes/No)": "smoke",
    "Dyspnea Score (MRC Scale)": "dyspnea",

    "Pelvic Tilt (Degrees)": "tilt",
    "Lumbar Lordosis Angle": "lordosis",
    "Sacral Slope": "slope",
    "Pelvic Radius": "radius",
    "Degree of Spondylolisthesis": "spondy",

    "pH Monitoring Test Results": "ph",
    "Endoscopy Observations": "endoscopy",
    "Symptoms Frequency": "frequency",
    "Smoking Intake": "smoke_intake",
    "Alcohol Intake": "alcohol",
    "H. pylori Infection Status": "h_pylori",

    "ANA (Antinuclear Antibody Test)": "ana",
    "CRP (C-Reactive Protein)": "crp",
    "ESR (Sedimentation Rate)": "esr",
    "Rheumatoid Factor": "rf",
    "WBC Count": "wbc",
    "Joint Pain Level": "joint_pain",
    "TSH (Thyroid Test)": "tsh"
}

# Encode categorical values
yes_no_map = {"yes": 1, "no": 0}
mal_benign_map = {"malignant": 1, "benign": 0}

def preprocess_input(form_data):
    inputs = []
    for key in form_data:
        value = form_data[key].strip().lower()
        if value in yes_no_map:
            inputs.append(yes_no_map[value])
        elif value in mal_benign_map:
            inputs.append(mal_benign_map[value])
        else:
            try:
                inputs.append(float(value))
            except:
                inputs.append(0)
    return np.array(inputs).reshape(1, -1)

def predict_by_test_values(form_data):
    disease = form_data.get("disease")
    values = dict(form_data)
    values.pop("disease", None)

    X = preprocess_input(values)

    model_path = f"models/{disease}_random_forest.pkl"
    try:
        model = joblib.load(model_path)
        score = model.predict_proba(X)[0][1] * 100 if hasattr(model, 'predict_proba') else model.predict(X)[0] * 100
        level = "High" if score > 70 else "Medium" if score > 40 else "Low"
        color = "red" if level == "High" else "yellow" if level == "Medium" else "green"
        return {disease: {"score": int(score), "level": level, "color": color}}
    except Exception as e:
        return {disease: {"score": 0, "level": "Unknown", "color": "gray", "error": str(e)}}
