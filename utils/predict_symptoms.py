import joblib
import os

SYMPTOM_WEIGHTS = {
    "chronic cough": {"copd": 0.9},
    "shortness of breath": {"copd": 0.8, "heart": 0.7},
    "wheezing": {"copd": 0.7},
    "chest tightness": {"copd": 0.6},
    "frequent respiratory infections": {"copd": 0.5},
    "chest pain": {"heart": 0.9},
    "palpitations": {"heart": 0.8},
    "swelling in legs or feet": {"heart": 0.7},
    "fatigue": {"diabetes": 0.5, "kidney": 0.6, "cancer": 0.5, "mental": 0.7, "autoimmune": 0.5},
    "dizziness": {"heart": 0.6},
    "fainting": {"heart": 0.7},
    "increased thirst": {"diabetes": 0.9},
    "frequent urination": {"diabetes": 0.8},
    "unexplained weight loss": {"diabetes": 0.6, "cancer": 0.6},
    "blurred vision": {"diabetes": 0.5},
    "slow healing wounds": {"diabetes": 0.6},
    "numbness or tingling": {"diabetes": 0.5, "neural": 0.6},
    "swelling around eyes": {"kidney": 0.6},
    "decreased appetite": {"kidney": 0.6},
    "nausea": {"kidney": 0.5, "gastro": 0.6},
    "itchy skin": {"kidney": 0.6, "liver": 0.4},
    "muscle cramps": {"kidney": 0.5},
    "weakness": {"cancer": 0.5},
    "jaundice": {"liver": 0.9},
    "abdominal swelling": {"liver": 0.7},
    "loss of appetite": {"liver": 0.6},
    "easy bruising": {"liver": 0.5},
    "tremors": {"neural": 0.6},
    "muscle weakness": {"neural": 0.7},
    "difficulty walking": {"neural": 0.6},
    "difficulty speaking": {"neural": 0.6},
    "cognitive changes": {"neural": 0.6},
    "pain": {"cancer": 0.6},
    "persistent cough": {"cancer": 0.6},
    "bowel or bladder habits": {"cancer": 0.5, "gastro": 0.6},
    "bleeding": {"cancer": 0.5},
    "joint pain": {"autoimmune": 0.8},
    "skin rashes": {"autoimmune": 0.7},
    "muscle aches": {"autoimmune": 0.6},
    "dry eyes": {"autoimmune": 0.5},
    "hair loss": {"autoimmune": 0.4},
    "abdominal pain": {"gastro": 0.9},
    "cramping": {"gastro": 0.7},
    "bloating": {"gastro": 0.6},
    "diarrhea": {"gastro": 0.7},
    "constipation": {"gastro": 0.6},
    "heartburn": {"gastro": 0.8},
    "regurgitation": {"gastro": 0.7},
    "sadness": {"mental": 0.9},
    "low mood": {"mental": 0.8},
    "loss of interest": {"mental": 0.7},
    "difficulty concentrating": {"mental": 0.7},
    "sleep disturbances": {"mental": 0.7},
    "irritability": {"mental": 0.6},
    "physical symptoms": {"mental": 0.5},
}

ALL_DISEASES = [
    "diabetes", "heart", "kidney", "cancer", "mental", "copd", "neural",
    "orthopedic", "gastro", "autoimmune"
]

RISK_FACTORS = {
    "diabetes": "BMI, Family History, Age",
    "heart": "Blood Pressure, Cholesterol, Smoking",
    "kidney": "Age, Hypertension, Diabetes",
    "cancer": "Smoking, Alcohol, Genetics",
    "mental": "Stress, Family History, Work Environment",
    "copd": "Age, Environmental Factors",
    "neural": "Brain Health, Age, Genetics",
    "orthopedic": "Bone Density, Posture, Age",
    "gastro": "Diet, Alcohol, Infection",
    "autoimmune": "Genetics, Gender, Environment"
}

def predict_by_symptoms(selected_symptoms):
    disease_scores = {disease: 0.0 for disease in ALL_DISEASES}

    if not selected_symptoms:
        return [
            {
                "name": disease.capitalize(),
                "percentage": 0,
                "level": "Low",
                "factors": RISK_FACTORS.get(disease, "")
            }
            for disease in ALL_DISEASES
        ]

    # Ensure consistent casing and trimming
    cleaned_symptoms = [sym.strip().lower() for sym in selected_symptoms if sym.strip()]

    for symptom in cleaned_symptoms:
        weights = SYMPTOM_WEIGHTS.get(symptom, {})
        for disease, weight in weights.items():
            if disease in disease_scores:
                disease_scores[disease] += weight

    all_disease_outputs = []
    for disease in ALL_DISEASES:
        score = min(disease_scores[disease], 3.0)
        percentage = round((score / 3.0) * 100)
        if percentage >= 66:
            level = "High"
        elif percentage >= 33:
            level = "Medium"
        else:
            level = "Low"

        all_disease_outputs.append({
            "name": disease.capitalize(),
            "percentage": percentage,
            "level": level,
            "factors": RISK_FACTORS.get(disease, "")
        })

    return all_disease_outputs