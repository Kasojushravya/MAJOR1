import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import glob

os.makedirs("models", exist_ok=True)

# Dictionary of datasets and their target columns
dataset_info = {
    "diabetes": ("datasets/pima-indians-diabetes-database/diabetes.csv", "Outcome"),
    "heart": ("datasets/heart-disease-data/heart_disease_uci.csv", "target"),
    "kidney": ("datasets/ckdisease/kidney_disease.csv", "classification"),
    "cancer": ("datasets/cancer-data/Cancer_Data.csv", "diagnosis"),
    "mental": ("datasets/mental-health-in-tech-survey/survey.csv", "treatment"),
    "copd": ("datasets/copd-student-dataset/dataset.csv", "COPD"),
    "neural": (None, None),  # No CSV found
    "orthopedic": ("datasets/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv", "class"),
    "gastro": ("datasets/acid-reflux-dataset-for-classification/preprocessed_data_final.csv", "Label"),
    "autoimmune": ("datasets/all-autoimmune-disorder-10k/Final_Balanced_Autoimmune_Disorder_Dataset.csv", "condition")
}

models_to_use = {
    "random_forest": RandomForestClassifier(),
    "svm": SVC(probability=True),
    "decision_tree": DecisionTreeClassifier(),
    "neural_net": MLPClassifier(max_iter=1000)
}

for disease, (file_path, target_col) in dataset_info.items():
    try:
        if not file_path or not os.path.exists(file_path):
            print(f"Skipping {disease}: file not found.")
            continue

        df = pd.read_csv(file_path)
        if target_col not in df.columns:
            print(f"Skipping {disease}: target column '{target_col}' not found.")
            continue

        df = df.dropna()

        # Convert datetime columns to numeric (timestamp)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype('int64') // 10**9
            elif df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[col] = df[col].astype('int64') // 10**9
                except Exception:
                    pass

        # Encode categorical features (excluding target)
        for col in df.columns:
            if df[col].dtype == 'object' and col != target_col:
                df[col] = df[col].astype('category').cat.codes

        y = df[target_col]
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        X = df.drop(columns=[target_col])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for name, model in models_to_use.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"{disease}_{name} Accuracy: {acc:.2f}")
            joblib.dump(model, f"models/{disease}_{name}.pkl")

    except Exception as e:
        print(f"Error processing {disease}: {e}")