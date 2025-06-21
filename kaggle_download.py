import kagglehub
import os

datasets = {
    "COPD": "prakharrathi25/copd-student-dataset",
    "MentalHealth": "osmi/mental-health-in-tech-survey",
    "HeartDisease": "redwankarimsony/heart-disease-data",
    "Diabetes": "uciml/pima-indians-diabetes-database",
    "Cancer": "erdemtaha/cancer-data",
    "Neural": "alifatahi/multi-class-neurological-disorder-mcnd-dataset",
    "Orthopedic": "uciml/biomechanical-features-of-orthopedic-patients",
    "Kidney": "mansoordaku/ckdisease",
    "Gastro": "ahmetsametgrkan/acid-reflux-dataset-for-classification",
    "Autoimmune": "abdullahragheb/all-autoimmune-disorder-10k"
}

os.makedirs("datasets", exist_ok=True)

for name, path in datasets.items():
    print(f"Downloading {name}...")
    dataset_path = kagglehub.dataset_download(path)
    print(f"{name} saved to: {dataset_path}")
