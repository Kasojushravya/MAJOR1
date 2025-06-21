import kagglehub
import os
import shutil

def download_and_extract(dataset_name, target_folder):
    print(f"Downloading {dataset_name}...")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Downloaded to: {path}")

    # Kagglehub usually downloads and extracts files under some folder or zip
    # Move/organize extracted files into your 'datasets' folder with consistent naming
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    
    dest_path = os.path.join("datasets", target_folder)
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    shutil.move(path, dest_path)
    print(f"Moved dataset to {dest_path}")

datasets = {
    "prakharrathi25/copd-student-dataset": "copd-student-dataset",
    "osmi/mental-health-in-tech-survey": "mental-health-in-tech-survey",
    "redwankarimsony/heart-disease-data": "heart-disease-data",
    "uciml/pima-indians-diabetes-database": "pima-indians-diabetes-database",
    "erdemtaha/cancer-data": "cancer-data",
    "alifatahi/multi-class-neurological-disorder-mcnd-dataset": "multi-class-neurological-disorder-mcnd-dataset",
    "uciml/biomechanical-features-of-orthopedic-patients": "biomechanical-features-of-orthopedic-patients",
    "mansoordaku/ckdisease": "ckdisease",
    "ahmetsametgrkan/acid-reflux-dataset-for-classification": "acid-reflux-dataset-for-classification",
    "abdullahragheb/all-autoimmune-disorder-10k": "all-autoimmune-disorder-10k",
}

for dataset, folder in datasets.items():
    download_and_extract(dataset, folder)
