import os
import glob

# Folders to check
folders = [
    "datasets/heart-disease-data",
    "datasets/cancer-data",
    "datasets/copd-student-dataset",
    "datasets/multi-class-neurological-disorder-mcnd-dataset",
    "datasets/acid-reflux-dataset-for-classification",
    "datasets/all-autoimmune-disorder-10k"
]

print("\n📂 Searching for CSV files...\n")

for folder in folders:
    print(f"🔎 In: {folder}")
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        print("❌ No CSV files found\n")
    else:
        for file in csv_files:
            print("✅ Found:", os.path.basename(file))
        print()
