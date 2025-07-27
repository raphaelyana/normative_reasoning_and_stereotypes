from sklearn.metrics import classification_report
import os
import pandas as pd

def save_classification_reports(df: pd.DataFrame, base_folder: str, label_col="true_label"):
    """
    Save classification reports for each profile in their corresponding result folders.
    """
    true = df[label_col].astype(str).str.strip().str.lower()
    profile_cols = [col for col in df.columns if col.startswith("profile")]

    for profile in profile_cols:
        pred = df[profile].astype(str).str.strip().str.lower()

        report = classification_report(true, pred, digits=3)

        profile_folder = os.path.join(base_folder, profile)
        os.makedirs(profile_folder, exist_ok=True)
        output_path = os.path.join(profile_folder, "classification_report.txt")

        with open(output_path, "w") as f:
            f.write(f"=== Classification Report for {profile} ===\n\n")
            f.write(report)
