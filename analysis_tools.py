import os
import glob
import re

import pandas as pd
import numpy as np

from enum import Enum
from profiles.schema import PersonSet


def load_and_merge_profiles(
    base_file_path: str,
    role_playing_glob_pattern: str,
    sample_df: pd.DataFrame,
    sample_id_col: str = "sample_id",
    label_col: str = "true_label",
    pred_col: str = "pred_label",
    extra_cols: list = ["stereotype_type", "original_dataset"]
) -> pd.DataFrame:

    df_base = pd.read_csv(base_file_path)
    merged = df_base[[sample_id_col, label_col, pred_col]].rename(columns={pred_col: "base_pred"})

    profile_files = glob.glob(role_playing_glob_pattern)
    print(f"=== Found {len(profile_files)} profile result files.")

    for file in profile_files:
        profile_folder = os.path.basename(os.path.dirname(file))
        df_profile = pd.read_csv(file)

        if sample_id_col not in df_profile.columns or pred_col not in df_profile.columns:
            print(f"=== Skipping {profile_folder} (missing required columns)")
            continue

        df_profile = df_profile[[sample_id_col, pred_col]].rename(columns={pred_col: profile_folder})

        if profile_folder in merged.columns:
            merged = merged.drop(columns=[profile_folder])

        merged = merged.merge(df_profile, on=sample_id_col, how="left")

    sample_df = sample_df.reset_index(drop=True)
    sample_df[sample_id_col] = sample_df.index

    merged = merged.merge(
        sample_df[[sample_id_col] + extra_cols],
        on=sample_id_col,
        how="left"
    )

    merged[label_col] = merged[label_col].astype(str).str.strip().str.lower()
    merged["base_pred"] = merged["base_pred"].astype(str).str.strip().str.lower()
    for col in merged.columns[3:]:
        merged[col] = merged[col].astype(str).str.strip().str.lower()

    fixed_columns = [sample_id_col, label_col, "base_pred"]
    meta_columns = [col for col in extra_cols if col in merged.columns]

    profile_columns = [
        col for col in merged.columns
        if col.startswith("profile") and col not in fixed_columns + meta_columns
    ]

    # Sort numerically by profile number
    def extract_profile_number(col_name):
        match = re.search(r"profile(\d+)", col_name)
        return int(match.group(1)) if match else float('inf')

    profile_columns = sorted(profile_columns, key=extract_profile_number)

    # Final column ordering
    final_columns = fixed_columns + meta_columns + profile_columns
    merged = merged[final_columns]

    print("=== Merged DataFrame ready with columns:\n", merged.columns.tolist())
    return merged


def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def get_demographic_info(profile_name: str, person_set) -> str:
    """
    Return a demographic signature like 'white_man_low_harm' or 'white_man_age_2'.
    Works whether person_set.get_traits returns a dict or a PersonMeta.
    """
    pid = profile_name.replace("_passive", "").replace("_active", "")
    traits = person_set.get_traits(pid)

    def read(tr, key):
        v = tr.get(key) if isinstance(tr, dict) else getattr(tr, key, None)
        if isinstance(v, Enum):
            v = v.value
        return None if v is None else str(v).lower()

    parts = []
    for k in ("gender", "ethnicity"): 
        v = read(traits, k)
        if v:
            parts.append(v)

    cs = read(traits, "cognitive_style")
    if cs:
        parts.append(cs)
    else:
        extra_fields = []
        if hasattr(traits, "__dataclass_fields__"):
            extra_fields = [f for f in traits.__dataclass_fields__ if f not in ("gender", "ethnicity", "cognitive_style")]
        elif isinstance(traits, dict):
            extra_fields = [f for f in traits.keys() if f not in ("gender", "ethnicity", "cognitive_style")]

        for f in extra_fields:
            v = read(traits, f)
            if v:
                parts.append(v)

    return "_".join(parts) if parts else "unknown"



def has_cognitive_style_data(person_set: PersonSet, sample_size: int = 10) -> bool:
    """
    Check if PersonSet contains cognitive style data by sampling profiles.
    
    Parameters:
    - person_set: PersonSet to check
    - sample_size: Number of profiles to sample for checking
    
    Returns:
    - True if cognitive style data found, False otherwise
    """
    if not person_set.metadata:
        return False
    
    # Sample some profile IDs to check
    profile_ids = list(person_set.metadata.keys())
    sample_ids = profile_ids[:min(sample_size, len(profile_ids))]
    
    for pid in sample_ids:
        traits = person_set.get_traits(pid, ["cognitive_style"])
        if traits.get("cognitive_style", "Unknown") != "Unknown":
            return True
    
    return False