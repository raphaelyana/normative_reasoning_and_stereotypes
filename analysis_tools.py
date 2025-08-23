import os
import glob
import re

import pandas as pd
import numpy as np

from enum import Enum
from profiles.schema import PersonSet
from cases.cases_config import CaseConfig
from typing import Tuple, List, Optional


def load_and_merge_profiles(
    base_file_path: str,
    role_playing_glob_pattern: str,
    sample_df: pd.DataFrame,
    case: CaseConfig,
    sample_id_col: str = "sample_id",
    label_col: str = "true_label",
    pred_col: str = "pred_label",
    extra_sample_cols: Optional[List[str]] = None,
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

        clean_profile_name = re.sub(r'_(passive|active)$', '', profile_folder)
        df_profile = df_profile[[sample_id_col, pred_col]].rename(columns={pred_col: clean_profile_name})

        if profile_folder in merged.columns:
            merged = merged.drop(columns=[profile_folder])

        merged = merged.merge(df_profile, on=sample_id_col, how="left")

    sample_df = sample_df.reset_index(drop=True)
    sample_df[sample_id_col] = sample_df.index

    meta_candidates = (case.category_cols if hasattr(case, "category_cols") else [])
    if extra_sample_cols:
        meta_candidates = list(meta_candidates) + list(extra_sample_cols)

    meta_columns = [col for col in meta_candidates if col in sample_df.columns]

    merged = merged.merge(
        sample_df[[sample_id_col] + meta_columns],
        on=sample_id_col,
        how="left"
    )

    merged[label_col] = merged[label_col].astype(str).str.strip().str.lower()
    merged["base_pred"] = merged["base_pred"].astype(str).str.strip().str.lower()
    for col in merged.columns[3:]:
        merged[col] = merged[col].astype(str).str.strip().str.lower()

    fixed_columns = [sample_id_col, label_col, "base_pred"]
    profile_columns = [
        col for col in merged.columns
        if col.startswith("profile") and col not in fixed_columns + meta_columns
    ]

    def extract_profile_number(col_name):
        match = re.search(r"profile(\d+)", col_name)
        return int(match.group(1)) if match else float("inf")

    profile_columns = sorted(profile_columns, key=extract_profile_number)

    final_columns = fixed_columns + meta_columns + profile_columns
    merged = merged[final_columns]

    print("=== Merged DataFrame ready with columns:\n", merged.columns.tolist())
    return merged


def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def get_demographic_info(profile_name: str, person_set: PersonSet) -> str:
    """
    Fixed version that works with your PersonSet.get_traits method
    """
    try:
        # Use your existing get_traits method
        traits = person_set.get_traits(profile_name, group_keys=("gender", "ethnicity", "cognitive_style", "age"))
        
        def normalize_trait(value):
            if value is None or value == "Unknown":
                return None
            return str(value).lower()
        
        parts = []
        
        # Always include ethnicity and gender first
        ethnicity = normalize_trait(traits.get("ethnicity"))
        gender = normalize_trait(traits.get("gender"))
        
        if ethnicity:
            parts.append(ethnicity)
        if gender:
            parts.append(gender)
        
        # Add cognitive style or age if available
        cognitive_style = normalize_trait(traits.get("cognitive_style"))
        age = normalize_trait(traits.get("age"))
        
        if cognitive_style:
            parts.append(cognitive_style)
        elif age:
            parts.append(f"age_{age}")
        
        return "_".join(parts) if parts else "unknown"
        
    except Exception as e:
        print(f"Warning: Could not get traits for {profile_name}: {e}")
        return "unknown"



def get_analysis_group_keys(person_set: PersonSet) -> Tuple[str, ...]:
    """
    Dynamically determine group keys for analysis based on available metadata
    """
    required_keys, optional_keys = get_available_traits(person_set)
    group_keys = tuple(required_keys + optional_keys)
    
    print(f"Available traits detected:")
    print(f"  Required: {required_keys}")
    print(f"  Optional: {optional_keys}")
    print(f"Using group keys: {group_keys}")
    
    return group_keys



def get_available_traits(person_set: PersonSet) -> Tuple[List[str], List[str]]:
    """
    Dynamically detect available traits from PersonMeta dataclass
    
    Returns:
    --------
    Tuple[List[str], List[str]]
        - required_keys: Always present traits (gender, ethnicity)
        - optional_keys: Optional traits that exist in the metadata
    """
    # Core traits that should always be present for bias analysis
    required_keys = ["gender", "ethnicity"]
    
    if not person_set.metadata:
        return required_keys, []
    
    sample_size = min(5, len(person_set.metadata))
    sample_profiles = list(person_set.metadata.keys())[:sample_size]
    
    first_meta = list(person_set.metadata.values())[0]
    all_possible_fields = list(first_meta.__dataclass_fields__.keys())
    
    available_traits = []
    for field in all_possible_fields:
        field_has_data = False
        
        for profile in sample_profiles:
            try:
                traits = person_set.get_traits(profile, group_keys=[field])
                if traits.get(field, "Unknown") != "Unknown":
                    field_has_data = True
                    break
            except:
                continue
        
        if field_has_data:
            available_traits.append(field)

    required_keys = []
    optional_keys = []
    
    priority_fields = ["gender", "ethnicity"]
    for field in priority_fields:
        if field in available_traits:
            required_keys.append(field)
            available_traits.remove(field)
    
    optional_keys = available_traits
    
    return required_keys, optional_keys



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