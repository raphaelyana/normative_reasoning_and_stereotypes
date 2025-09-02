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

    # ---------- BASE (baseline) ----------
    df_base = pd.read_csv(base_file_path)

    base_keep = [c for c in [sample_id_col, label_col, pred_col,
                             "prompt_tokens", "completion_tokens",
                             "tokens_used", "max_tokens"] if c in df_base.columns]

    df_base = df_base[base_keep].rename(columns={pred_col: "base_pred"})

    base_rename = {}
    if "prompt_tokens" in df_base.columns:
        base_rename["prompt_tokens"] = "prompt_tokens__base_pred"
    if "completion_tokens" in df_base.columns:
        base_rename["completion_tokens"] = "completion_tokens__base_pred"
    if "tokens_used" in df_base.columns:
        base_rename["tokens_used"] = "tokens_used__base_pred"
    if "max_tokens" in df_base.columns:
        base_rename["max_tokens"] = "max_tokens__base_pred"
    df_base = df_base.rename(columns=base_rename)

    merged = df_base.copy()

    # ---------- PROFILES ----------
    profile_files = glob.glob(role_playing_glob_pattern)
    print(f"=== Found {len(profile_files)} profile result files.")

    for file in profile_files:
        profile_folder = os.path.basename(os.path.dirname(file))
        clean_profile_name = re.sub(r'_(passive|active)$', '', profile_folder)

        df_profile = pd.read_csv(file)

        if sample_id_col not in df_profile.columns or pred_col not in df_profile.columns:
            print(f"=== Skipping {profile_folder} (missing required columns)")
            continue

        keep_cols = [c for c in [sample_id_col, pred_col,
                                 "prompt_tokens", "completion_tokens",
                                 "tokens_used", "max_tokens"] if c in df_profile.columns]
        df_p = df_profile[keep_cols].copy()

        ren = {pred_col: clean_profile_name}

        if "prompt_tokens" in df_p.columns:
            ren["prompt_tokens"] = f"prompt_tokens__{clean_profile_name}"
        if "completion_tokens" in df_p.columns:
            ren["completion_tokens"] = f"completion_tokens__{clean_profile_name}"
        if "tokens_used" in df_p.columns:
            ren["tokens_used"] = f"tokens_used__{clean_profile_name}"
        if "max_tokens" in df_p.columns:
            ren["max_tokens"] = f"max_tokens__{clean_profile_name}"

        df_p = df_p.rename(columns=ren)

        if profile_folder in merged.columns and profile_folder != clean_profile_name:
            merged = merged.drop(columns=[profile_folder])

        merged = merged.merge(df_p, on=sample_id_col, how="left")

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

    pred_cols = ["base_pred"] + [c for c in merged.columns if re.match(r"^profile\d+$", c)]
    for col in pred_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.strip().str.lower()

    fixed_columns = [sample_id_col, label_col, "base_pred"]
    profile_pred_cols = [c for c in merged.columns
                         if c.startswith("profile") and c not in fixed_columns + meta_columns]

    token_prefixes = ("prompt_tokens__", "completion_tokens__", "tokens_used__", "max_tokens__")
    token_columns = [c for c in merged.columns if c.startswith(token_prefixes)]

    def extract_profile_number(col_name: str):
        m = re.search(r"profile(\d+)$", col_name)
        return int(m.group(1)) if m else float("inf")
    profile_pred_cols = sorted(profile_pred_cols, key=extract_profile_number)

    final_columns = fixed_columns + meta_columns + profile_pred_cols + token_columns
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

def guarded_labelspace_analysis(
    analysis_func,
    merged_df,
    case,
    person_set=None,
    baseline_col="base_pred",
    profile_prefix="profile",
    **kwargs
):
    """
    Wrapper général pour l'analyse adaptative selon le label space.
    Applique la binarisation locale si multi-classes (>2 labels), sinon passe les arguments tels quels.
    Tous les arguments supplémentaires sont transmis à la fonction d'analyse via **kwargs.
    """
    n_labels = len(case.valid_labels)
    if n_labels == 2:
            return analysis_func(
                merged_df,
                person_set=person_set,
                case=case,
                baseline_col=baseline_col,
                profile_prefix=profile_prefix,
                **kwargs
            )
    elif n_labels > 2:
        print(f"[INFO] Multi-class label space detected ({n_labels} labels). Binarizing for analysis.")

        df_bin = merged_df.copy()
        for p in [c for c in merged_df.columns if c.startswith(profile_prefix)]:
            df_bin[p] = (merged_df[p] == merged_df["true_label"]).astype(int)
        
        df_bin[baseline_col] = (merged_df[baseline_col] == merged_df["true_label"]).astype(int)
        
        df_bin["true_label"] = 1
        case_bin = case.copy(update={
            "case_name": case.case_name + "_binarized",
            "valid_labels": [1, 0],
            "label_map": {"1": "1", "0": "0"},
            "label_col": "true_label",
        })
        
        return analysis_func(
            df_bin,
            person_set=person_set,
            case=case_bin,
            baseline_col=baseline_col,
            profile_prefix=profile_prefix,
            **kwargs
        )
    else:
        print("[WARN] case.valid_labels vide ou non reconnu. Analyse non effectuée.")
        return {}



def resolve_plot_dir(
    case,
    plots_root: Optional[str] = None,
    strategy: Optional[str] = None,   # e.g., "zero_shot", "few_shot", "cot"
    stage: Optional[str] = None,      # e.g., "preliminary", "tier1", "tier2"
    extra_subdir: Optional[str] = None,
    sub_case: Optional[str] = None,
) -> str:
    """
    Build: <plots_root or 'results/figs'>/[strategy]/<case_name>/[stage]/[extra_subdir]
    Creates the directory if missing.
    """
    base = plots_root or os.path.join("results", "figs")
    case_name = case if isinstance(case, str) else getattr(case, "case_name", str(case))
    parts = [base]
    if strategy:
        parts.append(strategy)
    parts.append(case_name)
    if sub_case:
        parts.append(sub_case)
    if stage:
        parts.append(stage)
    if extra_subdir:
        parts.append(extra_subdir)
    out = os.path.join(*parts)
    os.makedirs(out, exist_ok=True)
    return out
