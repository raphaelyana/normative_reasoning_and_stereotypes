import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from openai import RateLimitError

from clustering.clustering_train_test import (
    AdaptiveClusteringPipeline,
    RoutingPolicyConfig,
)

from profiles.schema import PersonSet
from cases.cases_config import CaseConfig
from few_shots import FewShot
from profiles.profile_sets import (
    PERSON_ETHNICS,
    PERSON_SYSTEMATIC,
    PERSON_SEEDS_CORE,
)
from cases.stereotypes_case import stereotypes_case
from cases.manipulation_case import manipulation_case
from cases.mmlu_case import mmlu_case

from data_loader import get_additional_fields


# -----------------------------
# Label mapping (shared helper)
# -----------------------------
def _safe_map_label(case: CaseConfig, raw: str):
    """
    Normalize raw model output and map to case.label_map.
    Returns None if unmapped so caller can skip the row.
    """
    if raw is None:
        return None
    if case.case_name == "mmlu":
        key = str(raw).strip().upper()[:1]  # "A"/"B"/"C"/"D"
    else:
        key = str(raw).strip().lower().capitalize()  # "Yes"/"No"
    return case.label_map.get(key)


# -----------------------------
# Few-shot gen_fns (cached)
# -----------------------------
def make_gen_fn(
    case: CaseConfig, client, model, examples_df, person_set=PERSON_ETHNICS, max_tokens=300
):
    """
    Returns a callable: gen_fn(text, profile_key_or_None, *, row=None) -> (mapped_label, token_usage).
    For MMLU we pass the row to FewShot.classify; output mapped A/B/C/D -> int by caller if desired.
    """
    cache: Dict[tuple, FewShot] = {}
    label_map = case.label_map

    def _get_fewshot(profile_key: Optional[str], role: str):
        key = (profile_key, role)
        if key not in cache:
            if profile_key is None:
                fs = FewShot(
                    case=case,
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                    task_definition="",
                    n_shots=3,
                    examples_df=examples_df,
                    person_key=None,
                    role_playing="none",
                    person_set=person_set,
                )
            else:
                fs = FewShot(
                    case=case,
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                    task_definition="",
                    n_shots=3,
                    examples_df=examples_df,
                    person_key=profile_key,
                    role_playing="passive",
                    person_set=person_set,
                )
            cache[key] = fs
        return cache[key]

    def _map_output(case_name: str, raw: str):
        if case_name == "mmlu":
            token = (raw or "").strip().upper()[:1]
            return label_map.get(token, list(label_map.values())[-1])
        else:
            token = (raw or "").strip().lower()
            return label_map.get(token, list(label_map.values())[-1])

    def gen_fn(text: str, profile_key: Optional[str], *, row: Optional[dict] = None):
        role = "none" if profile_key is None else "passive"
        fs = _get_fewshot(profile_key, role)

        if case.case_name == "mmlu":
            pred, stats = fs.classify(text, row=row or {})
        else:
            pred, stats = fs.classify(text)

        mapped = _map_output(case.case_name, pred)
        tok = stats.get("tokens_used") if isinstance(stats, dict) else None
        return mapped, tok

    return gen_fn


def make_online_gen_fn(
    case: CaseConfig,
    client,
    model,
    person_set: PersonSet,
    examples_df,
    task_definition=None,
    n_shots=3,
    max_tokens=300,
):
    """
    Returns gen_fn(text, profile_key, row=None) -> (mapped_label, tokens_used)
    - Caches FewShot classifiers per profile (incl. baseline with profile_key=None).
    - Applies MMLU uppercasing; others lowercasing.
    - Maps raw model outputs via case.label_map so it matches df['true_label'].
    """
    cls_cache: Dict[str, FewShot] = {}

    def get_cls(profile_key):
        key = profile_key or "__BASELINE__"
        if key not in cls_cache:
            role = "passive" if profile_key else "none"
            cls_cache[key] = FewShot(
                case=case,
                client=client,
                model=model,
                max_tokens=max_tokens,
                task_definition=task_definition,
                n_shots=n_shots,
                examples_df=examples_df,
                person_key=profile_key,
                role_playing=role,
                person_set=person_set,
            )
        return cls_cache[key]

    canon = dict(case.label_map) if hasattr(case, "label_map") else {}
    for k, v in list(canon.items()):
        canon.setdefault(str(k).lower(), v)
        canon.setdefault(str(k).upper(), v)

    def normalize_and_map(raw: str, case_name: str):
        if case_name == "mmlu":
            key = (raw or "").strip().upper()
        else:
            key = (raw or "").strip().lower().capitalize()
        return canon.get(key, canon.get(key.strip(), raw))

    def gen_fn(text, profile_key, row=None):
        cls = get_cls(profile_key)
        if case.case_name == "mmlu":
            pred, stats = cls.classify(text, row=row or {})
        else:
            pred, stats = cls.classify(text)

        mapped = normalize_and_map(pred, case.case_name)
        tokens = (stats or {}).get("tokens_used")
        return mapped, tokens

    return gen_fn


def _infer_sample_df_from_train(
    train_df: pd.DataFrame, case: CaseConfig
) -> Optional[pd.DataFrame]:
    """
    Return a minimal sample_df (sample_id + input text) derived from train_df
    *only if needed*. If train_df lacks either column, return None and let the
    pipeline raise a clear error if text is missing.
    """
    if "sample_id" in train_df.columns and case.input_col in train_df.columns:
        return train_df[["sample_id", case.input_col]].copy()
    return None


# ----------------------------------------------------------------------
# (MOD 2 + 3) Few-shot evaluator with tqdm + robust saving + robust metrics
# ----------------------------------------------------------------------
def _eval_and_save_few_shot(
    *,
    case_name: str,
    case: CaseConfig,
    data: pd.DataFrame,
    few_shot_examples: pd.DataFrame,
    client,
    model,
    person_set: PersonSet,
    person_key: Optional[str],
    role_playing: str,
    output_file: str,
    max_tokens: int = 300,
    tqdm_desc: Optional[str] = None,
):
    """
    Runs a guarded few-shot loop and writes CSV + prints metrics.
    - Skips unmapped predictions.
    - Handles empty outputs safely.
    - Shows a tqdm progress bar per row.
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    classifier = FewShot(
        case=case,
        client=client,
        model=model,
        max_tokens=max_tokens,
        task_definition="" if case_name != "manipulation" else "",  # provided via examples
        n_shots=3,
        examples_df=few_shot_examples,
        person_key=person_key,
        role_playing=role_playing,
        person_set=person_set,
    )

    rows = []
    data_iter = data.reset_index(drop=True).iterrows()
    desc = (
        tqdm_desc
        or f"{case_name} | {person_key or 'baseline'} | few_shot | {role_playing}"
    )

    for idx, row in tqdm(data_iter, total=len(data), desc=desc):
        try:
            # Basic schema checks
            if case.input_col not in row or case.label_col not in row:
                # Skip rows missing required fields
                continue

            text = row[case.input_col]
            true_label = row[case.label_col]
            if pd.isna(text):
                continue

            if case_name == "mmlu":
                pred, stats = classifier.classify(text, row=row.to_dict())
                raw = (pred or "").strip()
            else:
                pred, stats = classifier.classify(text)
                raw = (pred or "").strip()

            mapped = _safe_map_label(case, raw)
            if mapped is None:
                # Unmapped model output -> skip to keep metrics stable
                continue

            # Robust stats extraction
            tokens_used = (stats or {}).get("tokens_used")
            prompt_tokens = (stats or {}).get("prompt_tokens")
            completion_tokens = (stats or {}).get("completion_tokens")
            latency = (stats or {}).get("latency")

            additional = get_additional_fields(row, case_name)

            rows.append(
                {
                    "sample_id": idx,
                    "text": text,
                    "true_label": true_label,
                    "pred_label": mapped,
                    "max_tokens": classifier.max_tokens,
                    "tokens_used": tokens_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency": latency,
                    **additional,
                }
            )

        except RateLimitError as e:
            print(f"WARNING: Rate limit at sample {idx}: {e}")
            continue
        except Exception as e:
            print(f"ERROR at sample {idx}: {e}")
            continue

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df_out)} rows to {output_file}")

    # Metrics (guarded)
    if df_out.empty:
        print("No valid predictions mapped; skipping metrics.")
        return df_out

    df_out = df_out.dropna(subset=["true_label", "pred_label"]).copy()
    if df_out.empty:
        print("No valid rows after dropping NaNs; skipping metrics.")
        return df_out

    if case_name in {"manipulation", "mmlu"}:
        y_true = df_out["true_label"].astype(int)
        y_pred = df_out["pred_label"].astype(int)
    else:
        y_true = df_out["true_label"].astype(str).str.strip().str.lower()
        y_pred = df_out["pred_label"].astype(str).str.strip().str.lower()

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Confusion Matrix ===")
    labels = sorted(set(pd.Series(y_true).unique()) | set(pd.Series(y_pred).unique()))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(conf_matrix, index=labels, columns=labels))

    accuracy = accuracy_score(y_true, y_pred)
    print(
        f"\n=== Accuracy for {case_name} | {person_key or 'baseline'} | few_shot | {role_playing}: {accuracy:.3f}"
    )

    return df_out


def _role_folder_for_person_set(person_set: PersonSet) -> str:
    seeds = getattr(person_set, "seeds", None)
    if seeds is PERSON_SEEDS_CORE:
        return "role_playing_core"
    if seeds == PERSON_SYSTEMATIC.seeds:
        return "role_playing_system"
    if seeds == PERSON_ETHNICS.seeds:
        return "role_playing_ethnics"
    return "role_playing"


def run_few_shot_case(
    *,
    case_name: str,
    case: CaseConfig,
    data: pd.DataFrame,
    few_shot_examples: pd.DataFrame,
    selected_profiles: List[str],
    client,
    model,
    person_set: PersonSet = PERSON_ETHNICS,
    model_foldername: str = "default_model",
    results_root: str = "results",
    max_tokens: int = 300,
):
    """
    Executes few-shot classification for:
      - Baseline (no persona, role='none')
      - Each selected profile (role='passive')

    Produces CSVs in:
      results/{model_foldername}/clustering/classic/...
      results/{model_foldername}/clustering/{role_folder}/{profile_key}_passive/...

    Returns a dict with output file paths.
    """
    outputs = {}

    # File suffix (matches your original naming)
    type_suffix = "" if case_name == "mmlu" else "binary_"
    file_suffix = f"results_{case_name}_few_shot_prompt_short_3examples_{type_suffix}test.csv"

    # --- Baseline ---
    baseline_out = os.path.join(
        results_root,
        model_foldername,
        "clustering",
        "classic",
        file_suffix,
    )
    _eval_and_save_few_shot(
        case_name=case_name,
        case=case,
        data=data,
        few_shot_examples=few_shot_examples,
        client=client,
        model=model,
        person_set=person_set,
        person_key=None,
        role_playing="none",
        output_file=baseline_out,
        max_tokens=max_tokens,
        tqdm_desc=f"{case_name} | baseline | few_shot | none",
    )
    outputs["baseline"] = baseline_out

    # --- Profiles ---
    role_folder = _role_folder_for_person_set(person_set)
    for person_key in selected_profiles:
        prof_out = os.path.join(
            results_root,
            model_foldername,
            "clustering",
            role_folder,
            f"{person_key}_passive",
            file_suffix,
        )
        _eval_and_save_few_shot(
            case_name=case_name,
            case=case,
            data=data,
            few_shot_examples=few_shot_examples,
            client=client,
            model=model,
            person_set=person_set,
            person_key=person_key,
            role_playing="passive",
            output_file=prof_out,
            max_tokens=max_tokens,
            tqdm_desc=f"{case_name} | {person_key} | few_shot | passive",
        )
        outputs[person_key] = prof_out

    return outputs


# ----------------------------------------------------------------------
# (Unchanged) Clustering pipeline runner for one case
# ----------------------------------------------------------------------
def _infer_sample_df_from_train(
    train_df: pd.DataFrame, case: CaseConfig
) -> Optional[pd.DataFrame]:
    """
    Return a minimal sample_df (sample_id + input text) derived from train_df
    *only if needed*. If train_df lacks either column, return None and let the
    pipeline raise a clear error if text is missing.
    """
    if "sample_id" in train_df.columns and case.input_col in train_df.columns:
        return train_df[["sample_id", case.input_col]].copy()
    return None


def run_one_case(
    *,
    case_name: str,
    case: CaseConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    person_set: PersonSet,
    client,  # kept for parity with your original signature (not used here)
    model,   # kept for parity with your original signature (not used here)
    with_fdr: bool = False,
    risk_cap: Optional[float] = None,
    category_risk_cap: Optional[float] = None,
    min_category_n: int = 50,
    save_dir: str = "clustering_models",
    perf_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Trains and tests the clustering+routing pipeline for one case.
    - Does NOT require a separate sample_df.
    - If embeddings need text and the text isn't inside train/test, we
      auto-derive a minimal sample_df from train_df (sample_id + input text).
    """
    sample_df = _infer_sample_df_from_train(train_df, case)

    policy = RoutingPolicyConfig(
        enable_pareto=False,
        lambda_tok=5e-4,
        lambda_extra=2.0,
        risk_cap=risk_cap,
        category_risk_cap=category_risk_cap,
        min_category_n=min_category_n,
        require_fdr=with_fdr,
        q_threshold=0.10,
        enable_tier3_exploratory=False,
    )

    pipeline = AdaptiveClusteringPipeline(
        save_dir=save_dir,
        embedding_model_name="all-MiniLM-L6-v2",
        routing_policy=policy,
    )

    train_out = pipeline.train(
        merged_df=train_df,
        case=case,
        sample_df=sample_df,
        person_set=person_set,
        version_name=f"{case_name}_v1",
        perf_df=perf_df,
    )

    test_out = pipeline.test(
        test_df=test_df,
        sample_df=sample_df,
        case=case,
        person_set=person_set,
        eval_mode="offline",
    )

    return {
        "case": case_name,
        "train": train_out,
        "test": test_out,
        "policy": policy.__dict__,
    }
