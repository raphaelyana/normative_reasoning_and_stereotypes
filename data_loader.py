import pandas as pd
from typing import List, Optional
from datasets import load_dataset
import re
import numpy as np
import pathlib
import json


NORMATIVE_CATEGORY = [
    "moral_disputes",
    "philosophy",
    "world_religions",
    "us_foreign_policy",
    "sociology",
    "professional_psychology",
    "professional_law",
    "moral_scenarios",
    "human_sexuality",
    "international_law",
]

CONTROL_CATEGORY = [
    "college_mathematics",
    "college_physics",
    "formal_logic",
    "logical_fallacies",
    "college_computer_science",
]

SUBJECT_LIST = NORMATIVE_CATEGORY+CONTROL_CATEGORY




def fetch_categories_mmlu(subject_list: List[str] = SUBJECT_LIST,
                          normative_category: List[str] = NORMATIVE_CATEGORY,
                          control_category: List[str] = CONTROL_CATEGORY):

    if subject_list is None or len(subject_list)==0:
        return "ERROR: No valid subjects, please enter correct subjects."
    
    dfs = []

    for sub in subject_list:
        ds = load_dataset("cais/mmlu", sub)
        if "test" not in ds:
            print(f"Warning: Subject '{sub}' not found in the dataset.")
            continue
        df = ds["test"].to_pandas()
        df["subject"]=sub
        if sub in normative_category:
            df["category"] = "normative"
        elif sub in control_category:
            df["category"] = "control"
        else:
            raise ValueError(f"Subject '{sub}' is not in normative or control categories.")

        dfs.append(df)

    mmlu_full_df = pd.concat(dfs, ignore_index=True)

    return mmlu_full_df


QUOTE_BLOCK = re.compile(r"""(['"])(.*?)\1""", flags=re.DOTALL)

def parse_mmlu_choices(val):
    """Parse strings like "['Choice 1'\n  'Choice 2'\n  'Choice 3']" → ['Choice 1', 'Choice 2', 'Choice 3']"""
    # numpy arrays → flatten to list
    if isinstance(val, np.ndarray):
        val = val.ravel().tolist()

    # unwrap single-string list/tuple: ["'A' 'B' ..."] → "'A' 'B' ..."
    if isinstance(val, (list, tuple)) and len(val) == 1 and isinstance(val[0], str):
        val = val[0]

    # already a list[str]
    if isinstance(val, (list, tuple)) and all(isinstance(t, str) for t in val):
        return [t.strip() for t in val]

    if isinstance(val, str):
        s = val.strip()
        # strip outer brackets (works for [] or [[]])
        s = re.sub(r'^\s*\[\s*\[?\s*', '', s)
        s = re.sub(r'\s*\]?\s*\]\s*$', '', s)

        # core: extract every quoted block (handles '…' and "…", with newlines)
        parts = [m[1].strip() for m in QUOTE_BLOCK.findall(s)]
        if parts:
            return parts

        # fallback: split on newlines / big spaces if somehow unquoted
        parts = [p.strip(" '\"") for p in re.split(r'\r?\n|\s{2,}', s) if p.strip(" '\"")]
        return parts if parts else [s]

    # anything else → stringify
    return [str(val).strip()]


def normalize_answer(ans, choices):
    if not choices:
        return -1
    try:
        ai = int(ans)
        if 0 <= ai < len(choices): return ai          
        if 1 <= ai <= len(choices): return ai - 1     
    except Exception:
        pass

    if isinstance(ans, str) and len(ans) == 1 and ans.isalpha():
        idx = ord(ans.upper()) - ord('A')
        if 0 <= idx < len(choices): return idx

    if isinstance(ans, str):
        norm = lambda t: re.sub(r'\s+', ' ', str(t)).strip().lower()
        s = norm(ans)
        for i, c in enumerate(choices):
            if norm(c) == s:
                return i
    return -1



def format_mmlu_prompt(question: str, choices):
    """
    Use parse_mmlu_choices to ensure list[str], then emit:
    Question: ...
    Choices:
    A. ...
    B. ...
    ...
    """
    ch = parse_mmlu_choices(choices)
    ch = [str(c).strip() for c in ch]
    letters = [chr(65 + i) for i in range(len(ch))]  # A, B, C, ...
    choices_str = "\n".join(f"{L}. {c}" for L, c in zip(letters, ch))
    question = ("" if question is None else str(question)).strip()
    return f"Question: {question}\nChoices:\n{choices_str}"


def finalize_mmlu_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure choices are list[str], answers are 0-based indices,
    build formatted_prompt, and return the final ordered columns.
    Expects columns at least: question, subject, category, choices, answer
    """
    df = input_df.copy()

    df["choices"] = df["choices"].apply(parse_mmlu_choices)
    df["answer"] = df.apply(lambda r: normalize_answer(r["answer"], r["choices"]), axis=1).astype(int)
    df["formatted_prompt"] = df.apply(lambda r: format_mmlu_prompt(r["question"], r["choices"]), axis=1)

    df = df.reset_index(drop=True)
    df["sample_id"] = df.index

    final_cols = ["sample_id", "question", "subject", "category", "choices", "answer", "formatted_prompt"]
    existing = [c for c in final_cols if c in df.columns]
    other = [c for c in df.columns if c not in existing]
    df = df[existing + other]

    return df





def load_mgsd_dataset(
        input_dataframe: pd.DataFrame,
        sample_size: dict = {
            'stereotype': 200,
            'anti-stereotype': 200,
            'unrelated': 100
        },
        sample_size_examples: dict = {
            'stereotype': 5,
            'anti-stereotype': 5,
            'unrelated': 5
        },
        random_state: int = 42,
        random_state_examples: int = 0,
        print_balance: bool = False,     
):
    """
    Loads the MGSD dataset

    Inputs:
    - input_dataframe: pd.DataFrame
        The input dataframe containing the MGSD dataset.
    - sample_size: dict
        A dictionary specifying the number of samples to take for each label in the MGSD dataset. Making it balanced ensures equal representation of each label.
    - sample_size_examples: dict
        A dictionary specifying the number of few-shot examples to take for each label in the MGSD dataset.
    - random_state: int
        Random state for reproducibility when sampling the main dataset.
    - random_state_examples: int
        Random state for reproducibility when sampling few-shot examples.
    - print_balance: bool
        If True, prints the balance of the dataset after sampling.
    
    Returns:
    - sample_mgsd: pd.DataFrame, 
        A dataframe containing the dataset samples used for testing.
    - sample_examples_mgsd: pd.DataFrame
        A dataframe containing the few-shot examples for the MGSD dataset.
    """


    if not isinstance(input_dataframe, pd.DataFrame):
        raise ValueError("input_dataframe must be a pandas DataFrame")

    balanced_samples_mgsd = []
    used_indices_mgsd = set()

    for label, size in sample_size.items():
        subset = input_dataframe[input_dataframe["label"] == label].sample(n=size, random_state=random_state)
        balanced_samples_mgsd.append(subset)
        used_indices_mgsd.update(subset.index)
    
    sample_mgsd = pd.concat(balanced_samples_mgsd).sample(frac=1, random_state=42).reset_index(drop=True)
    
    example_samples_mgsd = []

    for label, n_examples in sample_size_examples.items():
        candidates = input_dataframe[(input_dataframe["label"] == label) & (~input_dataframe.index.isin(used_indices_mgsd))]
        sampled_examples = candidates.sample(n=n_examples, random_state=random_state_examples)
        example_samples_mgsd.append(sampled_examples)
    
    sample_examples_mgsd = pd.concat(example_samples_mgsd).reset_index(drop=True)
    
    if print_balance:
        print("MGSD Test set balance:\n", sample_mgsd["label"].value_counts())
        print("MGSD Few-shot examples balance:\n", sample_examples_mgsd["label"].value_counts())

    return sample_mgsd, sample_examples_mgsd



def load_mentalmanip_dataset(
        input_dataframe: pd.DataFrame,
        sample_size: dict = {1: 250, 0: 250},
        sample_size_examples: dict = {1: 5, 0: 5},
        max_len_examples: int = 1000,
        random_state: int = 42,
        random_state_examples: int = 42,
        print_balance: bool = False,
):
    """
    Loads the MentalManip dataset

    Inputs:
    - input_dataframe: pd.DataFrame
        The input dataframe containing the MentalManip dataset.
    - sample_size: dict
        A dictionary specifying the number of samples to take for each label in the MentalManip dataset. Making it balanced ensures equal representation of each label.
    - sample_size_examples: dict
        A dictionary specifying the number of few-shot examples to take for each label in the MentalManip dataset.
    - max_len_examples: int
        The maximum length of the examples to be included in the few-shot examples.
    - random_state: int
        Random state for reproducibility when sampling the main dataset.
    - random_state_examples: int
        Random state for reproducibility when sampling few-shot examples.
    - print_balance: bool
        If True, prints the balance of the dataset after sampling.
    
    Returns:
    - sample_mentalmanip: pd.DataFrame
        A dataframe containing the dataset samples used for testing.
    - sample_examples_mentalmanip: pd.DataFrame
        A dataframe containing the few-shot examples for the MentalManip dataset.
    """

    balanced_samples_manip = []
    example_samples_manip = []
    used_indices = set()
    
    for label, size in sample_size.items():
        subset = input_dataframe[input_dataframe["manipulative"] == label].sample(n=size, random_state=random_state)
        balanced_samples_manip.append(subset)
        used_indices.update(subset.index)
    
    for label, n_examples in sample_size_examples.items():
        candidates = input_dataframe[
            (input_dataframe["manipulative"] == label) &
            (input_dataframe["dialogue"].str.len() <= max_len_examples) &
            (~input_dataframe.index.isin(used_indices))
        ]
        example_samples_manip.append(candidates.sample(n=n_examples, random_state=0))
    
    sample_mentalmanip = pd.concat(balanced_samples_manip).sample(frac=1, random_state=random_state_examples).reset_index(drop=True)
    sample_examples_mentalmanip = pd.concat(example_samples_manip).reset_index(drop=True)

    if print_balance:
        print("MentalManip Test set balance:\n", sample_mentalmanip["manipulative"].value_counts())
        print("MentalManip Few-shot examples balance:\n", sample_examples_mentalmanip["manipulative"].value_counts())

    return sample_mentalmanip, sample_examples_mentalmanip



def load_mmlu_dataset(
        input_df: pd.DataFrame,
        sample_per_subject: int = 50,
        sample_examples_per_subject: int = 5,
        random_state: int = 42,
        random_state_examples: int = 0,
        print_balance: bool = False,
        save_dfs: bool = False,
        save_path: pathlib.Path = "data/"
    ):
    subject_list = input_df["subject"].unique().tolist()
    test_samples, examples_samples = [], []

    for subject in subject_list:
        subject_df = input_df[input_df["subject"] == subject]
        available = len(subject_df)

        n_ex = min(sample_examples_per_subject, available)
        n_test = min(sample_per_subject, max(available - n_ex, 0))

        if print_balance:
            print(f"{subject}: available={available} → test={n_test}, ex={n_ex}")

        examples = subject_df.sample(
            n=n_ex, random_state=random_state_examples, replace=False
        ) if n_ex > 0 else subject_df.iloc[0:0]

        remaining = subject_df.drop(examples.index)
        tests = remaining.sample(
            n=n_test, random_state=random_state, replace=False
        ) if n_test > 0 else subject_df.iloc[0:0]

        examples_samples.append(examples)
        test_samples.append(tests)

    sample_mmlu = pd.concat(test_samples).reset_index(drop=True) if test_samples else input_df.iloc[0:0].copy()
    sample_examples_mmlu = pd.concat(examples_samples).reset_index(drop=True) if examples_samples else input_df.iloc[0:0].copy()

    sample_mmlu = finalize_mmlu_df(sample_mmlu)
    sample_examples_mmlu = finalize_mmlu_df(sample_examples_mmlu)

    if save_dfs:
        sample_mmlu["choices"] = sample_mmlu["choices"].apply(json.dumps)
        sample_examples_mmlu["choices"] = sample_examples_mmlu["choices"].apply(json.dumps)

        sample_mmlu.to_csv(save_path / "mmlu_test.csv", index=False)
        sample_examples_mmlu.to_csv(save_path / "mmlu_examples.csv", index=False)

    return sample_mmlu, sample_examples_mmlu


def build_failure_augmented_sample(
    full_df: pd.DataFrame,
    failure_ids: List[int],
    total_size: int = 500,
    n_failures_to_include: int = 20,
    label_col: Optional[str] = None,
    label_subset: Optional[List[str]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Build a dataset of `total_size` by injecting `n_failures_to_include` known failures,
    then randomly sampling the rest (optionally filtered by labels).
    """
    failure_ids_set = set(failure_ids)
    df_failures_all = full_df[full_df.index.isin(failure_ids_set)]

    if len(df_failures_all) < n_failures_to_include:
        raise ValueError(f"Requested {n_failures_to_include} failures, but only found {len(df_failures_all)}.")

    df_failures = df_failures_all.sample(n=n_failures_to_include, random_state=random_state)

    df_remaining = full_df[~full_df.index.isin(df_failures.index)]

    if label_subset and label_col:
        df_remaining = df_remaining[df_remaining[label_col].isin(label_subset)]

    n_remaining = total_size - len(df_failures)
    if n_remaining < 0:
        raise ValueError(f"Total size {total_size} is smaller than number of failures {len(df_failures)}")

    df_sampled = df_remaining.sample(n=n_remaining, random_state=random_state)

    df_final = pd.concat([df_failures, df_sampled], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_final


def get_additional_fields(row: pd.Series, case_name: str) -> dict:
    """Return extra metadata fields for a result row."""
    key = case_name.lower().strip()
    if key == "stereotype":
        return {
            "stereotype_type": row.get("stereotype_type") if "stereotype_type" in row else None,
            "original_dataset": row.get("original_dataset") if "original_dataset" in row else None,
        }
    elif key == "manipulation":
        return {
            "technique": row.get("technique") if "technique" in row else None,
            "vulnerability": row.get("vulnerability") if "vulnerability" in row else None,
        }
    elif key == "mmlu":
        return {
            "subject": row.get("subject"),
            "category": row.get("category"),
            "question": row.get("question"),
            "choices": row.get("choices")
        }
    else:
        return {}