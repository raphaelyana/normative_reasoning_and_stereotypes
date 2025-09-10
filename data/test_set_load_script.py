
from datasets import load_dataset
import pandas as pd
from data_loader import load_mgsd_dataset, load_mentalmanip_dataset
from cases.cases_config import CaseConfig
from cases.stereotypes_case import stereotypes_case
from cases.manipulation_case import manipulation_case

from cases.mmlu_case import mmlu_case
from data_loader import fetch_categories_mmlu, load_mmlu_dataset, finalize_mmlu_df

def mgsd_train_test(save_files=False, save_path="data/"):
    ## Loading the MGSD full dataset

    dataset = load_dataset("wu981526092/MGSD")
 
    data = dataset['train']
    df = data.to_pandas()
 
 
    sample_sizes_mgsd = {
        'stereotype': 250,
        'unrelated': 250,
    }
 
    sample_size_examples_mgsd = {
        'stereotype': 5,
        'unrelated': 5
    }
 
    sample_mgsd, sample_examples_mgsd = load_mgsd_dataset(
        df, 
        sample_sizes_mgsd, 
        sample_size_examples_mgsd,
        random_state=42,
        random_state_examples=0,
        )
 
    # Loading the test set for MGSD
    if sample_mgsd.empty == True:
        sample_mgsd = pd.read_csv("data/mgsd.csv")

    if sample_examples_mgsd.empty == True:
        sample_examples_mgsd = pd.read_csv("data/mgsd_examples.csv")
 
    train_samples=set(sample_mgsd[stereotypes_case.input_col])
    train_examples=set(sample_examples_mgsd[stereotypes_case.input_col])
 
    pool_1 = df[~df[stereotypes_case.input_col].isin(train_samples)].copy()
    pool = pool_1[~pool_1[stereotypes_case.input_col].isin(train_examples)].copy()
    pool = pool.drop_duplicates(subset=[stereotypes_case.input_col])

    new_sample_sizes_mgsd = {"stereotype": 500, "unrelated": 500}
    new_sample_mgsd, _ = load_mgsd_dataset(
        input_dataframe=pool,
        sample_size=new_sample_sizes_mgsd,
        sample_size_examples={"stereotype": 5, "unrelated": 5},
        random_state=123,
        random_state_examples=0,
        print_balance=True,
    )
 
    assert (set(new_sample_mgsd[stereotypes_case.input_col]) & train_samples) == set()
    assert (set(new_sample_mgsd[stereotypes_case.input_col]) & train_examples) == set()


def mentalmanip_train_test(save_files=False, save_path="data/"):


    ## Loading the full MentalManip dataset

    dataset_2 = load_dataset("audreyeleven/MentalManip", "mentalmanip_maj")
    data_2 = dataset_2["train"]
    df_2 = data_2.to_pandas()

    sample_sizes_manip = {1: 250, 0: 250}
    sample_sizes_examples_manip = {1: 5, 0: 5}
    max_len_examples = 1000

    sample_mentalmanip, sample_examples_mentalmanip = load_mentalmanip_dataset(
        df_2, 
        sample_sizes_manip, 
        sample_sizes_examples_manip, 
        max_len_examples,
        random_state=42,
        random_state_examples=0,
    )

    if save_files:
        sample_mentalmanip.to_csv(save_path+"mentalmanip.csv", index=False)
        sample_examples_mentalmanip.to_csv(save_path+"mentalmanip_examples.csv", index=False)

    # Loading the test set for MentalManip
    if sample_mentalmanip.empty == True:
        sample_mentalmanip = pd.read_csv("data/mentalmanip.csv")

    if sample_examples_mentalmanip.empty == True:
        sample_examples_mentalmanip = pd.read_csv("data/mentalmanip_examples.csv")

    train_samples=set(sample_mentalmanip[manipulation_case.input_col])
    train_examples=set(sample_examples_mentalmanip[manipulation_case.input_col])

    pool_1 = df_2[~df_2[manipulation_case.input_col].isin(train_samples)].copy()
    pool = pool_1[~pool_1[manipulation_case.input_col].isin(train_examples)].copy()
    pool = pool.drop_duplicates(subset=[manipulation_case.input_col])

    counts = pool["manipulative"].value_counts().to_dict()
    available = {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}
    required = {0: 1500, 1: 1500}

    n_per_label = min(available[0], available[1], required[0], required[1])

    if n_per_label == 0:
        raise ValueError("After exclusions, one class has 0 samples; cannot build a balanced set.")

    if n_per_label < required[0] or n_per_label < required[1]:
        print(f"[info] Not enough data to sample 1500 per class without overlap. "
              f"Using {n_per_label} per class instead (avail: 0→{available[0]}, 1→{available[1]}).")

    rng = 123
    parts = []
    for lbl in (0, 1):
        sub = pool[pool["manipulative"] == lbl]
        parts.append(sub.sample(n=n_per_label, random_state=rng, replace=False))
    
    new_sample_mentalmanip = (
        pd.concat(parts, axis=0)
          .sample(frac=1.0, random_state=rng)
          .reset_index(drop=True)
    )

    assert (set(new_sample_mentalmanip[manipulation_case.input_col]) & train_samples) == set()
    assert (set(new_sample_mentalmanip[manipulation_case.input_col]) & train_examples) == set()

    if save_files:
        new_sample_mentalmanip.to_csv(save_path+"mentalmanip_test.csv", index=False)




def mmlu_train_test(save_files=False, save_path="data/"):

    normative_category = [
    "professional_law",
    "moral_scenarios",
    ]
    mmlu_full_raw = fetch_categories_mmlu(normative_category)
    mmlu_full_df  = finalize_mmlu_df(mmlu_full_raw)


    sample_mmlu_train, sample_mmlu_examples = load_mmlu_dataset(
        mmlu_full_df,
        sample_per_subject=500,
        sample_examples_per_subject=5,
        random_state=42,
        random_state_examples=0,
        print_balance=True,
        save_dfs=False,
    )

    
    used = set(sample_mmlu_train[mmlu_case.input_col]) | set(sample_mmlu_examples[mmlu_case.input_col])
    pool = (
        mmlu_full_df[~mmlu_full_df[mmlu_case.input_col].isin(used)]
        .drop_duplicates(subset=[mmlu_case.input_col])
        .copy()
    )

    new_mmlu_test, _ = load_mmlu_dataset(
        pool,
        sample_per_subject=1000,
        sample_examples_per_subject=5,
        random_state=123,
        random_state_examples=0,
        print_balance=True,
    )

    if save_files:
        new_mmlu_test.to_csv(save_path + "samples_mmlu_full_test.csv", index=False)

    return sample_mmlu_train, sample_mmlu_examples, new_mmlu_test
