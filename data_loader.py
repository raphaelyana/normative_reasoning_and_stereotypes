# Functions to load the datasets for experiments, in a balanced way.

import pandas as pd

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
    