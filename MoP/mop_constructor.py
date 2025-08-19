import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from profiles.schema import PersonSet
from cases.cases_config import CaseConfig
from analysis_tools import get_available_traits, get_analysis_group_keys


class base_MoP:
    """
    Minimal base class - just shared constructor and essential utilities.
    
    Acts as a common foundation for MoP strategies without imposing
    any specific implementation approach.
    """

    def __init__(self,
                 person_set: PersonSet,
                 objectives = None,
                 group_keys: tuple = ("gender", "ethnicity", "age"),
                 case: Optional[CaseConfig] = None):
        """Shared constructor for all MoP strategies."""
        self.person_set = person_set
        self.objectives = objectives
        self.group_keys = group_keys
        self.case = case

        self.is_fitted_ = False
        self.positive_label_ = None
        self.negative_label_ = None

    
    def configure_group_keys(self):
        if self.group_keys is None:
            print("Auto-detecting available traits from PersonSet...")
            self.group_keys = get_analysis_group_keys(self.person_set)
        else:
            print(f"Using provided group keys: {self.group_keys}")
            required_keys, optional_keys = get_available_traits(self.person_set)
            available_keys = set(required_keys + optional_keys)
            invalid_keys = [key for key in self.group_keys if key not in available_keys]
            if invalid_keys:
                print(f"WARNING: These group keys are not available in PersonSet: {invalid_keys}")
                print(f"Available keys: {sorted(available_keys)}")
                self.group_keys = tuple(key for key in self.group_keys if key in available_keys)
                print(f"Using filtered group keys: {self.group_keys}")


    def _detect_labels(self, df: pd.DataFrame) -> None:
        """Shared label detection logic."""
        if 'true_label' not in df.columns:
            raise ValueError("DataFrame must contain 'true_label' column")
    
        if self.case and hasattr(self.case, "label_map"):
            self.positive_label_ = self.case.label_map["Yes"]
            self.negative_label_ = self.case.label_map["No"]
        else:
            unique_labels = df['true_label'].dropna().unique()
            if len(unique_labels) != 2:
                raise ValueError(f"Expected exactly 2 unique labels, found: {unique_labels}")
            label_counts = df['true_label'].value_counts()
            self.positive_label_ = label_counts.idxmin()
            self.negative_label_ = label_counts.idxmax()


    def _get_profile_traits(self, pid: str) -> Dict[str, str]:
        try:
            traits = self.person_set.get_traits(pid, self.group_keys) or {}
            return {k: str(v.value if hasattr(v, 'value') else v) for k, v in traits.items()}
        except Exception:
            return {k: "Unknown" for k in self.group_keys}


    def _calculate_basic_metrics(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic performance metrics for all profiles."""
        profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        
        base_correct = (merged_df['base_pred'] == merged_df['true_label'])
        base_acc = float(base_correct.mean())

        profile_data = []

        for profile in profile_cols:
            if profile not in merged_df.columns:
                continue

            accuracy = float((merged_df[profile] == merged_df['true_label']).mean())
            profile_correct = (merged_df[profile] == merged_df['true_label'])

            rescued = int((~base_correct & profile_correct).sum())
            base_errors = int((~base_correct).sum())
            rescue_rate = float(rescued / base_errors) if base_errors > 0 else 0.0

            extra_errors = int((base_correct & (~profile_correct)).sum())
            extra_error_rate = float(extra_errors / len(merged_df))

            pred_counts = merged_df[profile].value_counts(normalize=True)
            volatility = float(-sum(p*np.log(p+1e-10) for p in pred_counts.values))

            disagreements = int((merged_df[profile] != merged_df['base_pred']).sum())
            bias_magnitude = float(disagreements / len(merged_df))

            pid = profile.replace("profile", "")
            traits = self._get_profile_traits(pid)

            row = {
                'profile': profile,
                'pid': pid,
                'accuracy': accuracy,
                'rescue_rate': rescue_rate,
                'extra_error_rate': extra_error_rate,
                'volatility': volatility,
                'bias_magnitude': bias_magnitude,
                'accuracy_delta': accuracy - base_acc,
            }

            row.update(traits)
            profile_data.append(row)

        return pd.DataFrame(profile_data)

    def evaluate(self, predictions: np.ndarray, true_labels: np.ndarray, baseline_preds: np.ndarray = None) -> Dict:
        """Shared evaluation logic."""
        pred_binary = (predictions == self.positive_label_).astype(int)
        true_binary = (true_labels == self.positive_label_).astype(int)
        
        accuracy = float(np.mean(pred_binary == true_binary))
        results = {'accuracy': accuracy, 'n_samples': len(predictions)}

        if baseline_preds is not None:
            baseline_binary = (baseline_preds == self.positive_label_).astype(int)
            baseline_accuracy = float(np.mean(baseline_binary == true_binary))
            baseline_correct = (baseline_binary == true_binary)
            model_correct = (pred_binary == true_binary)

            rescued = int((~baseline_correct & model_correct).sum())
            baseline_errors = int((~baseline_correct).sum())
            extra_errors = int((baseline_correct & ~model_correct).sum())

            results.update({
                'baseline_accuracy': baseline_accuracy,
                'accuracy_improvement': accuracy - baseline_accuracy,
                'rescue_rate': float(rescued / baseline_errors) if baseline_errors > 0 else 0.0,
                'extra_error_rate': float(extra_errors / len(predictions)),
                'rescued_cases': rescued,
                'extra_errors': extra_errors
            })

        return results
    
    def fit(self, merged_df: pd.DataFrame, **kwargs):
        pass

    def predict(self, merged_df: pd.DataFrame, **kwargs) -> np.ndarray:
        pass