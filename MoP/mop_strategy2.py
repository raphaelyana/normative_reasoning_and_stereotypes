import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from MoP.mop_constructor import base_MoP
from cases.cases_config import CaseConfig
from MoP.objectives import Routing


@dataclass
class RoutingConfiguration:
    min_category_samples: int = 10
    min_expert_accuracy: float = 0.72
    max_experts_per_category: int = 3
    fallback_ensemble_size: int = 3
    fallback_equal_weights: bool = True
    category_column: str = 'stereotype_type'
    require_demographic_diversity: bool = True


class SmartRoutingMoP(base_MoP):
    def __init__(self, person_set, group_keys=("gender", "ethnicity", "age"), case=None):
        super().__init__(
            person_set=person_set,
            objectives=RoutingConfiguration(),
            group_keys=group_keys,
            case=case
        )
        self.routing_rules_ = {}
        self.expert_profiles_ = {}
        self.category_performance_map_ = {}
        self.fallback_ensemble_ = None
        self.auto_configured_ = False

    def _get_profile_traits(self, profile_column: str) -> Dict[str, str]:
        try:
            return self.person_set.get_traits(profile_column, self.group_keys)
        except Exception:
            return {key: "Unknown" for key in self.group_keys}

    def _normalize_pid(self, pid: str) -> str:
        return pid.split('_')[0]

    def _find_profiles_for_group(self, group: str, merged_df: pd.DataFrame, valid_profiles=None):
        candidates = []
        for profile in merged_df.columns:
            if not profile.startswith("profile"):
                continue
            norm_pid = self._normalize_pid(profile.replace("profile", ""))
            traits = self.person_set.get_traits(f"profile{norm_pid}")
            if traits and f"{traits['gender']}_{traits['ethnicity']}" == group:
                if not valid_profiles or profile in valid_profiles:
                    candidates.append(profile)
        return candidates

    def _pick_best_profile_for_category(self, category: str, candidates: list, merged_df: pd.DataFrame):
        if not candidates:
            return None
        best_acc, best_profile = -1.0, None
        for profile in candidates:
            if profile in merged_df.columns:
                acc = (merged_df[profile] == merged_df["true_label"]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_profile = profile
        return best_profile

    def _extract_category_performance(self, result_tier2: dict) -> Dict[str, Dict]:
        ensemble_analysis = result_tier2.get('ensemble_analysis', {})
        category_results = ensemble_analysis.get('category_results', {})
        if not category_results:
            return {}
        category_performance = {}
        for category, cat_data in category_results.items():
            if 'ensembles' not in cat_data:
                continue
            best_group, best_accuracy = None, 0.0
            for group_name, group_data in cat_data['ensembles'].items():
                accuracy = group_data.get('accuracy', 0.0)
                if accuracy > best_accuracy:
                    best_accuracy, best_group = accuracy, group_name
            if best_group:
                category_performance[f"stereotype_type:{category}"] = {
                    'best_group': best_group,
                    'best_accuracy': best_accuracy,
                    'baseline_accuracy': cat_data.get('baseline_accuracy', 0.7),
                    'improvement': best_accuracy - cat_data.get('baseline_accuracy', 0.7)
                }
        return category_performance

    def _analyze_category_performance_direct(self, merged_df: pd.DataFrame) -> Dict[str, Dict]:
        if 'stereotype_type' not in merged_df.columns:
            return {}
        categories = merged_df['stereotype_type'].unique()
        profile_cols = [col for col in merged_df.columns if col.startswith('profile')]
        category_performance = {}
        for category in categories:
            cat_data = merged_df[merged_df['stereotype_type'] == category]
            if len(cat_data) < self.objectives.min_category_samples:
                continue
            best_profile, best_accuracy = None, 0.0
            for profile in profile_cols:
                accuracy = (cat_data[profile] == cat_data['true_label']).mean()
                if accuracy > best_accuracy:
                    best_accuracy, best_profile = accuracy, profile
            if best_profile:
                traits = self._get_profile_traits(best_profile)
                best_group = "_".join(traits.get(k, "").lower() for k in self.group_keys)
                baseline_accuracy = (cat_data['base_pred'] == cat_data['true_label']).mean()
                category_performance[f"stereotype_type:{category}"] = {
                    'best_group': best_group,
                    'best_profile': best_profile,
                    'best_accuracy': best_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'improvement': best_accuracy - baseline_accuracy
                }
        return category_performance

    def _find_expert_profile(self, demographics: Dict, category: str, merged_df: pd.DataFrame) -> Optional[Dict]:
        profile_cols = [col for col in merged_df.columns if col.startswith('profile')]
        matching_profiles = []
        for profile in profile_cols:
            traits = self._get_profile_traits(profile)
            if all(demographics.get(k) == traits.get(k) for k in demographics):
                matching_profiles.append(profile)
        if not matching_profiles:
            return None
        cat_data = merged_df[merged_df['stereotype_type'] == category.split(":")[-1]] if 'stereotype_type' in merged_df.columns else merged_df
        best_profile, best_accuracy = None, 0.0
        for profile in matching_profiles:
            accuracy = (cat_data[profile] == cat_data['true_label']).mean()
            if accuracy > best_accuracy:
                best_accuracy, best_profile = accuracy, profile
        return {"col_name": best_profile}

    def _build_routing_rules(self, category_performance: Dict, merged_df: pd.DataFrame) -> Dict[str, Routing]:
        routing_rules = {}
        for category, perf_data in category_performance.items():
            best_group = perf_data['best_group']
            group_parts = best_group.split('_')
            gender_part = next((p for p in group_parts if p in ['man', 'woman', 'nonbinary']), None)
            ethnicity_part = next((p for p in group_parts if p in ['white', 'black', 'asian', 'latine', 'middle_eastern', 'indian']), None)
            demographics = {}
            if gender_part:
                demographics['gender'] = gender_part
            if ethnicity_part:
                demographics['ethnicity'] = ethnicity_part
            expert_profile = self._find_expert_profile(demographics, category, merged_df)
            if expert_profile:
                routing_rules[category] = Routing(
                    category=category,
                    expert_profile=expert_profile['col_name'],
                    expert_accuracy=perf_data['best_accuracy'],
                    expert_demographics=demographics,
                    confidence=perf_data['improvement']
                )
        return routing_rules

    def _select_expert_profiles(self, merged_df: pd.DataFrame):
        valid_profiles = set(self.result_tier3.get("consistency_analysis", {}).get("valid_profiles", [])) if self.result_tier3 else None
        pareto_pool = set(self.result_tier1.get("pareto_profiles", [])) if self.result_tier1 else None
        self.expert_profiles_ = {}
        for category, perf_data in self.category_performance_map_.items():
            best_group = perf_data["best_group"]
            candidates = self._find_profiles_for_group(best_group, merged_df)
            if valid_profiles is not None:
                candidates = [p for p in candidates if p in valid_profiles]
            if pareto_pool is not None:
                candidates = [p for p in candidates if p in pareto_pool]
            if candidates:
                best_profile = self._pick_best_profile_for_category(category, candidates, merged_df)
                self.expert_profiles_[category] = best_profile
            else:
                continue
        if not hasattr(self, "fallback_profiles_") or not self.fallback_profiles_:
            self.fallback_profiles_ = list(pareto_pool or [])

    def _build_fallback_ensemble(self, merged_df: pd.DataFrame):
        profile_cols = [col for col in merged_df.columns if col.startswith('profile')]
        profile_accuracies = {p: (merged_df[p] == merged_df['true_label']).mean() for p in profile_cols}
        sorted_profiles = sorted(profile_accuracies.items(), key=lambda x: x[1], reverse=True)
        top_profiles = [profile for profile, _ in sorted_profiles[:self.objectives.fallback_ensemble_size]]
        self.fallback_ensemble_ = {
            'profiles': top_profiles,
            'weights': [1/len(top_profiles)] * len(top_profiles)
        }

    def fit(self, merged_df: pd.DataFrame, result_tier1=None, result_tier2=None, result_tier3=None, result_preliminary=None, **kwargs):
        self.result_tier1 = result_tier1
        self.result_tier2 = result_tier2
        self.result_tier3 = result_tier3
        self.result_preliminary = result_preliminary
        self._detect_labels(merged_df)
        if result_tier2 is not None:
            self.category_performance_map_ = self._extract_category_performance(result_tier2)
        if not self.category_performance_map_:
            self.category_performance_map_ = self._analyze_category_performance_direct(merged_df)
        self.routing_rules_ = self._build_routing_rules(self.category_performance_map_, merged_df)
        self._select_expert_profiles(merged_df)
        self._build_fallback_ensemble(merged_df)
        self.auto_configured_ = True
        self.is_fitted_ = True
        return self

    def _normalize_category(self, category: str) -> str:
        return category.strip().lower()

    def predict(self, merged_df: pd.DataFrame, category_column: str = 'stereotype_type') -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        predictions, routing_stats = [], {}
        for idx, row in merged_df.iterrows():
            raw_category = row[category_column] if category_column in row else 'unknown'
            category = self._normalize_category(f"{category_column}:{raw_category}")
            if category in self.expert_profiles_:
                expert_profile = self.expert_profiles_[category]
                if expert_profile in row:
                    prediction = row[expert_profile]
                    routing_stats[category] = routing_stats.get(category, 0) + 1
                else:
                    prediction = self._ensemble_predict(row)
                    routing_stats['fallback'] = routing_stats.get('fallback', 0) + 1
            else:
                prediction = self._ensemble_predict(row)
                routing_stats['fallback'] = routing_stats.get('fallback', 0) + 1
            predictions.append(prediction)
        return np.array(predictions)

    def _ensemble_predict(self, row) -> str:
        if not self.fallback_ensemble_:
            return self.negative_label_
        profiles, weights = self.fallback_ensemble_['profiles'], self.fallback_ensemble_['weights']
        score, total_weight = 0.0, 0.0
        for profile, weight in zip(profiles, weights):
            if profile in row:
                pred_binary = 1.0 if row[profile] == self.positive_label_ else 0.0
                score += weight * pred_binary
                total_weight += weight
        if total_weight > 0:
            return self.positive_label_ if (score / total_weight) >= 0.5 else self.negative_label_
        else:
            return self.negative_label_

    def evaluate(self, predictions: np.ndarray, true_labels: np.ndarray, baseline_preds: np.ndarray = None) -> Dict:
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

    def get_routing_summary(self) -> Dict:
        if not self.auto_configured_:
            return {"error": "Not configured"}
        summary = {
            'routing_rules': {},
            'expert_profiles': self.expert_profiles_,
            'fallback_ensemble': self.fallback_ensemble_,
            'category_performance': self.category_performance_map_
        }
        for category, rule in self.routing_rules_.items():
            summary['routing_rules'][category] = {
                'expert_profile': rule.expert_profile,
                'expert_accuracy': rule.expert_accuracy,
                'expert_demographics': rule.expert_demographics,
                'confidence': rule.confidence
            }
        return summary


def create_smart_routing_mop(person_set, group_keys=("gender", "ethnicity", "age"), case=None):
    return SmartRoutingMoP(person_set, group_keys, case)
