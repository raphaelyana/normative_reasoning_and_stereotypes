import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from MoP.mop_constructor import base_MoP
from cases.cases_config import CaseConfig
from profiles.schema import PersonSet


@dataclass
class ClusterRoutingConfiguration:
    min_cluster_samples: int = 10
    min_expert_accuracy: float = 0.72
    max_experts_per_cluster: int = 3
    fallback_ensemble_size: int = 5
    fallback_equal_weights: bool = True
    cluster_column: str = None
    use_ensemble_for_cluster: bool = True


@dataclass
class ClusterRoutingRule:
    cluster_id: int
    expert_profiles: List[str]
    expert_accuracy: float
    baseline_accuracy: float
    improvement: float
    rescue_rate: float
    cluster_size: int
    ensemble_type: str


class ClusterSmartRoutingMoP(base_MoP):
    def __init__(self, person_set, group_keys=("gender", "ethnicity", "age"), case=None):
        super().__init__(
            person_set=person_set,
            objectives=ClusterRoutingConfiguration(),
            group_keys=group_keys,
            case=case
        )
        self.routing_rules_ = {}
        self.cluster_performance_map_ = {}
        self.fallback_ensemble_ = None
        self.cluster_column_ = None
        self.is_fitted_ = False

    def _analyze_cluster_performance(self, merged_df: pd.DataFrame) -> Dict[int, Dict]:
        if self.cluster_column_ not in merged_df.columns:
            raise ValueError(f"Cluster column '{self.cluster_column_}' not found in DataFrame")
        
        profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        cluster_performance = {}
        
        for cluster_id, cluster_df in merged_df.groupby(self.cluster_column_):
            n_samples = len(cluster_df)
            if n_samples < self.objectives.min_cluster_samples:
                continue
            
            baseline_acc = (cluster_df['base_pred'] == cluster_df['true_label']).mean()
            best_single_profile = None
            best_single_acc = 0.0
            profile_accuracies = {}
            
            for profile in profile_cols:
                if profile in cluster_df.columns:
                    acc = (cluster_df[profile] == cluster_df['true_label']).mean()
                    profile_accuracies[profile] = acc
                    if acc > best_single_acc:
                        best_single_acc = acc
                        best_single_profile = profile
            
            sorted_profiles = sorted(profile_accuracies.items(), key=lambda x: x[1], reverse=True)
            top_k_profiles = [p for p, _ in sorted_profiles[:self.objectives.max_experts_per_cluster]]
            ensemble_acc = self._calculate_ensemble_accuracy(cluster_df, top_k_profiles)
            
            if ensemble_acc > best_single_acc:
                best_approach = "ensemble"
                best_profiles = top_k_profiles
                best_acc = ensemble_acc
            else:
                best_approach = "single"
                best_profiles = [best_single_profile]
                best_acc = best_single_acc
            
            rescue_rate = self._calculate_rescue_rate(cluster_df, best_profiles, best_approach == "ensemble")
            
            cluster_performance[cluster_id] = {
                'size': n_samples,
                'baseline_accuracy': baseline_acc,
                'best_single_profile': best_single_profile,
                'best_single_accuracy': best_single_acc,
                'best_ensemble_profiles': top_k_profiles,
                'best_ensemble_accuracy': ensemble_acc,
                'best_approach': best_approach,
                'best_accuracy': best_acc,
                'improvement': best_acc - baseline_acc,
                'rescue_rate': rescue_rate,
                'profile_accuracies': profile_accuracies
            }
        
        return cluster_performance
    
    def _calculate_ensemble_accuracy(self, df: pd.DataFrame, profiles: List[str]) -> float:
        if not profiles:
            return 0.0
        predictions = []
        for idx, row in df.iterrows():
            votes = []
            for profile in profiles:
                if profile in row:
                    votes.append(row[profile])
            if votes:
                from collections import Counter
                vote_counts = Counter(votes)
                prediction = vote_counts.most_common(1)[0][0]
            else:
                prediction = self.negative_label_
            predictions.append(prediction)
        predictions = pd.Series(predictions, index=df.index)
        accuracy = (predictions == df['true_label']).mean()
        return accuracy
    
    def _calculate_rescue_rate(self, df: pd.DataFrame, profiles: List[str], use_ensemble: bool) -> float:
        base_correct = (df['base_pred'] == df['true_label'])
        base_errors = (~base_correct).sum()
        if base_errors == 0:
            return 0.0
        if use_ensemble:
            predictions = []
            for _, row in df.iterrows():
                votes = []
                for profile in profiles:
                    if profile in row:
                        votes.append(row[profile])
                if votes:
                    from collections import Counter
                    vote_counts = Counter(votes)
                    prediction = vote_counts.most_common(1)[0][0]
                else:
                    prediction = self.negative_label_
                predictions.append(prediction)
            predictions = pd.Series(predictions, index=df.index)
            model_correct = (predictions == df['true_label'])
        else:
            profile = profiles[0]
            model_correct = (df[profile] == df['true_label'])
        rescued = (~base_correct & model_correct).sum()
        rescue_rate = rescued / base_errors
        return rescue_rate
    
    def _build_routing_rules(self, cluster_performance: Dict) -> Dict[int, ClusterRoutingRule]:
        routing_rules = {}
        for cluster_id, perf_data in cluster_performance.items():
            if perf_data['best_accuracy'] < self.objectives.min_expert_accuracy:
                continue
            if perf_data['best_approach'] == 'ensemble':
                expert_profiles = perf_data['best_ensemble_profiles']
                ensemble_type = 'top_k'
            else:
                expert_profiles = [perf_data['best_single_profile']]
                ensemble_type = 'single'
            routing_rules[cluster_id] = ClusterRoutingRule(
                cluster_id=cluster_id,
                expert_profiles=expert_profiles,
                expert_accuracy=perf_data['best_accuracy'],
                baseline_accuracy=perf_data['baseline_accuracy'],
                improvement=perf_data['improvement'],
                rescue_rate=perf_data['rescue_rate'],
                cluster_size=perf_data['size'],
                ensemble_type=ensemble_type
            )
        return routing_rules
    
    def _build_fallback_ensemble(self, merged_df: pd.DataFrame):
        profile_cols = [col for col in merged_df.columns if col.startswith('profile')]
        profile_accuracies = {}
        for profile in profile_cols:
            accuracy = (merged_df[profile] == merged_df['true_label']).mean()
            profile_accuracies[profile] = accuracy
        sorted_profiles = sorted(profile_accuracies.items(), key=lambda x: x[1], reverse=True)
        top_profiles = [profile for profile, _ in sorted_profiles[:self.objectives.fallback_ensemble_size]]
        self.fallback_ensemble_ = {
            'profiles': top_profiles,
            'weights': [1/len(top_profiles)] * len(top_profiles) if self.objectives.fallback_equal_weights else None,
            'accuracies': {p: profile_accuracies[p] for p in top_profiles}
        }
    
    def fit(self, merged_df: pd.DataFrame, cluster_column: Optional[str] = None, 
            clustering_results: Optional[Dict] = None, **kwargs):
        self._detect_labels(merged_df)
        if clustering_results and not cluster_column:
            cluster_column = self._auto_select_cluster_column(merged_df, clustering_results)
        if cluster_column:
            self.cluster_column_ = cluster_column
        else:
            cluster_cols = [col for col in merged_df.columns if col.startswith('synthetic_cluster_')]
            if cluster_cols:
                self.cluster_column_ = cluster_cols[0]
            else:
                self.cluster_column_ = self.objectives.cluster_column
        self.cluster_performance_map_ = self._analyze_cluster_performance(merged_df)
        self.routing_rules_ = self._build_routing_rules(self.cluster_performance_map_)
        self._build_fallback_ensemble(merged_df)
        if clustering_results:
            self.clustering_metadata_ = {
                'expected_accuracy': clustering_results.get('expected_accuracy', 0),
                'silhouette_score': clustering_results.get('silhouette_score', 0),
                'ensemble_metrics': clustering_results.get('ensemble_metrics', {})
            }
        self.is_fitted_ = True
        return self
    
    def _auto_select_cluster_column(self, merged_df: pd.DataFrame, clustering_results: Dict) -> str:
        if 'ensemble_metrics' in clustering_results:
            k = clustering_results['ensemble_metrics'].get('k', None)
            if k:
                cluster_col = f"synthetic_cluster_{k}"
                if cluster_col in merged_df.columns:
                    return cluster_col
        cluster_cols = [col for col in merged_df.columns if col.startswith('synthetic_cluster_')]
        if not cluster_cols:
            raise ValueError("No cluster columns found in DataFrame")
        return cluster_cols[0]
    
    def predict(self, merged_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        if self.cluster_column_ not in merged_df.columns:
            raise ValueError(f"Cluster column '{self.cluster_column_}' not found in DataFrame")
        predictions = []
        routing_stats = {'routed': 0, 'fallback': 0, 'by_cluster': {}}
        for idx, row in merged_df.iterrows():
            cluster_id = row[self.cluster_column_]
            if cluster_id in self.routing_rules_:
                rule = self.routing_rules_[cluster_id]
                if rule.ensemble_type == 'single':
                    profile = rule.expert_profiles[0]
                    if profile in row:
                        prediction = row[profile]
                    else:
                        prediction = self._fallback_predict(row)
                else:
                    prediction = self._ensemble_predict(row, rule.expert_profiles)
                routing_stats['routed'] += 1
                routing_stats['by_cluster'][cluster_id] = routing_stats.get('by_cluster', {}).get(cluster_id, 0) + 1
            else:
                prediction = self._fallback_predict(row)
                routing_stats['fallback'] += 1
            predictions.append(prediction)
        return np.array(predictions)
    
    def _ensemble_predict(self, row, profiles: List[str]) -> str:
        votes = []
        for profile in profiles:
            if profile in row:
                votes.append(row[profile])
        if votes:
            from collections import Counter
            vote_counts = Counter(votes)
            return vote_counts.most_common(1)[0][0]
        else:
            return self._fallback_predict(row)
    
    def _fallback_predict(self, row) -> str:
        if not self.fallback_ensemble_:
            return self.negative_label_
        return self._ensemble_predict(row, self.fallback_ensemble_['profiles'])
    
    def get_routing_summary(self) -> Dict:
        if not self.is_fitted_:
            return {"error": "Model not fitted"}
        summary = {
            'cluster_column': self.cluster_column_,
            'n_clusters_routed': len(self.routing_rules_),
            'routing_rules': {},
            'fallback_ensemble': {
                'n_profiles': len(self.fallback_ensemble_['profiles']),
                'avg_accuracy': np.mean(list(self.fallback_ensemble_['accuracies'].values()))
            },
            'overall_stats': {
                'avg_improvement': np.mean([r.improvement for r in self.routing_rules_.values()]),
                'avg_rescue_rate': np.mean([r.rescue_rate for r in self.routing_rules_.values()]),
                'total_samples_covered': sum(r.cluster_size for r in self.routing_rules_.values())
            }
        }
        for cluster_id, rule in self.routing_rules_.items():
            summary['routing_rules'][f'cluster_{cluster_id}'] = {
                'ensemble_type': rule.ensemble_type,
                'n_experts': len(rule.expert_profiles),
                'expert_accuracy': rule.expert_accuracy,
                'baseline_accuracy': rule.baseline_accuracy,
                'improvement': rule.improvement,
                'rescue_rate': rule.rescue_rate,
                'cluster_size': rule.cluster_size
            }
        return summary


def create_cluster_routing_mop(person_set, group_keys=("gender", "ethnicity", "age"), case=None):
    return ClusterSmartRoutingMoP(person_set, group_keys, case)
