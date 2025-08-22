import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from collections import Counter
from tqdm import tqdm

from clustering.clustering import add_text_clusters, evaluate_cluster_with_tier2_ensembles, evaluate_cluster_routing
from MoP.mop_strategy2_alt import create_cluster_routing_mop, ClusterSmartRoutingMoP
from profiles.schema import PersonSet
from cases.cases_config import CaseConfig


@dataclass
class ClusteringModel:
    kmeans_model: Any
    k: int
    embedding_model_name: str
    cluster_column: str
    cluster_centroids: np.ndarray
    training_metrics: Dict
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RoutingModel:
    routing_rules: Dict
    fallback_ensemble: Dict
    cluster_performance_map: Dict
    cluster_column: str
    positive_label: str
    negative_label: str
    training_metrics: Dict


class AdaptiveClusteringPipeline:
    def __init__(self, 
                 save_dir: str = "clustering_models",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.clustering_model = None
        self.routing_model = None
        self.training_history = []
        
    def train(self,
              merged_df: pd.DataFrame,
              case: CaseConfig,
              sample_df: pd.DataFrame,
              person_set: PersonSet,
              min_k: int = 3,
              max_k: int = 12,
              complexity_penalty: str = "bic",
              version_name: str = "v1") -> Dict:
        print(f"\n{'='*80}")
        print(f"TRAINING ADAPTIVE CLUSTERING MODEL - {version_name}")
        print(f"{'='*80}")
        if "sample_id" not in sample_df.columns:
            sample_df = sample_df.reset_index().rename(columns={"index": "sample_id"})
        if case.input_col not in merged_df.columns:
            merged_df = merged_df.merge(
                sample_df[["sample_id", case.input_col]],
                on="sample_id", how="left"
            )
        print(f"\nGenerating embeddings for {len(merged_df)} samples...")
        texts = merged_df[case.input_col].astype(str).tolist()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print("\nFinding optimal clustering...")
        best_k, best_kmeans, best_metrics = self._find_optimal_clustering(
            embeddings, merged_df, person_set, case,
            min_k, max_k, complexity_penalty
        )
        cluster_column = f"synthetic_cluster_{best_k}"
        merged_df[cluster_column] = best_kmeans.labels_
        print("\nConfiguring routing model...")
        routing_mop = create_cluster_routing_mop(person_set, case=case)
        routing_mop.fit(merged_df, cluster_column=cluster_column)
        predictions = routing_mop.predict(merged_df)
        training_metrics = routing_mop.evaluate(
            predictions, 
            merged_df['true_label'].values,
            merged_df['base_pred'].values if 'base_pred' in merged_df.columns else None
        )
        print(f"\nTraining Performance:")
        print(f"  Accuracy: {training_metrics['accuracy']:.3f}")
        if 'baseline_accuracy' in training_metrics:
            print(f"  Baseline: {training_metrics['baseline_accuracy']:.3f}")
            print(f"  Improvement: {training_metrics['accuracy_improvement']:+.3f}")
        self.clustering_model = ClusteringModel(
            kmeans_model=best_kmeans,
            k=best_k,
            embedding_model_name=self.embedding_model_name,
            cluster_column=cluster_column,
            cluster_centroids=best_kmeans.cluster_centers_,
            training_metrics=best_metrics
        )
        self.routing_model = RoutingModel(
            routing_rules=routing_mop.routing_rules_,
            fallback_ensemble=routing_mop.fallback_ensemble_,
            cluster_performance_map=routing_mop.cluster_performance_map_,
            cluster_column=cluster_column,
            positive_label=routing_mop.positive_label_,
            negative_label=routing_mop.negative_label_,
            training_metrics=training_metrics
        )
        self._save_models(version_name)
        self.training_history.append({
            'version': version_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(merged_df),
            'k': best_k,
            'accuracy': training_metrics['accuracy'],
            'metrics': training_metrics
        })
        return {
            'clustering_model': self.clustering_model,
            'routing_model': self.routing_model,
            'training_metrics': training_metrics,
            'merged_df': merged_df
        }
    
    def test(self,
             test_df: pd.DataFrame,
             sample_df: pd.DataFrame,
             case: CaseConfig,
             person_set: PersonSet,
             version_name: Optional[str] = None) -> Dict:
        print(f"\n{'='*80}")
        print(f"TESTING ON NEW DATA")
        print(f"{'='*80}")
        if version_name:
            self._load_models(version_name)
        if not self.clustering_model or not self.routing_model:
            raise ValueError("No trained models available. Run train() first.")
        if "sample_id" not in sample_df.columns:
            sample_df = sample_df.reset_index().rename(columns={"index": "sample_id"})
        if case.input_col not in test_df.columns:
            test_df = test_df.merge(
                sample_df[["sample_id", case.input_col]],
                on="sample_id", how="left"
            )
        print(f"\nEmbedding {len(test_df)} test samples...")
        texts = test_df[case.input_col].astype(str).tolist()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print("Assigning to clusters...")
        cluster_assignments = self.clustering_model.kmeans_model.predict(embeddings)
        test_df[self.clustering_model.cluster_column] = cluster_assignments
        cluster_dist = pd.Series(cluster_assignments).value_counts().sort_index()
        print(f"\nCluster distribution in test set:")
        for cluster_id, count in cluster_dist.items():
            print(f"  Cluster {cluster_id}: {count} samples ({100*count/len(test_df):.1f}%)")
        distances = []
        for i, embedding in enumerate(embeddings):
            cluster = cluster_assignments[i]
            centroid = self.clustering_model.cluster_centroids[cluster]
            dist = np.linalg.norm(embedding - centroid)
            distances.append(dist)
        avg_distance = np.mean(distances)
        print(f"\nAverage distance to centroids: {avg_distance:.3f}")
        print("\nApplying routing rules...")
        routing_mop = create_cluster_routing_mop(person_set, case=case)
        routing_mop.routing_rules_ = self.routing_model.routing_rules
        routing_mop.fallback_ensemble_ = self.routing_model.fallback_ensemble
        routing_mop.cluster_performance_map_ = self.routing_model.cluster_performance_map
        routing_mop.cluster_column_ = self.routing_model.cluster_column
        routing_mop.positive_label_ = self.routing_model.positive_label
        routing_mop.negative_label_ = self.routing_model.negative_label
        routing_mop.is_fitted_ = True
        predictions = routing_mop.predict(test_df)
        test_metrics = routing_mop.evaluate(
            predictions,
            test_df['true_label'].values,
            test_df['base_pred'].values if 'base_pred' in test_df.columns else None
        )
        print(f"\nTest Performance:")
        print(f"  Accuracy: {test_metrics['accuracy']:.3f}")
        if 'baseline_accuracy' in test_metrics:
            print(f"  Baseline: {test_metrics['baseline_accuracy']:.3f}")
            print(f"  Improvement: {test_metrics['accuracy_improvement']:+.3f}")
        if self.routing_model.training_metrics:
            train_acc = self.routing_model.training_metrics['accuracy']
            test_acc = test_metrics['accuracy']
            degradation = train_acc - test_acc
            print(f"\nGeneralization:")
            print(f"  Training accuracy: {train_acc:.3f}")
            print(f"  Test accuracy: {test_acc:.3f}")
            print(f"  Degradation: {degradation:.3f}")
        return {
            'test_metrics': test_metrics,
            'cluster_distribution': cluster_dist.to_dict(),
            'avg_distance_to_centroids': avg_distance,
            'test_df': test_df,
            'predictions': predictions
        }
    
    def adaptive_retrain(self,
                        new_data: pd.DataFrame,
                        sample_df: pd.DataFrame,
                        case: CaseConfig,
                        person_set: PersonSet,
                        retrain_threshold: int = 500,
                        version_prefix: str = "adaptive") -> List[Dict]:
        results = []
        accumulated_data = pd.DataFrame()
        n_samples = len(new_data)
        n_batches = (n_samples + retrain_threshold - 1) // retrain_threshold
        print(f"\n{'='*80}")
        print(f"ADAPTIVE RETRAINING PIPELINE")
        print(f"Processing {n_samples} samples in {n_batches} batches")
        print(f"{'='*80}")
        for batch_idx in range(n_batches):
            start_idx = batch_idx * retrain_threshold
            end_idx = min((batch_idx + 1) * retrain_threshold, n_samples)
            batch_data = new_data.iloc[start_idx:end_idx]
            print(f"\n{'='*60}")
            print(f"Batch {batch_idx + 1}/{n_batches}: samples {start_idx}-{end_idx}")
            print(f"{'='*60}")
            if self.clustering_model is not None:
                print(f"\nTesting batch with current model...")
                test_result = self.test(
                    test_df=batch_data,
                    sample_df=sample_df,
                    case=case,
                    person_set=person_set
                )
                batch_data = test_result['test_df']
            accumulated_data = pd.concat([accumulated_data, batch_data], ignore_index=True)
            version_name = f"{version_prefix}_v{batch_idx + 1}"
            print(f"\nRetraining on {len(accumulated_data)} accumulated samples...")
            train_result = self.train(
                merged_df=accumulated_data,
                case=case,
                sample_df=sample_df,
                person_set=person_set,
                version_name=version_name
            )
            results.append({
                'batch_idx': batch_idx,
                'n_samples_total': len(accumulated_data),
                'train_metrics': train_result['training_metrics'],
                'test_metrics': test_result['test_metrics'] if batch_idx > 0 else None,
                'version': version_name
            })
        self._generate_evolution_report(results)
        return results

    def assign_clusters_to_df(
        self,
        df: pd.DataFrame,
        case: CaseConfig,
        sample_df: Optional[pd.DataFrame] = None,
        sample_id_col: str = "sample_id",
    ) -> pd.DataFrame:
        if not self.clustering_model:
            raise ValueError("No trained clustering model available.")

        if case.input_col not in df.columns:
            if sample_df is None:
                raise ValueError(
                    f"Missing input column '{case.input_col}' in df and no sample_df provided to merge it."
                )
            if sample_id_col not in df.columns:
                raise ValueError(
                    f"df must contain '{sample_id_col}' to merge text from sample_df."
                )
            if sample_id_col not in sample_df.columns:
                sample_df = sample_df.reset_index().rename(columns={"index": sample_id_col})
            if case.input_col not in sample_df.columns:
                raise ValueError(
                    f"sample_df does not contain required text column '{case.input_col}'."
                )
            df = df.merge(
                sample_df[[sample_id_col, case.input_col]],
                on=sample_id_col,
                how="left"
            )
            if case.input_col not in df.columns:
                raise ValueError(
                    f"Failed to merge '{case.input_col}' from sample_df into df."
                )

        texts = df[case.input_col].astype(str).tolist()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        cluster_assignments = self.clustering_model.kmeans_model.predict(embeddings)
        df = df.copy()
        df[self.clustering_model.cluster_column] = cluster_assignments
        return df
    
    def predict_offline_from_profiles(
        self,
        df: pd.DataFrame,
        case: CaseConfig,
        person_set: PersonSet,
        sample_df: Optional[pd.DataFrame] = None,
        sample_id_col: str = "sample_id",
    ) -> Dict[str, Any]:
        if not self.routing_model or not self.clustering_model:
            raise ValueError("No trained routing/clustering models available.")
        if self.routing_model.cluster_column not in df.columns:
            df = self.assign_clusters_to_df(
                df=df,
                case=case,
                sample_df=sample_df,
                sample_id_col=sample_id_col,
            )
        routing_mop = create_cluster_routing_mop(person_set, case=case)
        routing_mop.routing_rules_ = self.routing_model.routing_rules
        routing_mop.fallback_ensemble_ = self.routing_model.fallback_ensemble
        routing_mop.cluster_performance_map_ = self.routing_model.cluster_performance_map
        routing_mop.cluster_column_ = self.routing_model.cluster_column
        routing_mop.positive_label_ = self.routing_model.positive_label
        routing_mop.negative_label_ = self.routing_model.negative_label
        routing_mop.is_fitted_ = True
        preds = routing_mop.predict(df)
        metrics = routing_mop.evaluate(
            predictions=preds,
            true_labels=df['true_label'].values,
            baseline_preds=df['base_pred'].values if 'base_pred' in df.columns else None
        )
        return {"predictions": preds, "metrics": metrics, "df": df}
    
    def route_texts_with_llm(self,
                             texts_df: pd.DataFrame,
                             case: CaseConfig,
                             person_set: PersonSet,
                             gen_fn,
                             label_map: Optional[Dict[str, str]] = None,
                             batch_size: int = 32,
                             progress_fn: Optional[callable] = None) -> Dict[str, Any]:
        if not self.routing_model or not self.clustering_model:
            raise ValueError("No trained routing/clustering models available.")
        if case.input_col not in texts_df.columns:
            raise ValueError(f"Missing input column '{case.input_col}' in texts_df.")
        df = self.assign_clusters_to_df(texts_df.copy(), case)
        cluster_col = self.routing_model.cluster_column
        preds = []
        n = len(df)
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=n, desc="Routing texts"), 1):
            cl = row[cluster_col]
            if cl in self.routing_model.routing_rules:
                rule = self.routing_model.routing_rules[cl]
                profiles = rule.expert_profiles if rule.ensemble_type != "single" else [rule.expert_profiles[0]]
            else:
                profiles = self.routing_model.fallback_ensemble['profiles']
            votes = []
            for p in profiles:
                y = gen_fn(row[case.input_col], p)
                if label_map and y in label_map:
                    y = label_map[y]
                votes.append(y)
            if len(votes) == 1:
                pred = votes[0]
            else:
                cnt = Counter(votes)
                pred = cnt.most_common(1)[0][0]
            preds.append(pred)

            if progress_fn:  
                progress_fn(i, n)
        df['routed_pred'] = preds
        return {"predictions": np.array(preds), "df": df}
    
    def _find_optimal_clustering(self,
                                embeddings: np.ndarray,
                                merged_df: pd.DataFrame,
                                person_set: PersonSet,
                                case: CaseConfig,
                                min_k: int,
                                max_k: int,
                                complexity_penalty: str) -> Tuple[int, Any, Dict]:
        from sklearn.metrics import silhouette_score
        best_score = -np.inf
        best_k = min_k
        best_kmeans = None
        best_metrics = {}
        n_samples = len(embeddings)
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            sil_score = silhouette_score(embeddings, labels)
            cluster_col = f"synthetic_cluster_{k}"
            merged_df[cluster_col] = labels
            ensemble_metrics = evaluate_cluster_with_tier2_ensembles(
                merged_df, person_set, case, cluster_col, k
            )
            routing_perf = evaluate_cluster_routing(merged_df, cluster_col)
            expected_acc = routing_perf.get("expected_accuracy", 0)
            penalty = 0
            if complexity_penalty == "bic":
                penalty = (k * np.log(n_samples)) / n_samples * 0.5
            elif complexity_penalty == "linear":
                penalty = 0.01 * k
            score = (
                expected_acc * 0.35 +
                ensemble_metrics.get('best_cluster_ensemble_acc', 0) * 0.35 +
                ensemble_metrics.get('avg_rescue_rate', 0) * 0.20 +
                sil_score * 0.10 -
                penalty
            )
            if score > best_score:
                best_score = score
                best_k = k
                best_kmeans = kmeans
                best_metrics = {
                    "expected_accuracy": expected_acc,
                    **ensemble_metrics,
                    "silhouette_score": sil_score,
                    "penalty": penalty,
                    "composite_score": score
                }
        print(f"Selected k={best_k} with score {best_score:.3f}")
        return best_k, best_kmeans, best_metrics
    
    def _save_models(self, version_name: str):
        model_path = self.save_dir / f"{version_name}_models.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'clustering_model': self.clustering_model,
                'routing_model': self.routing_model,
                'training_history': self.training_history
            }, f)
        print(f"Models saved to {model_path}")
    
    def _load_models(self, version_name: str):
        model_path = self.save_dir / f"{version_name}_models.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No saved models found at {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        self.clustering_model = saved_data['clustering_model']
        self.routing_model = saved_data['routing_model']
        self.training_history = saved_data.get('training_history', [])
        print(f"Models loaded from {model_path}")
    
    def _generate_evolution_report(self, results: List[Dict]):
        print(f"\n{'='*80}")
        print("EVOLUTION REPORT")
        print(f"{'='*80}")
        print(f"\n{'Version':<15} {'Samples':<10} {'Train Acc':<12} {'Test Acc':<12} {'Degradation':<12}")
        print("-" * 60)
        for result in results:
            version = result['version']
            n_samples = result['n_samples_total']
            train_acc = result['train_metrics']['accuracy']
            test_acc = result['test_metrics']['accuracy'] if result['test_metrics'] else 'N/A'
            if isinstance(test_acc, float):
                degradation = train_acc - test_acc
                print(f"{version:<15} {n_samples:<10} {train_acc:<12.3f} {test_acc:<12.3f} {degradation:<12.3f}")
            else:
                print(f"{version:<15} {n_samples:<10} {train_acc:<12.3f} {test_acc:<12} {'N/A':<12}")


def run_train_test_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    case: CaseConfig,
    person_set: PersonSet,
    save_dir: str = "clustering_models"
) -> Dict:
    pipeline = AdaptiveClusteringPipeline(save_dir=save_dir)
    train_results = pipeline.train(
        merged_df=train_df,
        case=case,
        sample_df=sample_df,
        person_set=person_set,
        version_name="model_v1"
    )
    test_results = pipeline.test(
        test_df=test_df,
        sample_df=sample_df,
        case=case,
        person_set=person_set
    )
    return {
        'train': train_results,
        'test': test_results,
        'pipeline': pipeline
    }


def run_adaptive_pipeline(
    initial_train_df: pd.DataFrame,
    new_data_stream: pd.DataFrame,
    sample_df: pd.DataFrame,
    case: CaseConfig,
    person_set: PersonSet,
    retrain_interval: int = 500,
    save_dir: str = "clustering_models"
) -> List[Dict]:
    pipeline = AdaptiveClusteringPipeline(save_dir=save_dir)
    pipeline.train(
        merged_df=initial_train_df,
        case=case,
        sample_df=sample_df,
        person_set=person_set,
        version_name="initial"
    )
    results = pipeline.adaptive_retrain(
        new_data=new_data_stream,
        sample_df=sample_df,
        case=case,
        person_set=person_set,
        retrain_threshold=retrain_interval
    )
    return results
