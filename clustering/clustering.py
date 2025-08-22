from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Literal
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from cases.cases_config import CaseConfig
from collections import Counter, defaultdict
from profiles.schema import PersonSet
from profiles.profile_sets import PERSON_ETHNICS
from analysis_2 import (
    ensemble_by_trait_analysis,
    build_trait_groups,
    majority_vote_ensemble
)


def add_text_clusters(merged_df: pd.DataFrame,
                     case: CaseConfig,
                     sample_df: pd.DataFrame,
                     person_set: PersonSet,
                     min_k=3,
                     max_k=10,
                     complexity_penalty="bic", 
                     model: Literal["all-MiniLM-L6-v2"] = "all-MiniLM-L6-v2"):
    
    print(f"\n{'='*80}")
    print(f"CLUSTERING ANALYSIS: k={min_k} to k={max_k}")
    print(f"{'='*80}")
    
    if "sample_id" not in sample_df.columns:
        sample_df = sample_df.reset_index().rename(columns={"index": "sample_id"})
    
    if case.input_col not in merged_df.columns:
        merged_df = merged_df.merge(
            sample_df[["sample_id", case.input_col]],
            on="sample_id", how="left"
        )
    
    print(f"\nGenerating embeddings for {len(merged_df)} samples...")
    embedding_model = SentenceTransformer(model)
    texts = merged_df[case.input_col].astype(str).tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    print(f"Embedding shape: {embeddings.shape}")
    
    best_k_results = []
    detailed_results = {}
    
    print(f"\n{'='*80}")
    print("EVALUATING K VALUES")
    print(f"{'='*80}")
    
    for k in range(min_k, max_k+1):
        print(f"\n{'='*60}")
        print(f"K = {k}")
        print(f"{'='*60}")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        merged_df[f"synthetic_cluster_{k}"] = cluster_labels
        
        sil_score = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette Score: {sil_score:.4f}")
        
        routing_perf = evaluate_cluster_routing(
            merged_df, cluster_col=f"synthetic_cluster_{k}"
        )
        
        ensemble_perf = evaluate_cluster_with_tier2_ensembles(
            merged_df,
            person_set=person_set,
            case=case,
            cluster_col=f"synthetic_cluster_{k}",
            k=k
        )
        
        routing_perf["ensemble_metrics"] = ensemble_perf
        routing_perf["silhouette_score"] = sil_score
        detailed_results[k] = {
            "routing": routing_perf,
            "ensemble": ensemble_perf,
            "silhouette": sil_score
        }
        
        best_k_results.append((k, routing_perf))
    
    best_k, best_perf = select_best_k(best_k_results, complexity_penalty=complexity_penalty)
    
    print(f"\n{'='*80}")
    print(f"FINAL SELECTION: k={best_k}")
    print(f"Expected Accuracy: {best_perf['expected_accuracy']:.4f}")
    print(f"Best Cluster Ensemble Accuracy: {best_perf['ensemble_metrics']['best_cluster_ensemble_acc']:.4f}")
    print(f"Average Rescue Rate: {best_perf['ensemble_metrics']['avg_rescue_rate']:.4f}")
    print(f"{'='*80}")
    
    keep_col = f"synthetic_cluster_{best_k}"
    drop_cols = [c for c in merged_df.columns if c.startswith("synthetic_cluster_") and c!=keep_col]
    merged_df = merged_df.drop(columns=drop_cols)
    
    return merged_df, best_perf, detailed_results


def evaluate_cluster_routing(df, cluster_col):
    
    print(f"\nCluster Routing Analysis:")
    print(f"{'Cluster':<10} {'Size':<8} {'Best Profile':<25} {'Best Acc':<10}")
    print("-" * 55)
    
    cluster_perfs = {}
    profile_cols = [col for col in df.columns if col.startswith("profile")]
    
    for c, subdf in df.groupby(cluster_col):
        prof_acc = {p: (subdf[p] == subdf["true_label"]).mean() for p in profile_cols}
        best_prof, best_acc = max(prof_acc.items(), key=lambda x: x[1])
        cluster_perfs[c] = {
            "best_prof": best_prof, 
            "best_acc": best_acc, 
            "size": len(subdf)
        }
        print(f"{c:<10} {len(subdf):<8} {best_prof[:25]:<25} {best_acc:<10.4f}")
    
    weighted_acc = sum(
        len(subdf)/len(df)*cluster_perfs[c]["best_acc"]
        for c, subdf in df.groupby(cluster_col)
    )
    
    print(f"\nWeighted Expected Accuracy: {weighted_acc:.4f}")
    
    return {"cluster_perfs": cluster_perfs, "expected_accuracy": weighted_acc}


def evaluate_cluster_with_tier2_ensembles(merged_df: pd.DataFrame,
                                          person_set: PersonSet,
                                          case: CaseConfig,
                                          cluster_col: str,
                                          k: int) -> Dict[str, Any]:
    
    print(f"\nTier-2 Ensemble Analysis for {k} clusters:")
    
    cluster_ensemble_metrics = {}
    all_cluster_metrics = []
    
    for cluster_id, cluster_df in merged_df.groupby(cluster_col):
        n_samples = len(cluster_df)
        print(f"\n  Analyzing Cluster {cluster_id} (n={n_samples}):")
        
        try:
            ensemble_results = ensemble_by_trait_analysis(
                cluster_df,
                person_set,
                case=case,
                group_keys=("gender", "ethnicity", "age")
            )
            
            baseline_acc = ensemble_results.get('baseline_accuracy', 0)
            
            if ensemble_results.get('ensemble_results'):
                best_ensemble_info = max(
                    ensemble_results['ensemble_results'].items(),
                    key=lambda x: x[1]['accuracy']
                )
                best_ensemble_name = best_ensemble_info[0]
                best_ensemble_metrics = best_ensemble_info[1]
                best_ensemble_acc = best_ensemble_metrics['accuracy']
                best_rescue_rate = best_ensemble_metrics['rescue_rate']
                best_extra_error_rate = best_ensemble_metrics['extra_error_rate']
                
                all_ensembles = ensemble_results['ensemble_results'].values()
                avg_ensemble_acc = np.mean([e['accuracy'] for e in all_ensembles])
                avg_rescue_rate = np.mean([e['rescue_rate'] for e in all_ensembles])
                avg_extra_error_rate = np.mean([e['extra_error_rate'] for e in all_ensembles])
            else:
                best_ensemble_name = "none"
                best_ensemble_acc = baseline_acc
                best_rescue_rate = 0
                best_extra_error_rate = 0
                avg_ensemble_acc = baseline_acc
                avg_rescue_rate = 0
                avg_extra_error_rate = 0
            
            cluster_ensemble_metrics[cluster_id] = {
                "size": n_samples,
                "baseline_acc": baseline_acc,
                "best_ensemble": best_ensemble_name,
                "best_ensemble_acc": best_ensemble_acc,
                "best_rescue_rate": best_rescue_rate,
                "best_extra_error_rate": best_extra_error_rate,
                "avg_ensemble_acc": avg_ensemble_acc,
                "avg_rescue_rate": avg_rescue_rate,
                "avg_extra_error_rate": avg_extra_error_rate,
                "improvement": best_ensemble_acc - baseline_acc,
                "n_ensembles_tested": len(ensemble_results.get('ensemble_results', {}))
            }
            
            print(f"    Baseline: {baseline_acc:.4f}")
            print(f"    Best Ensemble ({best_ensemble_name}): {best_ensemble_acc:.4f}")
            print(f"    Improvement: {best_ensemble_acc - baseline_acc:.4f}")
            print(f"    Rescue Rate: {best_rescue_rate:.4f}")
            print(f"    Extra Error Rate: {best_extra_error_rate:.4f}")
            print(f"    Ensembles Tested: {len(ensemble_results.get('ensemble_results', {}))}")
            
        except Exception as e:
            print(f"    ERROR in ensemble analysis: {e}")
            baseline_acc = accuracy_score(cluster_df['true_label'], cluster_df['base_pred'])
            cluster_ensemble_metrics[cluster_id] = {
                "size": n_samples,
                "baseline_acc": baseline_acc,
                "best_ensemble": "error",
                "best_ensemble_acc": baseline_acc,
                "best_rescue_rate": 0,
                "best_extra_error_rate": 0,
                "avg_ensemble_acc": baseline_acc,
                "avg_rescue_rate": 0,
                "avg_extra_error_rate": 0,
                "improvement": 0,
                "n_ensembles_tested": 0,
                "error": str(e)
            }
        
        all_cluster_metrics.append(cluster_ensemble_metrics[cluster_id])
    
    if all_cluster_metrics:
        valid_clusters = [m for m in all_cluster_metrics if "error" not in m]
        if valid_clusters:
            overall_metrics = {
                "k": k,
                "best_cluster_ensemble_acc": max(m["best_ensemble_acc"] for m in valid_clusters),
                "avg_ensemble_acc": np.mean([m["avg_ensemble_acc"] for m in valid_clusters]),
                "avg_baseline_acc": np.mean([m["baseline_acc"] for m in valid_clusters]),
                "avg_improvement": np.mean([m["improvement"] for m in valid_clusters]),
                "avg_rescue_rate": np.mean([m["avg_rescue_rate"] for m in valid_clusters]),
                "avg_extra_error_rate": np.mean([m["avg_extra_error_rate"] for m in valid_clusters]),
                "total_ensembles_tested": sum(m["n_ensembles_tested"] for m in valid_clusters),
                "clusters": cluster_ensemble_metrics
            }
        else:
            overall_metrics = {
                "k": k,
                "best_cluster_ensemble_acc": 0,
                "avg_ensemble_acc": 0,
                "avg_baseline_acc": 0,
                "avg_improvement": 0,
                "avg_rescue_rate": 0,
                "avg_extra_error_rate": 0,
                "total_ensembles_tested": 0,
                "clusters": cluster_ensemble_metrics
            }
    else:
        overall_metrics = {
            "k": k,
            "best_cluster_ensemble_acc": 0,
            "avg_ensemble_acc": 0,
            "avg_baseline_acc": 0,
            "avg_improvement": 0,
            "avg_rescue_rate": 0,
            "avg_extra_error_rate": 0,
            "total_ensembles_tested": 0,
            "clusters": {}
        }
    
    print(f"\n  Overall Metrics for k={k}:")
    print(f"    Average Baseline Accuracy: {overall_metrics['avg_baseline_acc']:.4f}")
    print(f"    Average Ensemble Accuracy: {overall_metrics['avg_ensemble_acc']:.4f}")
    print(f"    Average Improvement: {overall_metrics['avg_improvement']:.4f}")
    print(f"    Average Rescue Rate: {overall_metrics['avg_rescue_rate']:.4f}")
    print(f"    Total Ensembles Tested: {overall_metrics['total_ensembles_tested']}")
    
    return overall_metrics


def select_best_k(best_k_results, complexity_penalty="bic") -> Tuple[int, Dict]:
    
    print(f"\n{'='*80}")
    print("SCORING EACH K VALUE")
    print(f"{'='*80}")
    
    first_k, first_perf = best_k_results[0]
    n_samples = sum(cluster['size'] for cluster in first_perf['ensemble_metrics']['clusters'].values())
    print(f"Total samples: {n_samples}")
    print(f"Complexity penalty method: {complexity_penalty}")
    
    print(f"\n{'K':<5} {'Expected':<10} {'Ensemble':<10} {'Rescue':<10} {'Sil':<8} {'Penalty':<10} {'Score':<10}")
    print("-" * 75)
    
    scored_results = []
    
    for k, perf in best_k_results:
        expected_acc = perf["expected_accuracy"]
        ensemble_acc = perf["ensemble_metrics"]["best_cluster_ensemble_acc"]
        rescue_rate = perf["ensemble_metrics"]["avg_rescue_rate"]
        sil_score = perf["silhouette_score"]
        
        if complexity_penalty == "bic":
            base_penalty = (k*np.log(n_samples))/n_samples
            penalty = base_penalty*0.5
            penalty = min(penalty, 0.25)

        elif complexity_penalty == "aic":
            base_penalty = (2 * k)/n_samples
            penalty = base_penalty*0.3
            penalty = min(penalty, 0.20)

        elif complexity_penalty == "min_size":
            min_cluster_size = min(cluster['size'] for cluster in perf['ensemble_metrics']['clusters'].values())
            min_threshold = n_samples*0.05
            if min_cluster_size < min_threshold:
                penalty = 0.1 * (1 - min_cluster_size/min_threshold)
            else:
                penalty = 0.0

        elif complexity_penalty == "sqrt":
            penalty = 0.01*np.sqrt(k)

        elif complexity_penalty == "linear":
            penalty = 0.01*k
        else:
            penalty = 0.0
        
        raw_score = (
            expected_acc*0.35 +
            ensemble_acc*0.35 +
            rescue_rate*0.20 +
            sil_score*0.10
        )
        
        final_score = raw_score-penalty
        
        print(f"{k:<5} {expected_acc:<10.4f} {ensemble_acc:<10.4f} {rescue_rate:<10.4f} "
              f"{sil_score:<8.4f} {penalty:<10.4f} {final_score:<10.4f}")
        
        scored_results.append((k, perf, final_score))
    
    best_k, best_perf, best_score = max(scored_results, key=lambda x: x[2])
    
    print(f"\n{'='*80}")
    print("ELBOW ANALYSIS")
    print(f"{'='*80}")
    
    if len(scored_results) > 2:
        scores = [s[2] for s in scored_results]
        improvements = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        
        print("Score improvements by k:")
        for i, (k, _, _) in enumerate(scored_results[:-1]):
            print(f"  k={k} to k={k+1}: {improvements[i]:+.4f}")
        
        if len(improvements) > 1:
            positive_improvements = [imp for imp in improvements if imp > 0]
            if positive_improvements:
                avg_positive_improvement = np.mean(positive_improvements)
                for i, imp in enumerate(improvements):
                    if i > 0 and imp < avg_positive_improvement * 0.3 and improvements[i-1] > avg_positive_improvement * 0.5:
                        elbow_k = scored_results[i][0]
                        print(f"\nElbow detected at k={elbow_k} (improvement drops to {imp:.4f})")
                        break
    
    return best_k, best_perf
