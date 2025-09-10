# clustering/clustering.py

from typing import Optional, Dict, Any, Tuple, Literal
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score

from cases.cases_config import CaseConfig
from profiles.schema import PersonSet

from analysis_2 import (
    ensemble_by_trait_analysis,
)


def add_text_clusters(
    merged_df: pd.DataFrame,
    case: CaseConfig,
    sample_df: pd.DataFrame,
    person_set: PersonSet,
    min_k: int = 3,
    max_k: int = 10,
    complexity_penalty: Literal["bic", "aic", "min_size", "sqrt", "linear", "none"] = "bic",
    model: Literal["all-MiniLM-L6-v2"] = "all-MiniLM-L6-v2",
    *,
    perf_df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """
    Adds synthetic text clusters (KMeans on sentence embeddings) and evaluates:
      (1) best-profile routing per cluster
      (2) Tier-2 ensembles per cluster via analysis_2.ensemble_by_trait_analysis

    Returns:
      merged_df_with_best_k_cluster, best_perf_for_k, detailed_results_by_k
    """
    print(f"\n{'='*80}\nCLUSTERING ANALYSIS: k={min_k}..{max_k}\n{'='*80}")

    # Ensure sample_id exists in sample_df
    if "sample_id" not in sample_df.columns:
        sample_df = sample_df.reset_index().rename(columns={"index": "sample_id"})

    # Attach input text if missing
    if case.input_col not in merged_df.columns:
        merged_df = merged_df.merge(
            sample_df[["sample_id", case.input_col]],
            on="sample_id",
            how="left",
        )

    # Build embeddings
    print(f"\nGenerating embeddings for {len(merged_df)} items using {model} …")
    embedding_model = SentenceTransformer(model)
    texts = merged_df[case.input_col].astype(str).tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings)
    print(f"Embedding shape: {embeddings.shape}")

    best_k_results: list[tuple[int, Dict[str, Any]]] = []
    detailed_results: Dict[int, Dict[str, Any]] = {}

    print(f"\n{'='*80}\nEVALUATING K VALUES\n{'='*80}")
    for k in range(min_k, max_k + 1):
        print(f"\n{'-'*60}\nK = {k}\n{'-'*60}")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        col_k = f"synthetic_cluster_{k}"
        merged_df[col_k] = cluster_labels

        import warnings
        uniq, counts = np.unique(cluster_labels, return_counts=True)
        if len(uniq) >= 2 and (counts > 1).all():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
                sil_score = float(silhouette_score(embeddings, cluster_labels))
        else:
            sil_score = np.nan
        print(f"Silhouette Score: {sil_score:.4f}" if np.isfinite(sil_score) else "Silhouette Score: N/A")

        # (1) Routing: choose best single profile per cluster
        routing_perf = evaluate_cluster_routing(merged_df, cluster_col=col_k)

        # (2) Tier-2 ensembles per cluster (uses analysis_2 internals)
        ensemble_perf = evaluate_cluster_with_tier2_ensembles(
            merged_df,
            person_set=person_set,
            case=case,
            cluster_col=col_k,
            k=k,
            perf_df=perf_df,   # NEW
        )

        routing_perf["ensemble_metrics"] = ensemble_perf
        routing_perf["silhouette_score"] = sil_score
        detailed_results[k] = {
            "routing": routing_perf,
            "ensemble": ensemble_perf,
            "silhouette": sil_score,
        }
        best_k_results.append((k, routing_perf))

    # Pick best k with penalty
    best_k, best_perf = select_best_k(best_k_results, complexity_penalty=complexity_penalty)

    print(f"\n{'='*80}\nFINAL SELECTION: k={best_k}")
    print(f"Expected Accuracy (routing): {best_perf['expected_accuracy']:.4f}")
    em = best_perf.get("ensemble_metrics", {})
    print(f"Best Cluster Ensemble Accuracy: {em.get('best_cluster_ensemble_acc', float('nan')):.4f}")
    print(f"Average Rescue Rate: {em.get('avg_rescue_rate', float('nan')):.4f}")
    print(f"{'='*80}")

    # Keep only the best-k column
    keep_col = f"synthetic_cluster_{best_k}"
    drop_cols = [c for c in merged_df.columns if c.startswith("synthetic_cluster_") and c != keep_col]
    merged_df = merged_df.drop(columns=drop_cols, errors="ignore")

    return merged_df, best_perf, detailed_results


# -----------------------------
# Helpers
# -----------------------------
def evaluate_cluster_routing(df: pd.DataFrame, cluster_col: str) -> Dict[str, Any]:
    """
    For each cluster: find the single best profile (accuracy vs true_label).
    Returns weighted expected accuracy if you routed each cluster to its best profile.
    """
    print(f"\nCluster Routing Analysis ({cluster_col}):")
    print(f"{'Cluster':<10} {'Size':<8} {'Best Profile':<25} {'Best Acc':<10}")
    print("-" * 55)

    cluster_perfs: Dict[int, Dict[str, Any]] = {}
    profile_cols = [col for col in df.columns if col.startswith("profile")]
    if not profile_cols:
        raise ValueError("No profile columns found in merged_df for routing.")

    for c, subdf in df.groupby(cluster_col):
        # mean correctness per profile
        prof_acc = {p: float((subdf[p].astype(str) == subdf["true_label"].astype(str)).mean()) for p in profile_cols}
        best_prof, best_acc = max(prof_acc.items(), key=lambda x: x[1])
        cluster_perfs[int(c)] = {"best_prof": best_prof, "best_acc": float(best_acc), "size": int(len(subdf))}
        print(f"{str(c):<10} {len(subdf):<8} {best_prof[:25]:<25} {best_acc:<10.4f}")

    N = float(len(df))
    weighted_acc = sum(
        (len(subdf) / N) * cluster_perfs[int(c)]["best_acc"] for c, subdf in df.groupby(cluster_col)
    )
    print(f"\nWeighted Expected Accuracy: {weighted_acc:.4f}")

    return {"cluster_perfs": cluster_perfs, "expected_accuracy": float(weighted_acc)}


def evaluate_cluster_with_tier2_ensembles(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    case: CaseConfig,
    cluster_col: str,
    k: int,
    *,
    perf_df: Optional[pd.DataFrame] = None,  # NEW
) -> Dict[str, Any]:
    """
    For each cluster, run analysis_2.ensemble_by_trait_analysis on that subset
    (this computes accuracy/rescue/extra_error and can ingest token/cost via perf_df).
    Aggregates cluster-level summaries plus global averages.
    """
    print(f"\nTier-2 Ensemble Analysis for {k} clusters:")

    cluster_ensemble_metrics: Dict[int, Dict[str, Any]] = {}
    all_cluster_metrics: list[Dict[str, Any]] = []

    for cluster_id, cluster_df in merged_df.groupby(cluster_col):
        n_samples = len(cluster_df)
        print(f"\n  Analyzing Cluster {cluster_id} (n={n_samples}):")

        try:
            ens_results = ensemble_by_trait_analysis(
                cluster_df,
                person_set,
                case=case,
                group_keys=("gender", "ethnicity", "age"),
                perf_df=perf_df,
                print_indications=False,
            )

            baseline_acc = float(ens_results.get("baseline_accuracy", 0.0))

            ens_dict = ens_results.get("ensemble_results") or {}
            if ens_dict:
                # best by accuracy
                best_name, best_metrics = max(
                    ens_dict.items(), key=lambda kv: float(kv[1].get("accuracy", -np.inf))
                )
                best_acc = float(best_metrics.get("accuracy", baseline_acc))
                best_rr = float(best_metrics.get("rescue_rate", 0.0))
                best_er = float(best_metrics.get("extra_error_rate", 0.0))

                vals_acc = [e.get("accuracy") for e in ens_dict.values() if e.get("accuracy") is not None]
                avg_acc = float(np.mean(vals_acc)) if vals_acc else baseline_acc

                vals_rr = [float(e.get("rescue_rate", np.nan)) for e in ens_dict.values()]
                avg_rr = float(np.mean(vals_rr)) if vals_rr else 0.0

                vals_er = [float(e.get("extra_error_rate", np.nan)) for e in ens_dict.values()]
                avg_er = float(np.mean(vals_er)) if vals_er else 0.0

                n_ens = int(len(ens_dict))
            else:
                best_name, best_acc, best_rr, best_er = "none", baseline_acc, 0.0, 0.0
                avg_acc, avg_rr, avg_er, n_ens = baseline_acc, 0.0, 0.0, 0

            rec = {
                "size": int(n_samples),
                "baseline_acc": baseline_acc,
                "best_ensemble": best_name,
                "best_ensemble_acc": best_acc,
                "best_rescue_rate": best_rr,
                "best_extra_error_rate": best_er,
                "avg_ensemble_acc": avg_acc,
                "avg_rescue_rate": avg_rr,
                "avg_extra_error_rate": avg_er,
                "improvement": float(best_acc - baseline_acc),
                "n_ensembles_tested": n_ens,
            }
            cluster_ensemble_metrics[int(cluster_id)] = rec
            print(f"    Baseline: {baseline_acc:.4f}")
            print(f"    Best Ensemble ({best_name}): {best_acc:.4f} | Δ={best_acc - baseline_acc:+.4f}")
            print(f"    Rescue: {best_rr:.4f} | Extra: {best_er:.4f} | Ensembles: {n_ens}")

        except Exception as e:
            # Fallback: single baseline metric
            print(f"    ERROR in ensemble analysis: {e}")
            baseline_acc = float(accuracy_score(cluster_df["true_label"], cluster_df["base_pred"]))
            cluster_ensemble_metrics[int(cluster_id)] = {
                "size": int(n_samples),
                "baseline_acc": baseline_acc,
                "best_ensemble": "error",
                "best_ensemble_acc": baseline_acc,
                "best_rescue_rate": 0.0,
                "best_extra_error_rate": 0.0,
                "avg_ensemble_acc": baseline_acc,
                "avg_rescue_rate": 0.0,
                "avg_extra_error_rate": 0.0,
                "improvement": 0.0,
                "n_ensembles_tested": 0,
                "error": str(e),
            }

        all_cluster_metrics.append(cluster_ensemble_metrics[int(cluster_id)])

    valid = [m for m in all_cluster_metrics if "error" not in m]
    overall = {
        "k": int(k),
        "best_cluster_ensemble_acc": float(max((m["best_ensemble_acc"] for m in valid), default=0.0)),
        "avg_ensemble_acc": float(np.mean([m["avg_ensemble_acc"] for m in valid])) if valid else 0.0,
        "avg_baseline_acc": float(np.mean([m["baseline_acc"] for m in valid])) if valid else 0.0,
        "avg_improvement": float(np.mean([m["improvement"] for m in valid])) if valid else 0.0,
        "avg_rescue_rate": float(np.mean([m["avg_rescue_rate"] for m in valid])) if valid else 0.0,
        "avg_extra_error_rate": float(np.mean([m["avg_extra_error_rate"] for m in valid])) if valid else 0.0,
        "total_ensembles_tested": int(sum(m["n_ensembles_tested"] for m in valid)) if valid else 0,
        "clusters": cluster_ensemble_metrics,
    }

    print(f"\n  Overall Metrics for k={k}:")
    print(f"    Avg Baseline Acc: {overall['avg_baseline_acc']:.4f}")
    print(f"    Avg Ensemble Acc: {overall['avg_ensemble_acc']:.4f} | Δ={overall['avg_improvement']:+.4f}")
    print(f"    Avg Rescue: {overall['avg_rescue_rate']:.4f} | Avg Extra: {overall['avg_extra_error_rate']:.4f}")
    print(f"    Total Ensembles Tested: {overall['total_ensembles_tested']}")
    return overall


def select_best_k(
    best_k_results: list[tuple[int, Dict[str, Any]]],
    complexity_penalty: Literal["bic", "aic", "min_size", "sqrt", "linear", "none"] = "bic",
) -> Tuple[int, Dict[str, Any]]:
    """
    Score each k by a weighted mix of:
      expected routing acc (35%) + best cluster ensemble acc (35%) + avg rescue (20%) + silhouette (10%)
    minus a complexity penalty.
    """
    print(f"\n{'='*80}\nSCORING EACH K VALUE\n{'='*80}")
    if not best_k_results:
        raise ValueError("No k results to score.")

    first_k, first_perf = best_k_results[0]
    clusters = first_perf.get("ensemble_metrics", {}).get("clusters", {})
    n_samples = int(sum(c["size"] for c in clusters.values())) if clusters else 0
    n_samples = max(n_samples, 1)
    print(f"Total samples: {n_samples}")
    print(f"Complexity penalty: {complexity_penalty}")

    header = f"\n{'K':<5} {'Expected':<10} {'Ensemble':<10} {'Rescue':<10} {'Sil':<8} {'Penalty':<10} {'Score':<10}"
    print(header)
    print("-" * (len(header) - 1))

    scored: list[tuple[int, Dict[str, Any], float]] = []
    for k, perf in best_k_results:
        expected_acc = float(perf.get("expected_accuracy", 0.0))
        em = perf.get("ensemble_metrics", {})
        ensemble_acc = float(em.get("best_cluster_ensemble_acc", 0.0))
        rescue_rate = float(em.get("avg_rescue_rate", 0.0))
        sil_score = perf.get("silhouette_score", np.nan)
        sil = float(sil_score) if np.isfinite(sil_score) else 0.0

        if complexity_penalty == "bic":
            penalty = min(0.25, 0.5 * (k * np.log(n_samples) / n_samples))
        elif complexity_penalty == "aic":
            penalty = min(0.20, 0.3 * (2 * k / n_samples))
        elif complexity_penalty == "min_size":
            min_cluster_size = min((c["size"] for c in em.get("clusters", {}).values()), default=n_samples)
            min_threshold = n_samples * 0.05
            penalty = 0.1 * (1 - min_cluster_size / min_threshold) if min_cluster_size < min_threshold else 0.0
        elif complexity_penalty == "sqrt":
            penalty = 0.01 * np.sqrt(k)
        elif complexity_penalty == "linear":
            penalty = 0.01 * k
        else:
            penalty = 0.0

        raw = 0.35 * expected_acc + 0.35 * ensemble_acc + 0.20 * rescue_rate + 0.10 * sil
        score = float(raw - penalty)

        print(f"{k:<5} {expected_acc:<10.4f} {ensemble_acc:<10.4f} {rescue_rate:<10.4f} "
              f"{sil:<8.4f} {penalty:<10.4f} {score:<10.4f}")
        scored.append((k, perf, score))

    best_k, best_perf, _ = max(scored, key=lambda x: x[2])

    # Elbow hints (optional logging)
    if len(scored) > 2:
        scores = [s[2] for s in scored]
        improvements = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        print("\nScore improvements by k:")
        for i, (k, _, _) in enumerate(scored[:-1]):
            print(f"  k={k}→{k+1}: {improvements[i]:+.4f}")

    return int(best_k), best_perf
