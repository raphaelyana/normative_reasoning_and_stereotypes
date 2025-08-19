from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from cases.cases_config import CaseConfig
from collections import Counter
from profiles.schema import PersonSet
from profiles.profile_sets import PERSON_ETHNICS
from analysis_2 import run_full_tier2_analysis


def add_text_clusters(merged_df: pd.DataFrame, 
                      case: CaseConfig, 
                      sample_df: pd.DataFrame, 
                      person_set: PersonSet,
                      min_k=3, 
                      max_k=10):

    if "sample_id" not in sample_df.columns:
        sample_df = sample_df.reset_index().rename(columns={"index": "sample_id"})

    if case.input_col not in merged_df.columns:
        merged_df = merged_df.merge(
            sample_df[["sample_id", case.input_col]],
            on="sample_id", how="left"
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = merged_df[case.input_col].astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    best_k_results = []
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        merged_df[f"synthetic_cluster_{k}"] = kmeans.fit_predict(embeddings)

        routing_perf = evaluate_cluster_routing(
            merged_df, cluster_col=f"synthetic_cluster_{k}"
        )
        best_k_results.append((k, routing_perf))

    best_k, best_perf = max(best_k_results, key=lambda x: x[1]["expected_accuracy"])
    print(f"Best k={best_k}, Expected Acc={best_perf['expected_accuracy']:.3f}")

    keep_col = f"synthetic_cluster_{best_k}"
    drop_cols = [c for c in merged_df.columns if c.startswith("synthetic_cluster_") and c != keep_col]
    merged_df = merged_df.drop(columns=drop_cols)

    return merged_df, best_perf



def add_text_clusters_with_tier2(merged_df, case, sample_df, person_set, min_k=3, max_k=10):
    if "sample_id" not in sample_df.columns:
        sample_df = sample_df.reset_index().rename(columns={"index": "sample_id"})

    if case.input_col not in merged_df.columns:
        merged_df = merged_df.merge(
            sample_df[["sample_id", case.input_col]],
            on="sample_id", how="left"
        )

    # --- Generate embeddings ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = merged_df[case.input_col].astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    best_k_results = []
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        merged_df[f"synthetic_cluster_{k}"] = kmeans.fit_predict(embeddings)

        cluster_results = run_full_tier2_analysis(
            merged_df.rename(columns={f"synthetic_cluster_{k}": "synthetic_cluster"}),
            person_set=person_set,
            case=case,
            group_keys=("synthetic_cluster",),
            create_visualizations=False
        )

        expected_acc = cluster_results["cluster_analysis"].get("expected_accuracy", 0)
        best_k_results.append((k, expected_acc, cluster_results))

    # --- Pick best k by expected accuracy
    best_k, best_acc, best_cluster_results = max(best_k_results, key=lambda x: x[1])
    print(f"Best k={best_k}, Expected Acc={best_acc:.3f}")

    return merged_df, best_cluster_results



def evaluate_cluster_routing(df, cluster_col):
    cluster_perfs = {}
    for c, subdf in df.groupby(cluster_col):
        profile_cols = [col for col in df.columns if col.startswith("profile")]
        # accuracy of each profile in cluster
        prof_acc = {p: (subdf[p] == subdf["true_label"]).mean() for p in profile_cols}
        best_prof, best_acc = max(prof_acc.items(), key=lambda x: x[1])
        cluster_perfs[c] = {"best_prof": best_prof, "best_acc": best_acc}

    # weighted expected performance
    weighted_acc = sum(
        len(subdf)/len(df) * cluster_perfs[c]["best_acc"]
        for c, subdf in df.groupby(cluster_col)
    )
    return {"cluster_perfs": cluster_perfs, "expected_accuracy": weighted_acc}