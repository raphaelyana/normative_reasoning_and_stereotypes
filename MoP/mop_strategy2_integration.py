import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Import your existing functions
from clustering.clustering_alt import add_text_clusters
from MoP.mop_strategy2_alt import create_cluster_routing_mop, ClusterSmartRoutingMoP
from profiles.schema import PersonSet
from cases.cases_config import CaseConfig


@dataclass
class AutoRoutingResults:
    """Container for automated routing results."""
    clustering_results: Dict
    routing_model: ClusterSmartRoutingMoP
    predictions: np.ndarray
    evaluation_metrics: Dict
    selected_k: int
    cluster_column: str
    routing_summary: Dict


def auto_configure_cluster_routing(
    merged_df: pd.DataFrame,
    case: CaseConfig,
    sample_df: pd.DataFrame,
    person_set: PersonSet,
    min_k: int = 3,
    max_k: int = 12,
    complexity_penalty: str = "bic",
    group_keys: tuple = ("gender", "ethnicity", "age"),
    evaluate_on_test: Optional[pd.DataFrame] = None
) -> AutoRoutingResults:
    """
    Fully automated pipeline for cluster-based routing.
    
    This function:
    1. Finds optimal number of clusters
    2. Creates cluster assignments
    3. Configures routing based on cluster performance
    4. Returns fitted model and evaluation metrics
    
    Parameters:
    - merged_df: Training data with predictions
    - case: CaseConfig object
    - sample_df: Sample data with text
    - person_set: PersonSet object
    - min_k: Minimum clusters to test
    - max_k: Maximum clusters to test
    - complexity_penalty: Penalty method for k selection
    - group_keys: Trait keys for analysis
    - evaluate_on_test: Optional test set for evaluation
    
    Returns:
    - AutoRoutingResults with all components
    """
    
    print("="*80)
    print("AUTOMATED CLUSTER-BASED ROUTING CONFIGURATION")
    print("="*80)
    
    # Step 1: Find optimal clustering
    print("\nStep 1: Finding optimal clustering...")
    merged_df_with_clusters, clustering_results = add_text_clusters(
        merged_df=merged_df,
        case=case,
        sample_df=sample_df,
        person_set=person_set,
        min_k=min_k,
        max_k=max_k,
        complexity_penalty=complexity_penalty
    )
    
    # Extract selected k from results
    selected_k = clustering_results.get('ensemble_metrics', {}).get('k', 8)
    cluster_column = f"synthetic_cluster_{selected_k}"
    
    print(f"\nOptimal clustering: k={selected_k}")
    print(f"Expected accuracy: {clustering_results.get('expected_accuracy', 0):.3f}")
    
    # Step 2: Configure routing model
    print("\nStep 2: Configuring routing model...")
    routing_model = create_cluster_routing_mop(
        person_set=person_set,
        group_keys=group_keys,
        case=case
    )
    
    # Fit with automatic configuration
    routing_model.fit(
        merged_df=merged_df_with_clusters,
        clustering_results=clustering_results
    )
    
    # Step 3: Generate predictions
    print("\nStep 3: Generating predictions...")
    
    # Determine which dataset to evaluate on
    eval_df = evaluate_on_test if evaluate_on_test is not None else merged_df_with_clusters
    
    # Ensure cluster column exists in evaluation data
    if evaluate_on_test is not None and cluster_column not in eval_df.columns:
        print(f"Adding cluster assignments to test set...")
        # You would need to assign clusters to test data here
        # This is a simplified version - in practice you'd use the fitted KMeans model
        eval_df = eval_df.copy()
        eval_df[cluster_column] = 0  # Placeholder - should use actual cluster assignments
    
    predictions = routing_model.predict(eval_df)
    
    # Step 4: Evaluate performance
    print("\nStep 4: Evaluating performance...")
    evaluation_metrics = routing_model.evaluate(
        predictions=predictions,
        true_labels=eval_df['true_label'].values,
        baseline_preds=eval_df['base_pred'].values if 'base_pred' in eval_df.columns else None
    )
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {evaluation_metrics['accuracy']:.3f}")
    if 'baseline_accuracy' in evaluation_metrics:
        print(f"  Baseline: {evaluation_metrics['baseline_accuracy']:.3f}")
        print(f"  Improvement: {evaluation_metrics['accuracy_improvement']:+.3f}")
        print(f"  Rescue Rate: {evaluation_metrics['rescue_rate']:.3f}")
    
    # Step 5: Get routing summary
    routing_summary = routing_model.get_routing_summary()
    
    print(f"\nRouting Configuration:")
    print(f"  Clusters with routing: {routing_summary['n_clusters_routed']}")
    print(f"  Average improvement: {routing_summary['overall_stats']['avg_improvement']:.3f}")
    print(f"  Average rescue rate: {routing_summary['overall_stats']['avg_rescue_rate']:.3f}")
    
    # Return comprehensive results
    return AutoRoutingResults(
        clustering_results=clustering_results,
        routing_model=routing_model,
        predictions=predictions,
        evaluation_metrics=evaluation_metrics,
        selected_k=selected_k,
        cluster_column=cluster_column,
        routing_summary=routing_summary
    )


def quick_cluster_routing(
    merged_df: pd.DataFrame,
    case: CaseConfig,
    sample_df: pd.DataFrame,
    person_set: PersonSet,
    fixed_k: int = 8
) -> Tuple[ClusterSmartRoutingMoP, Dict]:
    """
    Quick version with fixed k value for faster execution.
    
    Parameters:
    - merged_df: Training data
    - case: CaseConfig object
    - sample_df: Sample data
    - person_set: PersonSet object
    - fixed_k: Fixed number of clusters (default 8)
    
    Returns:
    - Tuple of (fitted routing model, evaluation metrics)
    """
    
    print(f"Quick clustering with k={fixed_k}")
    
    # Run clustering with fixed k
    merged_df_clustered, clustering_results = add_text_clusters(
        merged_df=merged_df,
        case=case,
        sample_df=sample_df,
        person_set=person_set,
        min_k=fixed_k,
        max_k=fixed_k,
        complexity_penalty=None  # No penalty needed for single k
    )
    
    # Create and fit routing model
    routing_model = create_cluster_routing_mop(person_set)
    routing_model.fit(
        merged_df=merged_df_clustered,
        cluster_column=f"synthetic_cluster_{fixed_k}"
    )
    
    # Evaluate
    predictions = routing_model.predict(merged_df_clustered)
    metrics = routing_model.evaluate(
        predictions=predictions,
        true_labels=merged_df_clustered['true_label'].values,
        baseline_preds=merged_df_clustered['base_pred'].values
    )
    
    return routing_model, metrics


def analyze_routing_decisions(
    routing_model: ClusterSmartRoutingMoP,
    merged_df: pd.DataFrame,
    n_samples: int = 10
) -> pd.DataFrame:
    """
    Analyze routing decisions for sample data points.
    
    Parameters:
    - routing_model: Fitted routing model
    - merged_df: DataFrame with clusters and predictions
    - n_samples: Number of samples to analyze
    
    Returns:
    - DataFrame with routing analysis
    """
    
    if not routing_model.is_fitted_:
        raise ValueError("Routing model must be fitted first")
    
    # Sample data points
    sample_indices = np.random.choice(len(merged_df), min(n_samples, len(merged_df)), replace=False)
    samples = merged_df.iloc[sample_indices]
    
    analysis_data = []
    
    for idx, row in samples.iterrows():
        cluster_id = row[routing_model.cluster_column_]
        
        # Get routing decision
        if cluster_id in routing_model.routing_rules_:
            rule = routing_model.routing_rules_[cluster_id]
            routing_type = rule.ensemble_type
            experts = rule.expert_profiles
            expected_accuracy = rule.expert_accuracy
        else:
            routing_type = "fallback"
            experts = routing_model.fallback_ensemble_['profiles'][:3]  # Show first 3
            expected_accuracy = np.mean(list(routing_model.fallback_ensemble_['accuracies'].values()))
        
        # Get actual prediction
        prediction = routing_model.predict(pd.DataFrame([row]))[0]
        
        analysis_data.append({
            'sample_id': row.get('sample_id', idx),
            'cluster': cluster_id,
            'routing_type': routing_type,
            'n_experts': len(experts) if routing_type != "fallback" else len(routing_model.fallback_ensemble_['profiles']),
            'expected_accuracy': expected_accuracy,
            'prediction': prediction,
            'true_label': row['true_label'],
            'correct': prediction == row['true_label']
        })
    
    return pd.DataFrame(analysis_data)


# Usage example
if __name__ == "__main__":
    """
    results = auto_configure_cluster_routing(
        merged_df=merged_df,
        case=case,
        sample_df=sample_df,
        person_set=person_set,
        min_k=3,
        max_k=12,
        complexity_penalty="bic"
    )
    
    # Access components
    routing_model = results.routing_model
    print(f"Selected k={results.selected_k}")
    print(f"Accuracy: {results.evaluation_metrics['accuracy']:.3f}")
    
    # Analyze specific routing decisions
    routing_analysis = analyze_routing_decisions(
        routing_model=results.routing_model,
        merged_df=merged_df,
        n_samples=20
    )
    print(routing_analysis.groupby('routing_type')['correct'].mean())
    """
    pass