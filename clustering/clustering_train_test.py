#clustering/clustering_train_test.py

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from collections import Counter
from tqdm import tqdm
from datetime import datetime

from clustering.clustering import (
    evaluate_cluster_with_tier2_ensembles,
    evaluate_cluster_routing,
    
)

from profiles.schema import PersonSet
from cases.cases_config import CaseConfig

from analysis_2 import (
        ensemble_by_trait_analysis,
        pareto_frontier_for_ensembles,
        paired_bootstrap_report_global,
        paired_bootstrap_report_by_category,
        fdr_families,
    )
import math


@dataclass
class ClusteringModel:
    kmeans_model: Any
    k: int
    embedding_model_name: str
    cluster_column: str
    cluster_centroids: np.ndarray
    training_metrics: Dict
    timestamp: str = None
    normalized: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RoutingModel:
    cluster_column: str
    training_metrics: Dict
    deployment_plans: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    routing_rules: Dict = field(default_factory=dict)
    fallback_ensemble: Dict = field(default_factory=dict)
    cluster_performance_map: Dict = field(default_factory=dict)
    positive_label: Optional[str] = None
    negative_label: Optional[str] = None


@dataclass
class RoutingPolicyConfig:
    enable_pareto: bool = False
    lambda_tok: float = 5e-4
    lambda_extra: float = 2.0

    risk_cap: Optional[float] = None
    category_risk_cap: Optional[float] = None
    min_category_n: int = 50

    require_fdr: bool = False
    q_threshold: float = 0.10

    enable_tier3_exploratory: bool = False



def select_deployment_plan_for_cluster(
    cluster_df: pd.DataFrame,
    person_set,
    case,
    *,
    perf_df: Optional[pd.DataFrame],
    policy: RoutingPolicyConfig
) -> Dict[str, Any]:

    ens = ensemble_by_trait_analysis(
        cluster_df, person_set, case=case,
        group_keys=("gender", "ethnicity", "age"),
        perf_df=perf_df,
        print_indications=False,
    )
    ens_dict = ens.get("ensemble_results", {}) or {}
    baseline_acc = float(ens.get("baseline_accuracy", 0.0))

    if not ens_dict:
        prof_cols = [c for c in cluster_df.columns if c.startswith("profile")]
        if not prof_cols:
            return {'plan_type': 'baseline', 'name': 'base_pred',
                    'profiles': [], 'metrics': {}, 'reason': 'no_profiles'}
        accs = {p: float((cluster_df[p].astype(str) == cluster_df["true_label"].astype(str)).mean()) for p in prof_cols}
        best_prof = max(accs.items(), key=lambda kv: kv[1])[0]
        return {'plan_type': 'single_profile', 'name': best_prof,
                'profiles': [best_prof],
                'metrics': {'accuracy': accs[best_prof], 'baseline_accuracy': baseline_acc},
                'reason': 'no_ensembles'}


    pe=None
    if policy.enable_pareto and (perf_df is not None):
        pe = pareto_frontier_for_ensembles(
            ensemble_results=ens,
            merged_df=cluster_df,
            case=case,
            lambda_tok=policy.lambda_tok,
            lambda_extra=policy.lambda_extra,
            out_dir=None
        )
        rec = pe.get("recommended")
        cand_name = rec["ensemble"] if rec else max(ens_dict.items(), key=lambda kv: float(kv[1].get("accuracy", -np.inf)))[0]
        reason_pick = "pareto"
    else:
        cand_name = max(ens_dict.items(), key=lambda kv: float(kv[1].get("accuracy", -np.inf)))[0]
        reason_pick = "best_accuracy"

    cand = ens_dict[cand_name]
    er = float(cand.get("extra_error_rate", 0.0))


    if policy.risk_cap is not None and er > policy.risk_cap:
        reason_pick = "risk_cap_reject"
        cand = None


    if cand and policy.category_risk_cap is not None:

        cat_safety = pe.get("category_safety") if (policy.enable_pareto and pe is not None) else None
        bad_cat = False
        if cat_safety is not None and isinstance(cat_safety, pd.DataFrame) and not cat_safety.empty:

            rows = cat_safety[cat_safety['ensemble'] == cand_name]
            for _, r in rows.iterrows():
                n = int(r.get('n', 0))
                extra_cat = float(r.get('extra_error_rate', np.nan))
                if n >= policy.min_category_n and np.isfinite(extra_cat) and (extra_cat > policy.category_risk_cap):
                    bad_cat = True
                    break

        if bad_cat:
            reason_pick = "category_risk_cap_reject"
            cand = None


    if cand and policy.require_fdr and ("ensemble_preds" in cand):
        try:
            base = cluster_df["base_pred"].astype(str)
            true = cluster_df["true_label"].astype(str)
    
            ens = pd.Series(cand["ensemble_preds"], index=cluster_df.index).astype(str)
    
            pb_global = paired_bootstrap_report_global(
                base=base, ens=ens, true=true
            )
    
            category_cols = getattr(case, "category_cols", ["stereotype_type"]) or ["stereotype_type"]
            category_col = category_cols[0]
    
            percat = paired_bootstrap_report_by_category(
                merged_df=cluster_df.assign(_ens=ens),
                ens_preds=ens,
                category_col=category_col,
                min_n=policy.min_category_n
            )
    
            gtbl = pd.DataFrame([{"ensemble": cand_name, **pb_global}])
            fam = fdr_families(gtbl, percat)
    
            q_tbl = fam.get("q_global_delta_acc") or fam.get("q_global")
            ok = True
            if isinstance(q_tbl, pd.DataFrame) and not q_tbl.empty:
                q_col = next((c for c in q_tbl.columns if c.startswith("q_")), q_tbl.columns[-1])
                q = float(q_tbl.iloc[0][q_col])
                ok = (q <= policy.q_threshold)
    
            if not ok:
                reason_pick = "fdr_reject"
                cand = None
    
        except Exception as e:
            print(f"[FDR] skipped for {cand_name}: {e}")
     

    if not cand:
        prof_cols = [c for c in cluster_df.columns if c.startswith("profile")]
        accs = {p: float((cluster_df[p].astype(str) == cluster_df["true_label"].astype(str)).mean()) for p in prof_cols}
        best_prof = max(accs.items(), key=lambda kv: kv[1])[0]
        return {
            'plan_type': 'single_profile',
            'name': best_prof,
            'profiles': [best_prof],
            'metrics': {'accuracy': accs[best_prof], 'baseline_accuracy': baseline_acc},
            'reason': reason_pick
        }

    return {
        'plan_type': 'ensemble',
        'name': cand_name,
        'profiles': cand.get("profiles", []),
        'metrics': {
            'accuracy': float(cand.get("accuracy", baseline_acc)),
            'rescue_rate': float(cand.get("rescue_rate", 0.0)),
            'extra_error_rate': float(cand.get("extra_error_rate", 0.0)),
            'baseline_accuracy': baseline_acc,
            'n_profiles': int(cand.get("n_profiles", 0) or 0),
            'tokens_per_sample_sum': float(cand.get("tokens_per_sample_sum", np.nan)) if cand.get("tokens_per_sample_sum") is not None else np.nan
        },
        'reason': reason_pick
    }




class AdaptiveClusteringPipeline:
    def __init__(self, 
                 save_dir: str = "clustering_models",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 routing_policy: Optional[RoutingPolicyConfig] = None,
                 show_progress: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.clustering_model = None
        self.routing_model = None
        self.training_history = []
        self.policy = routing_policy or RoutingPolicyConfig()
        self.show_progress = show_progress

    def _pbar(self, iterable, *, total=None, desc:str=""):
        return tqdm(iterable, total=total, desc=desc, leave=False) if self.show_progress else iterable

    def _embed(self, texts: list[str], *, normalized: Optional[bool] = None) -> np.ndarray:
        if normalized is None:
            normalized = getattr(self.clustering_model, "normalized", True)
        return self.embedding_model.encode(
            texts, show_progress_bar=True, normalize_embeddings=normalized
        )
    
    def _plan_string(self, plan):
        if not plan or plan.get("plan_type") in (None, "baseline"):
           return "baseline"
        if plan["plan_type"] == "single_profile":
            return f"single:{plan['name']}"
        if plan["plan_type"] == "ensemble":
            profs = ",".join(plan.get("profiles", []))
            return f"ensemble:[{profs}]"
        return "baseline"

    def _true_series(self, df: pd.DataFrame, case: CaseConfig) -> tuple[pd.Series, str]:
        """
        Return (true_label_series_as_str, column_name_used) using 'true_label' if present,
        otherwise fall back to case.label_col.
        """
        if "true_label" in df.columns:
            return df["true_label"].astype(str), "true_label"
        if case.label_col in df.columns:
            return df[case.label_col].astype(str), case.label_col
        raise KeyError(
            f"Neither 'true_label' nor case.label_col ('{case.label_col}') found. "
            f"Columns available: {list(df.columns)}"
        )
    

    def _apply_deployment_plans_and_eval(self, df: pd.DataFrame, case:CaseConfig) -> Dict[str, Any]:
        """
        Apply saved deployment plans *offline* (no LLM calls).
        Produces per-row columns:
          - policy_pred
          - policy_plan_type ∈ {baseline, single_profile, ensemble}
          - policy_profiles_used (pipe-separated)
          - policy_selected_profile (baseline or profile for single; 'majority' for ensemble)
          - policy_votes (JSON: {profile: label})
        Saves a CSV next to the model.
        """
        if not self.routing_model.deployment_plans:
            return {}
    
        import json
        from collections import Counter
    
        cluster_col = self.routing_model.cluster_column
        df2 = df.copy()
        preds = pd.Series(index=df.index, dtype=object)
    
        df2["policy_plan_type"] = ""
        df2["policy_profiles_used"] = ""
        df2["policy_selected_profile"] = ""
        df2["policy_votes"] = ""
    
        for cl, sub in self._pbar(df.groupby(cluster_col),
                          total=df[cluster_col].nunique(),
                          desc="Applying plans (offline)"):
            
            plan = self.routing_model.deployment_plans.get(int(cl))
            if not plan:
                fb = self.routing_model.fallback_ensemble or {}
                cols = list(fb.get("profiles", []))
                if not cols:
                    preds.loc[sub.index] = sub["base_pred"].astype(str)
                    df2.loc[sub.index, "policy_plan_type"] = "baseline"
                    df2.loc[sub.index, "policy_profiles_used"] = ""
                    df2.loc[sub.index, "policy_selected_profile"] = "baseline"
                    df2.loc[sub.index, "policy_votes"] = "{}"
                elif len(cols) == 1:
                    p = cols[0]
                    preds.loc[sub.index] = sub[p].astype(str)
                    df2.loc[sub.index, "policy_plan_type"] = "single_profile"
                    df2.loc[sub.index, "policy_profiles_used"] = p
                    df2.loc[sub.index, "policy_selected_profile"] = p
                    df2.loc[sub.index, "policy_votes"] = sub[p].astype(str).apply(lambda y: json.dumps({p: y}))
                else:
                    votes_df = sub[cols].astype(str)
                    maj = votes_df.apply(lambda r: Counter(r).most_common(1)[0][0], axis=1)
                    preds.loc[sub.index] = maj
                    df2.loc[sub.index, "policy_plan_type"] = "ensemble"
                    df2.loc[sub.index, "policy_profiles_used"] = "|".join(cols)
                    df2.loc[sub.index, "policy_selected_profile"] = "majority"
                    df2.loc[sub.index, "policy_votes"] = votes_df.apply(lambda r: json.dumps({c: r[c] for c in cols}), axis=1)
                continue
    
            if plan["plan_type"] == "single_profile":
                p = plan["name"]
                preds.loc[sub.index] = sub[p].astype(str)
                df2.loc[sub.index, "policy_plan_type"] = "single_profile"
                df2.loc[sub.index, "policy_profiles_used"] = p
                df2.loc[sub.index, "policy_selected_profile"] = p
                df2.loc[sub.index, "policy_votes"] = sub[p].astype(str).apply(lambda y: json.dumps({p: y}))
            elif plan["plan_type"] == "ensemble":
                cols = plan.get("profiles", [])
                if len(cols) <= 1:
                    p = cols[0] if cols else "base_pred"
                    preds.loc[sub.index] = sub[p].astype(str)
                    df2.loc[sub.index, "policy_plan_type"] = "single_profile" if cols else "baseline"
                    df2.loc[sub.index, "policy_profiles_used"] = (p if cols else "")
                    df2.loc[sub.index, "policy_selected_profile"] = (p if cols else "baseline")
                    df2.loc[sub.index, "policy_votes"] = sub[p].astype(str).apply(lambda y: json.dumps({p: y}) if cols else "{}")
                else:
                    votes_df = sub[cols].astype(str)
                    maj = votes_df.apply(lambda r: Counter(r).most_common(1)[0][0], axis=1)
                    preds.loc[sub.index] = maj
                    df2.loc[sub.index, "policy_plan_type"] = "ensemble"
                    df2.loc[sub.index, "policy_profiles_used"] = "|".join(cols)
                    df2.loc[sub.index, "policy_selected_profile"] = "majority"
                    df2.loc[sub.index, "policy_votes"] = votes_df.apply(lambda r: json.dumps({c: r[c] for c in cols}), axis=1)
            else:
                preds.loc[sub.index] = sub["base_pred"].astype(str)
                df2.loc[sub.index, "policy_plan_type"] = "baseline"
                df2.loc[sub.index, "policy_profiles_used"] = ""
                df2.loc[sub.index, "policy_selected_profile"] = "baseline"
                df2.loc[sub.index, "policy_votes"] = "{}"
    
        df2["policy_pred"] = preds.values
        true, true_col = self._true_series(df2, case)
        base = df2["base_pred"].astype(str) if "base_pred" in df2.columns else None
        acc = float((df2["policy_pred"].astype(str) == true).mean())

        metrics = {"accuracy": acc}

        if base is not None:
            base = base.astype(str)
            base_acc = float((base == true).mean())
            rescue   = float(((base != true) & (df2["policy_pred"].astype(str) == true)).mean())
            extra    = float(((base == true) & (df2["policy_pred"].astype(str) != true)).mean())
            metrics.update({
                "baseline_accuracy": base_acc,
                "rescue_rate": rescue,
                "extra_error_rate": extra,
                "accuracy_improvement": acc - base_acc,
        })
    

        save_path = self.save_dir / f"{self.clustering_model.k}_policy_offline_decisions.csv"
        cols = [c for c in [
            "sample_id",
            getattr(self, "text_col", None),
        ] if c in df2.columns] + [
            self.routing_model.cluster_column, "true_label", "base_pred" if "base_pred" in df2.columns else None,
            "policy_pred", "policy_plan_type", "policy_profiles_used", "policy_selected_profile", "policy_votes"
        ]
        cols = [c for c in cols if c is not None]
        df2[cols].to_csv(save_path, index=False)
        print(f"[saved] {save_path}")
    
        return {"policy_preds": preds.values, "policy_metrics": metrics, "df": df2}

    
    def route_texts_with_plans(
        self,
        texts_df: pd.DataFrame,
        case: CaseConfig,
        gen_fn,                   
        label_map: Optional[Dict[str,str]] = None,
        progress_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        if not self.routing_model or not self.clustering_model:
            raise ValueError("Train first.")
        if case.input_col not in texts_df.columns:
            raise ValueError(f"Missing '{case.input_col}' in texts_df.")
    
        df = self.assign_clusters_to_df(texts_df.copy(), case)
        cluster_col = self.routing_model.cluster_column
    
        preds, token_logs = [], []
        it = df.iterrows() if progress_fn else self._pbar(df.iterrows(),
                                                  total=len(df),
                                                  desc="Policy routing")
        for i, (_, row) in enumerate(it, 1):
            cl = int(row[cluster_col])
            plan = self.routing_model.deployment_plans.get(cl)
    
            if not plan:
                fb = self.routing_model.fallback_ensemble
                profiles = (fb.get("profiles") if fb else []) or []
            else:
                if plan["plan_type"] == "single_profile":
                    profiles = [plan["name"]]
                else:
                    profiles = plan.get("profiles", [])
    
            votes = []
            if not profiles:
                if "base_pred" in df.columns:
                    pred, tok = row["base_pred"], None
                else:
                    pred, tok = gen_fn(row[case.input_col], None)
                votes = [pred]
                if tok is not None:
                    token_logs.append({"i": i, "profile": "baseline", "tokens": tok})
            else:
                for p in profiles:
                    y, tok = gen_fn(row[case.input_col], p)
                    if label_map and y in label_map:
                        y = label_map[y]
                    votes.append(y)
                    if tok is not None:
                        token_logs.append({"i": i, "profile": p, "tokens": tok})
    
            if len(votes) == 1:
                pred = votes[0]
            else:
                from collections import Counter
                pred = Counter(votes).most_common(1)[0][0]
    
            preds.append(pred)

            if progress_fn: 
                progress_fn(i, len(df))
    
        df["policy_pred"] = preds
    
        true = df["true_label"].astype(str)
        pred = df["policy_pred"].astype(str)
        acc = float((pred == true).mean())
        metrics = {"accuracy": acc}
    
        if "base_pred" in df.columns:
            base = df["base_pred"].astype(str)
            rescue = float(((base != true) & (pred == true)).mean())
            extra  = float(((base == true) & (pred != true)).mean())
            metrics.update({
                "baseline_accuracy": float((base == true).mean()),
                "rescue_rate": rescue,
                "extra_error_rate": extra,
                "accuracy_improvement": acc - metrics["baseline_accuracy"],
            })
    
        return {"df": df, "policy_metrics": metrics, "token_logs": token_logs}

        


    def train(self,
              merged_df: pd.DataFrame,
              case: CaseConfig,
              sample_df: pd.DataFrame,
              person_set: PersonSet,
              min_k: int = 3,
              max_k: int = 12,
              complexity_penalty: str = "bic",
              version_name: str = "v1",
              perf_df: Optional[pd.DataFrame] = None
              ) -> Dict:
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
        embeddings = self._embed(texts, normalized=True)
        print("\nFinding optimal clustering...")
        best_k, best_kmeans, best_metrics = self._find_optimal_clustering(
            embeddings, merged_df, person_set, case,
            min_k, max_k, complexity_penalty,
            perf_df=perf_df
        )
        cluster_column = f"synthetic_cluster_{best_k}"
        
        self.clustering_model = ClusteringModel(
            kmeans_model=best_kmeans,
            k=best_k,
            embedding_model_name=self.embedding_model_name,
            cluster_column=cluster_column,
            cluster_centroids=best_kmeans.cluster_centers_,
            training_metrics=best_metrics,
            normalized=True,
        )
        print("\nSelecting deployment plans per cluster...")
        deployment_plans = {}

        cluster_ids = sorted(merged_df[cluster_column].unique().tolist())
        for cl in self._pbar(cluster_ids, total=len(cluster_ids), desc="Selecting deployment plans"):
            cdf = merged_df[merged_df[cluster_column] == cl]
            plan = select_deployment_plan_for_cluster(
                    cdf, person_set, case, perf_df=perf_df, policy=self.policy
                )
            deployment_plans[int(cl)] = plan

        # Save plans on a lightweight routing model
        self.routing_model = RoutingModel(
            cluster_column=cluster_column,
            training_metrics={},  # filled after applying plans
            deployment_plans=deployment_plans,
        )

        # Compute training metrics by applying plans on train (no LLM)
        apply_out = self._apply_deployment_plans_and_eval(merged_df, case=case)
        training_metrics = apply_out.get("policy_metrics", {})
        self.routing_model.training_metrics = training_metrics

        print(f"\nTraining Performance (plans on train):")
        if training_metrics:
            print(f"  Accuracy: {training_metrics.get('accuracy', float('nan')):.3f}")
            if 'baseline_accuracy' in training_metrics:
                print(f"  Baseline: {training_metrics['baseline_accuracy']:.3f}")
                print(f"  Improvement: {training_metrics['accuracy_improvement']:+.3f}")

        self._save_models(version_name)
        self.training_history.append({
            'version': version_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(merged_df),
            'k': best_k,
            'accuracy': training_metrics.get('accuracy', float('nan')),
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
             version_name: Optional[str] = None,
             eval_mode: str = "offline",
             gen_fn=None,
             label_map=None,
             save_internal: bool = True,       
            run_tag: Optional[str] = None,) -> Dict:
    
        import json
        from collections import Counter
    
        print(f"\n{'='*80}")
        print(f"Testing on new data - mode: {eval_mode}")
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
        embeddings = self._embed(texts, normalized=self.clustering_model.normalized)
        print("Assigning to clusters...")
        cluster_assignments = self.clustering_model.kmeans_model.predict(embeddings)
        test_df.loc[:, self.clustering_model.cluster_column] = cluster_assignments

        cluster_dist = pd.Series(cluster_assignments).value_counts().sort_index()
        print(f"\nCluster distribution in test set:")
        for cluster_id, count in cluster_dist.items():
            print(f"  Cluster {cluster_id}: {count} samples ({100*count/len(test_df):.1f}%)")
        distances = []
        for i, embedding in enumerate(self._pbar(embeddings, total=len(embeddings), desc="Computing distances")):
            cluster = cluster_assignments[i]
            centroid = self.clustering_model.cluster_centroids[cluster]
            dist = np.linalg.norm(embedding - centroid)
            distances.append(dist)
        avg_distance = np.mean(distances)
        print(f"\nAverage distance to centroids: {avg_distance:.3f}")

        # ---- Policy evaluation ----
        policy_eval, policy_df, policy_tokens = {}, None, None

        if eval_mode == "policy_online":
            if gen_fn is None:
                 raise ValueError("gen_fn is required for eval_mode='policy_online'.")
            policy_eval = self._policy_online_predict(test_df, case, gen_fn, save_files=save_internal,
                run_tag=run_tag)

            # Build a DF that includes source plan per row
            policy_df = policy_eval["df"].copy()
            clcol = self.routing_model.cluster_column
            policy_df["policy_source"] = policy_df[clcol].map(
                lambda cl: self._plan_string(self.routing_model.deployment_plans.get(int(cl)))
            )
            policy_tokens = policy_eval.get("token_logs")

            pm = policy_eval["policy_metrics"]
            print("\nPolicy deployment (online) performance:")
            print(f"  Accuracy: {pm['accuracy']:.3f}")
            if 'baseline_accuracy' in pm:
                print(f"  Baseline: {pm['baseline_accuracy']:.3f}")
                print(f"  ΔAcc: {pm['accuracy_improvement']:+.3f}")
                print(f"  Rescue: {pm['rescue_rate']:.3f} | Extra: {pm['extra_error_rate']:.3f}")
 
        else:
            # Offline application of deployment plans (requires per-profile cols in test_df)
            policy_eval = self._apply_deployment_plans_and_eval(test_df, case=case)

            # Build a DF with offline policy preds + sources
            policy_df = test_df.copy()
            policy_df["policy_pred"] = policy_eval.get("policy_preds")
            clcol = self.routing_model.cluster_column
            policy_df["policy_source"] = policy_df[clcol].map(
                lambda cl: self._plan_string(self.routing_model.deployment_plans.get(int(cl)))
            )

        return {
            "test_metrics": policy_eval.get("policy_metrics"),
            "policy_metrics": policy_eval.get("policy_metrics"),
            "cluster_distribution": cluster_dist.to_dict(),
            "avg_distance_to_centroids": avg_distance,
            "test_df": test_df,
            "predictions": policy_eval.get("policy_preds"),           
            "policy_predictions": policy_eval.get("policy_preds"),
            "policy_df": policy_df,                
            "policy_token_logs": policy_tokens,
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
        
        for batch_idx in self._pbar(range(n_batches), total=n_batches, desc="Adaptive batches"):
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
        embeddings = self._embed(texts, normalized=self.clustering_model.normalized)
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
    
    def _policy_online_predict(
        self,
        df: pd.DataFrame,
        case: CaseConfig,
        gen_fn,
        *,
        ensure_base_pred: bool = True,
        save_files: bool = True,
        run_tag: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Online LLM calls per deployment plan.
        Adds per-row columns:
          - policy_pred
          - policy_plan_type
          - policy_profiles_used
          - policy_selected_profile
          - policy_votes (JSON)
        Saves decisions CSV + token logs CSV.
        """
        import json
        from collections import Counter
    
        if not self.routing_model or not self.routing_model.deployment_plans:
            raise ValueError("No deployment plans available. Train with policy flags enabled.")
    
        cluster_col = self.routing_model.cluster_column
        if cluster_col not in df.columns:
            raise ValueError(f"'{cluster_col}' not found on df. Run assign_clusters_to_df/test() first.")
    
        df2 = df.copy()
        token_logs = []  # to log tokens for baseline and profiles
    
        # ---- Baseline prediction (optional) -------------------------------------
        if ensure_base_pred and "base_pred" not in df2.columns:
            base_preds = []
            it = self._pbar(df2.iterrows(), total=len(df2), desc="Baseline inference")
            for j, (_, row) in enumerate(it, 1):
                y, tok = gen_fn(str(row[case.input_col]), None, row=row.to_dict())
                base_preds.append(y)
                if tok is not None:
                    token_logs.append({"i": j, "profile": "baseline", "tokens": tok})
            df2["base_pred"] = base_preds
    
        # ---- Online policy routing ----------------------------------------------
        policy_preds = []
        plan_type_col, profs_used_col, sel_prof_col, votes_col = [], [], [], []
    
        for i, (idx, row) in enumerate(
            self._pbar(df2.iterrows(), total=len(df2), desc="Online policy inference"), 1
        ):
            text = str(row[case.input_col])
            cl = int(row[cluster_col])
            plan = self.routing_model.deployment_plans.get(cl)
    
            if not plan or plan.get("plan_type") in (None, "baseline"):
                y, tok = gen_fn(text, None, row=row.to_dict())
                policy_preds.append(y)
                plan_type_col.append("baseline")
                profs_used_col.append("")
                sel_prof_col.append("baseline")
                votes_col.append("{}")
                if tok is not None:
                    token_logs.append({"i": i, "profile": "baseline", "tokens": tok})
                continue
    
            if plan["plan_type"] == "single_profile":
                prof = plan["name"]
                y, tok = gen_fn(text, prof, row=row.to_dict())
                policy_preds.append(y)
                plan_type_col.append("single_profile")
                profs_used_col.append(prof)
                sel_prof_col.append(prof)
                votes_col.append(json.dumps({prof: y}))
                if tok is not None:
                    token_logs.append({"i": i, "profile": prof, "tokens": tok})
    
            elif plan["plan_type"] == "ensemble":
                votes, vote_map = [], {}
                for prof in plan.get("profiles", []):
                    y, tok = gen_fn(text, prof, row=row.to_dict())
                    votes.append(y)
                    vote_map[prof] = y
                    if tok is not None:
                        token_logs.append({"i": i, "profile": prof, "tokens": tok})
                if votes:
                    from collections import Counter as _Counter
                    maj = _Counter(votes).most_common(1)[0][0]
                    policy_preds.append(maj)
                    plan_type_col.append("ensemble")
                    profs_used_col.append("|".join(plan.get("profiles", [])))
                    sel_prof_col.append("majority")
                    votes_col.append(json.dumps(vote_map))
                else:
                    y, tok = gen_fn(text, None, row=row.to_dict())
                    policy_preds.append(y)
                    plan_type_col.append("baseline")
                    profs_used_col.append("")
                    sel_prof_col.append("baseline")
                    votes_col.append("{}")
                    if tok is not None:
                        token_logs.append({"i": i, "profile": "baseline", "tokens": tok})
            else:
                y, tok = gen_fn(text, None, row=row.to_dict())
                policy_preds.append(y)
                plan_type_col.append("baseline")
                profs_used_col.append("")
                sel_prof_col.append("baseline")
                votes_col.append("{}")
                if tok is not None:
                    token_logs.append({"i": i, "profile": "baseline", "tokens": tok})
    
        df2["policy_pred"] = pd.Series(policy_preds, index=df2.index)
        df2["policy_plan_type"] = plan_type_col
        df2["policy_profiles_used"] = profs_used_col
        df2["policy_selected_profile"] = sel_prof_col
        df2["policy_votes"] = votes_col
    
        # ---- Metrics (robust to different GT column names) ----------------------
        true, true_col = self._true_series(df2, case)
        pred = df2["policy_pred"].astype(str)
        acc = float((pred == true).mean())
        metrics = {"accuracy": acc}
    
        if "base_pred" in df2.columns:
            base = df2["base_pred"].astype(str)
            rescue = float(((base != true) & (pred == true)).mean())
            extra  = float(((base == true) & (pred != true)).mean())
            base_acc = float((base == true).mean())
            metrics.update({
                "baseline_accuracy": base_acc,
                "rescue_rate": rescue,
                "extra_error_rate": extra,
                "accuracy_improvement": acc - base_acc
            })
    
        # ---- Unique file tag ----------------------------------------------------
        if run_tag is None:
            fdr = getattr(self, "policy", RoutingPolicyConfig()).require_fdr
            run_tag = f"{getattr(case, 'case_name', 'case')}_{self.clustering_model.k}_fdr_{str(fdr).lower()}_{datetime.now():%Y%m%d-%H%M%S}"
    
        # ---- Choose columns to persist (+ include input text if present) --------
        base_cols = [c for c in ["sample_id", case.input_col] if c in df2.columns]
        cols = base_cols + [
            cluster_col,
            true_col,
            "base_pred" if "base_pred" in df2.columns else None,
            "policy_pred",
            "policy_plan_type",
            "policy_profiles_used",
            "policy_selected_profile",
            "policy_votes",
        ]
        cols = [c for c in cols if c is not None]
    
        export_df = df2[cols].copy()
        if true_col != "true_label":
            export_df = export_df.rename(columns={true_col: "true_label"})
    
        # ---- Save (non-overwriting by timestamped run_tag) ----------------------
        if save_files:
            save_path = self.save_dir / f"{run_tag}_policy_online_decisions.csv"
            export_df.to_csv(save_path, index=False)
            print(f"[saved] {save_path}")
    
            if token_logs:
                tok_path = self.save_dir / f"{run_tag}_policy_online_token_logs.csv"
                pd.DataFrame(token_logs).to_csv(tok_path, index=False)
                print(f"[saved] {tok_path}")
    
        return {
            "policy_preds": df2["policy_pred"].values,
            "policy_metrics": metrics,
            "df": df2,
            "token_logs": token_logs,
        }

    
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
                                complexity_penalty: str,
                                perf_df: Optional[pd.DataFrame] = None,
                                ) -> Tuple[int, Any, Dict]:

        best_score = -np.inf
        best_k = min_k
        best_kmeans = None
        best_metrics = {}
        n_samples = len(embeddings)
        for k in self._pbar(range(min_k, max_k + 1),
                    total=max_k - min_k + 1,
                    desc="Grid search: k"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            import warnings
            uniq, counts = np.unique(labels, return_counts=True)
            if len(uniq) >= 2 and (counts > 1).all():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
                    sil_score = float(silhouette_score(embeddings, labels))
            else:
                sil_score = 0.0
            cluster_col = f"synthetic_cluster_{k}"
            merged_df[cluster_col] = labels
            ensemble_metrics = evaluate_cluster_with_tier2_ensembles(
                merged_df, person_set, case, cluster_col, k, perf_df=perf_df
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

        keep_col = f"synthetic_cluster_{best_k}"
        cols_to_drop = [
            c for c in merged_df.columns
            if c.startswith("synthetic_cluster_") and c != keep_col
        ]
        if cols_to_drop:
            merged_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

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
        if not hasattr(self.clustering_model, "normalized"):
            self.clustering_model.normalized = True
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
    save_dir: str = "clustering_models",
    perf_df: Optional[pd.DataFrame] = None,
    routing_policy: Optional[RoutingPolicyConfig] = None,
    eval_mode: str = "offline",
    gen_fn=None,
) -> Dict:
    
    pipeline = AdaptiveClusteringPipeline(save_dir=save_dir, routing_policy=routing_policy)

    train_results = pipeline.train(
        merged_df=train_df,
        case=case,
        sample_df=sample_df,
        person_set=person_set,
        version_name="model_v1",
        perf_df=perf_df,
    )
    test_results = pipeline.test(
        test_df=test_df,
        sample_df=sample_df,
        case=case,
        person_set=person_set,
        eval_mode=eval_mode,
        gen_fn=gen_fn,
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