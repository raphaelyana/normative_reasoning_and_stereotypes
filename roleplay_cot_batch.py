# roleplay_cot_batch.py
# End-to-end batching for Role-Playing CoT with OpenAI Batch API.
# - Prompts are EXACT (via ChainOfThoughts._build_optimized_cot_prompt).
# - Per-request system message (per profile, active/passive/none).
# - Output directory is configurable.
# - Works with ANY case: pass a CaseConfig and its DataFrame.
# - Utilities to collect successes, list failures (error_file_id), and re-run failures online.

import os
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
import openai

# === Your project imports ===
from chain_of_thought import ChainOfThoughts
from profiles.profile_message import make_system_message
from profiles.profile_sets import PERSON_ETHNICS  # default person set
# (Caller provides the case object + dataframe, e.g. from cases.stereotypes_case import stereotypes_case)

# --------------------------------------------------------------------------------------
# Utility class
# --------------------------------------------------------------------------------------

class RoleplayBatchRunner:
    """
    Notebook- and script-friendly helper to run Role-Playing CoT with OpenAI Batch.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        model_filename: str = "openai_4.1_mini",
        strategy: str = "optimized",
        max_tokens: int = 700,
        output_base_dir: Optional[str] = None,
        manifests_dir: str = "batch_jobs/manifests",
        reports_dir: str = "batch_jobs/reports",
        person_set = PERSON_ETHNICS,
        api_key_env: str = "API_KEY_OPENAI",
        client: Optional[openai.OpenAI] = None,
    ):
        """
        Parameters
        ----------
        output_base_dir : str
            Root directory where results are written. If None:
            f"results/{model_filename}/cot/role_playing"
            (Change this to .../role_playing_ethnics/etc as you wish.)
        """
        load_dotenv()
        self.client = client or openai.OpenAI(api_key=os.getenv(api_key_env))
        if not self.client:
            raise ValueError("OpenAI client not initialized; ensure your API key is set.")

        self.model = model
        self.model_filename = model_filename
        self.strategy = strategy
        self.max_tokens = int(max_tokens)

        self.output_base_dir = output_base_dir or f"results/{self.model_filename}/cot/role_playing"
        self.manifests_dir = manifests_dir
        self.reports_dir = reports_dir
        self.person_set = person_set

        os.makedirs(self.manifests_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    # ------------------------------- Paths & small utils -------------------------------

    @staticmethod
    def _timestamp() -> str:
        return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def output_paths(self, case_name: str, profile: str, role: str) -> Tuple[str, str]:
        """
        Per-profile, per-case file destinations (CSV + reasoning JSON).
        """
        base = os.path.join(self.output_base_dir, f"{profile}_{role}")
        self._ensure_dir(base)
        csv_path = os.path.join(base, f"results_{case_name.lower()}_{self.strategy}_cot.csv")
        reasoning_path = os.path.join(base, f"reasoning_{case_name.lower()}_{self.strategy}_cot.json")
        return csv_path, reasoning_path

    def is_profile_done(self, case_name: str, profile: str, role: str) -> bool:
        csv_path, _ = self.output_paths(case_name, profile, role)
        return os.path.exists(csv_path)

    def filter_profiles_not_done(self, case_name: str, profiles: List[str], role: str) -> List[str]:
        return [p for p in profiles if not self.is_profile_done(case_name, p, role)]

    # ------------------------------- Prompt builders -----------------------------------

    def _build_exact_user_prompt(self, case, text: str) -> str:
        """
        Build your exact CoT prompt text using your ChainOfThoughts class.
        (role_playing='none' here — system message handles persona.)
        """
        tmp = ChainOfThoughts(
            case=case, client=None, model=self.model, max_tokens=self.max_tokens,
            task_definition=None, person_key=None, role_playing="none", person_set=self.person_set
        )
        return tmp._build_optimized_cot_prompt(text)

    def _build_system_message(self, case_name: str, profile: Optional[str], role: str) -> str:
        if role in ("active", "passive") and profile:
            sm = make_system_message(case_name=case_name, person_key=profile, person_set=self.person_set)
            return sm["content"] if isinstance(sm, dict) and "content" in sm else str(sm)
        return f"You are an expert classifier for {case_name}. Think carefully and follow the prompt."

    @staticmethod
    def _write_jsonl_lines(lines: List[dict], path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return path

    # -------------------------------- Submit batches -----------------------------------

    def submit_batches(
        self,
        case,                      # CaseConfig (e.g., stereotypes_case)
        df: pd.DataFrame,          # DataFrame for that case
        profiles: List[str],
        role: str = "passive",
        chunk_size: Optional[int] = None,
        skip_completed: bool = True,
        completion_window: str = "24h",
        metadata_extra: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build JSONL request(s) & submit OpenAI Batch jobs.
        Returns: [{"batch_id": ..., "jsonl_path": ...}, ...]
        """
        case_name = case.case_name

        if skip_completed:
            profiles = self.filter_profiles_not_done(case_name, profiles, role)
        if not profiles:
            print(f"Nothing to submit for '{case_name}' (all selected profiles already done).")
            return []

        print(f"Submitting {len(profiles)} profile(s) for case='{case_name}' role='{role}' using model='{self.model}'.")

        req_lines: List[dict] = []
        for profile in profiles:
            sys_msg = self._build_system_message(case_name, profile, role)
            for idx, row in df.iterrows():
                user_prompt = self._build_exact_user_prompt(case, row[case.input_col])
                custom_id = f"{case_name}:{profile}:sample_{int(idx)}"
                req_lines.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "temperature": 0,
                        "max_tokens": self.max_tokens,
                        "messages": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_prompt}
                        ]
                    }
                })

        chunk_size = chunk_size or len(req_lines)
        chunks = [req_lines[i:i+chunk_size] for i in range(0, len(req_lines), chunk_size)]

        submissions = []
        for ci, chunk in enumerate(chunks, start=1):
            jsonl_path = f"batch_jobs/{case_name}_{role}_{self._timestamp()}_{ci:03d}.jsonl"
            self._write_jsonl_lines(chunk, jsonl_path)

            file_obj = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
            meta = {"case": case_name, "role": role, "chunk": str(ci)}
            if metadata_extra:
                meta.update(metadata_extra)

            batch = self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
                metadata=meta,
            )
            print(f"[SUBMITTED] batch_id={batch.id}  jsonl={jsonl_path}  requests={len(chunk)}")

            manifest = {
                "batch_id": batch.id,
                "jsonl_path": jsonl_path,
                "case_name": case_name,
                "role": role,
                "model": self.model,
                "strategy": self.strategy,
                "model_filename": self.model_filename,
                "submitted_at": self._timestamp(),
                "profiles": profiles,
                "output_base_dir": self.output_base_dir,
            }
            with open(os.path.join(self.manifests_dir, f"{batch.id}.manifest.json"), "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2, ensure_ascii=False)

            submissions.append({"batch_id": batch.id, "jsonl_path": jsonl_path})

        return submissions

    # --------------------------------- Collection --------------------------------------

    @staticmethod
    def _parse_success_choice_text(item: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any], str]:
        """
        Returns (ok, text, usage, fail_reason)
        ok=True  -> text contains the chat completion content
        ok=False -> fail_reason explains why it failed
        """
        if "error" in item and item["error"]:
            return False, "", {}, f"error_object:{item['error']}"
        resp = item.get("response") or {}
        status = resp.get("status_code", 200)
        if status and int(status) >= 400:
            return False, "", {}, f"http_{status}"
        body = resp.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            return False, "", body.get("usage") or {}, "no_choices"
        text = (choices[0]["message"]["content"] or "").strip()
        usage = body.get("usage") or {}
        if not text:
            return False, "", usage, "empty_text"
        return True, text, usage, ""

    def _merge_rows_csv(self, csv_path: str, new_rows: List[Dict[str, Any]]) -> None:
        df_new = pd.DataFrame(new_rows)
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all = df_all.sort_values("sample_id").drop_duplicates("sample_id", keep="last")
        else:
            df_all = df_new
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_all.to_csv(csv_path, index=False)

    def _merge_reasoning_json(self, reasoning_path: str, new_items: List[Dict[str, Any]]) -> None:
        by_id = {x["sample_id"]: x for x in new_items}
        if os.path.exists(reasoning_path):
            with open(reasoning_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            for x in old:
                sid = x.get("sample_id")
                if sid not in by_id:
                    by_id[sid] = x
        merged = [by_id[k] for k in sorted(by_id)]
        os.makedirs(os.path.dirname(reasoning_path), exist_ok=True)
        with open(reasoning_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    def collect_one_batch_merged(self, case, df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        """
        Collect a single COMPLETED batch and MERGE outputs per profile into your standard files.
        Returns a DataFrame of failures from the output file scan (HTTP errors/missing choices).
        """
        case_name = case.case_name
        manifest_path = os.path.join(self.manifests_dir, f"{batch_id}.manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Manifest not found for batch {batch_id} at {manifest_path}")

        b = self.client.batches.retrieve(batch_id)
        if b.status != "completed":
            print(f"Batch {batch_id} not completed yet (status={b.status}). Skipping.")
            return pd.DataFrame(columns=["batch_id","custom_id","case","profile","sample_id","reason"])

        # Download successes JSONL
        raw = self.client.files.content(b.output_file_id).read().decode("utf-8")
        lines = [json.loads(l) for l in raw.splitlines() if l.strip()]

        per_profile_rows: Dict[str, List[Dict[str, Any]]] = {}
        per_profile_reasoning: Dict[str, List[Dict[str, Any]]] = {}
        failures: List[Dict[str, Any]] = []

        cot_parser = ChainOfThoughts(
            case=case, client=None, model=self.model, max_tokens=self.max_tokens,
            task_definition=None, person_key=None, role_playing="none", person_set=self.person_set
        )

        for item in lines:
            cid = item.get("custom_id")
            if not cid:
                continue
            try:
                _case, profile, sample_part = cid.split(":")
                sample_id = int(sample_part.replace("sample_", ""))
            except Exception:
                failures.append({"batch_id": batch_id, "custom_id": cid, "case": case_name,
                                 "profile": None, "sample_id": None, "reason": "bad_custom_id"})
                continue

            ok, text, usage, fail_reason = self._parse_success_choice_text(item)
            if not ok:
                failures.append({"batch_id": batch_id, "custom_id": cid, "case": case_name,
                                 "profile": profile, "sample_id": sample_id, "reason": fail_reason})
                continue

            final_label = cot_parser._extract_label(text, case.valid_labels)
            mapped_label = case.label_map.get(final_label.strip(), list(case.label_map.values())[-1])
            steps, analysis, final_line = cot_parser._parse_steps_and_final(text)

            true_label = df.loc[sample_id, case.label_col]
            if isinstance(true_label, str):
                true_label = true_label.strip()
            sample_text = df.loc[sample_id, case.input_col]

            per_profile_rows.setdefault(profile, []).append({
                "sample_id": sample_id,
                "text": sample_text,
                "true_label": true_label,
                "pred_label": mapped_label,
                "raw_pred_label": final_label,
                "max_tokens": self.max_tokens,
                "tokens_used": usage.get("total_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency": None,
                "strategy": self.strategy,
            })
            per_profile_reasoning.setdefault(profile, []).append({
                "sample_id": sample_id,
                "raw_response": text,
                "parsed_steps": steps,
                "analysis": analysis,
                "final_line": final_line,
                "final_label": final_label,
                "mapped_label": mapped_label,
            })

        for profile, rows in per_profile_rows.items():
            csv_path, reasoning_path = self.output_paths(case_name, profile, manifest_meta_role(self.manifests_dir, batch_id))
            # Note: we read role from manifest to ensure correct folder
            self._merge_rows_csv(csv_path, rows)
            self._merge_reasoning_json(reasoning_path, per_profile_reasoning[profile])
            print(f"[MERGED] {len(rows)} rows → {csv_path}")
            print(f"[MERGED] reasoning → {reasoning_path}")

        return pd.DataFrame(failures)

    # --------------------------------- Failures ----------------------------------------

    @staticmethod
    def _parse_cid(custom_id: str):
        """custom_id format: {case}:{profile}:sample_{idx}"""
        case = profile = None
        sample_id = None
        try:
            case, profile, sample_part = custom_id.split(":")
            sample_id = int(sample_part.replace("sample_", ""))
        except Exception:
            pass
        return case, profile, sample_id

    def get_failed_samples_from_batch(self, batch_id: str) -> pd.DataFrame:
        """
        Returns a DataFrame of failures by reading error_file_id (and rare HTTP errors in output).
        Columns: batch_id, custom_id, case, profile, sample_id, reason
        """
        b = self.client.batches.retrieve(batch_id)
        if b.status != "completed":
            print(f"[SKIP] {batch_id} not completed (status={b.status})")
            return pd.DataFrame(columns=["batch_id","custom_id","case","profile","sample_id","reason"])

        failures: List[Dict[str, Any]] = []

        # A) error_file_id (primary)
        err_id = getattr(b, "error_file_id", None)
        if err_id:
            err_text = self.client.files.content(err_id).read().decode("utf-8")
            for line in err_text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec.get("custom_id")
                err = rec.get("error") or {}
                reason = err.get("message") or err.get("code") or "error_file_entry"
                case, profile, sample_id = self._parse_cid(cid or "")
                failures.append({
                    "batch_id": batch_id,
                    "custom_id": cid,
                    "case": case,
                    "profile": profile,
                    "sample_id": sample_id,
                    "reason": reason
                })

        # B) Fallback: scan output for HTTP errors (rare)
        out_id = getattr(b, "output_file_id", None)
        if out_id:
            out_text = self.client.files.content(out_id).read().decode("utf-8")
            for line in out_text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec.get("custom_id")
                resp = rec.get("response") or {}
                status = resp.get("status_code")
                if status and int(status) >= 400:
                    case, profile, sample_id = self._parse_cid(cid or "")
                    body = resp.get("body") or {}
                    msg = (body.get("error") or {}).get("message") or f"http_{status}"
                    failures.append({
                        "batch_id": batch_id,
                        "custom_id": cid,
                        "case": case,
                        "profile": profile,
                        "sample_id": sample_id,
                        "reason": msg
                    })

        df = pd.DataFrame(failures, columns=["batch_id","custom_id","case","profile","sample_id","reason"])
        return df.sort_values(["profile","sample_id"], ignore_index=True)

    def get_failed_samples_from_batches(self, batch_ids: List[str]) -> pd.DataFrame:
        dfs = [self.get_failed_samples_from_batch(bid) for bid in batch_ids]
        out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
            columns=["batch_id","custom_id","case","profile","sample_id","reason"]
        )
        out_path = os.path.join(self.reports_dir, "failed_samples.csv")
        out.to_csv(out_path, index=False)
        print(f"[REPORT] Saved failed samples → {out_path} (n={len(out)})")
        return out

    # ----------------------------- Re-run failed (online) ------------------------------

    def rerun_failed_samples(
        self,
        case,                 # CaseConfig
        df: pd.DataFrame,     # DataFrame for the case
        role: str,
        failures_df: pd.DataFrame,
        profiles: List[str] = None
    ):
        """
        Re-run a small set of failed samples without Batch and merge results.
        Expects failures_df columns: profile, sample_id (integers).
        """
        case_name = case.case_name
        if profiles is not None:
            failures_df = failures_df[failures_df["profile"].isin(profiles)].copy()

        for _, row in failures_df.iterrows():
            profile = row["profile"]
            sample_id = int(row["sample_id"]) if pd.notna(row["sample_id"]) else None
            if not profile or sample_id is None:
                continue

            cot = ChainOfThoughts(
                case=case, client=self.client, model=self.model, max_tokens=self.max_tokens,
                task_definition=None, person_key=profile, role_playing=role, person_set=self.person_set
            )

            text = df.loc[sample_id, case.input_col]
            true_label = df.loc[sample_id, case.label_col]
            if isinstance(true_label, str):
                true_label = true_label.strip()

            try:
                pred_label, metrics = cot.classify_with_strategy(text, strategy=self.strategy)
                mapped_label = case.label_map.get(pred_label.strip(), list(case.label_map.values())[-1])

                csv_path, reasoning_path = self.output_paths(case_name, profile, role)
                self._merge_rows_csv(csv_path, [{
                    "sample_id": sample_id,
                    "text": text,
                    "true_label": true_label,
                    "pred_label": mapped_label,
                    "raw_pred_label": pred_label,
                    "max_tokens": cot.max_tokens,
                    "tokens_used": metrics.get("tokens_used"),
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "completion_tokens": metrics.get("completion_tokens"),
                    "latency": metrics.get("latency"),
                    "strategy": self.strategy,
                }])

                steps, analysis, final_line = cot._parse_steps_and_final(metrics.get("raw_response", "") or "")
                self._merge_reasoning_json(reasoning_path, [{
                    "sample_id": sample_id,
                    "raw_response": metrics.get("raw_response", "") or "",
                    "parsed_steps": steps,
                    "analysis": analysis,
                    "final_line": final_line,
                    "final_label": pred_label,
                    "mapped_label": mapped_label,
                }])

                print(f"[FIXED] {case_name}:{profile}:sample_{sample_id}")
            except Exception as e:
                print(f"[RETRY FAIL] {case_name}:{profile}:sample_{sample_id} -> {e}")


# --------------------------------------------------------------------------------------
# Helpers to read role from manifest (so collected files go to correct folder)
# --------------------------------------------------------------------------------------

def manifest_meta_role(manifests_dir: str, batch_id: str, default: str = "passive") -> str:
    path = os.path.join(manifests_dir, f"{batch_id}.manifest.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return m.get("role", default)
    except Exception:
        return default

def ensure_manifest(runner: RoleplayBatchRunner, batch_id: str, case_name: str, role: str = "passive"):
    path = os.path.join(runner.manifests_dir, f"{batch_id}.manifest.json")
    if os.path.exists(path):
        return path
    manifest = {
        "batch_id": batch_id,
        "jsonl_path": None,
        "case_name": case_name,
        "role": role,
        "model": runner.model,
        "strategy": runner.strategy,
        "model_filename": runner.model_filename,
        "submitted_at": runner._timestamp(),
        "profiles": [],
        "output_base_dir": runner.output_base_dir,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def example1():
    """ 
    Example code showing the sending and retrieving of batches based on the non-completed role-playing.
    """
    from datasets import load_dataset
    from data_loader import load_mgsd_dataset, load_mentalmanip_dataset
    from cases.stereotypes_case import stereotypes_case
    from cases.manipulation_case import manipulation_case
    

    dataset = load_dataset("wu981526092/MGSD")
 
    data = dataset['train']
    df = data.to_pandas()


    dataset_2 = load_dataset("audreyeleven/MentalManip", "mentalmanip_maj")
    data_2 = dataset_2["train"]
    df_2 = data_2.to_pandas()

    sample_sizes_mgsd = {
        'stereotype': 250,
        'unrelated': 250,
    }

    df_st, _ = load_mgsd_dataset(
        df, 
        sample_sizes_mgsd, 
        {'stereotype': 5, 'unrelated': 5},
        random_state=42,
        random_state_examples=0,
    )



    sample_sizes_manip = {1: 250, 0: 250}
    max_len_examples = 1000

    df_man, _ = load_mentalmanip_dataset(
        df_2, 
        sample_sizes_manip, 
        {1: 5, 0: 5}, 
        max_len_examples,
        random_state=42,
        random_state_examples=0,
    )


    model="gpt-4.1-mini"
    model_filename="openai_4.1_mini"

    runner = RoleplayBatchRunner(
        model=model,
        model_filename=model_filename,
        strategy="optimized",
        max_tokens=700,
        output_base_dir=f"results/{model_filename}/cot/role_playing_ethnics", 
        )

    subs = runner.submit_batches(
        case=stereotypes_case,
        df=df_st,
        profiles=[f"profile{i}" for i in range(1, 61)],
        role="passive",
        chunk_size=3000,
        )

    runner.collect_one_batch_merged(stereotypes_case, df_st, subs[0]["batch_id"])

    fail_df = runner.get_failed_samples_from_batches([s["batch_id"] for s in subs])

    runner.rerun_failed_samples(stereotypes_case, df_st, role="passive", failures_df=fail_df)



def example2():
    """ 
    2nd example code showing how to just retrieve already sent batches based on batch id,           
    and re-run NOT IN BATCH the failed samples.
    """
    from datasets import load_dataset
    from data_loader import load_mgsd_dataset, load_mentalmanip_dataset
    from roleplay_cot_batch import RoleplayBatchRunner, manifest_meta_role, ensure_manifest
    from cases.stereotypes_case import stereotypes_case
    from cases.manipulation_case import manipulation_case

    dataset = load_dataset("wu981526092/MGSD")
 
    data = dataset['train']
    df = data.to_pandas()


    dataset_2 = load_dataset("audreyeleven/MentalManip", "mentalmanip_maj")
    data_2 = dataset_2["train"]
    df_2 = data_2.to_pandas()

    sample_sizes_mgsd = {
        'stereotype': 250,
        'unrelated': 250,
    }

    df_st, _ = load_mgsd_dataset(
        df, 
        sample_sizes_mgsd, 
        {'stereotype': 5, 'unrelated': 5},
        random_state=42,
        random_state_examples=0,
    )



    sample_sizes_manip = {1: 250, 0: 250}
    max_len_examples = 1000

    df_man, _ = load_mentalmanip_dataset(
        df_2, 
        sample_sizes_manip, 
        {1: 5, 0: 5}, 
        max_len_examples,
        random_state=42,
        random_state_examples=0,
    )


    model="gpt-4.1-mini"
    model_filename="openai_4.1_mini"

    runner = RoleplayBatchRunner(
        model=model,
        model_filename=model_filename,
        strategy="optimized",
        max_tokens=700,
        output_base_dir=f"results/{model_filename}/cot/role_playing_ethnics",
    )

    CASE_MAP = {
        "stereotype":   (stereotypes_case,   df_st),
        "manipulation": (manipulation_case,  df_man),     
    }

    batch_ids = [
        "",          # batch id
        ""
    ]

    for bid in batch_ids:
        b = runner.client.batches.retrieve(bid)
        meta = getattr(b, "metadata", {}) or {}
        case_name = meta.get("case")        
        role = meta.get("role", "passive")

        if not case_name:
            raise ValueError(f"Batch {bid} has no 'metadata.case' — set it manually below.")

        case_obj, df = CASE_MAP[case_name]
        ensure_manifest(runner, bid, case_name, role)
        runner.collect_one_batch_merged(case_obj, df, bid)


    fail_df = runner.get_failed_samples_from_batches(batch_ids)
    fail_df.head()

    fail_st  = fail_df[fail_df["case"] == "stereotype"].dropna(subset=["profile","sample_id"])
    fail_man = fail_df[fail_df["case"] == "manipulation"].dropna(subset=["profile","sample_id"])

    runner.rerun_failed_samples(stereotypes_case,  df_st,        role="passive", failures_df=fail_st)
    runner.rerun_failed_samples(manipulation_case, df_man, role="passive", failures_df=fail_man)
