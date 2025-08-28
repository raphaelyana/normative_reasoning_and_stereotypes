# roleplay_cot_batch.py
# End-to-end batching for Role-Playing CoT with OpenAI Batch API.
# - Prompts are EXACT (via ChainOfThoughts._build_optimized_cot_prompt).
# - Per-request system message (per profile, active/passive/none).
# - Output directory is configurable.
# - Works with ANY case: pass a CaseConfig and its DataFrame.
# - Utilities to collect successes, list failures (error_file_id), and re-run failures online.

import os, json, datetime, re
from typing import List, Dict, Any, Optional, Tuple, Callable

import pandas as pd
from dotenv import load_dotenv
import openai

# Project imports
from chain_of_thought import ChainOfThoughts
from profiles.profile_message import make_system_message
from profiles.profile_sets import PERSON_ETHNICS
from cases.cases_config import CaseConfig


# --------------------------- small utils ---------------------------

def _timestamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _slug_for_id(s: str) -> str:
    """Colon-safe short tag for custom_id; spaces/punct -> '-'."""
    s = (s or "").strip().lower()
    s = s.replace(":", "_")
    return re.sub(r'[^a-z0-9]+', '-', s).strip("-") or "case"

def _parse_cid(custom_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    Supports both:
      4-part: caseTag:profile:caseType:sample_{idx}
      3-part: caseTag:profile:sample_{idx}
    """
    try:
        parts = custom_id.split(":")
        if len(parts) == 4:
            case_tag, profile, case_type, sample_part = parts
        elif len(parts) == 3:
            case_tag, profile, sample_part = parts
            case_type = None
        else:
            return None, None, None, None
        sample_id = int(sample_part.replace("sample_", ""))
        return case_tag, profile, (None if case_type in (None, "na") else case_type), sample_id
    except Exception:
        return None, None, None, None


# --------------------------- runner ---------------------------

class RoleplayBatchRunner:
    """
    CaseConfig-aware batch runner for role-playing CoT.
    - Prompts are built via ChainOfThoughts._build_optimized_cot_prompt(text, case_type=?).
    - Per-request system persona (active/passive/none).
    - custom_id uses a colon-safe short tag (slug of case.case_name) or your override.
    - Collect merges per-profile CSV + JSON; failures are surfaced with exact custom_id.
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
        load_dotenv()
        self.client = client or openai.OpenAI(api_key=os.getenv(api_key_env))
        if not self.client:
            raise ValueError("OpenAI client not initialized; set API key in env.")

        self.model = model
        self.model_filename = model_filename
        self.strategy = strategy
        self.max_tokens = int(max_tokens)

        self.output_base_dir = output_base_dir or f"results/{self.model_filename}/cot/role_playing"
        self.manifests_dir = manifests_dir
        self.reports_dir = reports_dir
        self.person_set = person_set

        _ensure_dir(self.manifests_dir)
        _ensure_dir(self.reports_dir)

    # -------------------- paths --------------------

    def output_paths(self, case_name: str, profile: str, role: str) -> Tuple[str, str]:
        base = os.path.join(self.output_base_dir, f"{profile}_{role}")
        _ensure_dir(base)
        # Pretty filenames but keep full case_name; only custom_id uses slug
        csv_path = os.path.join(base, f"results_{case_name.lower()}_{self.strategy}_cot.csv")
        reasoning_path = os.path.join(base, f"reasoning_{case_name.lower()}_{self.strategy}_cot.json")
        return csv_path, reasoning_path

    def is_profile_done(self, case_name: str, profile: str, role: str) -> bool:
        csv_path, _ = self.output_paths(case_name, profile, role)
        return os.path.exists(csv_path)

    def filter_profiles_not_done(self, case_name: str, profiles: List[str], role: str) -> List[str]:
        return [p for p in profiles if not self.is_profile_done(case_name, p, role)]

    # -------------------- prompt builders --------------------

    def _build_user_prompt(self, case: CaseConfig, text: str, case_type: Optional[str]) -> str:
        tmp = ChainOfThoughts(
            case=case, client=None, model=self.model, max_tokens=self.max_tokens,
            task_definition=None, person_key=None, role_playing="none", person_set=self.person_set
        )
        return tmp._build_optimized_cot_prompt(text, case_type=case_type)

    def _system_message(self, case_name: str, profile: Optional[str], role: str) -> str:
        if role in ("active", "passive") and profile:
            sm = make_system_message(case_name=case_name, person_key=profile, person_set=self.person_set)
            return sm["content"] if isinstance(sm, dict) and "content" in sm else str(sm)
        return f"You are an expert classifier for {case_name}. Think carefully and follow the prompt."

    # -------------------- case_type inference --------------------

    def _infer_case_type_for_row(
        self,
        row: pd.Series,
        case: CaseConfig,
        *,
        case_type_col: Optional[str],
        case_type_map: Optional[Dict[Any, str]],
        case_type_default: Optional[str],
        case_type_from_row: Optional[Callable[[pd.Series, CaseConfig], Optional[str]]],
    ) -> Optional[str]:
        # 1) explicit callable wins
        if case_type_from_row:
            ct = case_type_from_row(row, case)
            if ct:
                return ct

        # 2) explicit column if provided
        ct = None
        if case_type_col and case_type_col in row.index:
            raw = row[case_type_col]
            ct = None if pd.isna(raw) else str(raw)

        # 3) else try the CaseConfig.category_cols in order
        if not ct and getattr(case, "category_cols", None):
            for col in case.category_cols:
                if col in row.index and pd.notna(row[col]):
                    ct = str(row[col])
                    break

        # 4) normalize via map if supplied
        if ct and case_type_map:
            ct = case_type_map.get(ct, ct)

        # 5) final fallback to case.case_type or provided default
        if not ct:
            ct = case_type_default or getattr(case, "case_type", None)

        return ct

    # -------------------- submit --------------------

    def submit_batches(
        self,
        case: CaseConfig,
        df: pd.DataFrame,
        profiles: List[str],
        role: str = "passive",
        chunk_size: Optional[int] = None,
        skip_completed: bool = True,
        completion_window: str = "24h",
        metadata_extra: Optional[Dict[str, str]] = None,
        *,
        # case_type controls
        case_type_col: Optional[str] = None,
        case_type_map: Optional[Dict[Any, str]] = None,
        case_type_default: Optional[str] = None,
        case_type_from_row: Optional[Callable[[pd.Series, CaseConfig], Optional[str]]] = None,
        # custom_id tag override
        custom_id_case_tag: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build JSONL & submit Batch jobs. Returns [{"batch_id":..., "jsonl_path":...}, ...].

        CaseConfig used:
          - case.case_name, case.input_col, case.label_col
          - case.case_type (default), case.category_cols (optional list)
          - case.valid_labels, case.label_map used later when collecting
        """
        # sanity checks
        for col in [case.input_col]:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col!r} for case {case.case_name!r}")

        case_name = case.case_name
        case_tag = _slug_for_id(custom_id_case_tag or case_name)  # short, colon-safe

        if skip_completed:
            profiles = self.filter_profiles_not_done(case_name, profiles, role)
        if not profiles:
            print(f"Nothing to submit for '{case_name}' (all selected profiles already done).")
            return []

        print(f"Submitting {len(profiles)} profile(s) for case='{case_name}' role='{role}' model='{self.model}'.")

        req_lines: List[dict] = []
        for profile in profiles:
            sys_msg = self._system_message(case_name, profile, role)
            for idx, row in df.iterrows():
                ctype = self._infer_case_type_for_row(
                    row, case,
                    case_type_col=case_type_col,
                    case_type_map=case_type_map,
                    case_type_default=case_type_default,
                    case_type_from_row=case_type_from_row,
                )
                user_prompt = self._build_user_prompt(case, row[case.input_col], case_type=ctype)
                ctype_tag = (ctype or "na").replace(":", "_")
                custom_id = f"{case_tag}:{profile}:{ctype_tag}:sample_{int(idx)}"
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

        # chunk & submit
        chunk_size = chunk_size or len(req_lines)
        chunks = [req_lines[i:i+chunk_size] for i in range(0, len(req_lines), chunk_size)]

        submissions = []
        for ci, chunk in enumerate(chunks, start=1):
            jsonl_path = f"batch_jobs/{case_tag}_{role}_{_timestamp()}_{ci:03d}.jsonl"
            _ensure_dir(os.path.dirname(jsonl_path))
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for line in chunk:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            file_obj = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
            meta = {"case": case_tag, "role": role, "chunk": str(ci)}  # store tag in metadata
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
                "case_name": case_name,                  # human name
                "case_tag": case_tag,                    # tag used in custom_id
                "role": role,
                "model": self.model,
                "strategy": self.strategy,
                "model_filename": self.model_filename,
                "submitted_at": _timestamp(),
                "profiles": profiles,
                "output_base_dir": self.output_base_dir,
                "case_type_col": case_type_col,
                "case_type_default": case_type_default,
            }
            with open(os.path.join(self.manifests_dir, f"{batch.id}.manifest.json"), "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2, ensure_ascii=False)

            submissions.append({"batch_id": batch.id, "jsonl_path": jsonl_path})

        return submissions

    # -------------------- collect --------------------

    @staticmethod
    def _parse_success_choice_text(item: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any], str]:
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
        _ensure_dir(os.path.dirname(csv_path))
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
        _ensure_dir(os.path.dirname(reasoning_path))
        with open(reasoning_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    def collect_one_batch_merged(self, case: CaseConfig, df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        """
        Merge successes per profile; return a small DataFrame of output-file failures (rare).
        """
        manifest_path = os.path.join(self.manifests_dir, f"{batch_id}.manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Manifest not found for batch {batch_id} at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        role = manifest.get("role", "passive")
        case_name = manifest.get("case_name", case.case_name)

        b = self.client.batches.retrieve(batch_id)
        if b.status != "completed":
            print(f"Batch {batch_id} not completed yet (status={b.status}). Skipping.")
            return pd.DataFrame(columns=["batch_id","custom_id","case","profile","sample_id","case_type","reason"])

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
            case_tag, profile, case_type, sample_id = _parse_cid(cid)
            if profile is None or sample_id is None:
                failures.append({
                    "batch_id": batch_id, "custom_id": cid, "case": case_tag,
                    "profile": profile, "sample_id": sample_id, "case_type": case_type, "reason": "bad_custom_id"
                })
                continue

            ok, text, usage, fail_reason = self._parse_success_choice_text(item)
            if not ok:
                failures.append({
                    "batch_id": batch_id, "custom_id": cid, "case": case_tag,
                    "profile": profile, "sample_id": sample_id, "case_type": case_type, "reason": fail_reason
                })
                continue

            # Parse with correct case_type (falls back to case.case_type if None)
            final_label = cot_parser._extract_label(text, case.valid_labels,
                                                   case_type=case_type or getattr(case, "case_type", None))
            mapped_label = case.label_map.get(str(final_label).strip(), list(case.label_map.values())[-1])
            steps, analysis, final_line = cot_parser._parse_steps_and_final(text)

            true_label = df.loc[sample_id, case.label_col] if case.label_col in df.columns else None
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
            csv_path, reasoning_path = self.output_paths(case_name, profile, role)
            self._merge_rows_csv(csv_path, rows)
            self._merge_reasoning_json(reasoning_path, per_profile_reasoning[profile])
            print(f"[MERGED] {len(rows)} rows → {csv_path}")
            print(f"[MERGED] reasoning → {reasoning_path}")

        return pd.DataFrame(failures)

    # -------------------- failures --------------------

    def get_failed_samples_from_batch(self, batch_id: str) -> pd.DataFrame:
        """
        Read error_file_id (and rare HTTP errors in output) and return failures WITH case_type.
        """
        b = self.client.batches.retrieve(batch_id)
        if b.status != "completed":
            print(f"[SKIP] {batch_id} not completed (status={b.status})")
            return pd.DataFrame(columns=["batch_id","custom_id","case","profile","sample_id","case_type","reason"])

        failures: List[Dict[str, Any]] = []

        # primary: error_file_id
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
                case_tag, profile, case_type, sample_id = _parse_cid(cid or "")
                failures.append({
                    "batch_id": batch_id,
                    "custom_id": cid,
                    "case": case_tag,
                    "profile": profile,
                    "sample_id": sample_id,
                    "case_type": case_type,
                    "reason": reason
                })

        # fallback: scan output for http errors
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
                    case_tag, profile, case_type, sample_id = _parse_cid(cid or "")
                    body = resp.get("body") or {}
                    msg = (body.get("error") or {}).get("message") or f"http_{status}"
                    failures.append({
                        "batch_id": batch_id,
                        "custom_id": cid,
                        "case": case_tag,
                        "profile": profile,
                        "sample_id": sample_id,
                        "case_type": case_type,
                        "reason": msg
                    })

        df = pd.DataFrame(failures, columns=["batch_id","custom_id","case","profile","sample_id","case_type","reason"])
        return df.sort_values(["profile","sample_id"], ignore_index=True)

    def get_failed_samples_from_batches(self, batch_ids: List[str]) -> pd.DataFrame:
        dfs = [self.get_failed_samples_from_batch(bid) for bid in batch_ids]
        out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
            columns=["batch_id","custom_id","case","profile","sample_id","case_type","reason"]
        )
        out_path = os.path.join(self.reports_dir, "failed_samples.csv")
        out.to_csv(out_path, index=False)
        print(f"[REPORT] Saved failed samples → {out_path} (n={len(out)})")
        return out

    # -------------------- re-run failed (online) --------------------

    def rerun_failed_samples(
        self,
        case: CaseConfig,
        df: pd.DataFrame,
        role: str,
        failures_df: pd.DataFrame,
        profiles: List[str] = None
    ):
        """
        Re-run small failure sets with correct per-row case_type if present (column 'case_type').
        """
        case_name = case.case_name
        if profiles is not None:
            failures_df = failures_df[failures_df["profile"].isin(profiles)].copy()

        for _, row in failures_df.iterrows():
            profile = row.get("profile")
            if not profile:
                continue
            sample_id = row.get("sample_id")
            if pd.isna(sample_id):
                continue
            sample_id = int(sample_id)
            case_type = row.get("case_type", None)
            if isinstance(case_type, float) and pd.isna(case_type):
                case_type = None

            cot = ChainOfThoughts(
                case=case, client=self.client, model=self.model, max_tokens=self.max_tokens,
                task_definition=None, person_key=profile, role_playing=role, person_set=self.person_set
            )

            text = df.loc[sample_id, case.input_col]
            true_label = df.loc[sample_id, case.label_col] if case.label_col in df.columns else None
            if isinstance(true_label, str):
                true_label = true_label.strip()

            try:
                pred_label, metrics = cot.classify_with_strategy(text, strategy=self.strategy, case_type=case_type)
                mapped_label = case.label_map.get(str(pred_label).strip(), list(case.label_map.values())[-1])

                csv_path, reasoning_path = self.output_paths(case_name, profile, role)
                self._merge_rows_csv(csv_path, [{
                    "sample_id": sample_id,
                    "text": text,
                    "true_label": true_label,
                    "pred_label": mapped_label,
                    "raw_pred_label": pred_label,
                    "max_tokens": self.max_tokens,
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

                print(f"[FIXED] {case_name}:{profile}:sample_{sample_id} (case_type={case_type})")
            except Exception as e:
                print(f"[RETRY FAIL] {case_name}:{profile}:sample_{sample_id} -> {e}")


# --------------------------- manifest helpers ---------------------------

def manifest_meta_role(manifests_dir: str, batch_id: str, default: str = "passive") -> str:
    path = os.path.join(manifests_dir, f"{batch_id}.manifest.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return m.get("role", default)
    except Exception:
        return default

def ensure_manifest(runner: RoleplayBatchRunner, batch_id: str, case_name: str, role: str = "passive"):
    """If you submitted elsewhere, create a minimal manifest so collect() knows the role/output path."""
    path = os.path.join(runner.manifests_dir, f"{batch_id}.manifest.json")
    if os.path.exists(path):
        return path
    manifest = {
        "batch_id": batch_id,
        "jsonl_path": None,
        "case_name": case_name,
        "case_tag": _slug_for_id(case_name),
        "role": role,
        "model": runner.model,
        "strategy": runner.strategy,
        "model_filename": runner.model_filename,
        "submitted_at": _timestamp(),
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
