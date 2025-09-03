import os, json, datetime, re
from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
from dotenv import load_dotenv
import openai
from cases.cases_config import CaseConfig
from profiles.profile_message import make_system_message
from profiles.profile_sets import PERSON_ETHNICS
from tree_of_thought import TreeOfThought
from tree_of_thought_v2 import TreeOfThoughtExplorer
from tree_of_thought_judge import PathSelectionJudge


def _timestamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _slug_for_id(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace(":", "_")
    return re.sub(r'[^a-z0-9]+', '-', s).strip("-") or "case"

def _parse_cid(custom_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
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


class TreeOfThoughtBatchRunner:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        model_filename: str,
        output_base_dir: str,
        person_set=PERSON_ETHNICS,
        max_tokens: int = 700,
        strategy: str = "tot",
        manifests_dir: str = "batch_jobs/manifests",
        reports_dir: str = "batch_jobs/reports"
    ):
        self.client = client
        self.model = model
        self.model_filename = model_filename
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.output_base_dir = output_base_dir
        self.manifests_dir = manifests_dir
        self.reports_dir = reports_dir
        self.person_set = person_set

        _ensure_dir(manifests_dir)
        _ensure_dir(reports_dir)

    def _system_message(self, case_name: str, profile: Optional[str], role: str) -> str:
        if role in ("active", "passive") and profile:
            return make_system_message(case_name=case_name, person_key=profile, person_set=self.person_set)["content"]
        return f"You are an expert classifier for {case_name}. Think carefully and follow the prompt."

    def _file_tag_for(self, case_name: str) -> str:
        return _slug_for_id(case_name)

    def output_paths(self, case_name: str, profile: str, role: str) -> Tuple[str, str]:
        base = os.path.join(self.output_base_dir, f"{profile}_{role}")
        _ensure_dir(base)
        tag = self._file_tag_for(case_name)
        csv_path = os.path.join(base, f"results_{tag}_{self.strategy}.csv")
        json_path = os.path.join(base, f"tree_{tag}_{self.strategy}.json")
        return csv_path, json_path

    def submit_batches(
        self,
        case: CaseConfig,
        df: pd.DataFrame,
        profiles: List[str],
        role: str = "passive",
        chunk_size: int = 1000,
        case_type_col: Optional[str] = None,
        case_type_map: Optional[Dict[Any, str]] = None,
        case_type_default: Optional[str] = None,
        custom_id_case_tag: Optional[str] = None
    ) -> List[Dict[str, str]]:

        case_name = case.case_name
        case_tag = _slug_for_id(custom_id_case_tag or case_name)

        req_lines = []
        for profile in profiles:
            sys_msg = self._system_message(case_name, profile, role)
            for idx, row in df.iterrows():
                case_type = case_type_default
                if case_type_col and case_type_col in row:
                    raw = row[case_type_col]
                    case_type = None if pd.isna(raw) else str(raw)
                    if case_type_map:
                        case_type = case_type_map.get(case_type, case_type)

                input_text = row[case.input_col]
                ctype_tag = (case_type or "na").replace(":", "_")
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
                            {"role": "user", "content": input_text}
                        ]
                    }
                })

        chunks = [req_lines[i:i+chunk_size] for i in range(0, len(req_lines), chunk_size)]
        submissions = []

        for ci, chunk in enumerate(chunks, start=1):
            jsonl_path = f"batch_jobs/{case_tag}_{role}_{_timestamp()}_{ci:03d}.jsonl"
            _ensure_dir(os.path.dirname(jsonl_path))
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for line in chunk:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            file_obj = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
            batch = self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"case": case_tag, "role": role, "chunk": str(ci)}
            )

            with open(os.path.join(self.manifests_dir, f"{batch.id}.manifest.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "batch_id": batch.id,
                    "jsonl_path": jsonl_path,
                    "case_name": case_name,
                    "case_tag": case_tag,
                    "role": role,
                    "model": self.model,
                    "strategy": self.strategy,
                    "model_filename": self.model_filename,
                    "submitted_at": _timestamp(),
                    "profiles": profiles,
                    "output_base_dir": self.output_base_dir
                }, f, indent=2)

            print(f"[SUBMITTED] batch_id={batch.id}  jsonl={jsonl_path}  requests={len(chunk)}")
            submissions.append({"batch_id": batch.id, "jsonl_path": jsonl_path})

        return submissions

    def collect_one_batch_merged(
        self,
        case: CaseConfig,
        df: pd.DataFrame,
        batch_id: str,
        role: str = "passive",
        solver_kwargs: Optional[dict] = None,
    ):
        manifest_path = os.path.join(self.manifests_dir, f"{batch_id}.manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        case_tag = manifest["case_tag"]
        profiles = manifest["profiles"]

        b = self.client.batches.retrieve(batch_id)
        if b.status != "completed":
            print(f"[SKIP] {batch_id} not completed (status={b.status})")
            return

        raw = self.client.files.content(b.output_file_id).read().decode("utf-8")
        lines = [json.loads(l) for l in raw.splitlines() if l.strip()]

        for item in lines:
            cid = item.get("custom_id")
            ok, content, usage, reason = self._parse_success_choice_text(item)
            if not ok:
                print(f"[FAIL] {cid}: {reason}")
                continue

            case_tag, profile, case_type, sample_id = _parse_cid(cid)
            if sample_id is None or profile is None:
                continue

            input_text = df.loc[sample_id, case.input_col]
            true_label = df.loc[sample_id, case.label_col]

            tot_solver = TreeOfThought(
                case=case,
                client=self.client,
                model=self.model,
                task_definition="",
                examples_df=None,
                **(solver_kwargs or {})
            )
            best_path = tot_solver.solve(initial_prompt=input_text)
            pred_label = tot_solver._get_majority_vote_from_path(best_path)

            csv_path, json_path = self.output_paths(case.case_name, profile, role)
            _ensure_dir(os.path.dirname(csv_path))

            row = {
                "sample_id": sample_id,
                "text": input_text,
                "true_label": true_label,
                "pred_label": case.label_map.get(pred_label, pred_label),
                "max_tokens": self.max_tokens,
                "tokens_used": usage.get("total_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "strategy": self.strategy,
            }

            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                df_out = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
                df_out = df_out.drop_duplicates(subset="sample_id", keep="last")
            else:
                df_out = pd.DataFrame([row])
            df_out.to_csv(csv_path, index=False)

            # Save tree
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    tree_all = json.load(f)
            except:
                tree_all = {}

            tree_all[str(sample_id)] = {
                "tree": tot_solver.get_tree_dict(),
                "best_path": [t.id for t in best_path],
                "pred_label": pred_label,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(tree_all, f, indent=2, ensure_ascii=False)

            print(f"[MERGED] sample_id={sample_id} → {csv_path}")

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


class TreeOfThoughtExplorerBatchRunner:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        model_filename: str,
        output_base_dir: str,
        person_set=PERSON_ETHNICS,
        max_tokens: int = 700,
        strategy: str = "tot_explorer",
        manifests_dir: str = "batch_jobs/manifests",
        reports_dir: str = "batch_jobs/reports"
    ):
        self.client = client
        self.model = model
        self.model_filename = model_filename
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.output_base_dir = output_base_dir
        self.manifests_dir = manifests_dir
        self.reports_dir = reports_dir
        self.person_set = person_set

        _ensure_dir(manifests_dir)
        _ensure_dir(reports_dir)

    def _system_message(self, case_name: str, profile: Optional[str], role: str) -> str:
        if role in ("active", "passive") and profile:
            return make_system_message(case_name=case_name, person_key=profile, person_set=self.person_set)["content"]
        return f"You are an expert classifier for {case_name}. Think carefully and follow the prompt."

    def _file_tag_for(self, case_name: str) -> str:
        return _slug_for_id(case_name)

    def output_paths(self, case_name: str, profile: str, role: str) -> Tuple[str, str]:
        base = os.path.join(self.output_base_dir, f"{profile}_{role}")
        _ensure_dir(base)
        tag = self._file_tag_for(case_name)
        csv_path = os.path.join(base, f"results_{tag}_{self.strategy}.csv")
        json_path = os.path.join(base, f"tree_{tag}_{self.strategy}.json")
        return csv_path, json_path

    def submit_batches(
        self,
        case: CaseConfig,
        df: pd.DataFrame,
        profiles: List[str],
        role: str = "passive",
        chunk_size: int = 1000,
        case_type_col: Optional[str] = None,
        case_type_map: Optional[Dict[Any, str]] = None,
        case_type_default: Optional[str] = None,
        custom_id_case_tag: Optional[str] = None
    ) -> List[Dict[str, str]]:

        case_name = case.case_name
        case_tag = _slug_for_id(custom_id_case_tag or case_name)

        req_lines = []
        for profile in profiles:
            sys_msg = self._system_message(case_name, profile, role)
            for idx, row in df.iterrows():
                case_type = case_type_default
                if case_type_col and case_type_col in row:
                    raw = row[case_type_col]
                    case_type = None if pd.isna(raw) else str(raw)
                    if case_type_map:
                        case_type = case_type_map.get(case_type, case_type)

                input_text = row[case.input_col]
                ctype_tag = (case_type or "na").replace(":", "_")
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
                            {"role": "user", "content": input_text}
                        ]
                    }
                })

        chunks = [req_lines[i:i+chunk_size] for i in range(0, len(req_lines), chunk_size)]
        submissions = []

        for ci, chunk in enumerate(chunks, start=1):
            jsonl_path = f"batch_jobs/{case_tag}_{role}_{_timestamp()}_{ci:03d}.jsonl"
            _ensure_dir(os.path.dirname(jsonl_path))
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for line in chunk:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            file_obj = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
            batch = self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"case": case_tag, "role": role, "chunk": str(ci)}
            )

            with open(os.path.join(self.manifests_dir, f"{batch.id}.manifest.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "batch_id": batch.id,
                    "jsonl_path": jsonl_path,
                    "case_name": case_name,
                    "case_tag": case_tag,
                    "role": role,
                    "model": self.model,
                    "strategy": self.strategy,
                    "model_filename": self.model_filename,
                    "submitted_at": _timestamp(),
                    "profiles": profiles,
                    "output_base_dir": self.output_base_dir
                }, f, indent=2)

            print(f"[SUBMITTED] batch_id={batch.id}  jsonl={jsonl_path}  requests={len(chunk)}")
            submissions.append({"batch_id": batch.id, "jsonl_path": jsonl_path})

        return submissions

    def collect_one_batch_merged(
        self,
        case: CaseConfig,
        df: pd.DataFrame,
        batch_id: str,
        role: str = "passive",
        solver_kwargs: Optional[dict] = None,
    ):
        manifest_path = os.path.join(self.manifests_dir, f"{batch_id}.manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        case_tag = manifest["case_tag"]
        profiles = manifest["profiles"]

        b = self.client.batches.retrieve(batch_id)
        if b.status != "completed":
            print(f"[SKIP] {batch_id} not completed (status={b.status})")
            return

        raw = self.client.files.content(b.output_file_id).read().decode("utf-8")
        lines = [json.loads(l) for l in raw.splitlines() if l.strip()]

        for item in lines:
            cid = item.get("custom_id")
            ok, content, usage, reason = self._parse_success_choice_text(item)
            if not ok:
                print(f"[FAIL] {cid}: {reason}")
                continue

            case_tag, profile, case_type, sample_id = _parse_cid(cid)
            if sample_id is None or profile is None:
                continue

            input_text = df.loc[sample_id, case.input_col]
            true_label = df.loc[sample_id, case.label_col]

            explorer = TreeOfThoughtExplorer(
                case=case,
                client=self.client,
                model=self.model,
                task_definition="",
                examples_df=None,
                **(solver_kwargs or {})
            )
            paths = explorer.solve(initial_prompt=input_text)
            judge = PathSelectionJudge(
                client=self.client,
                model=self.model,
                person_key=profile,
                role_playing=role,
                person_set=self.person_set
            )
            result = judge.choose_best_from_explorer(case, paths)
            pred_label = result["label"]

            best_path_id = result["path_id"]
            best_path = next((p for p in paths if "->".join(t.id for t in p) == best_path_id), None)

            csv_path, json_path = self.output_paths(case.case_name, profile, role)
            _ensure_dir(os.path.dirname(csv_path))

            row = {
                "sample_id": sample_id,
                "text": input_text,
                "true_label": true_label,
                "pred_label": case.label_map.get(pred_label, pred_label),
                "max_tokens": self.max_tokens,
                "tokens_used": usage.get("total_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "strategy": self.strategy,
            }

            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                df_out = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
                df_out = df_out.drop_duplicates(subset="sample_id", keep="last")
            else:
                df_out = pd.DataFrame([row])
            df_out.to_csv(csv_path, index=False)

            # Save tree
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    tree_all = json.load(f)
            except:
                tree_all = {}

            tree_all[str(sample_id)] = {
                "tree": explorer.get_tree_dict(),
                "best_path": [t.id for t in best_path],
                "pred_label": pred_label,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(tree_all, f, indent=2, ensure_ascii=False)

            print(f"[MERGED] sample_id={sample_id} → {csv_path}")

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
    


def run_solver(
    prompt: str,
    profile: str,
    role: str,
    case: CaseConfig,
    client,
    model: str,
    solver_kwargs: Dict = None,
    person_set = None
) -> Tuple[str, Dict]:

    explorer = TreeOfThoughtExplorer(
        case=case,
        client=client,
        model=model,
        task_definition="",
        examples_df=None,
        **(solver_kwargs or {})
    )
    paths = explorer.solve(initial_prompt=prompt)

    judge = PathSelectionJudge(
        client=client,
        model=model,
        person_key=profile,
        role_playing=role,
        person_set=person_set
    )
    result = judge.choose_best_from_explorer(case, paths)

    return result["label"], {
        "tree": explorer.get_tree_dict(),
        "best_path": [t.id for t in result["path"]],
        "path_id": result["path_id"]
    }

