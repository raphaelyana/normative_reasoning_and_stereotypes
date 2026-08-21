from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Literal
import openai
from utils.call_llm import call_llm
from profiles.schema import PersonSet
from profiles.profile_message import make_system_message
from profiles.profile_sets import PERSON_ETHNICS
from cases.cases_config import CaseConfig

class PathSelectionJudge:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        person_key: Optional[str] = None,
        role_playing: Literal["passive", "active", "none"] = "none",
        person_set: Optional[PersonSet] = None,
        provider: Optional[str] = None,
    ) -> None:

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.person_key = person_key
        self.role_playing = role_playing
        self.person_set = person_set if person_set is not None else PERSON_ETHNICS
        self.provider = provider

    def _system_message(self, case_name: str) -> str:

        if self.role_playing == "active" and self.person_key:
            return (
                f"You are a path selection arbiter for {case_name}. "
                f"Decide as if you were the following person:\n{self.person_set.seeds[self.person_key]}"
            )
        
        if self.role_playing == "passive" and self.person_key:
            return make_system_message(case_name=case_name, person_key=self.person_key, person_set=self.person_set)["content"]
        
        return f"You are a careful arbiter for {case_name}. Choose one best path and its final label."


    def _serialize_paths(self, paths: List[List[Any]]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for p in paths:
            payload.append({
                "path_id": "->".join(getattr(t, "id", "") for t in p),
                "steps": [
                    {
                        "id": getattr(t, "id", ""),
                        "content": getattr(t, "content", "")[:240],
                        "label": getattr(t, "verdict", None),
                    } for t in p
                ]
            })
        return payload

    def choose_best_path(
        self,
        case: CaseConfig,
        paths: List[List[Any]],
        max_paths: int = 12,
    ) -> Dict[str, Any]:
        if not paths:
            return {"path_id": None, "label": None, "raw": None}

        paths_trimmed = paths[:max_paths]
        payload = self._serialize_paths(paths_trimmed)

        valid_labels = ", ".join(case.valid_labels)
        user_prompt = (
            "You are given several candidate reasoning paths (root→leaf). "
            "Each leaf carries a proposed final label.\n\n"
            f"Valid labels: {valid_labels}\n"
            "Choose the SINGLE best path whose internal reasoning is most coherent and best supports its final label.\n\n"
            "CANDIDATE_PATHS_JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            'Return STRICT JSON ONLY: {"path_id": "<id>", "label": "<label>"}'
        )

        resp = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_prompt,
            system_message=self._system_message(case.case_name),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            provider=self.provider,
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
            pid = data.get("path_id")
            lab = data.get("label")
            return {"path_id": pid, "label": lab, "raw": raw}
        except Exception:
            return {"path_id": None, "label": None, "raw": raw}


    def choose_best_from_explorer(
        self,
        case: CaseConfig,
        explorer_paths: List[List[Any]],
        max_paths: int = 12,
    ) -> Dict[str, Any]:
        
        return self.choose_best_path(case=case, paths=explorer_paths, max_paths=max_paths)
