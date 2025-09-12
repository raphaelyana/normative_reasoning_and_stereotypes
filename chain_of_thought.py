import re
import time
import openai
import pandas as pd
from utils.call_llm import call_llm
from pydantic import BaseModel
from cases.cases_config import CaseConfig
from profiles.profile_message import make_system_message
from profiles.profile_sets import PERSON_ETHNICS
from profiles.schema import PersonSet
from typing import Literal, List, Optional, Dict, Any



DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}



class SampleStats(BaseModel):
    tokens_used: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency: float



class ChainOfThoughts:
    def __init__(
        self,
        case: CaseConfig,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 800,
        task_definition: Optional[str] = None,
        person_key: Optional[str] = None,
        role_playing: Literal["active", "passive", "none"] = "none",
        person_set: Optional[PersonSet] = None,
        provider: Optional[str] = None,
    ):
        self.case = case
        self.case_name = case.case_name
        self.task_definition = task_definition
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.person_key = person_key
        self.role_playing = role_playing
        self.person_set = PERSON_ETHNICS if person_set is None else person_set
        self.provider = provider

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0


    def _build_optimized_cot_prompt(
        self,
        input_text: str,
        case_type: Optional[Literal["normative", "math_logic"]] = None
    ) -> str:
        """Structured reasoning prompt, switched by case_type."""
        ctype = case_type or getattr(self.case, "case_type", "normative")

        focus = getattr(self.case, "cot_focus", "the target phenomenon")
        adversarial = getattr(self.case, "cot_adversarial", "subtle confounds")
        rules = "\n".join(f"- {r}" for r in getattr(self.case, "label_rules", []))
        label_list = ", ".join(str(v) for v in self.case.valid_labels)

        prompt = f"""Analyze this text for {self.case_name}:

\"\"\"{input_text}\"\"\"

Rules:
{rules}
"""
        if ctype == "normative":
            prompt += f"""
Step 1 - Evidence: What evidence of {focus} do you see?
Step 2 - Adversarial Check: Could {adversarial} be happening? Consider subtle implications.
Step 3 - Decision: State the final decision with a brief justification.

Final Classification (choose EXACTLY one of: {label_list}):
"""
        else:
            prompt += f"""
Step 1 — Understand: Extract key quantities, definitions, and givens.
Step 2 — Derivation: Compute or deduce the needed result succinctly.
Step 3 — Elimination: For A, B, C, D, eliminate three options with one explicit reason each (if applicable).
Step 4 — Decision: State the single best option.

Answer: <LABEL>
(Choose EXACTLY one of: {label_list})
"""
        return prompt


    def classify_with_strategy(self, text: str, strategy: Literal["optimized"] = "optimized",
                               case_type: Optional[str] = None):
        
        if strategy != "optimized":
            raise ValueError(f"Unknown strategy: {strategy}. Use 'optimized'.")
        
        user_prompt = self._build_optimized_cot_prompt(text, case_type=case_type)
    
    
        if self.role_playing == "active" and self.person_key:
            system_message = {
                "role": "system",
                "content": f"You are a {self.case_name} classifier. Answer as if you were the following person:\n{self.person_set.seeds[self.person_key]}"
            }
        elif self.role_playing == "passive" and self.person_key:
            system_message = make_system_message(
                case_name=self.case_name,
                person_key=self.person_key,
                person_set=self.person_set
            )
        else:
            system_message = {
                "role": "system",
                "content": f"You are an expert classifier for {self.case_name}. Think carefully and follow the prompt."
            }
    
        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_prompt,
            system_message=system_message["content"],
            max_tokens=self.max_tokens,
            provider=self.provider,
        )
        elapsed = time.time() - start_time
    
    
        self.total_latency += elapsed
        self.total_calls += 1
        if getattr(response, "usage", None):
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
    
        stats = SampleStats(
            tokens_used=getattr(getattr(response, "usage", None), "total_tokens", None),
            prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", None),
            completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", None),
            latency=elapsed
        )
    
        response_text = response.choices[0].message.content.strip()
    
    
        final_label = self._extract_label(response_text, self.case.valid_labels, case_type=case_type or getattr(self.case, "case_type", None))
    
        return final_label, {
            "tokens_used": stats.tokens_used,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "latency": stats.latency,
            "raw_response": response_text
        }


    def _extract_label(self, response_text: str, valid_labels: List[Any], case_type: Optional[str] = None) -> str:
        """Dispatcher: 'normative' -> normative parser; else -> math/logic parser."""
        ct = (case_type or "normative").lower()
        if ct == "normative":
            return self._extract_label_normative(response_text, valid_labels)
        else:
            return self._extract_label_mathlogic(response_text, valid_labels)


    def _extract_label_normative(self, response_text: str, valid_labels: List[Any]) -> str:
        m = re.search(r'(?:Final\s+Classification|Classification)\s*:?\s*(.+)', response_text, re.IGNORECASE)
        candidate_region = m.group(1).strip() if m else response_text

        escaped = [re.escape(str(lbl)) for lbl in valid_labels]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        m2 = re.search(pattern, candidate_region, re.IGNORECASE)
        if m2:
            picked = m2.group(1).strip().lower()
            for lbl in valid_labels:
                if str(lbl).strip().lower() == picked:
                    return lbl

        lower_resp = response_text.lower()
        for lbl in valid_labels:
            if str(lbl).strip().lower() in lower_resp:
                return lbl

        return valid_labels[0]

    def _extract_label_mathlogic(self, response_text: str, valid_labels: List[str]) -> str:
        """
        Math/Logic parser:
          1) Prefer strict 'Answer: <LABEL>' (or 'Final Answer: <LABEL>'), allows 'Option X' and trailing punctuation.
          2) If labels are A-D, check near the tail for a lone letter.
          3) Else reuse normative fallback over whole text.
        """
        alt = "|".join(re.escape(str(v)) for v in sorted(valid_labels, key=lambda x: len(str(x)), reverse=True))
        rx = rf'(?im)^\s*(?:final\s+answer|answer)\s*[:\-]\s*(?:option\s*)?({alt})\s*[\).:]*\s*$'
        m = re.search(rx, response_text)
        if m:
            canon = self._normalize_to_valid(m.group(1), valid_labels)
            if canon is not None:
                return canon

        labels_upper = [str(v).strip().upper() for v in valid_labels]
        if sorted(labels_upper) == ["A", "B", "C", "D"]:
            tail = response_text[-160:]  # small window at end
            m2 = re.search(r'(?i)\b([ABCD])\b', tail)
            if m2:
                canon = self._normalize_to_valid(m2.group(1).upper(), valid_labels)
                if canon is not None:
                    return canon

        return self._extract_label_normative(response_text, valid_labels)

    def _normalize_to_valid(self, picked: str, valid_labels: List[str]):
        """Return the canonical label from valid_labels that matches picked (case/str tolerant)."""
        p = str(picked).strip().lower()
        for v in valid_labels:
            if str(v).strip().lower() == p:
                return v
        return None
    
    def _parse_steps_and_final(self, resp_text: str):
        steps = []
        for m in re.finditer(r'^\s*Step\s*(\d+)[\s:\-\)]\s*(.*)$', resp_text, flags=re.IGNORECASE | re.MULTILINE):
            try:
                n = int(m.group(1))
            except:
                n = None
            steps.append({"step": n, "content": (m.group(2) or "").strip()})

        analysis = None
        analysis_match = re.search(
            r'Analysis\s*:\s*(.+?)(?=\n(?:Classification|Final\s+Classification|Answer|Final\s+Answer)\b|$)',
            resp_text, flags=re.IGNORECASE | re.DOTALL
        )
        if analysis_match:
            analysis = analysis_match.group(1).strip()

        final_line = None
        m_ans = re.search(r'(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$', resp_text)
        if m_ans:
            final_line = m_ans.group(1).strip()
        else:
            m_fc = re.search(r'(?:Final\s+Classification|Classification)\s*:?\s*(.+)', resp_text, flags=re.IGNORECASE)
            if m_fc:
                final_line = m_fc.group(1).strip()

        return steps, analysis, final_line

    def classify(self, text: str, strategy: Literal["optimized"] = "optimized") -> str:
        label, _metrics = self.classify_with_strategy(text, strategy=strategy)
        return label

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency": self.total_latency,
            "avg_latency_per_call": self.total_latency/max(1, self.total_calls),
            "avg_tokens_per_call": self.total_tokens/max(1, self.total_calls),
        }