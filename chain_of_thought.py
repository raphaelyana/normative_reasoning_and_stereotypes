import re
import time
import openai
import pandas as pd
from utils.call_llm import call_llm
from pydantic import BaseModel
from cases.cases_config import CaseConfig
from profiles.profile_message import make_system_message
from profiles.profile_dict import PERSON_SEEDS
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
    ):
        self.case = case
        self.case_name = case.case_name
        self.task_definition = task_definition
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.person_key = person_key
        self.role_playing = role_playing

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0


    def _build_optimized_cot_prompt(self, input_text: str) -> str:
        """3-step structured reasoning prompt."""
        task_config = {
            "stereotype": {
                "focus": "ANY generalizations about groups, even subtle ones, implied assumptions, or coded references",
                "adversarial": "someone making it seem factual when it's actually a generalization about people"
            },
            "manipulation": {
                "focus": "emotional pressure, hidden agendas, and coercive tactics",
                "adversarial": "someone using subtle psychological pressure while appearing helpful"
            }
        }
        config = task_config.get(self.case_name.lower(), {
            "focus": "normative violations and ethical concerns",
            "adversarial": "someone trying to appear reasonable while violating norms"
        })
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = ", ".join(self.case.valid_labels)

        prompt = f"""Analyze this text for {self.case_name}:

\"\"\"{input_text}\"\"\"

Rules:
{rules}

Step 1 - Evidence: What evidence of {config['focus']} do you see?
Step 2 - Adversarial Check: Could {config['adversarial']} be happening? Consider subtle implications.
Step 3 - Decision: State the final decision with a brief justification.

Final Classification (choose EXACTLY one of: {label_list}):
"""
        return prompt

    def _build_zero_shot_plus_prompt(self, input_text: str) -> str:
        """Zero-shot with rules and brief reasoning"""
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = ", ".join(self.case.valid_labels)
    
        prompt = f"""Definition of a {self.case_name}: {self.task_definition}

Input: {input_text}

Labeling rules:
{rules}

Provide a brief analysis and your classification:

Analysis: [1-2 sentences explaining your reasoning]
Classification: [Choose from {label_list}]
"""
        return prompt



    def classify_with_strategy(self, text: str, strategy: Literal["optimized"] = "optimized"):
        """
        Run one of the two supported strategies and return (final_label, metrics_dict).
        """
        if strategy == "optimized":
            user_prompt = self._build_optimized_cot_prompt(text)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'optimized' or 'zero_plus'.")


        if self.role_playing == "active" and self.person_key:
            system_message = {
                "role": "system",
                "content": f"You are a {self.case_name} classifier. Answer as if you were the following person:\n{PERSON_SEEDS[self.person_key]}"
            }
        elif self.role_playing == "passive" and self.person_key:
            system_message = make_system_message(
                case_name=self.case_name,
                person_key=self.person_key
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


        final_label = self._extract_label(response_text, self.case.valid_labels)

        return final_label, {
            "tokens_used": stats.tokens_used,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "latency": stats.latency,
            "raw_response": response_text
        }


    def _extract_label(self, response_text: str, valid_labels: List[str]) -> str:
        """
        Try to extract exactly one of the valid labels.
        Priority:
          1) Look after 'Final Classification' / 'Classification' prefix.
          2) Exact-case-insensitive match as a whole token.
          3) Fallback: first substring match (case-insensitive).
          4) Final fallback: the first label in the list.
        """

        prefix_match = re.search(r'(?:Final\s+Classification|Classification)\s*:\s*(.+)', response_text, re.IGNORECASE)
        candidate_region = prefix_match.group(1).strip() if prefix_match else response_text

        escaped = [re.escape(lbl) for lbl in valid_labels]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        m = re.search(pattern, candidate_region, re.IGNORECASE)
        if m:
            matched_lower = m.group(1).lower()
            for lbl in valid_labels:
                if lbl.lower() == matched_lower:
                    return lbl

        lower_resp = response_text.lower()
        for lbl in valid_labels:
            if lbl.lower() in lower_resp:
                return lbl

        return valid_labels[0]
    

    def _parse_steps_and_final(self, resp_text: str):
        steps = []
        for m in re.finditer(r'^\s*Step\s*(\d+)[\s:\-\)]\s*(.*)$', resp_text, flags=re.IGNORECASE | re.MULTILINE):
            try:
                n = int(m.group(1))
            except:
                n = None
            steps.append({"step": n, "content": m.group(2).strip() if m.group(2) else ""})
        
        analysis = None
        analysis_match = re.search(r'Analysis\s*:\s*(.+?)(?=\nClassification|$)', resp_text, flags=re.IGNORECASE | re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        
        final_line = None
        m = re.search(r'(?:Final\s+Classification|Classification)\s*:\s*(.+)', resp_text, flags=re.IGNORECASE)
        if m:
            final_line = m.group(1).strip()
        
        return steps, analysis, final_line


    def classify(self, text: str, strategy: Literal["optimized", "zero_plus"] = "optimized") -> str:
        """Return only the label."""
        label, _metrics = self.classify_with_strategy(text, strategy=strategy)
        return label

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency": self.total_latency,
            "avg_latency_per_call": self.total_latency / max(1, self.total_calls),
            "avg_tokens_per_call": self.total_tokens / max(1, self.total_calls),
        }