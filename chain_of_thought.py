import re
import json
import time
import openai
import pandas as pd
from utils.utils import call_llm
from pydantic import BaseModel
from dataclasses import dataclass
from cases.cases_config import CaseConfig
from profiles.profile_message import make_system_message
from profiles.profile_sets import PERSON_ETHNICS, PERSON_SYSTEMATIC
from profiles.profile_dict import PERSON_SEEDS, PERSON_SEEDS_ETHNICS, PERSON_SEEDS_SYSTEMATIC
from typing import Literal, List, Optional, Dict, Any
from collections import defaultdict

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

class SampleStats(BaseModel):
    tokens_used: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency: float

@dataclass
class ReasoningStep:
    step_number: int
    content: str

class CoTResult(BaseModel):
    reasoning_steps: List[ReasoningStep]
    final_reasoning: str
    final_label: str
    confidence: Optional[float] = None
    stats: Optional[SampleStats] = None
    person_key: Optional[str] = None
    role_playing: Optional[Literal["active", "passive", "none"]] = None

class ChainOfThoughts:
    def __init__(
        self,
        case: CaseConfig,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 800,
        task_definition: Optional[str] = None,
        num_reasoning_steps: int = 3,
        reasoning_budget: Optional[dict] = None,
        person_key: Optional[str] = None,
        role_playing: Literal["active", "passive", "none"] = "none",
        examples_df: Optional[pd.DataFrame] = None
    ):
        self.case = case
        self.case_name = case.case_name
        self.task_definition = task_definition
        self.num_reasoning_steps = num_reasoning_steps

        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.reasoning_budget = reasoning_budget
        self.person_key = person_key
        self.role_playing = role_playing
        self.examples_df = examples_df

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

    def _select_formatted_examples(self, 
                                   subject: Optional[str] = None) -> List[str]:
            
            # subject needs to be defined only if the dataset includes different subjects
            # requiring separate examples (example: MMLU dataset)
            label_col = self.case.label_col
            template = self.case.example_template_fewshots
            df = self.examples_df

            if subject and "subject" in df.columns:
                df = df[df["subject"] == subject]
        
            label_to_examples = defaultdict(list)
            for _, row in df.iterrows():
                label_to_examples[row[label_col]].append(template(row))
        
            selected = []
            for _, examples in label_to_examples.items():
                selected.extend(examples[:3])
            return selected


    def _build_cot_prompt(self, input_text: str, subject: Optional[str]=None) -> str:
        examples_str = "\n\n".join(self._select_formatted_examples(subject))
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = self.case.valid_labels

        examples_section = ""
        if self.examples_df is not None:
            examples_section = f"""
Examples:
{examples_str}
"""
        step_templates = [
            ("Initial Observation", [
                "What do you observe in this text?",
                "What are the key elements or patterns?"
            ]),
            ("Feature Analysis", [
                f"What specific features are relevant to {self.case_name} detection?",
                "How do these features align with or contradict the labeling rules?"
            ]),
            ("Rule Application", [
                "Apply each labeling rule to this specific case.",
                "Explain how the text satisfies or violates each relevant rule."
            ]),
            ("Interpretation and Context", [
                "Does the text reinforce or challenge generalized beliefs?",
                "How might it be interpreted in terms of social bias, fairness, or group identity?"
            ]),
        ]

        example_json = {
            "reasoning_steps": [
                {"step_number": 1, "content": "Your reasoning step here"},
                {"step_number": 2, "content": "Another step"},
            ],
            "final_reasoning": "A short summary of your reasoning",
            "final_label": "stereotype"
        }

        reasoning_steps_section = ""
        for i in range(min(self.num_reasoning_steps, len(step_templates))):
            step_title, questions = step_templates[i]
            reasoning_steps_section += f"\nStep {i+1} - {step_title}:\n"
            reasoning_steps_section += "\n".join(f"- {q}" for q in questions) + "\n"

        prompt = f"""Definition of a {self.case_name}: {self.task_definition}

{examples_section}
Labeling rules:
{rules}

Now analyze the following input step by step:

Input: \"{input_text}\"

Please follow this chain of reasoning format:
{reasoning_steps_section}

Final Reasoning:
- Synthesize your analysis from all steps
- Provide your final judgment

Final Label: [Choose from: {label_list}]

Respond **only** with a JSON object in the following format:
{json.dumps(example_json, indent=2)}"""

        return prompt

    def _parse_cot_response(self, response_text: str, stats: SampleStats) -> CoTResult:
        try:
            parsed = CoTResult.parse_raw(response_text)
            parsed.stats = stats
            parsed.person_key = self.person_key
            parsed.role_playing = self.role_playing
            return parsed
        except Exception as e:
            print(f"[WARN] Failed to parse structured output: {e}")
            # fallback to raw string as final_reasoning
            return CoTResult(
                reasoning_steps=[],
                final_reasoning=response_text,
                final_label=self.case.valid_labels[0],
                confidence=None,
                stats=stats,
                person_key=self.person_key,
                role_playing=self.role_playing
            )
    

    def classify(self, text: str) -> str:
        return self.classify_with_reasoning(text).final_label

    def classify_with_reasoning(self, text: str) -> CoTResult:
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
                "content": f"You are an expert classifier for {self.case_name}. Analyze the input step by step and show your reasoning before classifying."
            }

        user_prompt = self._build_cot_prompt(text)
        user_message = {"role": "user", "content": user_prompt}

        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_message["content"],
            system_message=system_message["content"],
            max_tokens=self.max_tokens,
        )
        elapsed = time.time() - start_time

        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        stats = SampleStats(
            tokens_used=response.usage.total_tokens if response.usage else None,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            latency=elapsed
        )

        response_text = response.choices[0].message.content.strip()
        return self._parse_cot_response(response_text, stats)

    def map_label(self, raw_label: str) -> str:
        if hasattr(self.case, "label_map"):
            return self.case.label_map.get(raw_label, raw_label)
        return raw_label

    def print_reasoning(self, cot_result: CoTResult):
        print(f"\n=== Chain of Thoughts Analysis for {self.case_name} ===")
        print("-" * 60)

        for step in cot_result.reasoning_steps:
            print(f"\nStep {step.step_number}:\n{step.content}")

        print(f"\nFinal Reasoning:\n  {cot_result.final_reasoning}")
        print(f"\nFinal Classification: {cot_result.final_label}")
        print("-" * 60)

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

    def _build_optimized_cot_prompt(self, input_text: str) -> str:
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
        label_list = " or ".join(self.case.valid_labels)

        prompt = f"""Analyze this text for {self.case_name}:

\"{input_text}\"

Rules:
{rules}

IMPORTANT: Be sensitive to subtle stereotypes. Even factual-sounding statements can reinforce stereotypes.

Step 1 - Evidence: What evidence of {config['focus']} do you see? Look for ANY group-based assumptions.
Step 2 - Adversarial Check: Could {config['adversarial']}? Even subtle implications count.
Step 3 - Decision: Make a final decision with a brief justification (no numeric confidence).

FINAL CLASSIFICATION: {label_list}

Your analysis:"""
        return prompt
    
    def _build_zero_shot_plus_prompt(self, input_text: str) -> str:
        """
        Fast classification with minimal guidance and *no numeric confidence*.
        If unsure, provide 1–2 sentences of reasoning. Else, just classify.
        """
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = self.case.valid_labels

        prompt = f"""Definition of a {self.case_name}: {self.task_definition}

Input: {input_text}

Labeling rules:
{rules}

Instructions:
1) First, give your immediate classification.
2) Then write "Certain: Yes" if you're confident in your judgment.
3) If not confident, write "Certain: No" and give 1–2 sentences explaining why.

Format STRICT:
Classification: [Choose from {label_list}]
Certain: [Yes/No]
Reasoning: [Only present if Certain: No]
"""
        return prompt


    # Integration method for your ChainOfThoughts class
    def classify_with_strategy(self, text: str, strategy: str = "optimized") -> CoTResult:
        """
        Choose reasoning strategy based on performance needs.
        
        Strategies:
        - "optimized": 3-step structured reasoning (best accuracy)
        - "zero_plus": Enhanced zero-shot with guidance (fastest, good accuracy)
        """
        
        if strategy == "optimized":
            user_prompt = self._build_optimized_cot_prompt(text)
        elif strategy == "zero_plus":
            user_prompt = self._build_zero_shot_plus_prompt(text)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Use existing classification logic from classify_with_reasoning
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
                "content": f"You are an expert classifier for {self.case_name}. Analyze the input step by step and show your reasoning before classifying."
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
    
        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
    
        stats = SampleStats(
            tokens_used=response.usage.total_tokens if response.usage else None,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            latency=elapsed
        )
    
        response_text = response.choices[0].message.content.strip()

        return self._parse_cot_response(response_text, stats)
    