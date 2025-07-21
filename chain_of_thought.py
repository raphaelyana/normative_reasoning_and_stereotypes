from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import openai
import time
from utils import call_llm
import re
import json
from profile_message import PERSON_SEEDS, make_system_message
from pydantic import BaseModel
from typing import Literal

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

class CoTStats(BaseModel):
    tokens_used: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency: float

class ReasoningStep(BaseModel):
    step_number: int
    content: str

class CoTResult(BaseModel):
    reasoning_steps: List[ReasoningStep]
    final_reasoning: str
    final_label: str
    confidence: float = 0.0
    stats: Optional[CoTStats] = None

class ChainOfThoughts:
    def __init__(
        self,
        case: dict,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 800,
        task_definition: Optional[str] = None,
        num_reasoning_steps: int = 3,
        reasoning_budget: Optional[dict] = None,
        person_key: Optional[str] = None,
        role_playing: Literal["passive", "active", "none"] = "none"
    ):
        self.case = case
        self.case_name = case["case_name"]
        self.task_definition = task_definition
        self.num_reasoning_steps = num_reasoning_steps

        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.reasoning_budget = reasoning_budget
        self.person_key = person_key
        self.role_playing = role_playing

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

    def _build_cot_prompt(self, input_text: str) -> str:
        rules = "\n".join(f"- {r}" for r in self.case["label_rules"])
        label_list = self.case["valid_labels"]

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
            ])
        ]

        reasoning_steps_section = ""
        for i in range(min(self.num_reasoning_steps, len(step_templates))):
            step_title, questions = step_templates[i]
            reasoning_steps_section += f"\nStep {i+1} - {step_title}:\n"
            reasoning_steps_section += "\n".join(f"- {q}" for q in questions) + "\n"

        prompt = f"""Definition of a {self.case_name}: {self.task_definition}

Labeling rules:
{rules}

Now analyze the following input step by step:

Input: \"{input_text}\"

Please follow this chain of reasoning format:
{reasoning_steps_section}

Final Reasoning:
- Synthesize your analysis from all steps
- Provide your final judgment with confidence level

Final Label: [Choose from: {label_list}]

Please provide your complete reasoning following this exact format."""

        return prompt

    def _get_system_message(self) -> Dict[str, str]:
        if self.person_key and self.role_playing == "passive":
            return make_system_message(self.case_name, self.person_key)
        elif self.person_key and self.role_playing == "active":
            return {
                "role": "system",
                "content": f"You are an expert classifier answering as if you were {PERSON_SEEDS[self.person_key]}. Apply the reasoning process to classify the input."
            }
        else:
            return {
                "role": "system",
                "content": f"You are an expert classifier for {self.case_name}. Show your reasoning step by step before giving a final classification."
            }

    def _parse_cot_response(self, response_text: str) -> CoTResult:
        reasoning_steps = []
        final_reasoning = ""
        final_label = ""
        confidence = 0.0

        step_pattern = r"Step (\d+)[^:]*:(.*?)(?=Step \d+|Final Reasoning:|$)"
        step_matches = re.findall(step_pattern, response_text, re.DOTALL | re.IGNORECASE)
        for step_num, step_content in step_matches:
            reasoning_steps.append(ReasoningStep(
                step_number=int(step_num),
                content=step_content.strip()
            ))

        final_reasoning_match = re.search(r"Final Reasoning:\s*(.*?)(?=Final Label:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if final_reasoning_match:
            final_reasoning = final_reasoning_match.group(1).strip()

        final_label_match = re.search(r"Final Label:\s*\[?([^\]]+)\]?", response_text, re.IGNORECASE)
        if final_label_match:
            raw_label = final_label_match.group(1).strip()
            for valid_label in self.case["valid_labels"]:
                if valid_label.lower() in raw_label.lower():
                    final_label = valid_label
                    break

        confidence_match = re.search(r"confidence[^0-9]*([0-9.]+)", response_text, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except ValueError:
                pass

        return CoTResult(
            reasoning_steps=reasoning_steps,
            final_reasoning=final_reasoning,
            final_label=final_label if final_label else self.case["valid_labels"][0],
            confidence=confidence
        )

    def classify_with_reasoning(self, text: str) -> CoTResult:
        system_message = self._get_system_message()
        user_prompt = self._build_cot_prompt(text)
        user_message = {"role": "user", "content": user_prompt}

        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_message["content"],
            system_message=system_message["content"],
            max_tokens=self.max_tokens,
            reasoning_budget=self.reasoning_budget,
        )
        elapsed = time.time() - start_time

        usage = response.usage
        stats = CoTStats(
            tokens_used=usage.total_tokens if usage else None,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency=elapsed
        )

        self.total_calls += 1
        self.total_latency += elapsed
        if usage:
            self.total_tokens += usage.total_tokens
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens

        result = self._parse_cot_response(response.choices[0].message.content.strip())
        result.stats = stats
        return result

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency": self.total_latency,
            "avg_latency_per_call": self.total_latency / max(1, self.total_calls),
            "avg_tokens_per_call": self.total_tokens / max(1, self.total_calls)
        }
