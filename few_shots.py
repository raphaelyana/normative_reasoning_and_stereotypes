from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import openai
import time
from utils import call_llm
import re

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

class FewShot:
    def __init__(
        self,
        case: dict,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 300,
        task_definition: Optional[str] = None,
        n_shots: int = 3,
        seed: int = 42
    ):
        self.case = case
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.task_definition = task_definition
        self.n_shots = n_shots
        self.seed = seed

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

        self.examples = self._select_examples()

    def _select_examples(self):
        import random
        random.seed(self.seed)
        return random.sample(self.case["examples"], self.n_shots)

    def _format_prompt(self, input_text: str) -> str:
        examples_str = "\n\n".join(self.examples)
        rules = "\n".join(f"- {r}" for r in self.case["label_rules"])
        label_list = [i for i in self.case["valid_labels"]]

        return f"""Definition of a {self.case['case_name']}: {self.task_definition}

Examples:
{examples_str}

Now evaluate the following case:

Input: {input_text}

Labeling rules:
{rules}

Return only one of: {label_list}.
"""

    def classify(self, text: str) -> str:
        prompt = self._format_prompt(text)

        system_message = {
            "role": "system",
            "content": f"You are a few-shot classifier for {self.case['case_name']}. Use the examples to decide the correct label."
        }

        user_message = {
            "role": "user",
            "content": prompt
        }

        import time
        start_time = time.time()

        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_message["content"],
            system_message=system_message["content"],
            max_tokens=self.max_tokens
        )

        elapsed = time.time() - start_time
        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        return response.choices[0].message.content.strip()
