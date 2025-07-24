from typing import List, Optional
from collections import defaultdict
from enum import Enum
import openai
from utils import call_llm
import re
import pandas as pd
from cases import CaseConfig
from profile_message import make_system_message
from profile_message import PERSON_SEEDS
from typing import Literal

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

class FewShot:
    def __init__(
        self,
        case: CaseConfig,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 300,
        task_definition: Optional[str] = None,
        n_shots: int = 1,
        examples_df: Optional[pd.DataFrame] = None,
        person_key: Optional[str] = None, 
        role_playing: Literal["passive", "active", "none"] = "none",
        person_seeds: Optional[dict[str, str]] = PERSON_SEEDS,
    ):
        self.case = case
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.task_definition = task_definition
        self.n_shots = n_shots
        self.person_key = person_key
        self.role_playing = role_playing
        self.person_seeds = person_seeds

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

        self.examples_df = examples_df

    
    def _select_formatted_examples(self) -> List[str]:
            label_col = self.case.label_col
            template = self.case.example_template_fewshots
            df = self.examples_df
        
            label_to_examples = defaultdict(list)
            for _, row in df.iterrows():
                label_to_examples[row[label_col]].append(template(row))
        
            selected = []
            for _, examples in label_to_examples.items():
                selected.extend(examples[:self.n_shots])
            return selected


    def _format_prompt(self, input_text: str) -> str:
        examples_str = "\n\n".join(self._select_formatted_examples())
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = [i for i in self.case.valid_labels]

        return f"""Definition of a {self.case.case_name}: {self.task_definition}
                    
                    Labeling rules:
                    {rules}

                    Examples:
                    {examples_str}

                    Now evaluate the following case:
                    
                    Input: {input_text}
                    
                    Return only one of: {label_list}.
                    """

    def classify(self, text: str) -> str:
        prompt = self._format_prompt(text)

        if self.person_key is not None and self.role_playing == "passive":
            system_message = make_system_message(
                case_name=self.case.case_name,
                person_key=self.person_key
            )
        
        elif self.person_key is not None and self.role_playing == "active":
            content = (
                f"You are a few-shot classifier for {self.case.case_name}.\n"
                f"Please answer as if you were the following person:\n{self.person_seeds[self.person_key]}"
            )
            system_message = {
                "role": "system",
                "content": content
            }
        
        else:
            system_message = {
                "role": "system",
                "content": f"You are a few-shot classifier for {self.case.case_name}. Use the provided examples and rules to decide the correct label."
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
            max_tokens=self.max_tokens,
        )

        elapsed = time.time() - start_time

        usage = response.usage
        stats = {
            "tokens_used": usage.total_tokens if usage else None,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "latency": elapsed
        }


        self.total_latency += elapsed
        self.total_calls += 1

        if usage:
            self.total_tokens += usage.total_tokens
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens

        return response.choices[0].message.content.strip(), stats
