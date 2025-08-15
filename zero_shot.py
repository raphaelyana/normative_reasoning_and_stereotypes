from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import openai
import time
from utils import call_llm
from profiles.profile_dict import PERSON_SEEDS
from profiles.profile_message import make_system_message
import re
from cases.cases import CaseConfig
from typing import Literal
from profiles.schema import PersonSet

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

class ZeroShot:
    def __init__(
            self,
            case: CaseConfig,
            client: openai.OpenAI,
            model: Optional[dict] = None,
            max_tokens: int = 300,
            task_definition: Optional[str] = None,
            person_key: Optional[str] = None,
            role_playing: Literal["passive", "active", "none"] = "none",
            person_set: Optional[PersonSet] = None
        ):
        
        self.case = case
        self.case_name = case.case_name
        self.task_definition = task_definition

        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.person_key = person_key
        self.role_playing = role_playing
        if person_set is None:
            self.person_set = PersonSet(seeds=PERSON_SEEDS, metadata={})
        else:
            self.person_set = person_set

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0
        

    def classify(self, text: str) -> str:

        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = [i for i in self.case.valid_labels]

        if self.person_key is not None and self.role_playing == "passive":
            system_message = make_system_message(
            case_name=self.case_name,
            person_key=self.person_key,
            person_set=self.person_set
        )
            
        elif self.person_key and self.role_playing == "active":

            content = f"You are a classifier for {self.case_name}.\n" + f"""Please answer as if you were the following person:\n{self.person_set.seeds[self.person_key]}\n"""
            system_message = {
                "role": "system",
                "content": content
            }

        else:
            system_message = {
                "role": "system",
                "content": f"You are a classifier for {self.case_name}. Decide if the reasoning is an example or not."
            }

        if self.task_definition:
            prompt_intro = f"Definition of a {self.case_name}: {self.task_definition}\n\n"
        else:
            prompt_intro = ""
            
        user_prompt = (
                f"{prompt_intro}"
                f"""Input: {text}

                Labeling rules:
                {rules}

                Return only the label contained in this list: {label_list}.
                """)

        user_message = {
            "role": "user",   
            "content":  user_prompt
        }

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

         
