from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import openai
import time
from utils import call_llm
from profile_message import PERSON_SEEDS
from profile_message import make_system_message
import re
from cases import CaseConfig
from typing import Literal

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
            role_playing: Literal["passive", "active", "none"] = "none"
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
        

    def classify(self, text: str) -> str:

        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = [i for i in self.case.valid_labels]

        if self.person_key is not None and self.role_playing == "passive":
            system_message = make_system_message(
            case_name=self.case_name,
            person_key=self.person_key
        )
            
        elif self.person_key and self.role_playing == "active":

            content = f"You are a classifier for {self.case_name}.\n" + f"""Please answer as if you were the following person:\n{PERSON_SEEDS[self.person_key]}\n"""
            system_message = {
                "role": "system",
                "content": content
            }

        else:
            system_message = {
                "role": "system",
                "content": f"You are a classifier for {self.case_name}. Decide if the reasoning is an example or not."
            }

        user_prompt = f"""Definition of a {self.case_name}: {self.task_definition}

                    
                Input: {text}

                Labeling rules:
                {rules}

                Return only the label contained in this list: {label_list}.
                """

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

        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
        
        return response.choices[0].message.content.strip()

         
