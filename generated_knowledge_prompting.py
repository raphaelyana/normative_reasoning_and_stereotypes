from typing import List, Optional, Dict, Any, Literal
from collections import defaultdict
import openai
from utils import call_llm
import re
import pandas as pd
from cases.cases_config import CaseConfig
from profiles.profile_message import make_system_message, PERSON_SEEDS
import time

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

class GeneratedKnowledgePrompting:
    def __init__(
        self,
        case: CaseConfig,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 300,
        knowledge_max_tokens: int = 200,
        task_definition: Optional[str] = None,
        n_knowledge_pieces: int = 3,
        examples_df: Optional[pd.DataFrame] = None,
        person_key: Optional[str] = None,
        role_playing: Literal["passive", "active", "none"] = "none",
        person_seeds: Optional[dict[str, str]] = PERSON_SEEDS,
        knowledge_type: Literal["general", "contextual", "definitional"] = "contextual",
        examples_per_label: int = 1
    ):
        self.case = case
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.knowledge_max_tokens = knowledge_max_tokens
        self.task_definition = task_definition
        self.n_knowledge_pieces = n_knowledge_pieces
        self.person_key = person_key
        self.role_playing = role_playing
        self.person_seeds = person_seeds
        self.knowledge_type = knowledge_type
        self.examples_df = examples_df
        self.examples_per_label = examples_per_label

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

    def _generate_knowledge_prompt(self, input_text: str) -> str:
        knowledge_prompts = {
            "general": f"""Given the following text, generate {self.n_knowledge_pieces} pieces of relevant background knowledge that would help identify {self.case.case_name}. Focus on general principles, patterns, and contextual information.

Text: {input_text}

Generate {self.n_knowledge_pieces} distinct pieces of knowledge (one per line, numbered):""",
            "contextual": f"""Given the following text about potential {self.case.case_name}, generate {self.n_knowledge_pieces} pieces of contextual knowledge that would help in making an accurate judgment. Consider social dynamics, communication patterns, and relevant contextual factors.

Text: {input_text}

Generate {self.n_knowledge_pieces} relevant contextual insights (one per line, numbered):""",
            "definitional": f"""Given the following text, generate {self.n_knowledge_pieces} pieces of definitional and explanatory knowledge about {self.case.case_name} that would help in classification. Focus on key characteristics, indicators, and distinguishing features.

Text: {input_text}

Generate {self.n_knowledge_pieces} definitional insights (one per line, numbered):"""
        }
        if self.knowledge_type not in knowledge_prompts:
            raise ValueError(f"Unknown knowledge_type: {self.knowledge_type}")
        return knowledge_prompts[self.knowledge_type]

    def _generate_knowledge(self, input_text: str) -> List[str]:
        knowledge_prompt = self._generate_knowledge_prompt(input_text)

        if self.person_key is not None and self.role_playing == "passive":
            system_message = make_system_message(
                case_name=self.case.case_name,
                person_key=self.person_key
            )["content"]
        elif self.person_key is not None and self.role_playing == "active":
            system_message = (
                f"You are an expert in {self.case.case_name} detection.\n"
                f"Generate knowledge as if you were the following person:\n{self.person_seeds[self.person_key]}"
            )
        else:
            system_message = f"You are an expert in {self.case.case_name} detection. Generate relevant background knowledge to help with classification."

        start_time = time.time()

        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=knowledge_prompt,
            system_message=system_message,
            max_tokens=self.knowledge_max_tokens,
        )

        elapsed = time.time() - start_time
        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        knowledge_text = response.choices[0].message.content.strip()
        knowledge_pieces = []
        lines = knowledge_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                clean_line = re.sub(r'^\d+\.?\s*|^[-•]\s*', '', line).strip()
                if clean_line:
                    knowledge_pieces.append(clean_line)

        return knowledge_pieces[:self.n_knowledge_pieces]

    def _select_formatted_examples(self) -> List[str]:
        if self.examples_df is None:
            return []

        label_col = self.case.label_col
        template = self.case.example_template_fewshots
        df = self.examples_df

        label_to_examples = defaultdict(list)
        for _, row in df.iterrows():
            label_to_examples[row[label_col]].append(template(row))

        selected = []
        for _, examples in label_to_examples.items():
            selected.extend(examples[:self.examples_per_label])
        return selected

    def _format_classification_prompt(self, input_text: str, knowledge_pieces: List[str]) -> str:
        knowledge_str = "\n".join([f"- {k}" for k in knowledge_pieces])
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = [i for i in self.case.valid_labels]

        examples_section = ""
        examples = self._select_formatted_examples()
        if examples:
            examples_str = "\n\n".join(examples)
            examples_section = f"\nExamples:\n{examples_str}\n"

        return f"""Definition of {self.case.case_name}: {self.task_definition}

Relevant Background Knowledge:
{knowledge_str}

Labeling rules:
{rules}
{examples_section}
Now evaluate the following case using the background knowledge and rules:

Input: {input_text}

Return only one of: {label_list}."""

    def classify(self, text: str) -> tuple[str, Dict[str, Any]]:
        knowledge_pieces = self._generate_knowledge(text)
        classification_prompt = self._format_classification_prompt(text, knowledge_pieces)

        if self.person_key is not None and self.role_playing == "passive":
            system_message = make_system_message(
                case_name=self.case.case_name,
                person_key=self.person_key
            )["content"]
        elif self.person_key is not None and self.role_playing == "active":
            system_message = (
                f"You are a classifier for {self.case.case_name}.\n"
                f"Please answer as if you were the following person:\n{self.person_seeds[self.person_key]}"
            )
        else:
            system_message = f"You are a classifier for {self.case.case_name}. Use the provided background knowledge, examples, and rules to decide the correct label."

        start_time = time.time()

        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=classification_prompt,
            system_message=system_message,
            max_tokens=self.max_tokens,
        )

        elapsed = time.time() - start_time
        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        stats = {
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "latency": elapsed,
            "generated_knowledge": knowledge_pieces,
            "total_calls": self.total_calls,
            "total_latency": self.total_latency,
            "total_tokens": self.total_tokens
        }

        return response.choices[0].message.content.strip(), stats

    def get_total_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency": self.total_latency,
            "avg_latency_per_call": self.total_latency / self.total_calls if self.total_calls > 0 else 0,
            "avg_tokens_per_call": self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        }
