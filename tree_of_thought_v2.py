from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import openai
import time
from utils.call_llm import call_llm
import pandas as pd
from pydantic import BaseModel, Field
from collections import defaultdict
from cases.cases_config import CaseConfig
import json

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

class ThoughtOutput(BaseModel):
    thought: str = Field(..., description="The reasoning step or reflection")
    label: Optional[str] = Field(None, description="Optional label (e.g., 'Yes' or 'No')")

class ThoughtState(Enum):
    PENDING = "pending"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Thought:
    content: str
    state: ThoughtState
    id: str = "" 
    parent_id: Optional[str] = None
    children: List['Thought'] = None
    verdict: Optional[str] = None
    def __post_init__(self):
        if self.children is None:
            self.children = []

class TreeOfThoughtExplorer:
    def __init__(
        self, 
        case: CaseConfig,
        client: openai.OpenAI, 
        model: Optional[str] = None, 
        max_branching_factor: int = 3, 
        max_depth: int = 3,
        task_definition: Optional[str] = "",
        max_tokens_dict: dict = {
            "generation": 500
        },
        n_shots: int = 1,
        examples_df: Optional[pd.DataFrame] = None,
        provider: Optional[str] = None,
    ):
        self.case = case
        self.case_name = self.case.case_name
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT['default']
        self.provider = provider
        self.max_branching_factor = max_branching_factor
        self.max_depth = max_depth
        self.task_definition = task_definition
        self.n_shots = n_shots
        self.examples_df = examples_df
        self.thoughts = {}
        self.root_id = None
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0
        self.id_counter = 0
        self.max_tokens_dict = max_tokens_dict

    def _select_formatted_examples(self) -> List[str]:
        if self.examples_df is None:
            return []
        label_col = self.case.label_col
        template = self.case.example_template_tot
        df = self.examples_df
        label_to_examples = defaultdict(list)
        for _, row in df.iterrows():
            label_to_examples[row[label_col]].append(template(row))
        selected = []
        for _, examples in label_to_examples.items():
            selected.extend(examples[:self.n_shots])
        return selected

    def build_prompt_gen_thought(self, reasoning: str, max_branching_factor: int, with_labels: bool = False) -> str:
        task = self.case.task
        task_definition = self.task_definition
        rule_lines = "\n".join([f"- {rule}" for rule in self.case.label_rules])
        label_instruction_map = getattr(
            self.case, 
            "label_instructions_tot", 
            {
                True: "At the end of each thought, add a field 'label' with a value of either 'Yes' or 'No', based on your judgment.",
                False: "Do not include any 'label' field — just output the 'thought' reasoning steps."
            }
        )
        label_instruction = label_instruction_map[with_labels]
        example_blocks = "\n\n".join(self._select_formatted_examples()) if with_labels else ""
        prompt_lines = [
            task,
            "",
            f"Definition of a {self.case.case_name}: {task_definition}",
            "",
            f"You are reasoning about: \"{reasoning}\"",
            "",
            f"Generate {max_branching_factor} distinct analytical thoughts.",
            label_instruction,
            "",
            "Labeling rules:",
            rule_lines,
        ]
        if with_labels:
            prompt_lines.append("")
            prompt_lines.append("Examples:")
            prompt_lines.append(example_blocks)
            prompt_lines.append("")
            prompt_lines.append("Return the thoughts as a **valid JSON list**, using this format:")
            prompt_lines.append("""
[
  { "thought": "First reasoning step", "label": "Yes" },
  { "thought": "Second reasoning step", "label": "No" }
]
""")
            prompt_lines.append("Do not add any extra text before or after the JSON.")
            prompt_lines.append("Each 'thought' must be a complete sentence.")
            prompt_lines.append("Valid labels are: 'Yes' or 'No'.")
        else:
            prompt_lines.append("")
            prompt_lines.append("Return the thoughts as a **valid JSON list**, using this format:")
            prompt_lines.append("""
[
  { "thought": "First reasoning step" },
  { "thought": "Second reasoning step" }
]
""")
            prompt_lines.append("Do not add any 'label' fields.")
            prompt_lines.append("Do not add any extra text before or after the JSON.")
            prompt_lines.append("Each 'thought' must be a complete sentence.")
        return "\n".join(prompt_lines)

    def generate_thoughts(self, reasoning: str, current_depth: int = 0, parent_id: Optional[str] = None) -> List[Thought]:
        if current_depth >= self.max_depth:
            return []
        is_leaf = (current_depth == self.max_depth - 1)
        system_message = {
            "role": "system",
            "content": """You are a Tree of Thought generator.

When the user requests labeled thoughts, always return a **valid JSON list** where each object contains BOTH a 'thought' and a 'label' field.

Each object must look like: { "thought": "...", "label": "Yes" } or { "thought": "...", "label": "No" }

If you omit the label field, the reasoning will be discarded.

The only valid label values are: "Yes" or "No".
Do NOT use any other label values. Do NOT omit the label field.
Do NOT add text before or after the JSON."""
        }
        user_prompt = self.build_prompt_gen_thought(
            reasoning=reasoning,
            max_branching_factor=self.max_branching_factor,
            with_labels=is_leaf
        )
        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_prompt,
            system_message=system_message["content"],
            max_tokens=self.max_tokens_dict["generation"],
            provider=self.provider,
        )
        elapsed = time.time() - start_time
        self.total_latency += elapsed
        self.total_calls += 1
        if getattr(response, "usage", None):
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
        content = response.choices[0].message.content.strip()
        try:
            parsed_list = json.loads(content)
            assert isinstance(parsed_list, list)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON from model output:\n{content[:300]}\n→ {e}")
            return []
        thoughts = []
        for i, obj in enumerate(parsed_list):
            try:
                parsed = ThoughtOutput.parse_obj(obj)
                if is_leaf and parsed.label is None:
                    print(f"[SKIP] Missing label at leaf node idx {i}")
                    continue
                label = parsed.label.strip().capitalize() if (is_leaf and parsed.label) else None
                new_thought = Thought(
                    id="",
                    parent_id=parent_id,
                    content=parsed.thought.strip(),
                    state=ThoughtState.COMPLETED,
                    verdict=label
                )
                thoughts.append(new_thought)
            except Exception as e:
                print(f"[SKIP] Invalid structured output at idx {i}: {e}")
                continue
        return thoughts

    def solve(self, initial_prompt: str) -> List[List[Thought]]:
        root = Thought(
            content=initial_prompt,
            state=ThoughtState.COMPLETED,
            id="0",
            parent_id=None,
        )
        self.id_counter += 1
        self.thoughts[root.id] = root
        self.root_id = root.id
        self._expand_thought(root, 0)
        return self.enumerate_leaf_paths()

    def _expand_thought(self, thought: Thought, depth: int):
        if depth >= self.max_depth:
            return
        self.thoughts[thought.id] = thought
        if depth == 0:
            self.root_id = thought.id
        child_thoughts = self.generate_thoughts(thought.content, depth, thought.id)
        for idx, child in enumerate(child_thoughts, start=1):
            child.id = f"{thought.id}.{idx}" if thought.id else str(idx)
        thought.children = child_thoughts
        for child in child_thoughts:
            self.thoughts[child.id] = child
            if depth + 1 < self.max_depth:
                self._expand_thought(child, depth + 1)

    def enumerate_leaf_paths(self) -> List[List[Thought]]:
        paths: List[List[Thought]] = []
        def dfs(node: Thought, cur: List[Thought]):
            cur.append(node)
            if not node.children:
                if node.verdict is not None:
                    paths.append(cur.copy())
            else:
                for ch in node.children:
                    dfs(ch, cur)
            cur.pop()
        root = self.thoughts.get(self.root_id)
        if root:
            dfs(root, [])
        return paths

    def get_tree_dict(self) -> dict:
        tree_dict = {}
        for node_id, t in self.thoughts.items():
            node_data = {"content": t.content}
            if t.verdict is not None:
                node_data["label"] = t.verdict
            tree_dict[node_id] = node_data
        return tree_dict

    def print_full_tree(self, root: Optional[Thought] = None, indent: int = 0):
        if root is None:
            root = self.thoughts.get(self.root_id)
            if not root:
                print("No root thought found.")
                return
        indent_str = "    " * indent
        print(f"{indent_str} Thought ID {root.id}")
        print(f"{indent_str} -- Content: {root.content}")
        if root.verdict:
            print(f"{indent_str} -- Label: {root.verdict}")
        print(f"{indent_str}{'-' * 60}")
        for child in root.children:
            self.print_full_tree(child, indent + 1)
