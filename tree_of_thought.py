from typing import List, Optional, Callable, Literal
from dataclasses import dataclass
from enum import Enum
import openai
import time
from utils.call_llm import call_llm
import re
import pandas as pd
from collections import defaultdict
from pydantic import BaseModel, Field
from cases.get_case_config import get_case_config
from cases.cases_config import CaseConfig
from cases.stereotypes_case import stereotypes_case
from cases.manipulation_case import manipulation_case
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
    score: float = 0.0                      
    children: List['Thought'] = None
    verdict: Optional[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []



class TreeOfThought:
    def __init__(self, 
                 case: CaseConfig,
                 client: openai.OpenAI, 
                 model: Optional[dict]=None, 
                 max_branching_factor: int = 3, 
                 max_depth: int = 3,
                 task_definition: Optional[str] = "",
                 max_tokens_dict: dict = {
                    "generation": 500,
                    "evaluation": 10
                    },
                n_shots: int = 1,
                examples_df: Optional[pd.DataFrame] = None,
                reasoning_budget: Optional[dict] = None
                 ):


        self.case = case
        self.case_name = self.case.case_name
        
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT['default']
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

        self.id_counter=0

        self.tie_pairs   = 0
        self.tie_events  = 0
        self.max_tie_group = 0

        self.max_tokens_dict = max_tokens_dict
        self.reasoning_budget = reasoning_budget

    def _select_formatted_examples(self) -> List[str]:
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
      { "thought": "Second reasoning step", "label": "No" },
      ...
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
      { "thought": "Second reasoning step" },
      ...
    ]
    """)
            prompt_lines.append("Do not add any 'label' fields.")
            prompt_lines.append("Do not add any extra text before or after the JSON.")
            prompt_lines.append("Each 'thought' must be a complete sentence.")

        return "\n".join(prompt_lines)



    



    def generate_thoughts(self, reasoning: str, current_depth: int = 0, parent_id: Optional[str]=None) -> List[Thought]:
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

        user_message = {
            "role": "user",
            "content": user_prompt
        }

        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_message["content"],
            system_message=system_message["content"],
            max_tokens=self.max_tokens_dict["generation"],
            reasoning_budget=self.reasoning_budget
        )
        elapsed = time.time() - start_time

        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
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
                    state=ThoughtState.PENDING,
                    score=0.0,
                    verdict=label
                )
                thoughts.append(new_thought)

            except Exception as e:
                print(f"[SKIP] Invalid structured output at idx {i}: {e}")
                continue

        return thoughts


    def evaluate_thought(self, thought: Thought, parent_thought: Optional[Thought] = None) -> float:

        evaluation_prompt = self.case.evaluation_prompt

        system_message = {
            "role": "system",
            "content": f"""You are evaluating the quality of a reasoning step in a Tree of Thoughts process for {self.case_name} detection.
    
            Use these scoring intervals (return only the number, e.g., 0.85):
            
            - 0.0–0.2 → Irrelevant, redundant, or off-topic
            - 0.2–0.4 → Weak or vague contribution, limited insight
            - 0.4–0.6 → Somewhat useful, partially relevant or accurate
            - 0.6–0.8 → Clear and relevant, helps the reasoning process
            - 0.8–1.0 → Highly insightful, adds substantial new reasoning

            Return **only the numeric score**, no text.
            """
        }
    
        if parent_thought:
            user_prompt = f"""Evaluate the usefulness and novelty of this child thought **compared to its parent**:
    
            Parent Thought:
            \"{parent_thought.content}\"
        
            Child Thought:
            \"{thought.content}\"
        
            Score between 0.0 (worst) and 1.0 (best). Return only the number."""
                
        else:
            user_prompt = f"""{evaluation_prompt}:
    
            \"{thought.content}\"
    
            Score between 0.0 (worst) and 1.0 (best). Return only the number."""
    

        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_prompt,
            system_message=system_message["content"],
            max_tokens=self.max_tokens_dict["evaluation"]
        )
        elapsed = time.time() - start_time
    
        self.total_latency += elapsed
        self.total_calls += 1
        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
    
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"[WARN] Failed to parse score: {e}. Defaulting to 0.25")
            return 0.25
 



    def solve(self, initial_prompt: str) -> List[Thought]:
        root = Thought(
            content=initial_prompt,
            state=ThoughtState.PENDING,
            score=0.0,
            id="0",
            parent_id=None,
        )
        
        self.id_counter += 1

        self.thoughts[root.id] = root
        self.root_id = root.id

        self._expand_thought(root, 0)

        return self._get_best_solution(root)
    


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
            child.state = ThoughtState.EVALUATING
            child.score = self.evaluate_thought(child, parent_thought=thought)
            if depth + 1 < self.max_depth - 1:
                child.verdict = None
            child.state = ThoughtState.COMPLETED
            self._expand_thought(child, depth+1)

        leaf_thoughts = [t for t in self.thoughts.values() if t.verdict is not None]

        score_groups = {}
        for t in leaf_thoughts:
            rounded_score = round(t.score, 2)  
            score_groups.setdefault(rounded_score, []).append(t)

        found_conflict = False
        for group in score_groups.values():
            labels = {t.verdict.strip().capitalize() for t in group}
            if len(labels) > 1:
                n = len(group)
                self.tie_pairs += n * (n - 1) // 2
                found_conflict = True
                self.max_tie_group = max(self.max_tie_group, n)

        if found_conflict:
            self.tie_events += 1


       
    ##############
    ## Different strategies to find the best solution path


    def _get_best_solution(self, node: Thought) -> list[Thought]:
        """
        Depth-first 'greedy' path following the highest-scoring child at each
        level until a leaf is reached. Used to tell the user 'this is the line
        of reasoning I believe in most'.
        """
        if not node.children:
            return [node]

        best_child = max(
            node.children,
            key=lambda t: (t.score, t.verdict is not None)
        )
        return [node] + self._get_best_solution(best_child)



    def _get_best_solution_greedy(self, node: Thought) -> list[Thought]:
        "Greedy: pick the highest-scoring child at each depth. Tie-breaker prefers nodes that already have a verdict."
        if not node.children:
            return [node]
    
        best_child = max(
            node.children,
            key=lambda t: (t.score, t.verdict is not None)
        )
        return [node] + self._get_best_solution(best_child)
    


    def _get_best_solution_sum(self, node: Thought) -> list[Thought]:
        "Depth-first search to find the path with the highest cumulative score."
        def dfs(n: Thought, acc: float) -> tuple[float, list[Thought]]:
            if not n.children:
                return acc + n.score, [n]
    
            best_total = float("-inf")
            best_path: list[Thought] = []
            for child in n.children:
                tot, path = dfs(child, acc + n.score)
                avg = tot / len(path)
                best_avg = best_total / len(best_path) if best_path else float("-inf")
                if (tot > best_total) or (tot == best_total and avg > best_avg):
                    best_total, best_path = tot, path
            return best_total, [n] + best_path
        _, path = dfs(node, 0.0)
        return path



    def _get_best_solution_average(self, node: Thought) -> list[Thought]:
        "Returns the best path based on average score per node."
        def dfs(n: Thought) -> tuple[float, list[Thought]]:
            if not n.children:
                return n.score, [n]
    
            best_avg = float("-inf")
            best_path = []
            for child in n.children:
                child_avg, child_path = dfs(child)
                path_avg = (n.score + child_avg * len(child_path)) / (len(child_path) + 1)
                if path_avg > best_avg:
                    best_avg, best_path = path_avg, [n] + child_path
            return best_avg, best_path
    
        _, path = dfs(node)
        return path

    ## End of different strategies to find the best solution path
    ###########




    def _get_majority_vote_from_path(self, solution_path: list[Thought], weighted: bool = False) -> str:
        """Return the majority label from the best reasoning path."""
        votes = {label.capitalize(): 0.0 for label in self.case.valid_labels}
    
        for thought in solution_path:
            label = (thought.verdict or "").strip().capitalize()
            if label in votes:
                if weighted:
                    votes[label] += thought.score
                else:
                    votes[label] += 1
    
        return max(votes.items(), key=lambda x: x[1])[0]




    def _get_majority_vote_from_tree(self, weighted: bool = False) -> str:
        """Return the majority label across all nodes in the tree (with optional score weighting)."""
        votes = {label.capitalize(): 0.0 for label in self.case.valid_labels}
    
        def collect_votes(thought: Thought):
            label = (thought.verdict or "").strip().capitalize()
            if label in votes:
                if weighted:
                    votes[label] += thought.score
                else:
                    votes[label] += 1
            for child in thought.children:
                collect_votes(child)
    
        root = self.thoughts.get(self.root_id)
        if root:
            collect_votes(root)
    
        return max(votes.items(), key=lambda x: x[1])[0]



    def _get_majority_vote_from_leafs(self, weighted: bool = False) -> str:
        """Return the majority label only among the leaf nodes."""
        votes = {label.capitalize(): 0.0 for label in self.case.valid_labels}
    
        def collect_leaf_votes(thought: Thought):
            if not thought.children:
                label = (thought.verdict or "").strip().capitalize()
                if label in votes:
                    if weighted:
                        votes[label] += thought.score
                    else:
                        votes[label] += 1
            for child in thought.children:
                collect_leaf_votes(child)
    
        root = self.thoughts.get(self.root_id)
        if root:
            collect_leaf_votes(root)
    
        return max(votes.items(), key=lambda x: x[1])[0]

    

    def map_label(self, raw_label: str):
        """Convert the LLM label (Yes/No/…) to the dataset label using case['label_map']."""

        if self.case.case_name == "manipulation":
            return int(self.case.label_map.get(raw_label, 0))
        return self.case.label_map.get(raw_label, raw_label)


    def enumerate_leaf_paths(tot) -> list[list]:
        paths = []
        root = tot.thoughts.get(tot.root_id)
        if not root:
            return paths
    
        def dfs(node, cur):
            cur.append(node)
            if not node.children:
                if getattr(node, "verdict", None) is not None:
                    paths.append(list(cur))
            else:
                for ch in node.children:
                    dfs(ch, cur)
            cur.pop()
    
        dfs(root, [])
        return paths


    def print_full_tree(self, root: Optional[Thought] = None, indent: int = 0):
        if root is None:
            root = self.thoughts.get(self.root_id)
            if not root:
                print("No root thought found.")
                return
    
        indent_str = "    " * indent
        print(f"{indent_str} Thought ID {root.id}")
        print(f"{indent_str} --  Score: {root.score:.2f}")
        print(f"{indent_str} -- Content: {root.content}")
        if root.verdict:
            print(f"{indent_str} -- Label: {root.verdict}")
        print(f"{indent_str}{'-' * 60}")
    
        for child in root.children:
            self.print_full_tree(child, indent+1)
    
        if indent == 0:
            print("\n == Tie Summary == ")
            print(f"  - Tie events (groups with same score but conflicting labels): {self.tie_events}")
            print(f"  - Total tie pairs (within those groups): {self.tie_pairs}")
            print(f"  - Max tie group size observed: {self.max_tie_group}")

    def get_tree_dict(self) -> dict:
        """Return the tree as a flat dictionary for export."""
        tree_dict = {}
        for node_id, t in self.thoughts.items():
            node_data = {
                "content": t.content,
                "score": t.score,
            }
            if t.verdict is not None:
                node_data["label"] = t.verdict
            tree_dict[node_id] = node_data
        return tree_dict