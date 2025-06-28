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

##############################################
###           ThoughState Class            ###
##############################################

class ThoughtState(Enum):
    PENDING = "pending"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"



##############################################
###           dataclass Thought            ###
###                                        ###
###  Defines the different attributes at   ###
###         every node of the tree.        ###
##############################################

@dataclass
class Thought:
    content: str
    state: ThoughtState
    score: float = 0.0
    feature: str = ""
    description: str = ""
    id: str = ""                         
    parent_id: Optional[str] = None
    children: List['Thought'] = None
    verdict: Optional[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []



##############################################
###           class TreeOfThought          ###
###                                        ###
###  Defines the different attributes at   ###
###         every node of the tree.        ###
##############################################

class TreeOfThought:
    def __init__(self, 
                 case: dict,
                 client: openai.OpenAI, 
                 model: Optional[dict]=None, 
                 max_branching_factor: int = 3, 
                 max_depth: int = 3,
                 task_definition: Optional[str] = None
                 ):


        self.case = case
        self.case_name = case["case_name"]
        
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT['default']
        self.max_branching_factor = max_branching_factor
        self.max_depth = max_depth
        self.task_definition = task_definition

        self.thoughts = {}
        self.root_id = None

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

        self.id_counter=0



    def build_prompt_gen_thought(
            self,
            case_study: dict,
            reasoning: str,
            task_definition: str,
            max_branching_factor: int,
        ) -> str:

        task = self.case["task"]
        task_definition = task_definition or ""

        field_lines = "\n".join([f"- {field}" for field in case_study["fields"]])
        rule_lines = "\n".join([f"- {rule}" for rule in case_study["label_rules"]])
        example_blocks = "\n\n".join(case_study["examples"])

        prompt_built = f"""{task}
        
            Definition of a {case_study['case_name']}: {task_definition}

            You are reasoning about: "{reasoning}"
            
            Generate {max_branching_factor} independent thoughts that extend the reasoning.

            Return each thought with these fields ONLY:
            {field_lines}

            Labeling rules:
            {rule_lines}

            Examples:
            {example_blocks}

            Now, generate {max_branching_factor} thoughts in this same format, and nothing else.
            """
        return prompt_built
    



    def generate_thoughts(self, reasoning: str, current_depth: int = 0, parent_id: Optional[str]=None) -> List[Thought]:
        if current_depth >= self.max_depth:
            return []


        system_message = {
            "role": "system",
            "content": """You are a structured Tree of Thought reasoner. 
            Your job is to explore the possible paths of reasoning for a given query, each with a distinct perspective."""
        }

        user_prompt = self.build_prompt_gen_thought(
                                                    case_study = self.case,
                                                    reasoning = reasoning,
                                                    task_definition = self.task_definition,
                                                    max_branching_factor = self.max_branching_factor
                                                    )

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
            max_tokens=500
        )
        elapsed = time.time() - start_time

        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        content = response.choices[0].message.content.strip()
    
        thoughts = []
        
        sections = re.split(r"###\s*Thought\s+\d+\s*", content)

        field1 = self.case["fields"][0]
        field2 = self.case["fields"][1]
        field3 = self.case["fields"][2]
        field4 = self.case["fields"][3].split(":")[0] 
        
        
        for i, section in enumerate(sections[1:]):  

            parsed = {
                "thought": "",
                "feature": "",
                "description": "",
                "label": ""
            }
        
            try:
                lines = section.strip().split("\n")
                for line in lines:
                    if match := re.match(rf"-\s*{re.escape(field1)}\s*:\s*(.*)", line, re.IGNORECASE):
                        parsed["thought"] = match.group(1).strip()
                    elif match := re.match(rf"-\s*{re.escape(field2)}\s*:\s*(.*)", line, re.IGNORECASE):
                        parsed["feature"] = match.group(1).strip()
                    elif match := re.match(rf"-\s*{re.escape(field3)}\s*:\s*(.*)", line, re.IGNORECASE):
                        parsed["description"] = match.group(1).strip()
                    elif match := re.match(rf"-\s*{re.escape(field4)}\s*:\s*(.*)", line, re.IGNORECASE):
                        parsed["label"] = match.group(1).strip()

            except Exception as e:
                print(f"[WARN] Failed to parse section {i+1}: {e}")
                print(f"[RAW]:\n{section}")
                continue
        

            if not parsed["thought"]:
                print(f"[SKIP] Empty thought in section {i+1}")
                continue

            valid_labels = {item.capitalize() for item in self.case["valid_labels"]}
            label = parsed["label"].capitalize()

            if label not in valid_labels:
                print(f"[WARN] Invalid label: '{parsed['label']} -> default to '{self.case['valid_labels'][-1]}'")
                label = self.case["valid_labels"][-1].capitalize()
            
            parsed["thought"] = parsed["thought"].strip()
            parsed["description"] = parsed["description"].strip()
            parsed["feature"] = parsed["feature"].strip()
                
            
            new_thought = Thought(
                id="",
                parent_id=parent_id,
                content=parsed["thought"],
                state=ThoughtState.PENDING,
                score=0.0,
                feature=parsed["feature"],
                description=parsed["description"],
                verdict=label
            )

            thoughts.append(new_thought)
    
        return thoughts


    def evaluate_thought(self, thought: Thought, parent_thought: Optional[Thought] = None) -> float:

        evaluation_prompt = self.case["evaluation_prompt"]

        system_message = {
            "role": "system",
            "content": f"""You are evaluating the *usefulness and novelty* of a reasoning step in a Tree of Thoughts process for {self.case_name} detection.
    
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
            max_tokens=10
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
            feature="",
            description="",
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

        child_thoughts = self.generate_thoughts(thought.content, depth + 1, thought.id)
        for idx, child in enumerate(child_thoughts, start=1):
            child.id = f"{thought.id}.{idx}" if thought.id else str(idx)

        thought.children = child_thoughts
        

        for child in child_thoughts:
            child.state = ThoughtState.EVALUATING
            child.score = self.evaluate_thought(child, parent_thought=thought)
            child.state = ThoughtState.COMPLETED
            self._expand_thought(child, depth+1)



    def _get_best_solution(self, thought: Thought) -> List[Thought]:
        if not thought.children:
            return [thought]
        
        best_child = max(thought.children, key=lambda x: x.score)
        if best_child:
            return [thought] + self._get_best_solution(best_child)
        else:
            return [thought]



    def _get_majority_vote_from_path(self, solution_path: List[Thought], weighted: bool = False) -> str:
        votes = {label.capitalize(): 0.0 for label in self.case["valid_labels"]}

        for thought in solution_path:
            label = (thought.verdict or "").strip().capitalize()
            if label in votes:
                if not weighted:
                    votes[label] += 1
                else:
                    votes[label] += thought.score
    
        consensus_label = max(votes.items(), key=lambda x: x[1])[0]
        return consensus_label



    def _get_majority_vote_from_tree(self, weighted: bool = False) -> str:
        votes = {label.capitalize(): 0.0 for label in self.case["valid_labels"]}
        
        def collect_votes(thought: Thought):
            label = (thought.verdict or "").strip().capitalize()
            if label in votes:
                if not weighted:
                    votes[label] += 1
                else:
                    votes[label] += thought.score

            for child in thought.children:
                collect_votes(child)
    
        root = self.thoughts.get(self.root_id)

        if not root:
            print("[WARN] No root found for majority vote.")
            return "Unrelated"

        collect_votes(root)
        
        return max(votes.items(), key=lambda x: x[1])[0]



    def _get_majority_vote_from_leafs(self, weighted: bool = False) -> str:
        votes = {label.capitalize(): 0.0 for label in self.case["valid_labels"]}

        def collect_leaf_votes(thought: Thought):
            if not thought.children:
                label = (thought.verdict or "").strip().capitalize()
                if label in votes:
                    if not weighted:
                        votes[label] += 1
                    else:
                        votes[label] += thought.score
            
            for child in thought.children:
                collect_leaf_votes(child)

        root = self.thoughts.get(self.root_id)

        if not root:
            print("[WARN] No root found for majority vote.")
            return "Unrelated"
        
        collect_leaf_votes(root)

        return max(votes.items(), key=lambda x: x[1])[0]



    def print_full_tree(self, root: Optional[Thought] = None, indent: int = 0):
        """
        Recursively prints the entire tree with ID, score, content, feature, and description.
        """
        if root is None:
            root = self.thoughts.get(self.root_id)
            if not root:
                print("No root thought found.")
                return
    
        indent_str = "    " * indent

        field2_label = self.case["fields"][1]
        field4_label = self.case["fields"][3].split(":")[0]

        print(f"{indent_str} Thought ID {root.id}")
        print(f"{indent_str} --  Score: {root.score:.2f}")
        print(f"{indent_str} -- Content: {root.content}")
        if root.feature:
            print(f"{indent_str} -- {field2_label}: {root.feature}")
        if root.description:
            print(f"{indent_str} -- Description: {root.description}")
        if root.verdict:
            print(f"{indent_str} -- {field4_label}: {root.verdict}")
        print(f"{indent_str}{'-' * 60}")
        
        for child in root.children:
            self.print_full_tree(child, indent+1)






##############################################
###                                        ###
###              Example usage             ###
###                                        ###
##############################################

if __name__ == "__main__":

    import os
    from dotenv import load_dotenv

    from case_studies import manipulation, stereotypes
    from stereotype_def import stereotype_definition
    from mentalmanip_def import mental_manipulation_def1, mental_manipulation_def2



    load_dotenv()
    client = openai.OpenAI(api_key=os.environ.get("API_KEY_OPENAI"))




    print(" ===== Stereotype detection ===== \n")

    tot_stereotype = TreeOfThought(case=stereotypes, client=client, max_branching_factor=2, max_depth=3, task_definition=stereotype_definition)
    #prompt_stereotype = "Russians are born in Russia."
    prompt_stereotype = "French people are good kissers."
    solution_stereotype = tot_stereotype.solve(prompt_stereotype)

    print("\n📊 Metrics Summary:")
    print(f"  Total LLM Calls        : {tot_stereotype.total_calls}")
    print(f"  Total Tokens Used      : {tot_stereotype.total_tokens}")
    print(f"    └─ Prompt Tokens     : {tot_stereotype.total_prompt_tokens}")
    print(f"    └─ Completion Tokens : {tot_stereotype.total_completion_tokens}")
    print(f"  Total Latency (sec)    : {tot_stereotype.total_latency:.2f}")
    if tot_stereotype.total_calls > 0:
        print(f"  Avg Latency per Call   : {tot_stereotype.total_latency / tot_stereotype.total_calls:.2f} sec")

    print("\nSolution path (Stereotype):\n")
    for i, thought in enumerate(solution_stereotype):
        print(f"Level {i}:")
        print(f"  Thought: {thought.content}")
        print(f"  Feature: {thought.feature}")
        print(f"  Description: {thought.description}")
        print(f"  Label: {thought.verdict}")
        print(f"  Score: {thought.score:.2f}")
        print("-" * 60)


    print("\nFull Tree (Stereotype):\n")
    tot_stereotype.print_full_tree()

    print("\n\n ===== Manipulation detection ===== \n")

    tot_manipulation = TreeOfThought(case=manipulation, client=client, max_branching_factor=2, max_depth=2, task_definition=mental_manipulation_def1)
    prompt_manipulation = """Person1: You think I'm a fake.
    Person2: I think it's what you think.
    Person1: No, it isn't what I think.
    Person2: Look... You made a very calculated move, and then made me feel embarrassed for responding to you. That wasn't necessary."""
    solution_manipulation = tot_manipulation.solve(prompt_manipulation)

    print("\n📊 Metrics Summary (Manipulation):")
    print(f"  Total LLM Calls        : {tot_manipulation.total_calls}")
    print(f"  Total Tokens Used      : {tot_manipulation.total_tokens}")
    print(f"    └─ Prompt Tokens     : {tot_manipulation.total_prompt_tokens}")
    print(f"    └─ Completion Tokens : {tot_manipulation.total_completion_tokens}")
    print(f"  Total Latency (sec)    : {tot_manipulation.total_latency:.2f}")
    if tot_manipulation.total_calls > 0:
        print(f"  Avg Latency per Call   : {tot_manipulation.total_latency / tot_manipulation.total_calls:.2f} sec")

    print("\nSolution path (Manipulation):\n")
    for i, thought in enumerate(solution_manipulation):
        print(f"Level {i}:")
        print(f"  Thought: {thought.content}")
        print(f"  Feature: {thought.feature}")
        print(f"  Description: {thought.description}")
        print(f"  Label: {thought.verdict}")
        print(f"  Score: {thought.score:.2f}")
        print("-" * 60)

    print("\nFull Tree (Manipulation):\n")
    tot_manipulation.print_full_tree()