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
    stereotypes_biases: str = ""
    description: str = ""
    id: str = ""                         
    parent_id: Optional[str] = None
    children: List['Thought'] = None
    is_stereotyping: Optional[str] = None

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
                 client: openai.OpenAI, 
                 model: Optional[dict]=None, 
                 max_branching_factor: int = 3, 
                 max_depth: int = 3,
                 stereotype_definition: Optional[str] = None):

        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT['default']
        self.max_branching_factor = max_branching_factor
        self.max_depth = max_depth
        self.stereotype_definition = stereotype_definition

        self.thoughts = {}
        self.root_id = None

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

        self.id_counter=0

    def generate_thoughts(self, reasoning: str, current_depth: int = 0, parent_id: Optional[str]=None) -> List[Thought]:
        if current_depth >= self.max_depth:
            return []


        system_message = {
            "role": "system",
            "content": """You are a structured Tree of Thought reasoner. 
            Your job is to explore the possible paths of reasoning for a given query, each with a distinct perspective."""
        }

        user_message = {
            "role": "user",   
            "content": f"""Definition of a stereotype: {self.stereotype_definition}
                
                You are reasoning about: "{reasoning}"
                
                Generate {self.max_branching_factor} independent thoughts that extend the reasoning.
                
                Return each thought with these fields ONLY:
                - Thought
                - Stereotypes/Biases
                - Description
                - Is Stereotyping: Must be exactly one of ["Yes", "No", "Unrelated"]
                
                Labeling rules:
                - Choose "Yes" if the thought includes, implies, or reinforces a stereotype — even subtly or positively (e.g., assumptions based on group identity, occupation, culture, gender, age, etc.).
                - Choose "No" if it directly challenges, refutes, or undermines a stereotype — especially if it emphasizes individual variation or contradicts common assumptions.
                - Choose "Unrelated" if the thought doesn't pertain to social/generalized assumptions at all (e.g., it is factual, neutral, or off-topic).
                
                Examples:
                
                ### Thought 1
                - Thought: Engineers are usually introverted and focused on technical work rather than social interaction.
                - Stereotypes/Biases: Occupational stereotype about engineers.
                - Description: This thought generalizes personality traits to a profession, reinforcing a stereotype.
                - Is Stereotyping: Yes
                
                ### Thought 2
                - Thought: Engineers can be highly sociable and creative, breaking the mold of traditional technical stereotypes.
                - Stereotypes/Biases: Challenges the stereotype of engineers being introverted.
                - Description: This thought explicitly refutes a common stereotype by highlighting individual differences.
                - Is Stereotyping: No
                
                ### Thought 3
                - Thought: Most engineers use CAD software to assist with their design work.
                - Stereotypes/Biases: None
                - Description: This is a factual statement about tools used in engineering, unrelated to social assumptions.
                - Is Stereotyping: Unrelated
                
                Now, generate {self.max_branching_factor} thoughts in this same format, and nothing else.                        """
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
        
        for i, section in enumerate(sections[1:]):  

            parsed = {
                "thought": "",
                "stereotypes": "",
                "description": "",
                "is_stereotyping": ""
            }
        
            try:
                lines = section.strip().split("\n")
                for line in lines:
                    if match := re.match(r"-\s*Thought\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["thought"] = match.group(1).strip()
                    elif match := re.match(r"-\s*Stereotypes/Biases\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["stereotypes"] = match.group(1).strip()
                    elif match := re.match(r"-\s*Description\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["description"] = match.group(1).strip()
                    elif match := re.match(r"-\s*Is Stereotyping\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["is_stereotyping"] = match.group(1).strip()
            except Exception as e:
                print(f"[WARN] Failed to parse section {i+1}: {e}")
                print(f"[RAW]:\n{section}")
                continue
        

            if not parsed["thought"]:
                print(f"[SKIP] Empty thought in section {i+1}")
                continue

            valid_labels = {"Yes", "No", "Unrelated"}
            label = parsed["is_stereotyping"].strip().capitalize()

            if label not in valid_labels:
                print(f"[WARN] Invalid stereotyping label: '{parsed['is_stereotyping']}' → defaulting to 'Unrelated'")
                label = "Unrelated"
            
            parsed["thought"] = parsed["thought"].strip()
            parsed["description"] = parsed["description"].strip()
            parsed["stereotypes"] = parsed["stereotypes"].strip()
                
            
            new_thought = Thought(
                id="",
                parent_id=parent_id,
                content=parsed["thought"],
                state=ThoughtState.PENDING,
                score=0.0,
                stereotypes_biases=parsed["stereotypes"],
                description=parsed["description"],
                is_stereotyping=label
            )

            thoughts.append(new_thought)
    
        return thoughts


    def evaluate_thought(self, thought: Thought, parent_thought: Optional[Thought] = None) -> float:
        system_message = {
            "role": "system",
            "content": """You are evaluating the *usefulness and novelty* of a reasoning step in a Tree of Thoughts process for stereotype detection.
    
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
            user_prompt = f"""Evaluate the usefulness of this thought **for detecting or refuting stereotypes**:
    
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
            stereotypes_biases="",
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
        votes = {"Yes": 0.0, "No": 0.0, "Unrelated": 0.0}
        for thought in solution_path:
            label = (thought.is_stereotyping or "").strip().capitalize()
            if label in votes:
                if not weighted:
                    votes[label] += 1
                else:
                    votes[label] += thought.score
    
        consensus_label = max(votes.items(), key=lambda x: x[1])[0]
        return consensus_label



    def _get_majority_vote_from_tree(self, weighted: bool = False) -> str:
        votes = {"Yes": 0.0, "No": 0.0, "Unrelated": 0.0}
        
        def collect_votes(thought: Thought):
            label = (thought.is_stereotyping or "").strip().capitalize()
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
        votes = {"Yes": 0.0, "No": 0.0, "Unrelated": 0.0}

        def collect_leaf_votes(thought: Thought):
            if not thought.children:
                label = (thought.is_stereotyping or "").strip().capitalize()
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
        Recursively prints the entire tree with ID, score, content, stereotypes, and description.
        """
        if root is None:
            root = self.thoughts.get(self.root_id)
            if not root:
                print("No root thought found.")
                return
    
        indent_str = "    " * indent
        print(f"{indent_str} Thought ID {root.id}")
        print(f"{indent_str} --  Score: {root.score:.2f}")
        print(f"{indent_str} -- Content: {root.content}")
        if root.stereotypes_biases:
            print(f"{indent_str} -- Stereotypes: {root.stereotypes_biases}")
        if root.description:
            print(f"{indent_str} -- Description: {root.description}")
        if hasattr(root, "is_stereotyping") and root.is_stereotyping:
            print(f"{indent_str} -- Is Stereotyping: {root.is_stereotyping}")
        print(f"{indent_str}{'-' * 60}")
        
        for child in root.children:
            self.print_full_tree(child, indent + 1)






##############################################
###                                        ###
###              Example usage             ###
###                                        ###
##############################################

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = openai.OpenAI(api_key=os.environ.get("API_KEY_OPENAI"))

    tot = TreeOfThought(client, max_branching_factor=2, max_depth=2)
    initial_prompt = "Russians are born in Russia."
    solution = tot.solve(initial_prompt)

    print("\n📊 Metrics Summary:")
    print(f"  Total LLM Calls        : {tot.total_calls}")
    print(f"  Total Tokens Used      : {tot.total_tokens}")
    print(f"    └─ Prompt Tokens     : {tot.total_prompt_tokens}")
    print(f"    └─ Completion Tokens : {tot.total_completion_tokens}")
    print(f"  Total Latency (sec)    : {tot.total_latency:.2f}")
    if tot.total_calls > 0:
        print(f"  Avg Latency per Call   : {tot.total_latency / tot.total_calls:.2f} sec")

    print("Solution path:\n")
    for i, thought in enumerate(solution):
        print(f"Level {i}:")
        print(f"  Thought: {thought.content}")
        print(f"  Reasoning Type: {thought.reasoning_type}")
        print(f"  Normative Framework: {thought.normative_framework}")
        print(f"  Principles: {thought.principles}")
        print(f"  Score: {thought.score:.2f}")
        print("-" * 60)
