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

        # Metrics
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
            #TODO: Choose the prompt, probably make something around saying that it is designed to evaluate possible paths of thinking.
        }

        user_message = {
            "role": "user",    # REMOVE TYPE OF LOGICAL REASONING - The type of logical reasoning used: deductive, inductive, abductive, heuristic, or other
            "content": f"""Definition of a stereotype: {self.stereotype_definition}.
            
                        Current reasoning path: '{reasoning}'

                        Generate {self.max_branching_factor} possible next thoughts or steps to solve this question. 
                        Each thought should correspond to either:
                        - a small but meaningful reflection on one dimension of stereotype identification or critique (e.g., exaggerated generalization, social power dynamics, etc.);
                        - or a path of thinking that resists stereotyping — by highlighting complexity, individuality, or context that challenges overgeneralized assumptions.
                        
                        For each, include:
                        - Thought: [The thought content itself]
                        - Stereotypes/Biases: [Any stereotypes or biases that might appear, if relevant]
                        - Description: [A short description of the reasoning (1-3 sentences)]
                        - Is Stereotyping: [Yes/No/Borderline]
                
                        Format each thought as a separate section.
                        """
                        #- The thought itself
                        #- The stereotypes or biases that might appear
                        #- A short description of the reasoning (1-3 sentences)
                       # 
                        #Format:
                        #- 'Thought: <thought>'
                        #- 'Stereotypes/Biases: <stereotypes or biases>'
                        #- 'Description: <description>'

        }

        # To track the time taken for the API call
        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_message["content"],
            system_message=system_message["content"],
            max_tokens=500
        )
        elapsed = time.time() - start_time

        # Tracking the API call metrics (tokens, latency)
        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        content = response.choices[0].message.content.strip()
    
        # Fix 1: Improved parsing of thoughts
        thoughts = []
        sections = re.split(r"###\s*Thought\s+\d+\s*:", content)
        #sections = re.split(r"\b[Tt]hought\s*:", content)
        
        for i, section in enumerate(sections[0:]):  # Skip the first empty split
            if not section.strip():
                continue
                
            lines = section.strip().split("\n")
            #thought_text = lines[0].strip()  # First line is the thought content
            thought_text = ""
            for line in lines:
                if re.search(r"\*\*Thought\*\*", line, re.IGNORECASE):
                    parts = line.split("**Thought**", 1)[-1].split(":", 1)
                    if len(parts) > 1:
                        thought_text = parts[1].strip()
                    else:
                        thought_text = parts[0].strip()
                    break
            if not thought_text:
                thought_text = lines[0].strip()
            stereotypes_text = ""
            description_text = ""
            is_stereotyping = ""
            
            # Fix 2: Clean up the thought text to remove any leftover formatting
            thought_text = thought_text.replace("'", "").replace('"', "")
            
            for line in lines[1:]:
                #if "Stereotypes/Biases" in line:
                #    stereotypes_text = line.split(":", 1)[1].strip() if len(line.split(":", 1)) > 1 else ""
                #elif "Description" in line:
                #    description_text = line.split(":", 1)[1].strip() if len(line.split(":", 1)) > 1 else ""
                #elif "Is Stereotyping" in line:
                #    is_stereotyping = line.split(":", 1)[1].strip().capitalize() if len(line.split(":", 1)) > 1 else "" 
                parsed = {
                    "thought": "",
                    "stereotypes": "",
                    "description": "",
                    "is_stereotyping": ""
                }
                
                for line in lines:
                    if match := re.match(r"-\s*\*\*Thought\*\*\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["thought"] = match.group(1).strip()
                    elif match := re.match(r"-\s*\*\*Stereotypes/Biases\*\*\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["stereotypes"] = match.group(1).strip()
                    elif match := re.match(r"-\s*\*\*Description\*\*\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["description"] = match.group(1).strip()
                    elif match := re.match(r"-\s*\*\*Is Stereotyping\*\*\s*:\s*(.+)", line, re.IGNORECASE):
                        parsed["is_stereotyping"] = match.group(1).strip().capitalize()
                
            
            new_thought = Thought(
                id="",
                parent_id=parent_id,
                content=parsed["thought"],
                state=ThoughtState.PENDING,
                score=0.0,
                stereotypes_biases=parsed["stereotypes"],
                description=parsed["description"],
                is_stereotyping=parsed["is_stereotyping"]
            )

            thoughts.append(new_thought)
    
        return thoughts


    def evaluate_thought(self, thought: Thought) -> float:
        system_message = {
            "role": "system",
            "content": """You are an ethical and logical evaluator. You will be given a single thought. Rate it from 0.0 to 1.0 based on:
            - Ethical soundness (0.0: highly unethical; 0.3: unethical generally; 0.5: neutral, 0.7: somehow ethical, 1.0: highly ethical)
            - Logical coherence: Is the thought logically consistent and well-structured? (0.0: highly incoherent; 0.3: incoherent generally; 0.5: neutral, 0.7: somewhat coherent, 1.0: highly coherent)
            - Sensitivity to biases and stereotypes (0.0: highly biased; 0.3: biased generally; 0.5: neutral, 0.7: somewhat unbiased, 1.0: highly unbiased)

            Combine these scores by making a simple unweighted average, resulting in a final score between 0.0 and 1.0.
            Output only the final numeric score (0.0 to 1.0).
            """
        }

        alt_system_message = {
            "role": "system",
            "content": f"""You are a logic evaluator. You are given a single thought. 
            
            Rate it from 0.0 to 1.0 depending on its alignment with stereotype definition: {self.stereotype_definition}
            Output only the final score (scale from 0 to 1).
        """
        }

        user_message = {
            "role": "user",
            "content": f"""Evaluate the following thought on a scale from 0.0 (worst) to 1.0 (best):

                "{thought.content}"

                Provide only the final score (no explanation)."""
        }

        # To track the time taken for the API call
        start_time = time.time()
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=user_message["content"],
            system_message=system_message["content"],
            max_tokens=500
        )
        elapsed = time.time() - start_time

        # Tracking the API call metrics (tokens, latency)
        self.total_latency += elapsed
        self.total_calls += 1

        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens


        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5 



    def solve(self, initial_prompt: str) -> List[Thought]:
        root = Thought(
            content=initial_prompt,
            state=ThoughtState.PENDING,
            score=0.0,
            stereotypes_biases="",
            description="",
            id="0",  # Use the counter for the id
            parent_id=None,  # Root has no parent
            #children=[]
            )
        
        self.id_counter += 1  # Increment counter for next use

        self.thoughts[root.id] = root
        self.root_id = root.id

        self._expand_thought(root, 0)
        #print(f"DEBUG: Total thoughts in tree: {len(self.thoughts)}")
        #print(f"DEBUG: Root has {len(root.children)} children")

        return self._get_best_solution(root)
    


    def _expand_thought(self, thought: Thought, depth: int):
        if depth >= self.max_depth:
            return

        #print(f"DEBUG: Expanding thought at depth {depth}: {thought.content[:30]}...")
    
        self.thoughts[thought.id] = thought
        if depth == 0:
            self.root_id = thought.id

        child_thoughts = self.generate_thoughts(thought.content, depth + 1, thought.id)
        for idx, child in enumerate(child_thoughts, start=1):
            child.id = f"{thought.id}.{idx}" if thought.id else str(idx)
        #print(f"DEBUG: Generated {len(child_thoughts)} child thoughts")
        thought.children = child_thoughts
        

        for child in child_thoughts:
            child.state = ThoughtState.EVALUATING
            child.score = self.evaluate_thought(child)
            child.state = ThoughtState.COMPLETED
            self._expand_thought(child, depth + 1)

            if depth + 1 < self.max_depth:
                self._expand_thought(child, depth + 1)


    def _get_best_solution(self, thought: Thought) -> List[Thought]:
        if not thought.children:
            return [thought]
        
        #print(f"DEBUG: Finding best child among {len(thought.children)} children")
        #for i, child in enumerate(thought.children):
        #    print(f"DEBUG: Child {i}: score={child.score}, content={child.content[:30]}...")
    
        best_child = max(thought.children, key=lambda x: x.score)
        if best_child:
            #print(f"DEBUG: Best child has score {best_child.score}")
            return [thought] + self._get_best_solution(best_child)
        else:
            return [thought]

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
        print(f"{indent_str}🧠 Thought ID {root.id}")
        print(f"{indent_str}→ Score: {root.score:.2f}")
        print(f"{indent_str}→ Content: {root.content}")
        if root.stereotypes_biases:
            print(f"{indent_str}→ Stereotypes: {root.stereotypes_biases}")
        if root.description:
            print(f"{indent_str}→ Description: {root.description}")
        if hasattr(root, "is_stereotyping") and root.is_stereotyping:
            print(f"{indent_str}→ Is Stereotyping: {root.is_stereotyping}")
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
