from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import openai
import time
from utils import call_llm

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
    #reasoning_type: str = "unspecified"  # deductive, inductive, abductive, etc.
    #normative_framework: str = "unspecified"  # utilitarianism, deontology, virtue ethics, etc.
    #principles: str = ""
    stereotypes_biases: str = ""
    description: str = ""
    id: str = ""                         
    parent_id: Optional[str] = None
    children: List['Thought'] = None

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
                 max_depth: int = 3):

        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT['default']
        self.max_branching_factor = max_branching_factor
        self.max_depth = max_depth

        self.thoughts = {}
        self.root_id = None

        # Metrics
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

        self.id_counter=0

    def generate_thoughts(self, prompt: str, current_depth: int = 0, parent_id: Optional[str]=None) -> List[Thought]:
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
            "content": f"""Given the current state of reasoning:

                        '{prompt}'

                        Generate {self.max_branching_factor} possible next thoughts or steps to solve this question. 
                        For each thought, provide the following questions:

                        - Thought: [The thought content itself]
                        - Stereotypes/Biases: [Any stereotypes or biases that might appear]
                        - Description: [A short description of the reasoning (1-3 sentences)]
                
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
            max_tokens=300
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
        #sections = content.split("- Thought:")
        sections = content.split("Thought:")
        
        for i, section in enumerate(sections[1:]):  # Skip the first empty split
            if not section.strip():
                continue
                
            lines = section.strip().split("\n")
            thought_text = lines[0].strip()  # First line is the thought content
            stereotypes_text = ""
            description_text = ""
            
            # Fix 2: Clean up the thought text to remove any leftover formatting
            thought_text = thought_text.replace("'", "").replace('"', "")
            
            for line in lines[1:]:
                if "Stereotypes/Biases" in line:
                    stereotypes_text = line.split(":", 1)[1].strip() if len(line.split(":", 1)) > 1 else ""
                elif "Description" in line:
                    description_text = line.split(":", 1)[1].strip() if len(line.split(":", 1)) > 1 else ""
            
            new_thought = Thought(
                id=str(self.id_counter),
                parent_id=parent_id,
                content=thought_text,
                state=ThoughtState.PENDING,
                score=0.0,
                stereotypes_biases=stereotypes_text,
                description=description_text
            )
            self.id_counter += 1  # Increment counter
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
            
            #"You are a moral reasoning assistant evaluating thoughts based on ethical soundness, logical coherence, and principle alignment. Rate on a scale from 0 to 1. Example of ranges: "
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
            max_tokens=300
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
            id=str(self.id_counter),  # Use the counter for the id
            parent_id=None,  # Root has no parent
            #children=[]
            )
        
        self.id_counter += 1  # Increment counter for next use

        self.thoughts[root.id] = root
        self.root_id = root.id

        self._expand_thought(root, 0)
        print(f"DEBUG: Total thoughts in tree: {len(self.thoughts)}")
        print(f"DEBUG: Root has {len(root.children)} children")

        return self._get_best_solution(root)
    


    def _expand_thought(self, thought: Thought, depth: int):
        if depth >= self.max_depth:
            return

        print(f"DEBUG: Expanding thought at depth {depth}: {thought.content[:30]}...")
    
        self.thoughts[thought.id] = thought
        if depth == 0:
            self.root_id = thought.id

        child_thoughts = self.generate_thoughts(thought.content, depth+1, thought.id)
        print(f"DEBUG: Generated {len(child_thoughts)} child thoughts")
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
        
        print(f"DEBUG: Finding best child among {len(thought.children)} children")
        for i, child in enumerate(thought.children):
            print(f"DEBUG: Child {i}: score={child.score}, content={child.content[:30]}...")
    
        best_child = max(thought.children, key=lambda x: x.score)
        if best_child:
            print(f"DEBUG: Best child has score {best_child.score}")
            return [thought] + self._get_best_solution(best_child)
        else:
            return [thought]





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
