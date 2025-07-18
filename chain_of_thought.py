from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import openai
import time
from utils import call_llm
import re
import json

DEFAULT_MODEL_DICT = {
    'default': 'gpt-4o-mini',
}

@dataclass
class ReasoningStep:
    """Represents a single step in the chain of reasoning."""
    step_number: int
    content: str


@dataclass
class CoTResult:
    """Contains the full chain of thought reasoning and final classification."""
    reasoning_steps: List[ReasoningStep]
    final_reasoning: str
    final_label: str
    confidence: float = 0.0

class ChainOfThoughts:
    def __init__(
        self,
        case: dict,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 800,
        task_definition: Optional[str] = None,
        num_reasoning_steps: int = 3, 
        reasoning_budget: Optional[dict] = None,
        person_key: Optional[str] = None,
    ):
        self.case = case
        self.case_name = case["case_name"]
        self.task_definition = task_definition
        self.num_reasoning_steps = num_reasoning_steps
        
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.reasoning_budget = reasoning_budget
        self.person_key = person_key
        
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

    def _build_cot_prompt(self, input_text: str) -> str:
        """Build the chain-of-thoughts prompt."""
        rules = "\n".join(f"- {r}" for r in self.case["label_rules"])
        label_list = self.case["valid_labels"]
        
        examples_section = ""
        if "examples" in self.case and self.case["examples"]:
            examples_section = f"""
Examples:
{chr(10).join(self.case["examples"])}
"""
        step_templates = [
        ("Initial Observation", [
            "What do you observe in this text?",
            "What are the key elements or patterns?"
        ]),
        ("Feature Analysis", [
            f"What specific features are relevant to {self.case_name} detection?",
            "How do these features align with or contradict the labeling rules?"
        ]),
        ("Rule Application", [
            "Apply each labeling rule to this specific case.",
            "Explain how the text satisfies or violates each relevant rule."
        ]),
        ("Interpretation and Context", [
            "Does the text reinforce or challenge generalized beliefs?",
            "How might it be interpreted in terms of social bias, fairness, or group identity?"
        ]),
        ]
        
        reasoning_steps_section = ""
        for i in range(min(self.num_reasoning_steps, len(step_templates))):
            step_title, questions = step_templates[i]
            reasoning_steps_section += f"\nStep {i+1} - {step_title}:\n"
            reasoning_steps_section += "\n".join(f"- {q}" for q in questions) + "\n"
    
        prompt = f"""Definition of a {self.case_name}: {self.task_definition}

{examples_section}
Labeling rules:
{rules}

Now analyze the following input step by step:

Input: "{input_text}"

Please follow this chain of reasoning format:
{reasoning_steps_section}

Final Reasoning:
- Synthesize your analysis from all steps
- Provide your final judgment with confidence level

Final Label: [Choose from: {label_list}]

Please provide your complete reasoning following this exact format."""

        return prompt

    def _parse_cot_response(self, response_text: str) -> CoTResult:
        """Parse the chain-of-thoughts response into structured format."""
        reasoning_steps = []
        final_reasoning = ""
        final_label = ""
        confidence = 0.0
        
        try:
            step_pattern = r"Step (\d+)[^:]*:(.*?)(?=Step \d+|Final Reasoning:|$)"
            step_matches = re.findall(step_pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            for step_num, step_content in step_matches:
                reasoning_steps.append(ReasoningStep(
                    step_number=int(step_num),
                    content=step_content.strip()
                ))
            
            # Extract final reasoning
            final_reasoning_match = re.search(r"Final Reasoning:\s*(.*?)(?=Final Label:|$)", response_text, re.DOTALL | re.IGNORECASE)
            if final_reasoning_match:
                final_reasoning = final_reasoning_match.group(1).strip()
            
            # Extract final label
            final_label_match = re.search(r"Final Label:\s*\[?([^\]]+)\]?", response_text, re.IGNORECASE)
            if final_label_match:
                raw_label = final_label_match.group(1).strip()
                # Clean and validate the label
                for valid_label in self.case["valid_labels"]:
                    if valid_label.lower() in raw_label.lower():
                        final_label = valid_label
                        break
                if not final_label:
                    final_label = self.case["valid_labels"][0]
            
            # Try to extract confidence if mentioned
            confidence_match = re.search(r"confidence[^0-9]*([0-9.]+)", response_text, re.IGNORECASE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    if confidence > 1.0:
                        confidence = confidence/100.0
                except ValueError:
                    confidence = 0.0
                    
        except Exception as e:
            print(f"[WARN] Failed to parse CoT response: {e}")
            final_reasoning = response_text
            for valid_label in self.case["valid_labels"]:
                if valid_label.lower() in response_text.lower():
                    final_label = valid_label
                    break
            if not final_label:
                final_label = self.case["valid_labels"][0]
        
        return CoTResult(
            reasoning_steps=reasoning_steps,
            final_reasoning=final_reasoning,
            final_label=final_label,
            confidence=confidence
        )

    def classify(self, text: str) -> str:
        """Classify text using chain-of-thoughts reasoning."""
        return self.classify_with_reasoning(text).final_label

    def classify_with_reasoning(self, text: str) -> CoTResult:
        """Classify text and return full reasoning chain."""
        system_message = {
            "role": "system",
            "content": f"""You are an expert classifier for {self.case_name}. 
Your task is to analyze the input text step by step, showing your complete reasoning process before making a final classification.

Be thorough in your analysis, consider multiple perspectives, and clearly explain how you apply the labeling rules to reach your conclusion."""
        }

        user_prompt = self._build_cot_prompt(text)
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
            max_tokens=self.max_tokens,
            reasoning_budget=self.reasoning_budget,
        )
        elapsed = time.time() - start_time


        self.total_latency += elapsed
        self.total_calls += 1
        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        response_text = response.choices[0].message.content.strip()
        return self._parse_cot_response(response_text)

    def map_label(self, raw_label: str) -> str:
        """Convert the LLM label to the dataset label using case['label_map']."""
        if "label_map" in self.case:
            return self.case["label_map"].get(raw_label, raw_label)
        return raw_label

    def print_reasoning(self, cot_result: CoTResult):
        """Print the full chain of reasoning in a readable format."""
        print(f"\n=== Chain of Thoughts Analysis for {self.case_name} ===")
        print("-" * 60)
        
        for step in cot_result.reasoning_steps:
            print(f"\nStep {step.step_number}:\n{step.content}")
        
        print(f"\nFinal Reasoning:")
        print(f"  {cot_result.final_reasoning}")
        
        print(f"\nFinal Classification: {cot_result.final_label}")
        if cot_result.confidence > 0:
            print(f"Confidence: {cot_result.confidence:.2f}")
        
        print("-" * 60)

    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency": self.total_latency,
            "avg_latency_per_call": self.total_latency/max(1, self.total_calls),
            "avg_tokens_per_call": self.total_tokens/max(1, self.total_calls)
        }