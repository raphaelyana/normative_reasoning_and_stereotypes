from typing import Optional, Dict, Literal, List
import time
import openai
from utils import call_llm
from profiles.profile_message import make_system_message
from profiles.profile_dict import PERSON_SEEDS
from pydantic import BaseModel
from dataclasses import dataclass
import random
from cases.cases_config import CaseConfig
import pandas as pd
from collections import defaultdict

DEFAULT_MODEL_DICT = {
    "default": "gpt-4o-mini"
}

class SampleStats(BaseModel):
    tokens_used: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency: float
    stage1_tokens: Optional[int] = None
    stage2_tokens: Optional[int] = None
    stage1_prompt_tokens: Optional[int] = None
    stage1_completion_tokens: Optional[int] = None
    stage2_prompt_tokens: Optional[int] = None
    stage2_completion_tokens: Optional[int] = None

class ReflexionResult(BaseModel):
    initial_prediction: str
    final_prediction: str
    reflection_text: str = ""
    retried: bool = False
    confidence: float = 0.0
    certainty: Optional[str] = None
    stats: Optional[SampleStats] = None
    person_key: Optional[str] = None
    role_playing: Optional[Literal["active", "passive", "none"]] = None
    label_changed: Optional[bool] = False

class SmartReflexion:
    """
    Simplified reflexion that uses self-assessment for retry decisions.
    No external judge needed - the model decides if it should reconsider.
    """
    
    def __init__(
        self,
        case: CaseConfig,
        client: openai.OpenAI,
        model: Optional[str] = None,
        max_tokens: int = 300,
        task_definition: Optional[str] = None,
        person_key: Optional[str] = None,
        role_playing: Literal["passive", "active", "none"] = "none",
        person_seeds: Optional[Dict[str, str]] = PERSON_SEEDS,
        uncertainty_threshold: str = "UNSURE",  # CERTAIN, CONFIDENT, UNSURE, CONFLICTED
        random_seed: int = 42,
        random_threshold: float = 0.5,
        n_shots: int = 0,
        examples_df: Optional[pd.DataFrame] = None,
    ):
        self.case = case
        self.case_name = case.case_name
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.task_definition = task_definition
        self.person_key = person_key
        self.role_playing = role_playing
        self.person_seeds = person_seeds
        self.uncertainty_threshold = uncertainty_threshold
        self.random_seed = random_seed
        random.seed(self.random_seed)

        if not (0.0 <= random_threshold <= 1.0):
            raise ValueError(f"random_threshold must be between 0.0 and 1.0, got {random_threshold}")
        self.random_threshold = random_threshold

        self.n_shots = n_shots
        self.examples_df = examples_df
        
        self.total_tokens = 0
        self.total_latency = 0.0
        self.total_calls = 0

    def _build_system_message(self) -> str:
        if self.person_key and self.role_playing == "passive":
            return make_system_message(self.case_name, self.person_key)["content"]
        elif self.person_key and self.role_playing == "active":
            return (
                f"You are a classifier for {self.case_name}.\n"
                f"Answer as if you were the following person:\n{self.person_seeds[self.person_key]}"
            )
        else:
            return f"You are an expert classifier for {self.case_name}. Make careful, thoughtful decisions."

    def _build_initial_prompt(self, input_text: str) -> str:
        """Stage 1: Classification with self-assessment"""
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        labels = self.case.valid_labels
        
        prompt = f"""Definition of {self.case_name}: {self.task_definition}

Rules:
{rules}

Input: "{input_text}"

First classify, then assess your certainty:

Classification: {labels[0]} or {labels[1]}
Certainty: CERTAIN, CONFIDENT, UNSURE, or CONFLICTED

Where:
- CERTAIN: Textbook example, no doubt
- CONFIDENT: Clear case with minor considerations  
- UNSURE: Borderline, could reasonably go either way
- CONFLICTED: Very ambiguous, multiple valid interpretations

Most cases are either CONFIDENT or UNSURE.
"""
        
        return prompt

    def _build_reflection_prompt(self, input_text: str, initial_pred: str, certainty: str) -> str:
        """Stage 2: Reflection and potential revision"""
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        labels = self.case.valid_labels
    
        if len(labels) != 2:
            raise ValueError("SmartReflexion currently supports exactly two labels.")
        
        label_1, label_2 = labels[0], labels[1]
        
        opposite_label = label_2 if initial_pred == label_1 else label_1

        prompt = f"""
You initially classified this input as "{initial_pred}" with certainty "{certainty}".

However, a domain expert reviewing your response suggested it might be better classified as "{opposite_label}".

They are not necessarily correct, but it's important to consider their reasoning.

Input: "{input_text}"

Rules:
{rules}
"""
        if self.n_shots > 0 and self.examples_df is not None:
            examples_str = "\n\nExamples:\n" + "\n\n".join(self._select_formatted_examples())
            prompt += f"\n{examples_str}\n"

        prompt += f"""
Please reflect carefully:

1. **Defense**: Why did you choose "{initial_pred}"? What evidence supports this decision?
2. **Expert critique**: What might the expert have seen that points to "{opposite_label}"?
3. **Reassessment**: Weigh both views. What is your most reasoned final classification?

Final classification: {label_1} or {label_2}
"""
        
        return prompt

    def _parse_initial_response(self, response_text: str) -> tuple[str, str]:
        """Extract classification and certainty from initial response"""
        import re
        
        # Extract classification
        classification = ""
        for label in self.case.valid_labels:
            if label.lower() in response_text.lower():
                classification = label
                break
        
        if not classification:
            classification = self.case.valid_labels[0]
        
        # Extract certainty
        certainty = "CONFIDENT"  # Default
        certainty_options = ["CERTAIN", "CONFIDENT", "UNSURE", "CONFLICTED"]
        for cert in certainty_options:
            if cert.lower() in response_text.lower():
                certainty = cert
                break
        
        return classification, certainty

    def _parse_reflection_response(self, response_text: str) -> tuple[str, str]:
        """Extract reflection and final classification"""
        import re
        
        # Extract reflection
        reflection_match = re.search(r"Reflection:\s*(.*?)(?=Alternative perspective:|Final classification:|$)", 
                                   response_text, re.DOTALL | re.IGNORECASE)
        reflection = reflection_match.group(1).strip() if reflection_match else ""
        
        # Extract final classification
        final_classification = ""
        for label in self.case.valid_labels:
            if f"Final classification: {label}" in response_text or \
               f"final classification: {label.lower()}" in response_text.lower():
                final_classification = label
                break
        
        if not final_classification:
            # Fallback: look for any mention of valid labels
            for label in self.case.valid_labels:
                if label.lower() in response_text.lower():
                    final_classification = label
                    break
        
        if not final_classification:
            final_classification = self.case.valid_labels[0]  # Default
        
        return reflection, final_classification
    
    def _select_formatted_examples(self) -> List[str]:
        label_col = self.case.label_col
        template = self.case.example_template_fewshots
        df = self.examples_df
    
        label_to_examples = defaultdict(list)
        for _, row in df.iterrows():
            label_to_examples[row[label_col]].append(template(row))
    
        selected = []
        for _, examples in label_to_examples.items():
            selected.extend(examples[:self.n_shots])
        return selected

    def classify_with_reflexion(self, text: str) -> ReflexionResult:
        """Main reflexion classification method"""
        system_message = self._build_system_message()
        
        # Stage 1: Initial classification with self-assessment
        initial_prompt = self._build_initial_prompt(text)
        
        start_time = time.time()
        initial_response = call_llm(
            client=self.client,
            model=self.model,
            prompt=initial_prompt,
            system_message=system_message,
            max_tokens=self.max_tokens,
        )
        stage1_time = time.time() - start_time
        
        self.total_calls += 1
        
        # Parse initial response
        initial_pred, certainty = self._parse_initial_response(initial_response.choices[0].message.content)
        
        # Decide if reflexion is needed
        valid_levels = ["CERTAIN", "CONFIDENT", "UNSURE", "CONFLICTED"]
        threshold_idx = valid_levels.index(self.uncertainty_threshold.upper())
        certainty_idx = valid_levels.index(certainty.upper())
        
        needs_reflexion = certainty_idx >= threshold_idx
        
        random_num = random.random()
        force_reflexion = random_num < self.random_threshold

        if not needs_reflexion and not force_reflexion:
            # High confidence - return initial prediction
            stage1_tokens = initial_response.usage.total_tokens if initial_response.usage else 0
            stage1_prompt = initial_response.usage.prompt_tokens if initial_response.usage else 0
            stage1_completion = initial_response.usage.completion_tokens if initial_response.usage else 0
            
            self.total_tokens += stage1_tokens
            
            stats = SampleStats(
                tokens_used=stage1_tokens,
                prompt_tokens=stage1_prompt,
                completion_tokens=stage1_completion,
                latency=stage1_time,
                stage1_tokens=stage1_tokens,
                stage2_tokens=0,
                stage1_prompt_tokens=stage1_prompt,
                stage1_completion_tokens=stage1_completion,
                stage2_prompt_tokens=0,
                stage2_completion_tokens=0
            )
            
            return ReflexionResult(
                initial_prediction=initial_pred,
                final_prediction=initial_pred,
                reflection_text=certainty,
                retried=False,
                confidence=0.9 if certainty == "CERTAIN" else 0.8,
                certainty=certainty,
                stats=stats,
                person_key=self.person_key,
                role_playing=self.role_playing
            )
        
        # Stage 2: Reflexion for uncertain cases
        reflection_prompt = self._build_reflection_prompt(text, initial_pred, certainty)
        
        start_time = time.time()
        reflection_response = call_llm(
            client=self.client,
            model=self.model,
            prompt=reflection_prompt,
            system_message=system_message,
            max_tokens=self.max_tokens,
        )
        stage2_time = time.time() - start_time
        
        self.total_calls += 1
        
        # Parse reflection response
        reflection_text, final_pred = self._parse_reflection_response(reflection_response.choices[0].message.content)
        
        label_changed = initial_pred != final_pred

        # Calculate usage for each stage
        stage1_tokens = initial_response.usage.total_tokens if initial_response.usage else 0
        stage1_prompt = initial_response.usage.prompt_tokens if initial_response.usage else 0
        stage1_completion = initial_response.usage.completion_tokens if initial_response.usage else 0
        
        stage2_tokens = reflection_response.usage.total_tokens if reflection_response.usage else 0
        stage2_prompt = reflection_response.usage.prompt_tokens if reflection_response.usage else 0
        stage2_completion = reflection_response.usage.completion_tokens if reflection_response.usage else 0
        
        total_tokens = stage1_tokens + stage2_tokens
        total_prompt_tokens = stage1_prompt + stage2_prompt
        total_completion_tokens = stage1_completion + stage2_completion
        
        self.total_tokens += total_tokens
        self.total_latency += stage1_time + stage2_time
        
        stats = SampleStats(
            tokens_used=total_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            latency=stage1_time + stage2_time,
            stage1_tokens=stage1_tokens,
            stage2_tokens=stage2_tokens,
            stage1_prompt_tokens=stage1_prompt,
            stage1_completion_tokens=stage1_completion,
            stage2_prompt_tokens=stage2_prompt,
            stage2_completion_tokens=stage2_completion
        )
        
        return ReflexionResult(
            initial_prediction=initial_pred,
            final_prediction=final_pred,
            reflection_text=reflection_text,
            retried=True,
            confidence=0.6 if certainty == "UNSURE" else 0.4,
            certainty=certainty,
            stats=stats,
            person_key=self.person_key,
            role_playing=self.role_playing,
            label_changed=label_changed,
        )

    def classify(self, text: str) -> str:
        """Simple interface that returns just the final prediction"""
        result = self.classify_with_reflexion(text)
        return result.final_prediction

    def get_metrics(self) -> Dict[str, any]:
        """Get performance metrics"""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency": self.total_latency,
            "avg_latency_per_call": self.total_latency/max(1, self.total_calls),
            "avg_tokens_per_call": self.total_tokens/max(1, self.total_calls),
        }