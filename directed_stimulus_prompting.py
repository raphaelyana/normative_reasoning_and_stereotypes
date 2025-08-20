from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from utils.call_llm import call_llm
from cases.cases_config import CaseConfig
import numpy as np
import pandas as pd
import openai 

class StimulusType(Enum):
    LINGUISTIC_MARKERS = "linguistic_markers"
    CONTEXTUAL_CUES = "contextual_cues"
    REASONING_CHAIN = "reasoning_chain"
    COUNTER_EXAMPLES = "counter_examples"

@dataclass
class DirectionalStimulus:
    """Container for different types of hints/cues"""
    linguistic_markers: List[str]  # Key phrases, words indicating bias
    contextual_cues: List[str]     # Context that changes interpretation
    reasoning_steps: List[str]     # Step-by-step reasoning hints
    attention_focus: List[str]     # What to pay attention to


class DSPClassifier:
    def __init__(
        self,
        case: CaseConfig,
        client: openai.OpenAI,
        policy_model: str = "gpt-4.1-mini",  # For generating stimulus
        target_model: str = "gpt-4.1-mini",        # Main classification model
        max_tokens: int = 300,
        task_definition: Optional[str] = None,
        examples_df: Optional[pd.DataFrame] = None,
        use_rl: bool = False,
        stimulus_type: StimulusType = StimulusType.LINGUISTIC_MARKERS,
    ):
        self.case = case
        self.client = client
        self.policy_model = policy_model
        self.target_model = target_model
        self.max_tokens = max_tokens
        self.task_definition = task_definition
        self.examples_df = examples_df
        self.use_rl = use_rl
        self.stimulus_type = stimulus_type
        
        self.policy_stats = {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_latency": 0.0,
            "total_calls": 0
        }
        
        self.target_stats = {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_latency": 0.0,
            "total_calls": 0
        }
        
        if examples_df is not None:
            self._train_policy_model()
    
    def _train_policy_model(self):
        """Train the policy model to generate appropriate stimulus"""
        training_data = []
        
        for _, row in self.examples_df.iterrows():
            text = row[self.case.text_col]
            label = row[self.case.label_col]
            
            if self.stimulus_type == StimulusType.LINGUISTIC_MARKERS:
                stimulus = self._extract_linguistic_markers(text, label)
            elif self.stimulus_type == StimulusType.CONTEXTUAL_CUES:
                stimulus = self._extract_contextual_cues(text, label)
            elif self.stimulus_type == StimulusType.REASONING_CHAIN:
                stimulus = self._generate_reasoning_chain(text, label)
            
            training_data.append({
                "input": text,
                "stimulus": stimulus,
                "label": label
            })
        
        self.policy_training_data = training_data
    
    def _generate_stimulus(self, text: str) -> DirectionalStimulus:
        """Use policy model to generate directional stimulus"""
        
        prompt = f"""Given the following text for {self.case.case_name} classification, 
        generate {self.stimulus_type.value} that will help identify the correct label.
        
        Text: {text}
        
        Generate directional hints to guide a classifier toward the correct label. 
        Do not reveal the label. Focus on:
        - Key bias-indicating phrases or generalizations
        - Social context or speaker intentions that alter interpretation
        - Step-by-step logic that would help disambiguate

        Format as a comma-separated list."""
        
        import time
        start_time = time.time()
        
        response = call_llm(
            client=self.client,
            model=self.policy_model,
            prompt=prompt,
            system_message=f"You are generating classification hints for {self.case.case_name}.",
            max_tokens=150,
        )
        
        elapsed = time.time() - start_time
        
        if response.usage:
            self.policy_stats["total_tokens"] += response.usage.total_tokens
            self.policy_stats["total_prompt_tokens"] += response.usage.prompt_tokens
            self.policy_stats["total_completion_tokens"] += response.usage.completion_tokens
        self.policy_stats["total_latency"] += elapsed
        self.policy_stats["total_calls"] += 1
        
        stimulus_text = response.choices[0].message.content.strip()
        stimulus_items = [item.strip() for item in stimulus_text.split(",")]
        
        return DirectionalStimulus(
            linguistic_markers=stimulus_items[:3] if len(stimulus_items) >= 3 else stimulus_items,
            contextual_cues=stimulus_items[3:6] if len(stimulus_items) >= 6 else [],
            reasoning_steps=stimulus_items[6:9] if len(stimulus_items) >= 9 else [],
            attention_focus=stimulus_items[9:] if len(stimulus_items) > 9 else []
        )
    
    def classify(self, text: str) -> Tuple[str, Dict]:
        """Classify using DSP approach"""
        
        stimulus = self._generate_stimulus(text)
        
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        label_list = [i for i in self.case.valid_labels]
        
        prompt = f"""Definition of a {self.case.case_name}: {self.task_definition}

Labeling rules:
{rules}

Input: {text}

Consider these hints when making your classification:
- Key patterns: {', '.join(stimulus.linguistic_markers)}
- Important context: {', '.join(stimulus.contextual_cues)}
- Reasoning approach: {', '.join(stimulus.reasoning_steps)}

Based on the input and these guiding hints, classify this text.
Return only one of: {label_list}."""

        import time
        start_time = time.time()
        
        response = call_llm(
            client=self.client,
            model=self.target_model,
            prompt=prompt,
            system_message=f"You are a classifier for {self.case.case_name}. Use the provided hints to guide your decision.",
            max_tokens=self.max_tokens,
        )
        
        elapsed = time.time() - start_time
        
        if response.usage:
            self.target_stats["total_tokens"] += response.usage.total_tokens
            self.target_stats["total_prompt_tokens"] += response.usage.prompt_tokens
            self.target_stats["total_completion_tokens"] += response.usage.completion_tokens
        self.target_stats["total_latency"] += elapsed
        self.target_stats["total_calls"] += 1
        
        stats = {
            "policy_model": {
                "tokens_used": self.policy_stats["total_tokens"],
                "latency": self.policy_stats["total_latency"],
            },
            "target_model": {
                "tokens_used": self.target_stats["total_tokens"],
                "latency": self.target_stats["total_latency"],
            },
            "stimulus_generated": stimulus,
            "total_tokens": self.policy_stats["total_tokens"] + self.target_stats["total_tokens"],
            "total_latency": self.policy_stats["total_latency"] + self.target_stats["total_latency"]
        }
        
        return response.choices[0].message.content.strip(), stats
    
class StereotypeDetectionDSP(DSPClassifier):
    """Specialized DSP for stereotype detection"""
    
    def _extract_linguistic_markers(self, text: str, label: str) -> List[str]:
        """Extract markers specific to stereotypes"""
        markers = []
        
        generalization_words = ["all", "every", "always", "never", "typical", "usually"]
        markers.extend([w for w in generalization_words if w in text.lower()])
        
        group_patterns = ["women", "men", "asian", "black", "white", "old", "young"]
        markers.extend([p for p in group_patterns if p in text.lower()])
        
        if "are" in text or "is" in text:
            markers.append("attribute_assignment")
        
        return markers[:5]  
    

class ManipulationDetectionDSP(DSPClassifier):
    """Specialized DSP for manipulation detection"""
    
    def _extract_contextual_cues(self, text: str, label: str) -> List[str]:
        """Extract manipulation-specific cues"""
        cues = []
        
        emotion_words = ["fear", "anger", "shame", "guilt", "pride"]
        if any(w in text.lower() for w in emotion_words):
            cues.append("emotional_appeal")
        
        urgency_words = ["now", "immediately", "limited time", "act fast"]
        if any(w in text.lower() for w in urgency_words):
            cues.append("false_urgency")
        
        if any(w in text.lower() for w in ["expert", "study shows", "research proves"]):
            cues.append("authority_claim")
        
        return cues
    
class RLEnhancedDSP(DSPClassifier):
    """DSP with RL-based policy improvement"""
    
    def optimize_policy(self, validation_df: pd.DataFrame, n_iterations: int = 10, origin: bool = False, llm_judge: bool = False):
        """Use RL to improve stimulus generation"""
        
        for iteration in range(n_iterations):
            total_reward = 0
            
            for _, row in validation_df.iterrows():
                text = row[self.case.text_col]
                true_label = row[self.case.label_col]
                
                predicted_label, _ = self.classify(text)
                
                if origin: 
                    reward = 1.0 if predicted_label == true_label else -0.5
                else:
                    reward = self.compute_reward(
                        predicted_label=predicted_label,
                        true_label=true_label,
                        explanation=predicted_label,
                        use_judge=llm_judge,
                        judge_confidence=0.0
                    )
                
                total_reward += reward
                self._update_policy(text, reward)
            
            avg_reward = total_reward / len(validation_df)
            print(f"Iteration {iteration + 1}: Average reward = {avg_reward:.3f}")

    
    def compute_reward(
        self,
        predicted_label: str,
        true_label: str,
        explanation: str,
        use_judge: bool = False,
        judge_confidence: float = 0.0,
        max_length: int = 100
    ) -> float:
        """
        Compute reward for RL policy based on predicted label, explanation, and optional LLM-as-a-Judge score.
    
        Args:
            predicted_label (str): The predicted label by the model.
            true_label (str): The ground truth label.
            explanation (str): The explanation provided by the model.
            use_judge (bool): Whether to use the judge confidence as a reward component.
            judge_confidence (float): Confidence score from the LLM-as-a-Judge (0 to 1).
            max_length (int): Maximum acceptable explanation length before applying penalty.
    
        Returns:
            float: The computed reward.
        """
        reward = 0.0

        if predicted_label == true_label:
            reward += 1.0
        else:
            reward -= 0.5

        if "because" in explanation.lower():
            reward += 0.2
    
        if len(explanation.split()) > max_length:
            reward -= 0.3
    
        if use_judge:
            reward += 1.0 * (judge_confidence - 0.5)
    
        return round(reward, 3)