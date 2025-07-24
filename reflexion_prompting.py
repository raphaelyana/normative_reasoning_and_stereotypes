from typing import Optional, Dict, Literal
import time
import openai
from utils import call_llm
from profile_message import make_system_message, PERSON_SEEDS
from cases import CaseConfig
from llm_judge import NormativeJudge

DEFAULT_MODEL_DICT = {
    "default": "gpt-4o-mini"
}

class ReflexionSingleRetry:
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
        judge: Optional[NormativeJudge] = None,
        detection_type: Literal["stereotype", "manipulation"] = "stereotype",
    ):
        self.case = case
        self.client = client
        self.model = model if model else DEFAULT_MODEL_DICT["default"]
        self.max_tokens = max_tokens
        self.task_definition = task_definition
        self.person_key = person_key
        self.role_playing = role_playing
        self.person_seeds = person_seeds
        self.detection_type = detection_type
        self.judge = judge or NormativeJudge(client=client, model=self.model)

    def _build_system_message(self) -> str:
        if self.person_key and self.role_playing == "passive":
            return make_system_message(self.case.case_name, self.person_key)["content"]
        elif self.person_key and self.role_playing == "active":
            return (
                f"You are a classifier for {self.case.case_name}.\n"
                f"Answer as if you were the following person:\n{self.person_seeds[self.person_key]}"
            )
        else:
            return f"You are a classifier for {self.case.case_name}. Use the rules to decide the correct label."

    def _format_prompt(self, input_text: str, reflection: Optional[str] = None) -> str:
        rules = "\n".join(f"- {r}" for r in self.case.label_rules)
        labels = self.case.valid_labels

        base = f"""Definition of {self.case.case_name}: {self.task_definition}

Labeling rules:
{rules}

Input:
{input_text}
"""
        if reflection:
            base += f"\nReflection on past mistake:\n{reflection}\n"

        base += f"\nReturn only one of: {labels}."
        return base

    def _get_reflection(self, input_text: str, wrong_pred: str, correct_label: str) -> str:
        prompt = f"""You previously misclassified the following input:

Input: {input_text}
Predicted: {wrong_pred}
True label: {correct_label}

Please reflect on the error. What was likely misunderstood? What should be done differently?"""

        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=prompt,
            system_message="You are a careful analyst who reflects on misclassifications.",
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    def classify(self, input_text: str, true_label: str) -> Dict[str, any]:
        """Run a prediction, judge it, and retry with reflection if needed."""
        system_message = self._build_system_message()
        initial_prompt = self._format_prompt(input_text)

        t0 = time.time()
        initial_response = call_llm(
            client=self.client,
            model=self.model,
            prompt=initial_prompt,
            system_message=system_message,
            max_tokens=self.max_tokens
        )
        t1 = time.time()

        initial_pred = initial_response.choices[0].message.content.strip()

        judge_func = self.judge._ster if self.detection_type == "stereotype" else self.judge._manip
        is_wrong = judge_func.is_likely_wrong(input_text, target_label=true_label)

        if not is_wrong:
            return {
                "final_pred": initial_pred,
                "initial_pred": initial_pred,
                "retried": False,
                "reflection": None,
                "latency": t1 - t0,
            }

        reflection = self._get_reflection(input_text, wrong_pred=initial_pred, correct_label=true_label)
        retry_prompt = self._format_prompt(input_text, reflection=reflection)

        t2 = time.time()
        retry_response = call_llm(
            client=self.client,
            model=self.model,
            prompt=retry_prompt,
            system_message=system_message,
            max_tokens=self.max_tokens
        )
        t3 = time.time()

        retry_pred = retry_response.choices[0].message.content.strip()

        return {
            "initial_pred": initial_pred,
            "final_pred": retry_pred,
            "retried": True,
            "reflection": reflection,
            "latency": (t1 - t0) + (t3 - t2),
        }
