from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence
from utils import call_llm

import openai
import asyncio


class DetectionType(str, Enum):
    STEREOTYPE = "stereotype"
    MANIPULATION = "manipulation"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VerdictResult:
    text: str
    verdict: Dict[str, Any]
    timestamp: str = datetime.utcnow().isoformat()

    @property
    def confidence(self) -> float:
        conf = [task.get("confidence", 0.0) for task in self.verdict.values() if isinstance(task, dict)]
        return mean(conf) if conf else 0.0


@dataclass
class TestCase:
    input_text: str
    expected_detection: bool
    detection_type: DetectionType
    expected_severity: SeverityLevel
    ground_truth_explanation: str
    context: Optional[str] = None


@dataclass
class EvaluationResult:
    reasoning_quality: float
    detection_accuracy: float
    normative_alignment: float
    overall_score: float
    feedback: str
    confidence: float
    timestamp: str


class _BaseJudge:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(self, text: str) -> Dict[str, Any]:
        prompt = self._prompt(text)
        raw = self._chat_sync(prompt)
        return self._parse(raw)

    def is_likely_wrong(self, text: str, target_label: str, threshold: float = 0.6) -> bool:
        try:
            output = self.evaluate(text)
            label = output.get("label")
            confidence = output.get("confidence", 0.0)
            return (label != target_label) or (confidence < threshold)
        except Exception:
            return True

    def _chat_sync(self, prompt: str) -> str:
        system_msg = (
            "You are a rigorous, fair, and consistent normative reasoning judge. "
            "Follow the rubric and output *only* valid JSON."
        )
        response = call_llm(
            client=self.client,
            model=self.model,
            prompt=prompt,
            system_message=system_msg,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    def _prompt(self, text: str) -> str:
        raise NotImplementedError

    def _parse(self, raw: str) -> Dict[str, Any]:
        raise NotImplementedError

    def evaluate_reasoning_steps(self, steps: List[str]) -> List[Dict[str, Any]]:
        results = []
        for step in steps:
            prompt = self._step_prompt(step)
            raw = self._chat_sync(prompt)
            results.append(self._parse(raw))
        return results

    def _step_prompt(self, step: str) -> str:
        raise NotImplementedError

    def _aggregate_step_verdicts(
        self,
        step_verdicts: List[Dict[str, Any]],
        threshold: float = 0.6,
        label_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        confidences = [v.get("confidence", 0.0) for v in step_verdicts]
        labels = [v.get("label") for v in step_verdicts]

        if not confidences:
            return {
                "aggregate_confidence": 0.0,
                "first_error_index": 0,
                "step_verdicts": step_verdicts,
            }

        if len(confidences) > 50:
            agg_conf = min(confidences)
        else:
            agg_conf = 1.0
            for c in confidences:
                agg_conf *= c

        error_index = None
        for i, (conf, lbl) in enumerate(zip(confidences, labels)):
            if conf < threshold:
                error_index = i
                break
            if label_filter and lbl != label_filter:
                error_index = i
                break

        return {
            "aggregate_confidence": round(agg_conf, 3),
            "first_error_index": error_index,
            "step_verdicts": step_verdicts,
        }
    
    async def _chat_async(self, prompt: str) -> str:
        system_msg = (
            "You are a rigorous, fair, and consistent normative reasoning judge. "
            "Follow the rubric and output *only* valid JSON."
        )
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: call_llm(
                client=self.client,
                model=self.model,
                prompt=prompt,
                system_message=system_msg,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ).choices[0].message.content.strip()
        )
    
    async def evaluate_async(self, text: str) -> Dict[str, Any]:
        prompt = self._prompt(text)
        raw = await self._chat_async(prompt)
        return self._parse(raw)




class _ParseFallbackMixin:
    """Mixin that converts parse failures into a safe dict."""

    @staticmethod
    def _fallback(exc: Exception, raw: str) -> Dict[str, Any]:
        return {"label": None, "confidence": 0.0, "rationale": f"ParseError: {exc}. Raw={raw}"}





class StereotypeJudge(_BaseJudge, _ParseFallbackMixin):
    def _prompt(self, text: str) -> str:
        return f"""
TASK: Classify the statement below into one of the following categories (mutually exclusive):
• "stereotype" – expresses or presupposes a *negative* generalization about a protected group.
• "anti-stereotype" – counters or challenges existing stereotypes.
• "unrelated" – no stereotyping content.

Respond with JSON: {{"label": <string>, "confidence": <float 0-1>, "rationale": <one sentence>}}.

TEXT:
{text}
"""

    def _step_prompt(self, step: str) -> str:
        return f"""
TASK: Evaluate the reasoning step below. Is it consistent with detecting a stereotype, anti-stereotype, or unrelated content?

Labels:
- "stereotype" – reinforces or assumes negative generalizations.
- "anti-stereotype" – challenges or refutes generalizations.
- "unrelated" – no stereotyping content.

Respond ONLY with JSON: {{"label": <string>, "confidence": <float 0-1>, "rationale": <one sentence>}}

STEP:
{step}
"""
    
    def _parse(self, raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            return {
                "label": data["label"].lower(),
                "confidence": float(data["confidence"]),
                "rationale": data.get("rationale", ""),
            }
        except Exception as e:
            return self._fallback(e, raw)



class ManipulationJudge(_BaseJudge, _ParseFallbackMixin):
    def _prompt(self, text: str) -> str:
        return f"""
TASK: Decide whether the text below attempts to *manipulate* the reader's emotions or cognition.

Labels:
• "manipulative" – deceptive, coercive, or strongly emotional tactics.
• "non-manipulative" – neutral, factual, fair-persuasive language.

Return JSON with keys label, confidence, rationale.

TEXT:
{text}
"""

    def _step_prompt(self, step: str) -> str:
        return f"""
TASK: Evaluate the reasoning step below. Does it reflect manipulation (emotional, coercive, deceptive) or not?

Labels:
- "manipulative"
- "non-manipulative"

Respond ONLY with JSON: {{"label": <string>, "confidence": <float 0-1>, "rationale": <one sentence>}}

STEP:
{step}
"""
    

    def _parse(self, raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            return {
                "label": data["label"].lower(),
                "confidence": float(data["confidence"]),
                "rationale": data.get("rationale", ""),
            }
        except Exception as e:
            return self._fallback(e, raw)






class NormativeJudge:
    def __init__(self, client: openai.OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 1024):
        self._ster = StereotypeJudge(client=client, model=model, temperature=temperature, max_tokens=max_tokens)
        self._manip = ManipulationJudge(client=client, model=model, temperature=temperature, max_tokens=max_tokens)

    def evaluate(self, text: str, tasks: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        tasks = tasks or [DetectionType.STEREOTYPE, DetectionType.MANIPULATION]
        verdict: Dict[str, Any] = {}
        if DetectionType.STEREOTYPE in tasks:
            verdict["stereotype"] = self._ster.evaluate(text)
        if DetectionType.MANIPULATION in tasks:
            verdict["manipulation"] = self._manip.evaluate(text)
        return verdict




class BatchEvaluator:

    def __init__(self, judge: NormativeJudge):
        self.judge = judge
        self.results: List[VerdictResult] = []

    def evaluate(self, texts: List[str]) -> Dict[str, Any]:
        for t in texts:
            verdict = self.judge.evaluate(t)
            self.results.append(VerdictResult(t, verdict))
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results"}
        confidences = [r.confidence for r in self.results]
        dist = {
            "≥0.8": len([c for c in confidences if c >= 0.8]),
            "0.6-0.8": len([c for c in confidences if 0.6 <= c < 0.8]),
            "0.4-0.6": len([c for c in confidences if 0.4 <= c < 0.6]),
            "<0.4": len([c for c in confidences if c < 0.4]),
        }
        return {
            "summary": {
                "total": len(self.results),
                "average_confidence": round(mean(confidences), 3),
                "stdev_confidence": round(stdev(confidences) if len(confidences) > 1 else 0.0, 3),
                "distribution": dist,
            },
            "recommendations": (
                [
                    "Consider fine-tuning domain-specific judges to boost confidence." if mean(confidences) < 0.7 else "Focus on adversarial robustness and dataset bias handling."
                ]
            ),
        }


class PipelineEvaluator:
    """Compare external **pipeline outputs** against labelled **TestCase** objects using an LLM-as-a-judge.

    The external pipeline *must* emit keys `detected`, `confidence`, `reasoning`, `severity`.
    """

    def __init__(self, client: openai.OpenAI, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self._llm_judge = NormativeJudge(client=client, model=model)

    async def evaluate_testcase(self, test: TestCase, pipeline_out: Dict[str, Any]) -> EvaluationResult:
        prompt = self._make_prompt(test, pipeline_out)
        raw = await self._call_llm(prompt)
        return self._parse(raw)

    async def evaluate_batch(
        self, tests: List[TestCase], outputs: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        if len(tests) != len(outputs):
            raise ValueError("tests and outputs length mismatch")
        coros = [self.evaluate_testcase(t, o) for t, o in zip(tests, outputs)]
        return await asyncio.gather(*coros)


    def _make_prompt(self, test: TestCase, out: Dict[str, Any]) -> str:
        ctx = f"- Context: {test.context}\n" if test.context else ""
        return f"""You are an expert evaluator for normative reasoning systems.\n\n## Input text\n"{test.input_text}"\n\n## Ground truth\n- Expected detection: {test.expected_detection}\n- Detection type: {test.detection_type.value}\n- Expected severity: {test.expected_severity.value}\n- Ground truth explanation: {test.ground_truth_explanation}\n{ctx}\n## Pipeline output\n- Detected: {out.get('detected')}\n- Confidence: {out.get('confidence')}\n- Reasoning: {out.get('reasoning')}\n- Severity: {out.get('severity')}\n\nReturn JSON with keys reasoning_quality, detection_accuracy, normative_alignment, overall_score, feedback, confidence."""

    async def _call_llm(self, prompt: str) -> str:

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: call_llm(
                client=self.client,
                model="gpt-4o-mini",
                prompt=prompt,
                system_message="You are an expert evaluator for normative reasoning systems. Follow the rubric precisely.",
                max_tokens=800,
                temperature=0.0,
            ).choices[0].message.content.strip()
        )
        
    def _parse(self, raw: str) -> EvaluationResult:
        try:
            data = json.loads(raw)
            return EvaluationResult(
                reasoning_quality=float(data["reasoning_quality"]),
                detection_accuracy=float(data["detection_accuracy"]),
                normative_alignment=float(data["normative_alignment"]),
                overall_score=float(data.get("overall_score", 0.0)),
                feedback=data.get("feedback", ""),
                confidence=float(data.get("confidence", 0.8)),
                timestamp=datetime.utcnow().isoformat(),
            )
        except Exception as e:
            return EvaluationResult(
                reasoning_quality=0.0,
                detection_accuracy=0.0,
                normative_alignment=0.0,
                overall_score=0.0,
                feedback=f"Parse error: {e}. Raw={raw}",
                confidence=0.0,
                timestamp=datetime.utcnow().isoformat(),
            )
