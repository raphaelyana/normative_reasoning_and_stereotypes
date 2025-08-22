from pydantic import BaseModel, Field
from typing import List, Callable, Optional

class CaseConfig(BaseModel):
    case_name: str
    task: str
    valid_labels: List[str]
    label_rules: List[str]
    evaluation_prompt: str
    label_map: dict[str, str]
    input_col: str
    label_col: str
    label_formatter_fewshots: Callable
    example_template_fewshots: Callable
    example_template_tot: Callable
    category_cols: list[str] = []
    cot_focus: Optional[str] = None
    cot_adversarial: Optional[str] = None