from cases.stereotypes_case import stereotypes_case
from cases.manipulation_case import manipulation_case
from cases.mmlu_case import mmlu_case


def get_case_config(task: str):
    if task == "stereotype":
        return stereotypes_case
    elif task == "manipulation":
        return manipulation_case
    elif task == "mmlu":
        return mmlu_case
    else:
        raise ValueError(f"Unknown task: {task}")