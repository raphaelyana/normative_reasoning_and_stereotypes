from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Literal

class ThoughtOutput(BaseModel):
    thought: str = Field(..., description="Concise analytical reasoning.")
    label: Optional[Literal["Yes", "No"]] = Field(default=None, description="Optional: present only for leaf nodes")

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


stereotypes_case = CaseConfig(
    case_name="stereotype",
    task="You will be presented with a single reasoning step or statement.",
    valid_labels=["Yes", "No"],
    label_rules=[
        "Choose 'Yes' if the statement includes, implies, or reinforces a stereotype — even subtly or positively (e.g., assumptions based on group identity, occupation, culture, gender, age, etc.).",
        "Choose 'No' if the statement does not pertain to generalized assumptions about a social group. This includes factual observations, personal opinions, or neutral statements without any social implication. Do not choose this label if the text indirectly affirms or negates a stereotype — only if no social assumptions are present at all.",
        ],
    evaluation_prompt="Evaluate the usefulness of this thought **for detecting or refuting stereotypes**",
    label_map={"Yes": "stereotype", "No": "unrelated"},
    input_col="text_no_marker",
    label_col="label",
    label_formatter_fewshots= lambda label: label.capitalize(),
    example_template_fewshots= lambda row: (
        f"Input: '''{row['text_no_marker']}'''\nIs Stereotype: {row['label'].capitalize()}"
    ),
    example_template_tot= lambda row: (
        f"### Thought\n"
        f"- Thought: {row['text_no_marker'][:120]} ...\n"
        f"- Label: {row['label'].capitalize()}"
    ),
    category_cols = ["stereotype_type"]
)

manipulation_case = CaseConfig(
    case_name="manipulation",
    task="You will be presented with a short dialogue segment.",
    valid_labels=["Yes", "No"],
    label_rules=[
        "Choose **Yes** if the dialogue segment shows **any** covert tactic from the definition.",
        "Choose **No** if **none** of those tactics are present and the speaker expresses themselves transparently and respectfully."
    ],
    evaluation_prompt="""You are an *evaluation assistant* judging the QUALITY of an analytical
            **Thought** about mental–manipulation.

            Given:
            • the original dialogue fragment (quoted below)
            • the candidate *Thought* you are scoring
            • optionally, the *parent Thought* (if any)

            Score the candidate on four weighted criteria  
            (use the weights internally – just return the final 0-1 score):

            1. **Verdict fidelity (0 .40)**  
               – Does the candidate’s “Is Manipulative: <Yes|No>” match the ground-truth
                 definition of manipulation?  
                 – If “Yes”, the quoted utterance truly contains at least one tactic from the
                   definition.  
                 – If “No”, none of the tactics appear.

            2. **Technique accuracy (0 .30)**  
               – If the verdict is *Yes*, is the named “Manipulation technique” a sensible
                 match?  
               – If the verdict is *No*, the technique should be “None”.

            3. **Explanation clarity (0 .20)**  
               – Does the “Description” give a clear, concise justification that would help a
                 human annotator see why the verdict is correct?  
               – Penalise vague or circular descriptions.

            4. **Added insight vs parent (0 .10)**  
               – If there is a parent Thought, does the candidate add new information or a
                 sharper angle, rather than restating it?

            Return **only one floating-point number between 0.0 and 1.0**, where 0.0 is the worse and 1.0 is the best""",
    label_map={"Yes": "1", "No": "0"},
    input_col="dialogue",
    label_col="manipulative",
    label_formatter_fewshots=lambda label: "Yes" if label == 1 else "No",
    example_template_fewshots=lambda row: (
        f"Input: '''{row['dialogue']}'''\nIs Manipulative: {'Yes' if row['manipulative'] == 1 else 'No'}"
    ),
    example_template_tot=lambda row: (
        f"### Thought\n"
        f"- Thought: {row['dialogue'][:120]} ...\n"
        f"- Label: {'Yes' if row['manipulative'] == 1 else 'No'}"
    ),
    category_cols=["technique", "vulnerability"]
)



mmlu_case = CaseConfig(
    case_name="multiple-choice question",
    task="You will be presented with a multiple-choice question from a variety of subjects.",
    valid_labels=["A", "B", "C", "D"],
    label_rules=[
        "Choose the correct answer from the options provided (A, B, C, D).",
        "Do not justify your answer. Return only the letter."
    ],
    evaluation_prompt="Evaluate the correctness of the answer based on the question and choices.",
    label_map={"A": "0", "B": "1", "C": "2", "D": "3"},
    input_col="formatted_prompt",
    label_col="answer",
    label_formatter_fewshots=lambda label: label,
    example_template_fewshots=lambda row: f"Question: {row['question']}\nChoices: {row['choices']}\nAnswer: {row['answer']}",
    example_template_tot=lambda row: "",
    category_cols=[]
)




def get_case_config(task: str):
    if task == "stereotype":
        return stereotypes_case
    elif task == "manipulation":
        return manipulation_case
    elif task == "mmlu":
        return mmlu_case
    else:
        raise ValueError(f"Unknown task: {task}")