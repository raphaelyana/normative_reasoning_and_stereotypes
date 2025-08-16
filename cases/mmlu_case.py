from cases.cases_config import CaseConfig


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

    category_cols=[],       # will actually be updated depending on the chosen categories
                            # in the implementation part, so to leave empty     
)
