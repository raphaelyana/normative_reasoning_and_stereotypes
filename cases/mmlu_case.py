from cases.cases_config import CaseConfig


mmlu_case = CaseConfig(

    case_name="mmlu",

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

    example_template_tot=lambda row: (
        f"### Thought\n"
        f"- Thought: {row['question']} Choices: {row['choices']}\n"
        f"- Label: {row['answer']}"
    ),

    label_instructions_tot={
    True: "At the end of each thought, add a field 'label' with a value of either 'A', 'B', 'C', or 'D'.",
    False: "Do not include any 'label' field — just output the 'thought' reasoning steps."
    },

    category_cols=["subject"],  # will actually be updated depending on the chosen categories
                                # in the implementation part, so to leave empty    

    # Not sure yet these work for the CoT using MMLU but it is worth trying it.
    cot_focus="key facts, definitions, and logical eliminations",
    cot_adversarial="distractor options that look plausible but violate a key fact",

    case_type="math_logic",  # "normative" or "math_logic"
)
