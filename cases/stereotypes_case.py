from cases.cases_config import CaseConfig


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
        
        category_cols = ["stereotype_type"],

        cot_focus="ANY generalizations about groups, even subtle ones, implied assumptions, or coded references",
        
        cot_adversarial="someone making it seem factual when it's actually a generalization about people",
    )