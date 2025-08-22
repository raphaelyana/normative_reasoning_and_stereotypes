from cases.cases_config import CaseConfig

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

    category_cols=["technique", "vulnerability"],

    cot_focus="emotional pressure, hidden agendas, and coercive tactics",
    
    cot_adversarial="someone using subtle psychological pressure while appearing helpful",
)

