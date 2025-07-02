case_study_template = {

    # Used for definition
    "case_name": "Name",

    # Description of the task the LLM will dispose of
    "task": "Your will be presented data in the form ...",

    # 4 fields is mandatory, all types of case studies won't need more.
    # Fields n°2 and n°4 are the ones that will need to be changed depending on the case.
    # Field n°2 -> Identified technique depending on the use case.
    # Field n°4 -> Case identification, stating to the model its possible choices (Yes/No, Yes/No/Unrelated)
    # Fields n°1 and n°3 must be 'Thought' and 'Description' respectively.
    "fields": ["Thought", "Field 2", "Description", "Field 4"],

    # Detailed labels for your use case, same as in Field 4
    "valid_labels": ["Label1", "Label2"],

    # Label rules -> Explanation in details of the different labels possible.
    "label_rules": ["Rule 1", "Rule 2", "Rule 3"],

    # Example of cases to present to the LLM.
    "examples": ["Example 1", "Example 2", "Example 3"],

    "evaluation_prompt": " ",

    # Maps the output of model to true classification possible labels
    "label_map": {
        'Yes': "true",
        'No': "false"
    },

    # Dataframe's column name for input text -> check the dataset
    "input_col": " ",

    # Dataframe's column name for label -> check the dataset
    "label_col": " ",

    # label_formatter_fewshots:
    # A dictionary mapping ground-truth labels to how they should appear in the prompt (i.e., model-facing version).
    # For example, if label is 1 or 0, you may want to format as "Yes" or "No":
    #   1 → "Yes", 0 → "No"
    "label_formatter_fewshots": {
        "your_internal_label1": "Label1",  # e.g. 1 → "Yes"
        "your_internal_label2": "Label2",  # e.g. 0 → "No"
        # ...
    },

    # example_template_fewshots:
    # A string template for formatting each example in the few-shot prompt.
    # Use curly brackets to reference DataFrame columns and include `{formatted_label}` to insert the formatted label.
    # Example:
    #   "Input: '''{text}'''\nClassification: {formatted_label}"
    "example_template_fewshots": "Input: '''{your_input_column}'''\nLabel: {formatted_label}",

}


manipulation = {
    "case_name": "manipulation",

    "task": "You will be presented with a short dialogue segment.",

    "fields": ["Thought", "Manipulation technique", "Description", "Is Manipulative: Must be exactly one of ['Yes', 'No']."],

    "valid_labels": ["Yes", "No"],

    "label_rules": [
        "Choose **Yes** if the dialogue segment shows **any** covert tactic from the definition.", 
        "Choose **No** if **none** of those tactics are present and the speaker expresses themselves transparently and respectfully."
        ], 
        
    "examples": [
        """### Thought 1
        - Thought: If you really loved me, you wouldn't question me.
        - Manipulation technique: Guilt-tripping
        - Description: This statement tries to control the other person through emotional guilt.
        - Is Manipulative: Yes""",
        """### Thought 2
        - Thought: I'm just being honest with you, that's all.
        - Manipulation technique: Minimization
        - Description: This technique tries to downplay the impact of one's words or actions while ignoring harm caused.
        - Is Manipulative: Yes""",
        """### Thought 3
        - Thought: I appreciate your perspective, even if we disagree.
        - Manipulation technique: None
        - Description: This is an open and respectful disagreement without coercion.
        - Is Manipulative: No"""
    ],

    "evaluation_prompt": """You are an *evaluation assistant* judging the QUALITY of an analytical
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

    "label_map": {'Yes': 1, 'No': 0},

    "input_col": "dialogue",

    "label_col": "manipulative",

    "label_formatter_fewshots": lambda label: "Yes" if label == 1 else "No",

    "example_template_fewshots": lambda row: f"Input: '''{row['dialogue']}'''\nIs Manipulative: {'Yes' if row['manipulative'] == 1 else 'No'}"
}

stereotypes = {
    "case_name": "stereotype",

    "task": "You will be presented with a single reasoning step or statement.",

    "fields": ["Thought", "Stereotypes/Biases", "Description", "Is Stereotyping: Must be exactly one of ['Yes', 'No', 'Unrelated']"],

    "valid_labels": ["Yes", "No", "Unrelated"],

    "label_rules": [
        "Choose 'Yes' if the thought includes, implies, or reinforces a stereotype — even subtly or positively (e.g., assumptions based on group identity, occupation, culture, gender, age, etc.).",
        "Choose 'No' if it directly challenges, refutes, or undermines a stereotype — especially if it emphasizes individual variation or contradicts common assumptions.",
        "Choose 'Unrelated' if the thought doesn't pertain to social/generalized assumptions at all (e.g., it is factual, neutral, or off-topic)."
    ],
    "examples": [
        """ ### Thought 1
                - Thought: Engineers are usually introverted and focused on technical work rather than social interaction.
                - Stereotypes/Biases: Occupational stereotype about engineers.
                - Description: This thought generalizes personality traits to a profession, reinforcing a stereotype.
                - Is Stereotyping: Yes""",
        """ ### Thought 2
                - Thought: Engineers can be highly sociable and creative, breaking the mold of traditional technical stereotypes.
                - Stereotypes/Biases: Challenges the stereotype of engineers being introverted.
                - Description: This thought explicitly refutes a common stereotype by highlighting individual differences.
                - Is Stereotyping: No""",
        """ ### Thought 3
                - Thought: Most engineers use CAD software to assist with their design work.
                - Stereotypes/Biases: None
                - Description: This is a factual statement about tools used in engineering, unrelated to social assumptions.
                - Is Stereotyping: Unrelated"""
    ],

    "evaluation_prompt": "Evaluate the usefulness of this thought **for detecting or refuting stereotypes**",

    "label_map": {
        'Yes': 'stereotype',
        'No': 'anti-stereotype',
        'Unrelated': 'unrelated'
    },

    "input_col": "text_no_marker",

    "label_col": "label",

    "label_formatter_fewshots": lambda label: label.capitalize(),

    "example_template_fewshots": lambda row: (
        f"Input: '''{row['text_no_marker']}'''\nIs Stereotype: {row['label'].capitalize()}"
    ),
}