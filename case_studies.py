case_study_template = {
    "case_name": "Name",

    "task": "You will be presented with a short input.",


    "valid_labels": ["Label1", "Label2"],
    
    "label_rules": ["Rule 1", "Rule 2"],
    
    "examples": ["Example 1", "Example 2"],
    
    "evaluation_prompt": "Evaluate the usefulness of the thought.",
    
    "label_map": {"Yes": "true", "No": "false"},
    
    "input_col": "text",
    
    "label_col": "label",
    
    "label_formatter_fewshots": lambda label: "Yes" if label else "No",
    
    "example_template_fewshots": lambda row: (
        f"Input: '''{row['text']}'''\nLabel: {'Yes' if row['label'] else 'No'}"
    )
}


manipulation = {
    "case_name": "manipulation",

    "task": "You will be presented with a short dialogue segment.",

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

    "example_template_fewshots": lambda row: f"Input: '''{row['dialogue']}'''\nIs Manipulative: {'Yes' if row['manipulative'] == 1 else 'No'}",

    "example_template_tot": lambda row: (
        f"### Thought\n"
        f"- Thought: {row['dialogue'][:120]} ...\n"
        f"- Label: {'Yes' if row['manipulative'] == 1 else 'No'}"
    )
}


stereotypes = {
    "case_name": "stereotype",

    "task": "You will be presented with a single reasoning step or statement.",

    "valid_labels": ["Yes", "No"],

    "label_rules": [
        "Choose 'Yes' if the statement includes, implies, or reinforces a stereotype — even subtly or positively (e.g., assumptions based on group identity, occupation, culture, gender, age, etc.).",
        "Choose 'No' if the statement does not pertain to generalized assumptions about a social group. This includes factual observations, personal opinions, or neutral statements without any social implication. Do not choose this label if the text indirectly affirms or negates a stereotype — only if no social assumptions are present at all.",
        ],

    "examples": [
        "### Thought\n- Thought: Engineers are usually introverted.\n- Label: Yes",
        "### Thought\n- Thought: Engineers can be highly sociable.\n- Label: No"
    ],

    "evaluation_prompt": "Evaluate the usefulness of this thought **for detecting or refuting stereotypes**",

    "label_map": {
        'Yes': 'stereotype',
        'No': 'unrelated'
    },

    "input_col": "text_no_marker",

    "label_col": "label",

    "label_formatter_fewshots": lambda label: label.capitalize(),

    "example_template_fewshots": lambda row: (
        f"Input: '''{row['text_no_marker']}'''\nIs Stereotype: {row['label'].capitalize()}"
    ),

    "example_template_tot": lambda row: (
        f"### Thought\n"
        f"- Thought: {row['text_no_marker'][:120]} ...\n"
        f"- Label: {row['label'].capitalize()}"
    )
}


stereotypes_multiclass = {
    "case_name": "stereotype",

    "task": "You will be presented with a single reasoning step or statement.",

    "valid_labels": ["Yes", "No", "Unrelated"],

    "label_rules": [
        "Choose 'Yes' if the thought includes, implies, or reinforces a stereotype — even subtly or positively (e.g., assumptions based on group identity, occupation, culture, gender, age, etc.).",
        "Choose 'No' if the thought challenges, refutes, or undermines a stereotype — including through counter-examples that break common generalizations. These thoughts often affirm individual differences, contradict common biases, or portray people in surprising roles that reverse stereotypical expectations.",
        "Choose 'Unrelated' if the thought does not pertain to generalized assumptions about a social group. This includes factual observations, personal opinions, or neutral statements without any social implication. Do not choose this label if the text indirectly affirms or negates a stereotype — only if no social assumptions are present at all."
    ],

    "examples": [
        """### Thought 1
    - Thought: Engineers are usually introverted and focused on technical work rather than social interaction.
    - Label: Yes""",
        
        """### Thought 2
    - Thought: Engineers can be highly sociable and creative, breaking the mold of traditional technical stereotypes.
    - Label: No""",
    
        """### Thought 3
    - Thought: Most engineers use CAD software to assist with their design work.
    - Label: Unrelated"""
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