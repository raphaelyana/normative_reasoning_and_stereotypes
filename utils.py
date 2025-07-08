
def call_llm(client, model, prompt, system_message=None, max_tokens=300, temperature=0.0):
    if "openai" in str(type(client)).lower():
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response
    #response.choices[0].message.content.strip()

    elif "mistral" in str(type(client)).lower():
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response
    #response.choices[0].message.content.strip()

    else:
        raise ValueError("Unsupported client type for call_llm")
    

print_old_results = """
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Parameters
regimes = ["low", "medium", "high"]
results_dir = "results/zero_shot"
case_name = "manipulation"  # or "stereotype"
prompt_type = "long"  # or "long" if needed
results_template = "results_{case}_zero_shot_prompt_{prompt}.csv"

# Evaluation loop
for regime in regimes:
    file_path = os.path.join(results_dir, regime, results_template.format(case=case_name, prompt=prompt_type))
    
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        continue

    print(f"\n=== Regime: {regime.upper()} ===")
    
    df = pd.read_csv(file_path)
    
    y_true = df["true_label"].astype(str).str.strip().str.lower()
    y_pred = df["pred_label"].astype(str).str.strip().str.lower()

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, digits=3))

    print("\n--- Confusion Matrix ---")
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    accuracy = (y_true == y_pred).mean()
    print(f"\n--- Accuracy: {accuracy:.2%} ---")

    # Optional: Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {regime} ({prompt_type} prompt)")
    plt.show()
"""