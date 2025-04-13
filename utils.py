
def call_llm(client, model, prompt, system_message=None, max_tokens=300):
    if "openai" in str(type(client)).lower():
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
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
        )
        return response
    #response.choices[0].message.content.strip()

    else:
        raise ValueError("Unsupported client type for call_llm")