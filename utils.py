from xai_sdk.chat import user, system

def call_llm(client, model, prompt, system_message=None, max_tokens=300, temperature=0.0):
    
    client_type = str(type(client)).lower()

    if "openai" in client_type:
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

    elif "mistral" in client_type:
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
    
    elif "xai_sdk" in client_type or "Client" in client_type:

        # Create a chat session for the Grok model
        chat = client.chat.create(model=model)

        if system_message:
            chat.append(system(system_message))
        chat.append(user(prompt))

        # Grok's `sample()` handles generation — you can pass temperature/max_tokens
        response = chat.sample(temperature=temperature, max_output_tokens=max_tokens)
        return response

    else:
        raise ValueError("Unsupported client type for call_llm")