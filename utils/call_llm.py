from xai_sdk.chat import user, system
from typing import Optional
from types import SimpleNamespace

def _pack_response(content: str,
                   prompt_tokens: Optional[int] = None,
                   completion_tokens: Optional[int] = None,
                   raw=None):
    usage = None
    if (prompt_tokens is not None) or (completion_tokens is not None):
        total = (prompt_tokens or 0) + (completion_tokens or 0)
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total
        )

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
        raw=raw
    )


def call_llm(client,
             model: str,
             prompt: str,
             system_message: Optional[str] = None,
             max_tokens: int = 300,
             temperature: float = 0.0,
             provider: Optional[str] = None):

    client_mod = getattr(client, "__class__", type(client)).__module__.lower() if client else ""
    client_name = getattr(client, "__class__", type(client)).__name__.lower() if client else ""
    fingerprint = f"{client_mod}.{client_name}"

    def with_sys(msgs):
        if system_message:
            return [{"role": "system", "content": system_message}, *msgs]
        return msgs

    # OpenAI models
    if (provider == "openai") or ("openai" in fingerprint):
        messages = with_sys([{"role": "user", "content": prompt}])
        r = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        content = r.choices[0].message.content
        pu = getattr(getattr(r, "usage", None), "prompt_tokens", None)
        cu = getattr(getattr(r, "usage", None), "completion_tokens", None)
        return _pack_response(content, pu, cu, raw=r)

    # Mistral
    if (provider == "mistral") or ("mistral" in fingerprint):
        messages = with_sys([{"role": "user", "content": prompt}])
        r = client.chat.complete(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )

        msg = r.choices[0].message
        content = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", "")
        usage = getattr(r, "usage", None)
        pu = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        cu = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        return _pack_response(content, pu, cu, raw=r)

    # Anthropic
    if (provider == "anthropic") or ("anthropic" in fingerprint):

        msgs = [{"role": "user", "content": prompt}]
        r = client.messages.create(
            model=model, system=system_message, messages=msgs,
            max_tokens=max_tokens, temperature=temperature
        )

        parts = []
        for block in getattr(r, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        content = "".join(parts)
        usage = getattr(r, "usage", None)
        pu = getattr(usage, "input_tokens", None)
        cu = getattr(usage, "output_tokens", None)
        return _pack_response(content, pu, cu, raw=r)

    # Gemini api
    if (provider == "gemini") or ("google.generativeai" in fingerprint) or ("google.genai" in fingerprint):
        try:
            import google.generativeai as genai
        except Exception:
            from google import genai as genai
    
        #model_obj = genai.GenerativeModel(
        #    model if model else "gemini-2.5-flash",
        #    system_instruction=system_message if system_message else None
        #)

        #safety_settings = [
        #    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        #]
    
        #r = model_obj.generate_content(
        #    prompt,
        #    generation_config={
        #        "max_output_tokens": max_tokens,
        #        "temperature": temperature,
        #        "candidate_count": 1,
        #        "response_mime_type": "text/plain",
        #    },
        #    safety_settings=safety_settings,
        #)
    
        # Reuse the model object if caller passed one
        if hasattr(client, "generate_content"):
            model_obj = client
        else:
            model_obj = genai.GenerativeModel(
                model if model else "gemini-2.5-flash",
                system_instruction=system_message if system_message else None
            )
    
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
        r = model_obj.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 50,
                "temperature": temperature,
                "candidate_count": 1,
                "response_mime_type": "text/plain",
            },
            safety_settings=safety_settings,
        )
    
        def _extract_text(resp):
            try:
                t = getattr(resp, "text", None)
                if t:
                    return t
            except Exception:
                pass
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    out = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
                    if out:
                        return out
            return ""
    
        content = _extract_text(r)
        usage = getattr(r, "usage_metadata", None) or getattr(r, "usageMetadata", None)
        pu = getattr(usage, "prompt_token_count", None)
        cu = getattr(usage, "candidates_token_count", None)
        return _pack_response(content, pu, cu, raw=r)


    # cohere
    if (provider == "cohere") or ("cohere" in fingerprint):
        messages = with_sys([{"role": "user", "content": prompt}])
        r = client.chat(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        msg = getattr(r, "message", None)
        parts = getattr(msg, "content", []) if msg else []
        content = ""
        if parts and isinstance(parts, list):
            first = parts[0]
            content = getattr(first, "text", "") if not isinstance(first, dict) else first.get("text", "")
        meta = getattr(r, "meta", None)
        tokens = getattr(meta, "tokens", None)
        pu = getattr(tokens, "input_tokens", None) if tokens else None
        cu = getattr(tokens, "output_tokens", None) if tokens else None
        return _pack_response(content, pu, cu, raw=r)

    # Qwen models
    if (provider == "dashscope") or ("dashscope" in fingerprint) or ("tongyi" in fingerprint) or ("alibaba" in fingerprint):
        from dashscope import Generation
        messages = with_sys([{"role": "user", "content": prompt}])
        r = Generation.call(model=model, messages=messages, result_format="message")
        out = getattr(r, "output", None)
        choices = getattr(out, "choices", []) if out else []
        message = choices[0].message if choices else None
        content = getattr(message, "content", "") if message else ""
        usage = getattr(r, "usage", None)
        pu = getattr(usage, "input_tokens", None) if usage else None
        cu = getattr(usage, "output_tokens", None) if usage else None
        return _pack_response(content, pu, cu, raw=r)

    # Grok models
    if (provider == "xai") or ("xai_sdk" in fingerprint) or ("x.ai" in fingerprint) or ("xai" in fingerprint):
        from xai_sdk.chat import user as x_user, system as x_system
        import inspect

        chat = client.chat.create(model=model)
        if system_message:
            chat.append(x_system(system_message))
        chat.append(x_user(prompt))

        sig = inspect.signature(chat.sample)
        kwargs = {}
        if "max_output_tokens" in sig.parameters:
            kwargs["max_output_tokens"] = max_tokens
        elif "max_tokens" in sig.parameters:
            kwargs["max_tokens"] = max_tokens
        if "temperature" in sig.parameters:
            kwargs["temperature"] = temperature
        elif "temp" in sig.parameters:
            kwargs["temp"] = temperature

        r = chat.sample(**kwargs) if kwargs else chat.sample()
        content = (
            getattr(r, "output_text", None)
            or getattr(r, "text", None)
            or getattr(getattr(r, "message", None), "text", None)
            or str(r)
        )
        return _pack_response(content, None, None, raw=r)

    raise ValueError(f"Unsupported client/provider for call_llm (fingerprint='{fingerprint}', provider='{provider}')")