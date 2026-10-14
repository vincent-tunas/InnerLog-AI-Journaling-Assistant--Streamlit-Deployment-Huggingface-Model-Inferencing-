"""Shared inference backend for InnerLog: Hugging Face Inference or Ollama.

`generate_response` always returns text. Agents remain responsible for validating
parsed JSON against their own output schema.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()


def inference_provider():
    provider = os.getenv("LLM_PROVIDER", "huggingface").lower().strip()
    if provider not in {"huggingface", "ollama"}:
        raise ValueError("LLM_PROVIDER must be 'huggingface' or 'ollama'")
    return provider


def _unsupported_schema_error(exc):
    """Only fall back if an API rejects the structured-output parameter.

    Authentication, throttling, model errors, timeouts, and other failures must
    propagate unchanged rather than being hidden by a second paid request.
    """
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status not in (400, 422):
        return False

    detail = str(exc).lower()
    return any(term in detail for term in (
        "response_format", "json_schema", "structured output",
        "structured_output", "guided_json", "unsupported format",
    )) and any(term in detail for term in (
        "unsupported", "not supported", "not available", "not implemented",
        "unknown", "unrecognized", "invalid", "extra", "not permitted",
    ))


def generate_response(system_prompt, user_prompt, model, temperature=0.3,
                      max_tokens=1000, timeout=180, json_schema=None):
    """Generate text from the selected inference provider.

    Parameters are backwards-compatible with the original InnerLog agents.
    When `json_schema` is provided, request constrained JSON output where
    supported. For HF providers explicitly rejecting structured output, retry
    once without the API constraint, with schema guidance in the prompt.
    Agents must *still* parse and validate the returned text.
    """
    if json_schema is not None and not isinstance(json_schema, dict):
        raise TypeError("json_schema must be a dictionary or None")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if inference_provider() == "ollama":
        from ollama import Client

        args = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_schema is not None:
            args["format"] = json_schema
        response = Client(
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout=timeout,
        ).chat(**args)
        text = response.message.content
    else:
        from huggingface_hub import InferenceClient

        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("Missing HF_TOKEN for Hugging Face inference")

        client = InferenceClient(api_key=token, provider="auto", timeout=timeout)
        request_options = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "qwen3.5" in model.lower():
            # This is provider-dependent; unsupported providers may ignore it.
            request_options["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        if json_schema is not None:
            request_options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "innerlog_response",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        try:
            response = client.chat.completions.create(**request_options)
        except Exception as exc:
            if json_schema is None or not _unsupported_schema_error(exc):
                raise
            # A provider rejected schema enforcement; use prompt-only JSON.
            # Avoid mutating the caller's messages or original request.
            fallback_options = dict(request_options)
            fallback_options.pop("response_format", None)
            fallback_options["messages"] = [
                {
                    "role": "system",
                    "content": system_prompt + "\nReturn only valid JSON matching this schema: "
                    + json.dumps(json_schema, ensure_ascii=False),
                },
                messages[1],
            ]
            response = client.chat.completions.create(**fallback_options)

        if not response.choices:
            raise ValueError("Inference provider returned no completion choices")
        choice = response.choices[0]
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            raise ValueError(
                "Inference provider returned empty content "
                f"(finish_reason={choice.finish_reason!r}, "
                f"completion_tokens={completion_tokens!r}). "
                "Check output-token limits or provider support for non-thinking mode."
            )
        if choice.finish_reason == "length":
            raise ValueError(
                "Inference provider stopped at its output-token limit; "
                "raise max_tokens or shorten the prompt."
            )

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Inference provider returned an empty response")
    return text
