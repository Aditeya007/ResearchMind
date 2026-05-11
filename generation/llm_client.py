from typing import Optional, Protocol, Any, cast

try:
    from groq import Groq  # type: ignore[import-not-found]
except ImportError:
    class Groq:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("groq is not installed")
from configs.settings import get_settings

settings = get_settings()

class _GroqClient(Protocol):
    chat: Any


_client: Optional[_GroqClient] = None


def get_client() -> _GroqClient:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in your .env file.")
        _client = cast(_GroqClient, Groq(api_key=settings.GROQ_API_KEY))

    assert _client is not None
    return _client


def generate(prompt: str, system_prompt: Optional[str] = None) -> str:
    client = get_client()

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    answer = generate(
        prompt="What is Retrieval Augmented Generation in 2 sentences?",
        system_prompt="You are a helpful AI assistant. Be concise.",
    )
    print(f"Response:\n{answer}")