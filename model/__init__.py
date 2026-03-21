"""Model initialization - centralized LLM setup."""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Default model from env or fallback
DEFAULT_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
# OpenRouter uses OpenAI provider
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")


def get_llm(model: str | None = None, model_provider: str | None = None):
    """Get initialized LLM model.

    Args:
        model: Model name (uses DEFAULT_MODEL if None)
        model_provider: Provider (uses MODEL_PROVIDER if None)

    Returns:
        Initialized chat model
    """
    return init_chat_model(
        model or DEFAULT_MODEL,
        model_provider=model_provider or MODEL_PROVIDER
    )
