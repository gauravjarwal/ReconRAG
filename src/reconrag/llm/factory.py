"""Builds the router from settings. A provider with no key is left out, not failed at runtime."""

from __future__ import annotations

import logging

from reconrag.config import Settings
from reconrag.llm.base import Provider
from reconrag.llm.breaker import CircuitBreaker
from reconrag.llm.router import LLMRouter

logger = logging.getLogger(__name__)


def make_router(settings: Settings) -> LLMRouter:
    providers: list[Provider] = []
    for name in settings.providers:
        if name == "gemini" and settings.google_api_key:
            from reconrag.llm.gemini import GeminiProvider

            providers.append(GeminiProvider(settings))
        elif name == "anthropic" and settings.anthropic_api_key:
            from reconrag.llm.anthropic import AnthropicProvider

            providers.append(AnthropicProvider(settings))
    if not providers:
        logger.warning("no model provider configured: answers are disabled, search still works")
    breakers = {
        p.name: CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_cooldown_s)
        for p in providers
    }
    return LLMRouter(providers, breakers)
