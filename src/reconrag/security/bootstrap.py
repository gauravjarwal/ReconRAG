"""Builds the PHI stack, or None when this deployment has no PHI support."""

from __future__ import annotations

from reconrag.config import Settings
from reconrag.security import phi_detect
from reconrag.security.phi import Pseudonymizer
from reconrag.security.vault import PhiVault


def make_pseudonymizer(settings: Settings) -> Pseudonymizer | None:
    """Fails loudly if NER was asked for and can't load. Regex alone would miss names."""
    if not settings.phi_available:
        return None
    phi_detect.load_ner(settings.phi_ner_model)
    return Pseudonymizer(settings.phi_hmac_key, PhiVault(settings.phi_encryption_key))
