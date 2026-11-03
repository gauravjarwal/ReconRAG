"""Failures the service knows how to handle. Anything else is a bug."""

from __future__ import annotations


class ReconRagError(Exception):
    """Base class. Messages are safe to show a client."""


class NotFound(ReconRagError):
    pass


class Conflict(ReconRagError):
    pass


class InvalidInput(ReconRagError):
    pass


class RejectedFile(ReconRagError):
    """The file can never succeed, so retrying is pointless. `code` is stable for clients."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class UpstreamUnavailable(ReconRagError):
    """A model provider failed in a way another attempt or provider might survive."""


class UpstreamRejected(ReconRagError):
    """A model provider refused the request itself. Trying again would fail the same way."""


class AllProvidersFailed(ReconRagError):
    pass


class LeaseLost(ReconRagError):
    """Another worker owns this job now. The current attempt must stop without writing."""
