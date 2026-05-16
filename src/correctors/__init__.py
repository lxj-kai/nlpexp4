"""Noise-robust correctors."""

from .base import BaseCorrector, register_corrector, get_corrector, list_correctors
from .prompt_corrector import PromptCorrector  # noqa: F401  (注册副作用)
from .iterative_corrector import IterativeCorrector  # noqa: F401
from .confidence_corrector import ConfidenceCorrector  # noqa: F401
from .selfrag_baseline import SelfRAGBaseline  # noqa: F401
from .voting_corrector import EvidenceVotingCorrector  # noqa: F401

__all__ = [
    "BaseCorrector",
    "register_corrector",
    "get_corrector",
    "list_correctors",
    "PromptCorrector",
    "IterativeCorrector",
    "ConfidenceCorrector",
    "SelfRAGBaseline",
    "EvidenceVotingCorrector",
]
