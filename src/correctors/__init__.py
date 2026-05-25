"""Noise-robust correctors."""
from __future__ import annotations

from .base import BaseCorrector, register_corrector, get_corrector, list_correctors
from .prompt_corrector import PromptCorrector  # noqa: F401  (注册副作用)
from .iterative_corrector import IterativeCorrector  # noqa: F401
from .confidence_corrector import ConfidenceCorrector  # noqa: F401
from .selfrag_baseline import SelfRAGBaseline  # noqa: F401
from .voting_corrector import EvidenceVotingCorrector  # noqa: F401
from .adaptive_corrector import AdaptiveCorrector  # noqa: F401
from .ablated_confidence import (  # noqa: F401
    AblatedFullCorrector,
    AblatedNoDecomposeCorrector,
    AblatedNoEvidenceCorrector,
    AblatedNoTagCorrector,
)
from .iterative_self_correct import IterativeSelfCorrectCorrector  # noqa: F401

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
    "AdaptiveCorrector",
    "AblatedFullCorrector",
    "AblatedNoDecomposeCorrector",
    "AblatedNoEvidenceCorrector",
    "AblatedNoTagCorrector",
    "IterativeSelfCorrectCorrector",
]
