"""
Ranking configuration — weight presets, dimension formulas, and config.

Adapted from ``phase2/ranking_config.py`` for Django context.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION NAMES
# ═══════════════════════════════════════════════════════════════════════════════

DIMENSION_NAMES = [
    "emotional_resonance",
    "visual_engagement",
    "attention_capture",
    "sustained_focus",
    "novelty_salience",
    "auditory_impact",
    "memory_encoding",
    "narrative_language",
]

DIMENSION_DESCRIPTIONS = {
    "emotional_resonance":  "Emotional processing — orbitofrontal, insula, cingulate activation",
    "visual_engagement":    "Visual saliency — occipital, fusiform, calcarine cortex activation",
    "attention_capture":    "Hook strength — first 3 seconds engagement and onset speed",
    "sustained_focus":      "DAN activation + DMN suppression = absorbed in content",
    "novelty_salience":     "Salience Network spikes — novelty, surprise, scroll-stopping",
    "auditory_impact":      "Auditory processing — Heschl's, superior temporal cortex",
    "memory_encoding":      "Memory formation — parahippocampal, precuneus, posterior cingulate",
    "narrative_language":   "Language comprehension — Broca's, Wernicke's, mid-temporal cortex",
}


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

WEIGHT_PRESETS = {
    "default": {
        "emotional_resonance": 0.20, "visual_engagement": 0.18,
        "attention_capture": 0.15, "sustained_focus": 0.12,
        "novelty_salience": 0.10, "auditory_impact": 0.10,
        "memory_encoding": 0.08, "narrative_language": 0.07,
    },
    "game_ads": {
        "visual_engagement": 0.25, "attention_capture": 0.20,
        "emotional_resonance": 0.18, "novelty_salience": 0.12,
        "sustained_focus": 0.10, "auditory_impact": 0.08,
        "memory_encoding": 0.05, "narrative_language": 0.02,
    },
    "narrative_ads": {
        "emotional_resonance": 0.22, "narrative_language": 0.18,
        "memory_encoding": 0.15, "sustained_focus": 0.15,
        "attention_capture": 0.10, "visual_engagement": 0.10,
        "auditory_impact": 0.05, "novelty_salience": 0.05,
    },
    "music_video": {
        "auditory_impact": 0.25, "emotional_resonance": 0.22,
        "visual_engagement": 0.15, "novelty_salience": 0.12,
        "sustained_focus": 0.10, "attention_capture": 0.08,
        "memory_encoding": 0.05, "narrative_language": 0.03,
    },
    "brand_awareness": {
        "memory_encoding": 0.22, "emotional_resonance": 0.20,
        "narrative_language": 0.15, "visual_engagement": 0.15,
        "sustained_focus": 0.10, "attention_capture": 0.08,
        "novelty_salience": 0.05, "auditory_impact": 0.05,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION FORMULAS
# ═══════════════════════════════════════════════════════════════════════════════

DIMENSION_FORMULAS = {
    "emotional_resonance": {
        "features": {
            "emotional_mean": {"weight": 0.30, "invert": False},
            "insula_short_mean": {"weight": 0.20, "invert": False},
            "orbital_mean": {"weight": 0.20, "invert": False},
            "emotional_peak": {"weight": 0.15, "invert": False},
            "emotional_std": {"weight": 0.15, "invert": False},
        },
    },
    "visual_engagement": {
        "features": {
            "visual_mean": {"weight": 0.30, "invert": False},
            "calcarine_mean": {"weight": 0.20, "invert": False},
            "fusiform_mean": {"weight": 0.20, "invert": False},
            "visual_peak": {"weight": 0.15, "invert": False},
            "visual_std": {"weight": 0.15, "invert": False},
        },
    },
    "attention_capture": {
        "features": {
            "attention_hook_ratio": {"weight": 0.30, "invert": False},
            "attention_first3s_mean": {"weight": 0.30, "invert": False},
            "attention_onset_second": {"weight": 0.25, "invert": True},
            "engagement_slope_first_half": {"weight": 0.15, "invert": False},
        },
    },
    "sustained_focus": {
        "features": {
            "dorsattn_net_mean": {"weight": 0.25, "invert": False},
            "absorption_score": {"weight": 0.25, "invert": False},
            "longest_sustained_above_mean": {"weight": 0.20, "invert": False},
            "pct_above_mean": {"weight": 0.15, "invert": False},
            "engagement_slope_second_half": {"weight": 0.15, "invert": False},
        },
    },
    "novelty_salience": {
        "features": {
            "salventattn_net_mean": {"weight": 0.25, "invert": False},
            "salience_spike_count": {"weight": 0.25, "invert": False},
            "salience_max_spike": {"weight": 0.25, "invert": False},
            "engagement_peak_count": {"weight": 0.25, "invert": False},
        },
    },
    "auditory_impact": {
        "features": {
            "auditory_mean": {"weight": 0.30, "invert": False},
            "heschl_mean": {"weight": 0.25, "invert": False},
            "auditory_peak": {"weight": 0.25, "invert": False},
            "auditory_std": {"weight": 0.20, "invert": False},
        },
    },
    "memory_encoding": {
        "features": {
            "memory_mean": {"weight": 0.30, "invert": False},
            "memory_peak": {"weight": 0.25, "invert": False},
            "limbic_net_mean": {"weight": 0.25, "invert": False},
            "emotional_peak": {"weight": 0.20, "invert": False},
        },
    },
    "narrative_language": {
        "features": {
            "language_mean": {"weight": 0.30, "invert": False},
            "broca_mean": {"weight": 0.25, "invert": False},
            "language_peak": {"weight": 0.25, "invert": False},
            "language_std": {"weight": 0.20, "invert": False},
        },
    },
}

DIMENSION_FORMULAS_FALLBACK = {
    "sustained_focus": {
        "features": {
            "pct_above_mean": {"weight": 0.30, "invert": False},
            "longest_sustained_above_mean": {"weight": 0.30, "invert": False},
            "engagement_slope_second_half": {"weight": 0.20, "invert": False},
            "engagement_variance": {"weight": 0.20, "invert": False},
        },
    },
    "novelty_salience": {
        "features": {
            "engagement_peak_count": {"weight": 0.35, "invert": False},
            "global_activation_range": {"weight": 0.35, "invert": False},
            "engagement_variance": {"weight": 0.30, "invert": False},
        },
    },
    "memory_encoding": {
        "features": {
            "memory_mean": {"weight": 0.35, "invert": False},
            "memory_peak": {"weight": 0.30, "invert": False},
            "emotional_peak": {"weight": 0.35, "invert": False},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NeuralRankingConfig:
    """User-adjustable configuration for the neural ranking system."""

    dimension_weights: dict[str, float] = field(
        default_factory=lambda: deepcopy(WEIGHT_PRESETS["default"])
    )
    normalization: str = "minmax"
    use_schaefer: bool = True
    preset_name: str = "default"

    def __post_init__(self):
        self._validate_and_normalize()

    def _validate_and_normalize(self):
        missing = set(DIMENSION_NAMES) - set(self.dimension_weights.keys())
        if missing:
            raise ValueError(f"Missing dimension weights: {missing}")

        for dim, w in self.dimension_weights.items():
            if w < 0:
                raise ValueError(f"Negative weight for '{dim}': {w}")

        total = sum(self.dimension_weights.values())
        if total <= 0:
            raise ValueError("All dimension weights are zero.")
        if abs(total - 1.0) > 1e-6:
            for dim in self.dimension_weights:
                self.dimension_weights[dim] /= total

        if self.normalization not in {"minmax", "zscore", "percentile"}:
            raise ValueError(f"Invalid normalization: {self.normalization}")

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> NeuralRankingConfig:
        if preset_name not in WEIGHT_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {', '.join(WEIGHT_PRESETS.keys())}"
            )
        weights = deepcopy(WEIGHT_PRESETS[preset_name])
        weights.update(overrides.get("dimension_weights", {}))
        return cls(
            dimension_weights=weights,
            preset_name=preset_name,
            normalization=overrides.get("normalization", "minmax"),
            use_schaefer=overrides.get("use_schaefer", True),
        )

    @classmethod
    def from_session(cls, session) -> NeuralRankingConfig:
        """Build config from a RankingSession model instance."""
        config = cls.from_preset(session.preset, normalization=session.normalization)
        if session.custom_weights:
            config.dimension_weights.update(session.custom_weights)
            config._validate_and_normalize()
        return config

    def to_dict(self) -> dict:
        return {
            "dimension_weights": dict(self.dimension_weights),
            "normalization": self.normalization,
            "use_schaefer": self.use_schaefer,
            "preset_name": self.preset_name,
        }


def get_effective_formulas(use_schaefer: bool) -> dict:
    """Return formulas with Destrieux-only fallbacks when Schaefer is unavailable."""
    formulas = deepcopy(DIMENSION_FORMULAS)
    if not use_schaefer:
        for dim, fallback in DIMENSION_FORMULAS_FALLBACK.items():
            formulas[dim] = fallback
    return formulas
