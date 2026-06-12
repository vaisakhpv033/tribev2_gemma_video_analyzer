"""
Ranking engine — normalization, dimension scoring, and insight generation.

Adapted from ``phase2/rank_videos.py`` for Django context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .ranking_config import (
    DIMENSION_DESCRIPTIONS,
    DIMENSION_NAMES,
    NeuralRankingConfig,
    get_effective_formulas,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoResult:
    """Per-video ranking result."""
    filename: str
    rank: int = 0
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "rank": self.rank,
            "overall_score": round(self.overall_score, 2),
            "dimension_scores": {k: round(v, 2) for k, v in self.dimension_scores.items()},
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


@dataclass
class RankingReport:
    """Complete ranking report for a set of videos."""
    videos: list[VideoResult] = field(default_factory=list)
    config: Optional[NeuralRankingConfig] = None
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict() if self.config else {},
            "summary": self.summary,
            "videos": [v.to_dict() for v in self.videos],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard sigmoid squashing function."""
    return 1.0 / (1.0 + np.exp(-x))


def normalize_features(
    df: pd.DataFrame,
    method: str = "minmax",
    exclude_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Normalize numeric features across videos (row-wise comparison).

    Args:
        df: DataFrame where each row is a video, columns are features.
        method: One of 'minmax', 'zscore', 'percentile'.
        exclude_cols: Columns to skip (metadata, filenames).

    Returns:
        Normalized DataFrame with same shape and column names.
    """
    exclude = set(exclude_cols or [])
    result = df.copy()

    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        vals = df[col].values.astype(float)
        valid_mask = ~np.isnan(vals)

        if valid_mask.sum() < 2:
            result[col] = 0.5
            continue

        valid = vals[valid_mask]

        if method == "minmax":
            vmin, vmax = valid.min(), valid.max()
            if abs(vmax - vmin) < 1e-12:
                normed = np.full_like(vals, 0.5)
            else:
                normed = (vals - vmin) / (vmax - vmin)

        elif method == "zscore":
            mu, sigma = valid.mean(), valid.std()
            if sigma < 1e-12:
                normed = np.full_like(vals, 0.5)
            else:
                normed = _sigmoid((vals - mu) / sigma)

        elif method == "percentile":
            from scipy.stats import rankdata
            normed = np.full_like(vals, 0.5)
            ranks = rankdata(valid, method="average")
            normed[valid_mask] = (ranks - 1) / max(len(ranks) - 1, 1)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normed[~valid_mask] = 0.5
        result[col] = np.clip(normed, 0.0, 1.0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dimension_scores(
    norm_df: pd.DataFrame,
    formulas: dict,
) -> pd.DataFrame:
    """
    Compute 8 dimension scores (0–100) for each video.

    Each dimension is a weighted combination of normalized features.
    """
    scores = pd.DataFrame(index=norm_df.index)

    for dim_name, formula in formulas.items():
        feature_specs = formula["features"]
        dim_score = np.zeros(len(norm_df))
        total_weight = 0.0

        for feat_name, spec in feature_specs.items():
            if feat_name not in norm_df.columns:
                continue

            weight = spec["weight"]
            invert = spec.get("invert", False)

            vals = norm_df[feat_name].values.astype(float)
            if invert:
                vals = 1.0 - vals

            dim_score += vals * weight
            total_weight += weight

        if total_weight > 0:
            dim_score = (dim_score / total_weight) * 100.0
        else:
            dim_score = np.full(len(norm_df), 50.0)

        scores[dim_name] = np.clip(dim_score, 0.0, 100.0)

    return scores


def compute_overall_scores(
    dimension_scores: pd.DataFrame,
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted sum of dimension scores → single overall score (0–100)."""
    overall = np.zeros(len(dimension_scores))

    for dim, weight in weights.items():
        if dim in dimension_scores.columns:
            overall += dimension_scores[dim].values * weight

    return np.clip(overall, 0.0, 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights(
    dimension_scores: pd.DataFrame,
    top_pct: float = 0.75,
    bottom_pct: float = 0.25,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    For each video, identify strengths (above top_pct quantile) and
    weaknesses (below bottom_pct quantile) across the comparison set.

    Returns:
        (strengths_per_video, weaknesses_per_video) — each a list of lists.
    """
    n = len(dimension_scores)
    all_strengths: list[list[str]] = []
    all_weaknesses: list[list[str]] = []

    for i in range(n):
        strengths: list[str] = []
        weaknesses: list[str] = []

        for dim in dimension_scores.columns:
            score = dimension_scores.iloc[i][dim]
            col_vals = dimension_scores[dim].values
            q75 = np.quantile(col_vals, top_pct)
            q25 = np.quantile(col_vals, bottom_pct)

            if score >= q75 and score >= 50.0:
                strengths.append(f"{dim} ({score:.0f}/100)")
            elif score <= q25 and score < 40.0:
                weaknesses.append(f"{dim} ({score:.0f}/100)")

        all_strengths.append(strengths)
        all_weaknesses.append(weaknesses)

    return all_strengths, all_weaknesses


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RANKING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def rank_videos(
    features_by_video: dict[str, dict[str, float]],
    config: NeuralRankingConfig,
) -> RankingReport:
    """
    Rank a set of videos by neurological impact.

    Args:
        features_by_video: Dict mapping filename → flat feature dict (75 features).
        config: Ranking configuration (weights, normalization, etc.).

    Returns:
        RankingReport with sorted VideoResult objects.
    """
    if len(features_by_video) < 2:
        raise ValueError("At least 2 videos are required for comparative ranking.")

    # Build DataFrame
    filenames = list(features_by_video.keys())
    rows = [features_by_video[f] for f in filenames]
    df = pd.DataFrame(rows, index=filenames)

    logger.info("Feature matrix: %d videos × %d features", *df.shape)

    # Determine which formulas to use
    schaefer_cols = [c for c in df.columns if c.startswith(("vis_net", "sommot_net", "dorsattn_net",
                                                            "salventattn_net", "limbic_net",
                                                            "control_net", "default_net",
                                                            "dmn_dan_", "absorption_",
                                                            "network_engagement_", "salience_"))]
    has_schaefer = any(
        not df[c].isna().all() for c in schaefer_cols if c in df.columns
    )
    formulas = get_effective_formulas(use_schaefer=has_schaefer and config.use_schaefer)

    # Normalize
    norm_df = normalize_features(df, method=config.normalization)

    # Compute dimension scores
    dim_scores = compute_dimension_scores(norm_df, formulas)

    # Compute overall scores
    overall = compute_overall_scores(dim_scores, config.dimension_weights)

    # Generate insights
    strengths_list, weaknesses_list = generate_insights(dim_scores)

    # Build results sorted by overall score
    results: list[VideoResult] = []
    sort_order = np.argsort(-overall)

    for rank_idx, orig_idx in enumerate(sort_order):
        results.append(VideoResult(
            filename=filenames[orig_idx],
            rank=rank_idx + 1,
            overall_score=float(overall[orig_idx]),
            dimension_scores={
                dim: float(dim_scores.iloc[orig_idx][dim])
                for dim in dim_scores.columns
            },
            strengths=strengths_list[orig_idx],
            weaknesses=weaknesses_list[orig_idx],
        ))

    # Build summary
    summary = _build_summary(dim_scores, overall, filenames, config)

    report = RankingReport(
        videos=results,
        config=config,
        summary=summary,
    )

    logger.info("Ranking complete: %d videos ranked.", len(results))
    return report


def _build_summary(
    dim_scores: pd.DataFrame,
    overall: np.ndarray,
    filenames: list[str],
    config: NeuralRankingConfig,
) -> dict:
    """Build overall ranking summary statistics."""
    summary: dict = {
        "n_videos": len(filenames),
        "preset": config.preset_name,
        "normalization": config.normalization,
        "score_stats": {
            "min": round(float(overall.min()), 2),
            "max": round(float(overall.max()), 2),
            "mean": round(float(overall.mean()), 2),
            "std": round(float(overall.std()), 2),
        },
        "best_per_dimension": {},
        "dimension_stats": {},
    }

    for dim in dim_scores.columns:
        vals = dim_scores[dim].values
        best_idx = int(np.argmax(vals))
        summary["best_per_dimension"][dim] = {
            "filename": filenames[best_idx],
            "score": round(float(vals[best_idx]), 2),
        }
        summary["dimension_stats"][dim] = {
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "mean": round(float(vals.mean()), 2),
        }

    return summary
