"""
Brain feature extraction and CTR prediction service.

Provides two lazily-initialized singleton classes:

    ``BrainAnalyzer``
        Extracts the 6 model features and 5 per-second timeseries arrays
        from a TRIBEv2 ``.npz`` prediction file.

    ``BrainPredictor``
        Loads the trained XGBoost models (regressor, classifier, P10/P90
        quantile) from ``analyzer/ml_models/`` and predicts CTR, class,
        confidence, tier, and quantile bounds from the 6 brain features.

Thread-safety: both singletons rely on lazy initialisation guarded by the
GIL (single boolean check). Atlas / model loading happens at most once per
worker process.
"""

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from nilearn import datasets

logger = logging.getLogger(__name__)

# Path to the bundled XGBoost model artefacts
_ML_MODELS_DIR = Path(__file__).resolve().parent.parent / "ml_models"


# ======================================================================
# Brain Feature Extractor
# ======================================================================

class BrainAnalyzer:
    """Extracts brain features and timeseries from TRIBEv2 .npz predictions.

    Thread-safe: atlas is loaded lazily on first call and reused.
    """

    def __init__(self):
        # ── Region groups for timeseries extraction ───────────────────
        self.region_groups = {
            "visual": [
                "G_occipital_middle", "G_occipital_sup", "Pole_occipital",
                "G_and_S_occipital_inf", "G_cuneus", "S_calcarine",
                "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal",
                "G_oc-temp_lat-fusifor",
            ],
            "emotional": [
                "G_orbital", "S_orbital-H_Shaped", "S_orbital_med-olfact",
                "S_orbital_lateral", "G_rectus", "S_circular_insula_inf",
                "S_circular_insula_sup", "S_circular_insula_ant",
                "G_insular_short", "G_Ins_lg_and_S_cent_ins",
                "G_and_S_cingul-Ant", "G_subcallosal", "G_front_inf-Orbital",
            ],
        }

        # ── Individual regions for model features ─────────────────────
        self.individual_regions = {
            "orbital":      ["G_orbital"],
            "insula_short": ["G_insular_short"],
        }

        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        """Lazily load the Destrieux atlas and precompute vertex indices."""
        if self._initialized:
            return

        logger.info("Loading Destrieux surface atlas…")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
                break
            except Exception as exc:
                logger.warning(
                    "Atlas fetch failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, exc,
                )
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(2)

        self.roi_map = np.concatenate([
            self.destrieux_atlas["map_left"],
            self.destrieux_atlas["map_right"],
        ])
        self.labels = [
            lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl
            for lbl in self.destrieux_atlas["labels"]
        ]

        # Precompute vertex indices for region groups
        self.region_indices = {
            group: self._get_indices_for_labels(labels)
            for group, labels in self.region_groups.items()
        }

        # Precompute vertex indices for individual model regions
        self.individual_indices = {
            name: self._get_indices_for_labels(labels)
            for name, labels in self.individual_regions.items()
        }

        self._initialized = True
        logger.info("Brain atlas initialised successfully.")

    def _get_indices_for_labels(self, target_labels: list[str]) -> np.ndarray:
        """Map atlas label names to vertex indices on the fsaverage5 mesh."""
        indices = []
        for label in target_labels:
            try:
                roi_idx = self.labels.index(label)
                idx = np.where(self.roi_map == roi_idx)[0]
                indices.extend(idx)
            except ValueError:
                pass  # Label not found in atlas — skip silently
        return np.array(indices)

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------

    def analyze(self, npz_path: str) -> dict:
        """Extract brain features and per-second timeseries from an .npz file.

        Args:
            npz_path: Path to ``.npz`` file with ``preds`` array of shape
                      ``(n_seconds, 20484)``.

        Returns:
            dict with keys:
                ``model_features`` — the 6 XGBoost input features (floats).
                ``timeseries``     — per-second arrays keyed by region name
                                     (emotional, orbital, visual, insula_short,
                                     global).
        """
        self._ensure_initialized()
        logger.info("Loading predictions from %s …", npz_path)

        loaded = np.load(npz_path, allow_pickle=True)
        preds = loaded["preds"]  # (n_timesteps, 20484)
        n_timesteps = preds.shape[0]

        # ── 1. Build per-second timeseries for each region group ──────
        region_ts = {}
        for group, indices in self.region_indices.items():
            if len(indices) > 0:
                region_ts[group] = np.mean(preds[:, indices], axis=1)
            else:
                region_ts[group] = np.zeros(n_timesteps)

        # Global (whole-brain) timeseries
        global_ts = np.mean(preds, axis=1)  # (n_timesteps,)
        overall_mean = float(np.mean(global_ts))
        global_std = float(np.std(global_ts))

        # Individual region timeseries
        orbital_indices = self.individual_indices["orbital"]
        if len(orbital_indices) > 0:
            orbital_ts = np.mean(preds[:, orbital_indices], axis=1)
        else:
            orbital_ts = np.zeros(n_timesteps)

        insula_indices = self.individual_indices["insula_short"]
        if len(insula_indices) > 0:
            insula_ts = np.mean(preds[:, insula_indices], axis=1)
        else:
            insula_ts = np.zeros(n_timesteps)

        # ── 2. Compute the 6 model features ──────────────────────────

        # Feature 1: longest_sustained_above_mean
        above_mean = global_ts > overall_mean
        longest_sustained = self._longest_consecutive_true(above_mean)

        # Feature 2: emotional_mean
        emotional_mean = float(np.mean(region_ts["emotional"]))

        # Feature 3: orbital_mean
        orbital_mean = float(np.mean(orbital_ts))

        # Feature 4: visual_std
        visual_std = float(np.std(region_ts["visual"]))

        # Feature 5: insula_short_mean
        insula_short_mean = float(np.mean(insula_ts))

        # Feature 6: attention_onset_second
        threshold = overall_mean + 0.5 * global_std
        onset_second = n_timesteps  # default: never reached
        for i, val in enumerate(global_ts):
            if val > threshold:
                onset_second = i
                break
        attention_onset_second = float(onset_second)

        model_features = {
            "longest_sustained_above_mean": float(longest_sustained),
            "emotional_mean":               emotional_mean,
            "orbital_mean":                 orbital_mean,
            "visual_std":                   visual_std,
            "insula_short_mean":            insula_short_mean,
            "attention_onset_second":       attention_onset_second,
        }

        # ── 3. Build serialisable timeseries dict ─────────────────────
        # Convert numpy arrays to plain Python lists for JSON storage.
        timeseries = {
            "emotional":    region_ts["emotional"].tolist(),
            "orbital":      orbital_ts.tolist(),
            "visual":       region_ts["visual"].tolist(),
            "insula_short": insula_ts.tolist(),
            "global":       global_ts.tolist(),
        }

        logger.info(
            "Feature extraction complete — %d timesteps, 6 features.",
            n_timesteps,
        )
        return {
            "model_features": model_features,
            "timeseries":     timeseries,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _longest_consecutive_true(arr: np.ndarray) -> int:
        """Return the length of the longest consecutive True streak."""
        max_run = current_run = 0
        for val in arr:
            if val:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        return max_run


# ======================================================================
# XGBoost CTR Predictor
# ======================================================================

class BrainPredictor:
    """Loads XGBoost models and predicts CTR from extracted brain features.

    Thread-safe: models are loaded lazily on first call and reused.
    """

    def __init__(self, model_dir: Path | None = None):
        self._model_dir = model_dir or _ML_MODELS_DIR
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        """Load XGBoost model artefacts from disk on first use."""
        if self._initialized:
            return

        model_dir = Path(self._model_dir)
        logger.info("Loading XGBoost models from %s …", model_dir)

        # Required artefacts
        reg_path = model_dir / "xgb_regressor.json"
        clf_path = model_dir / "xgb_classifier.json"
        feat_path = model_dir / "selected_features.json"

        for path in (reg_path, clf_path, feat_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required model file: {path}. "
                    "Ensure the trained models are deployed to analyzer/ml_models/."
                )

        self._regressor = xgb.XGBRegressor()
        self._regressor.load_model(str(reg_path))

        self._classifier = xgb.XGBClassifier()
        self._classifier.load_model(str(clf_path))

        with open(feat_path) as fh:
            self._features = json.load(fh)

        # Optional quantile models
        p10_path = model_dir / "xgb_quantile_p10.json"
        p90_path = model_dir / "xgb_quantile_p90.json"
        if p10_path.exists() and p90_path.exists():
            self._p10 = xgb.XGBRegressor()
            self._p10.load_model(str(p10_path))
            self._p90 = xgb.XGBRegressor()
            self._p90.load_model(str(p90_path))
            logger.info("Quantile models (P10/P90) loaded — confidence bounds enabled.")
        else:
            self._p10 = self._p90 = None
            logger.info("Quantile models not found — confidence bounds disabled.")

        self._initialized = True
        logger.info(
            "XGBoost models loaded. Required features: %s", self._features,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, feature_values: dict) -> dict:
        """Run CTR prediction on a single set of brain features.

        Args:
            feature_values: dict mapping feature name → float value
                (e.g. ``{"emotional_mean": 0.025, …}``).

        Returns:
            dict with keys:
                predicted_ctr, predicted_class, predicted_confidence,
                prediction_tier, ctr_lower_bound, ctr_upper_bound.
        """
        self._ensure_initialized()

        # Validate all required features are present
        missing = [f for f in self._features if f not in feature_values]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Build input array in the trained feature order
        X = np.array([[feature_values[f] for f in self._features]])

        # Regression — predict exact CTR
        predicted_log_ctr = self._regressor.predict(X)[0]
        predicted_ctr = float(np.expm1(predicted_log_ctr))

        # Classification — predict High/Low
        predicted_class_int = int(self._classifier.predict(X)[0])
        predicted_proba_high = float(self._classifier.predict_proba(X)[0][1])

        predicted_class = "High" if predicted_class_int == 1 else "Low"

        # Classifier confidence = probability of the *predicted* class
        classifier_confidence = (
            predicted_proba_high * 100
            if predicted_class_int == 1
            else (1 - predicted_proba_high) * 100
        )

        tier = self._classify_tier(predicted_proba_high)

        # Quantile bounds (if available)
        ctr_lower = ctr_upper = None
        if self._p10 is not None and self._p90 is not None:
            p10_log = self._p10.predict(X)[0]
            p90_log = self._p90.predict(X)[0]
            ctr_lower = float(np.expm1(p10_log))
            ctr_upper = float(np.expm1(p90_log))
            # Clamp so lower ≤ predicted ≤ upper
            ctr_lower = min(ctr_lower, predicted_ctr)
            ctr_upper = max(ctr_upper, predicted_ctr)
            ctr_lower = round(ctr_lower, 4)
            ctr_upper = round(ctr_upper, 4)

        return {
            "predicted_ctr":        round(predicted_ctr, 4),
            "predicted_class":      predicted_class,
            "predicted_confidence": round(classifier_confidence, 2),
            "prediction_tier":      tier,
            "ctr_lower_bound":      ctr_lower,
            "ctr_upper_bound":      ctr_upper,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_tier(proba_high: float) -> str:
        """Map classifier probability to a human-readable tier label."""
        if proba_high >= 0.80:
            return "Strong High"
        if proba_high >= 0.60:
            return "Likely High"
        if proba_high >= 0.40:
            return "Borderline"
        if proba_high >= 0.20:
            return "Likely Low"
        return "Strong Low"


# ======================================================================
# Module-level singletons — loaded once per worker process
# ======================================================================

analyzer = BrainAnalyzer()
predictor = BrainPredictor()
