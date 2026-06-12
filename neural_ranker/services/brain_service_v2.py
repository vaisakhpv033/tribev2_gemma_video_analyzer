"""
Brain feature extraction service (v2) — dual-atlas.

Lazily-initialized singleton that loads Destrieux + Schaefer/Yeo atlases
once per worker process and extracts 75 brain features from TRIBEv2 .npz
prediction files.

Thread-safety: atlas loading is guarded by a boolean flag checked under
the GIL. Atlas data is read-only after initialisation.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

FSAVERAGE5_N_VERTICES = 20484
FSAVERAGE5_N_VERTICES_PER_HEMI = 10242

# Region groups (Destrieux)
REGION_GROUPS = {
    "visual": [
        "G_occipital_middle", "G_occipital_sup", "Pole_occipital",
        "G_and_S_occipital_inf", "G_cuneus", "S_calcarine",
        "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal",
        "G_oc-temp_lat-fusifor",
    ],
    "auditory": [
        "S_temporal_transverse", "G_temp_sup-G_T_transv",
        "G_temp_sup-Lateral", "G_temp_sup-Plan_tempo",
        "S_temporal_sup", "G_temp_sup-Plan_polar",
    ],
    "emotional": [
        "G_orbital", "S_orbital-H_Shaped", "S_orbital_med-olfact",
        "S_orbital_lateral", "G_rectus", "S_circular_insula_inf",
        "S_circular_insula_sup", "S_circular_insula_ant",
        "G_insular_short", "G_Ins_lg_and_S_cent_ins",
        "G_and_S_cingul-Ant", "G_subcallosal", "G_front_inf-Orbital",
    ],
    "language": [
        "G_front_inf-Opercular", "G_front_inf-Triangul",
        "G_temporal_middle", "G_temporal_inf",
        "S_front_inf", "G_and_S_cingul-Mid-Ant",
    ],
    "memory": [
        "G_oc-temp_med-Parahip", "G_precuneus",
        "G_cingul-Post-dorsal", "G_oc-temp_med-Lingual",
        "S_collat_transv_ant",
    ],
}

KEY_INDIVIDUAL_REGIONS = {
    "fusiform": "G_oc-temp_lat-fusifor",
    "insula_short": "G_insular_short",
    "orbital": "G_orbital",
    "calcarine": "S_calcarine",
    "heschl": "G_temp_sup-G_T_transv",
    "broca": "G_front_inf-Opercular",
}

DIMENSION_WEIGHTS = {
    "visual": 0.25, "auditory": 0.15,
    "emotional": 0.30, "attention": 0.20, "language": 0.10,
}

YEO_7_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

YEO_7_FEATURE_PREFIX = {
    "Vis": "vis_net", "SomMot": "sommot_net", "DorsAttn": "dorsattn_net",
    "SalVentAttn": "salventattn_net", "Limbic": "limbic_net",
    "Cont": "control_net", "Default": "default_net",
}

SCHAEFER_BASE_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage5/label/"
)
SCHAEFER_FILES = {
    "lh": "lh.Schaefer2018_400Parcels_7Networks_order.annot",
    "rh": "rh.Schaefer2018_400Parcels_7Networks_order.annot",
}

ATLAS_CACHE_DIR = Path(__file__).resolve().parent.parent / "atlases" / "schaefer400_7networks"


# ═══════════════════════════════════════════════════════════════════════════════
# BRAIN ANALYZER V2 (Singleton)
# ═══════════════════════════════════════════════════════════════════════════════

class BrainAnalyzerV2:
    """
    Extracts 75 brain features from a TRIBEv2 .npz file using dual atlases.

    Lazy-initialized: atlases are loaded on first ``extract_features()`` call
    and reused for all subsequent calls within the same process.
    """

    def __init__(self):
        self._initialized = False
        self._destrieux_label_to_idx: dict[str, np.ndarray] = {}
        self._network_to_idx: dict[str, np.ndarray] = {}
        self._schaefer_available = False

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        if self._initialized:
            return

        self._load_destrieux()
        self._load_schaefer()
        self._initialized = True

    def _load_destrieux(self):
        from nilearn import datasets

        logger.info("Loading Destrieux surface atlas…")
        atlas = datasets.fetch_atlas_surf_destrieux()
        roi_map = np.concatenate([atlas["map_left"], atlas["map_right"]])
        labels = [
            l.decode("utf-8") if isinstance(l, bytes) else l
            for l in atlas["labels"]
        ]

        for roi_idx, name in enumerate(labels):
            verts = np.where(roi_map == roi_idx)[0]
            if len(verts) > 0:
                self._destrieux_label_to_idx[name] = verts

        logger.info("Destrieux atlas ready: %d labels.", len(labels))

    def _load_schaefer(self):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            logger.warning("nibabel not installed — Schaefer atlas unavailable.")
            return

        lh_path = ATLAS_CACHE_DIR / SCHAEFER_FILES["lh"]
        rh_path = ATLAS_CACHE_DIR / SCHAEFER_FILES["rh"]

        if not lh_path.exists() or not rh_path.exists():
            logger.info("Downloading Schaefer atlas from CBIG GitHub…")
            if not self._download_schaefer():
                logger.warning("Schaefer download failed — functional network features will be NaN.")
                return

        try:
            self._parse_schaefer_annots(lh_path, rh_path)
            self._schaefer_available = True
            logger.info(
                "Schaefer/Yeo atlas ready: %d networks, %d vertices mapped.",
                len(self._network_to_idx),
                sum(len(v) for v in self._network_to_idx.values()),
            )
        except Exception as exc:
            logger.error("Failed to load Schaefer atlas: %s", exc)

    def _download_schaefer(self) -> bool:
        ATLAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for hemi, filename in SCHAEFER_FILES.items():
            url = SCHAEFER_BASE_URL + filename
            dest = ATLAS_CACHE_DIR / filename
            try:
                urlretrieve(url, str(dest))
                if dest.stat().st_size < 1000:
                    dest.unlink(missing_ok=True)
                    return False
            except (URLError, OSError) as exc:
                logger.warning("Download failed for %s: %s", filename, exc)
                return False
        return True

    def _parse_schaefer_annots(self, lh_path: Path, rh_path: Path):
        import nibabel.freesurfer as fs

        network_vertices: dict[str, list[int]] = defaultdict(list)

        for hemi_path, offset in [(lh_path, 0), (rh_path, FSAVERAGE5_N_VERTICES_PER_HEMI)]:
            labels, ctab, names = fs.read_annot(str(hemi_path))
            decoded = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in names
            ]
            for v_idx in range(len(labels)):
                label_idx = int(labels[v_idx])
                if label_idx < 0 or label_idx >= len(decoded):
                    continue
                name = decoded[label_idx]
                if not name or "background" in name.lower():
                    continue
                parts = name.split("_")
                if len(parts) >= 3 and parts[0] == "7Networks":
                    network = parts[2]
                    if network in YEO_7_NETWORKS:
                        network_vertices[network].append(v_idx + offset)

        for network, verts in network_vertices.items():
            self._network_to_idx[network] = np.array(sorted(verts), dtype=int)

    # ------------------------------------------------------------------
    # Vertex helpers
    # ------------------------------------------------------------------

    def _vertices_for(self, label: str) -> np.ndarray:
        return self._destrieux_label_to_idx.get(label, np.array([], dtype=int))

    def _vertices_for_group(self, labels: list[str]) -> np.ndarray:
        parts = [self._vertices_for(l) for l in labels if l in self._destrieux_label_to_idx]
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    def _vertices_for_network(self, network: str) -> np.ndarray:
        return self._network_to_idx.get(network, np.array([], dtype=int))

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, npz_path: str) -> dict[str, float]:
        """
        Extract all 75 brain features from a TRIBEv2 .npz file.

        Returns a flat dict of feature_name → float value.
        """
        self._ensure_initialized()

        data = np.load(npz_path, allow_pickle=True)
        preds = data["preds"]  # (n_seconds, 20484)

        features: dict[str, float] = {}
        features.update(self._dimension_features(preds))
        features.update(self._attention_features(preds))
        features.update(self._temporal_features(preds))
        features.update(self._global_features(preds))
        features.update(self._individual_region_features(preds))
        features.update(self._weighted_scores(features))
        features.update(self._functional_network_features(preds))
        features.update(self._cross_network_features(preds))

        return features

    def extract_engagement_curve(self, npz_path: str) -> dict[str, list[float]]:
        """Extract per-second engagement timeseries for visualization."""
        self._ensure_initialized()

        data = np.load(npz_path, allow_pickle=True)
        preds = data["preds"]
        n_seconds = preds.shape[0]

        curve: dict[str, list[float]] = {
            "second": list(range(n_seconds)),
            "global": np.mean(preds, axis=1).tolist(),
        }

        for dim_name, region_labels in REGION_GROUPS.items():
            verts = self._vertices_for_group(region_labels)
            if len(verts) > 0:
                curve[dim_name] = np.mean(preds[:, verts], axis=1).tolist()
            else:
                curve[dim_name] = [0.0] * n_seconds

        if self._schaefer_available:
            for network in YEO_7_NETWORKS:
                prefix = YEO_7_FEATURE_PREFIX[network]
                verts = self._vertices_for_network(network)
                if len(verts) > 0:
                    valid = verts[verts < preds.shape[1]]
                    curve[prefix] = np.mean(preds[:, valid], axis=1).tolist() if len(valid) > 0 else [0.0] * n_seconds
                else:
                    curve[prefix] = [0.0] * n_seconds

        return curve

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _dimension_features(self, preds: np.ndarray) -> dict[str, float]:
        features = {}
        for dim_name, region_labels in REGION_GROUPS.items():
            verts = self._vertices_for_group(region_labels)
            if len(verts) == 0:
                for stat in ("mean", "peak", "std"):
                    features[f"{dim_name}_{stat}"] = float("nan")
                continue
            ts = np.mean(preds[:, verts], axis=1)
            features[f"{dim_name}_mean"] = float(np.mean(ts))
            features[f"{dim_name}_peak"] = float(np.max(ts))
            features[f"{dim_name}_std"] = float(np.std(ts))
        return features

    def _attention_features(self, preds: np.ndarray) -> dict[str, float]:
        n = preds.shape[0]
        global_ts = np.mean(preds, axis=1)
        m, s = float(np.mean(global_ts)), float(np.std(global_ts))

        first3 = global_ts[:min(3, n)]
        first3_mean = float(np.mean(first3))
        hook_ratio = first3_mean / m if abs(m) > 1e-9 else 1.0

        threshold = m + 0.5 * s
        above = np.where(global_ts > threshold)[0]
        onset = float(above[0]) if len(above) > 0 else float(n)

        return {
            "attention_hook_ratio": hook_ratio,
            "attention_onset_second": onset,
            "attention_first3s_mean": first3_mean,
        }

    def _temporal_features(self, preds: np.ndarray) -> dict[str, float]:
        n = preds.shape[0]
        global_ts = np.mean(preds, axis=1)
        m, s = float(np.mean(global_ts)), float(np.std(global_ts))

        features: dict[str, float] = {
            "peak_second": float(np.argmax(global_ts)),
            "engagement_variance": s,
        }

        mid = n // 2
        if mid >= 2:
            features["engagement_slope_first_half"] = float(np.polyfit(np.arange(mid), global_ts[:mid], 1)[0])
            features["engagement_slope_second_half"] = float(np.polyfit(np.arange(n - mid), global_ts[mid:], 1)[0])
        else:
            features["engagement_slope_first_half"] = 0.0
            features["engagement_slope_second_half"] = 0.0

        if s > 1e-9:
            peaks, _ = find_peaks(global_ts, prominence=s * 0.5)
            features["engagement_peak_count"] = float(len(peaks))
        else:
            features["engagement_peak_count"] = 0.0

        above_mean = global_ts > m
        longest, current = 0, 0
        for val in above_mean:
            current = current + 1 if val else 0
            longest = max(longest, current)
        features["longest_sustained_above_mean"] = float(longest)
        features["pct_above_mean"] = float(np.sum(above_mean) / n)
        features["global_activation_range"] = float(np.max(global_ts) - np.min(global_ts))
        return features

    def _global_features(self, preds: np.ndarray) -> dict[str, float]:
        return {
            "total_neural_energy": float(np.sum(np.maximum(preds, 0))),
            "global_mean_activation": float(np.mean(preds)),
            "global_peak_activation": float(np.max(preds)),
            "video_duration_seconds": float(preds.shape[0]),
            "n_vertices": float(preds.shape[1]),
        }

    def _individual_region_features(self, preds: np.ndarray) -> dict[str, float]:
        features = {}
        for short, label in KEY_INDIVIDUAL_REGIONS.items():
            verts = self._vertices_for(label)
            if len(verts) == 0:
                features[f"{short}_mean"] = float("nan")
                features[f"{short}_peak"] = float("nan")
                continue
            ts = np.mean(preds[:, verts], axis=1)
            features[f"{short}_mean"] = float(np.mean(ts))
            features[f"{short}_peak"] = float(np.max(ts))
        return features

    def _weighted_scores(self, features: dict[str, float]) -> dict[str, float]:
        dim_keys = {
            "visual": "visual_mean", "auditory": "auditory_mean",
            "emotional": "emotional_mean", "language": "language_mean",
            "attention": "attention_first3s_mean",
        }
        scores: dict[str, float] = {}
        overall = 0.0
        for dim, feat_key in dim_keys.items():
            raw = features.get(feat_key, 0.0)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                raw = 0.0
            scores[f"{dim}_score_raw"] = raw
            overall += raw * DIMENSION_WEIGHTS[dim]
        scores["overall_engagement_score_raw"] = overall
        return scores

    def _functional_network_features(self, preds: np.ndarray) -> dict[str, float]:
        features = {}
        for network in YEO_7_NETWORKS:
            prefix = YEO_7_FEATURE_PREFIX[network]
            if not self._schaefer_available:
                features[f"{prefix}_mean"] = float("nan")
                features[f"{prefix}_peak"] = float("nan")
                features[f"{prefix}_std"] = float("nan")
                continue
            verts = self._vertices_for_network(network)
            valid = verts[verts < preds.shape[1]] if len(verts) > 0 else np.array([], dtype=int)
            if len(valid) == 0:
                features[f"{prefix}_mean"] = float("nan")
                features[f"{prefix}_peak"] = float("nan")
                features[f"{prefix}_std"] = float("nan")
                continue
            ts = np.mean(preds[:, valid], axis=1)
            features[f"{prefix}_mean"] = float(np.mean(ts))
            features[f"{prefix}_peak"] = float(np.max(ts))
            features[f"{prefix}_std"] = float(np.std(ts))
        return features

    def _cross_network_features(self, preds: np.ndarray) -> dict[str, float]:
        features: dict[str, float] = {}

        def _net_ts(network: str) -> Optional[np.ndarray]:
            if not self._schaefer_available:
                return None
            verts = self._vertices_for_network(network)
            valid = verts[verts < preds.shape[1]] if len(verts) > 0 else np.array([], dtype=int)
            return np.mean(preds[:, valid], axis=1) if len(valid) > 0 else None

        dmn, dan, sal = _net_ts("Default"), _net_ts("DorsAttn"), _net_ts("SalVentAttn")
        cont, limbic = _net_ts("Cont"), _net_ts("Limbic")
        n = preds.shape[0]

        # DMN↔DAN anticorrelation
        if dmn is not None and dan is not None and n >= 3:
            if np.std(dmn) > 1e-9 and np.std(dan) > 1e-9:
                features["dmn_dan_anticorrelation"] = float(np.corrcoef(dmn, dan)[0, 1])
            else:
                features["dmn_dan_anticorrelation"] = 0.0
        else:
            features["dmn_dan_anticorrelation"] = float("nan")

        # Absorption score
        if dan is not None and dmn is not None:
            features["absorption_score"] = float(np.mean(dan) - np.mean(dmn))
        else:
            features["absorption_score"] = float("nan")

        # Network engagement ratio
        pos = [ts for ts in [dan, sal, cont] if ts is not None]
        neg = [ts for ts in [dmn, limbic] if ts is not None]
        if pos and neg:
            pos_m = float(np.mean([np.mean(t) for t in pos]))
            neg_m = float(np.mean([np.mean(t) for t in neg]))
            features["network_engagement_ratio"] = (
                pos_m / abs(neg_m) if abs(neg_m) > 1e-9
                else float(np.sign(pos_m) * 10.0 if abs(pos_m) > 1e-9 else 1.0)
            )
        else:
            features["network_engagement_ratio"] = float("nan")

        # Salience spikes
        if sal is not None and n >= 3:
            sal_m, sal_s = np.mean(sal), np.std(sal)
            if sal_s > 1e-9:
                spike_peaks, _ = find_peaks(sal, height=sal_m + sal_s, prominence=sal_s * 0.5)
                features["salience_spike_count"] = float(len(spike_peaks))
                features["salience_max_spike"] = float(np.max(sal[spike_peaks] - sal_m)) if len(spike_peaks) > 0 else 0.0
            else:
                features["salience_spike_count"] = 0.0
                features["salience_max_spike"] = 0.0
        else:
            features["salience_spike_count"] = float("nan")
            features["salience_max_spike"] = float("nan")

        return features


# Module-level singleton — loaded once per worker process
brain_analyzer_v2 = BrainAnalyzerV2()
