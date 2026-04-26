from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ParkingLotConfig, StallDefinition, load_config
from .detectors import BaseVehicleDetector, DetectionMask, create_detector
from .geometry import get_image_polygon, polygon_to_mask, solve_homography, stall_metrics


class ParkingVisionService:
    def __init__(self) -> None:
        self._config_cache: dict[Path, ParkingLotConfig] = {}
        self._detector_cache: dict[tuple[Path, str], BaseVehicleDetector] = {}
        self._temporal_state: dict[tuple[str, str], float] = {}

    def load_config(self, config_path: str | Path) -> ParkingLotConfig:
        resolved = Path(config_path).expanduser().resolve()
        config = self._config_cache.get(resolved)
        if config is None:
            config = load_config(resolved)
            self._config_cache[resolved] = config
        return config

    def _get_detector(self, config: ParkingLotConfig, backend: str) -> BaseVehicleDetector:
        cache_key = (config.config_path or Path(config.camera_id), backend)
        detector = self._detector_cache.get(cache_key)
        if detector is None:
            detector = create_detector(config, backend)
            self._detector_cache[cache_key] = detector
        return detector

    def _smooth_occupancy(self, config: ParkingLotConfig, stall: StallDefinition, raw_ratio: float) -> float:
        smoothing = config.temporal_smoothing
        if not smoothing.enabled:
            return raw_ratio

        state_key = (config.camera_id, stall.stall_id)
        previous_ratio = self._temporal_state.get(state_key)
        if previous_ratio is None:
            smoothed = raw_ratio
        else:
            alpha = smoothing.alpha
            smoothed = alpha * previous_ratio + (1.0 - alpha) * raw_ratio
        self._temporal_state[state_key] = smoothed
        return smoothed

    @staticmethod
    def _occupancy_confidence(ratio: float, threshold: float) -> float:
        margin = abs(ratio - threshold)
        scale = max(threshold, 1.0 - threshold, 1e-6)
        return round(min(0.99, 0.5 + margin / (2.0 * scale)), 3)

    def analyze(self, image: Image.Image, config: ParkingLotConfig, backend: str = "heuristic") -> dict[str, Any]:
        image_rgb = image.convert("RGB")
        if image_rgb.size != (config.image_width, config.image_height):
            image_rgb = image_rgb.resize((config.image_width, config.image_height), Image.Resampling.BILINEAR)

        detector = self._get_detector(config, backend)
        detections = detector.detect(image_rgb, config)
        spillover_threshold = float(config.detector_settings.get("multi_stall_overlap_threshold", 0.32))

        homography = None
        inverse_homography = None
        if config.calibration.is_configured:
            homography = solve_homography(config.calibration.image_points, config.calibration.ground_points_m)
            inverse_homography = np.linalg.inv(homography)

        stall_contexts: list[dict[str, Any]] = []
        for stall in config.stalls:
            image_polygon = get_image_polygon(stall, config.calibration, inverse_homography)
            stall_mask = polygon_to_mask(image_rgb.size, image_polygon)
            stall_contexts.append(
                {
                    "stall": stall,
                    "image_polygon": image_polygon,
                    "stall_mask": stall_mask,
                    "stall_pixels": max(1, int(stall_mask.sum())),
                }
            )

        assigned_detections: dict[str, list[tuple[DetectionMask, float, np.ndarray]]] = {
            context["stall"].stall_id: [] for context in stall_contexts
        }
        for detection in detections:
            overlaps: list[tuple[str, float, np.ndarray]] = []
            for context in stall_contexts:
                intersection = detection.mask & context["stall_mask"]
                if not intersection.any():
                    continue
                overlap_ratio = float(intersection.sum()) / context["stall_pixels"]
                overlaps.append((context["stall"].stall_id, overlap_ratio, intersection))

            if not overlaps:
                continue

            primary_stall_id, primary_ratio, _ = max(overlaps, key=lambda item: item[1])
            for stall_id, overlap_ratio, intersection in overlaps:
                if stall_id == primary_stall_id or overlap_ratio >= spillover_threshold:
                    assigned_detections[stall_id].append((detection, overlap_ratio, intersection))

        results: list[dict[str, Any]] = []
        for context in stall_contexts:
            stall = context["stall"]
            image_polygon = context["image_polygon"]
            stall_pixels = context["stall_pixels"]
            union_mask = np.zeros_like(context["stall_mask"], dtype=bool)
            max_instance_ratio = 0.0

            overlapping_detections = assigned_detections[stall.stall_id]
            for detection, instance_ratio, intersection in overlapping_detections:
                union_mask |= intersection
                max_instance_ratio = max(max_instance_ratio, instance_ratio)

            raw_coverage_ratio = float(union_mask.sum()) / stall_pixels
            smoothed_ratio = self._smooth_occupancy(config, stall, raw_coverage_ratio)
            occupied = smoothed_ratio >= stall.occupancy_threshold

            metrics = stall_metrics(
                stall=stall,
                image_polygon=image_polygon,
                calibration=config.calibration,
                homography=homography,
                thresholds=config.size_thresholds,
            )

            results.append(
                {
                    "stall_id": stall.stall_id,
                    "size": metrics["size"],
                    "availability": "unavailable" if occupied else "available",
                    "occupied": occupied,
                    "coverage_ratio": round(smoothed_ratio, 3),
                    "raw_coverage_ratio": round(raw_coverage_ratio, 3),
                    "max_vehicle_overlap_ratio": round(max_instance_ratio, 3),
                    "vehicle_count": len(overlapping_detections),
                    "occupancy_confidence": self._occupancy_confidence(smoothed_ratio, stall.occupancy_threshold),
                    "occupancy_threshold": stall.occupancy_threshold,
                    "polygon_area_pixels": metrics["polygon_area_pixels"],
                    "polygon_area_m2": metrics["polygon_area_m2"],
                    "length_m": metrics["length_m"],
                    "width_m": metrics["width_m"],
                }
            )

        available_count = sum(1 for result in results if result["availability"] == "available")
        occupied_count = len(results) - available_count

        return {
            "camera_id": config.camera_id,
            "backend": backend,
            "image_width": config.image_width,
            "image_height": config.image_height,
            "summary": {
                "total_stalls": len(results),
                "available_stalls": available_count,
                "occupied_stalls": occupied_count,
            },
            "stalls": results,
        }

    def analyze_path(self, image_path: str | Path, config_path: str | Path, backend: str = "heuristic") -> dict[str, Any]:
        config = self.load_config(config_path)
        with Image.open(Path(image_path)) as image:
            return self.analyze(image=image, config=config, backend=backend)
