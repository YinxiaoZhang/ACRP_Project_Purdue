from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


Point = tuple[float, float]


@dataclass(slots=True)
class TemporalSmoothingConfig:
    enabled: bool = True
    alpha: float = 0.7


@dataclass(slots=True)
class SizeThresholds:
    compact_max_area_m2: float = 12.5
    standard_max_area_m2: float = 16.5
    large_max_area_m2: float = 21.0


@dataclass(slots=True)
class CameraCalibration:
    image_points: list[Point] = field(default_factory=list)
    ground_points_m: list[Point] = field(default_factory=list)
    pixels_per_meter: float | None = None

    @property
    def is_configured(self) -> bool:
        return len(self.image_points) >= 4 and len(self.image_points) == len(self.ground_points_m)


@dataclass(slots=True)
class StallDefinition:
    stall_id: str
    polygon: list[Point] | None = None
    ground_polygon_m: list[Point] | None = None
    size_label: str | None = None
    occupancy_threshold: float = 0.18
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParkingLotConfig:
    camera_id: str
    image_width: int
    image_height: int
    stalls: list[StallDefinition]
    calibration: CameraCalibration = field(default_factory=CameraCalibration)
    temporal_smoothing: TemporalSmoothingConfig = field(default_factory=TemporalSmoothingConfig)
    size_thresholds: SizeThresholds = field(default_factory=SizeThresholds)
    reference_image_path: Path | None = None
    yolo_model_path: Path | None = None
    detector_settings: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None


def _as_point_list(raw_points: list[list[float]] | None) -> list[Point] | None:
    if raw_points is None:
        return None
    return [tuple(float(value) for value in point) for point in raw_points]


def _resolve_optional_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path: str | Path) -> ParkingLotConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    calibration_payload = payload.get("calibration", {})
    smoothing_payload = payload.get("temporal_smoothing", {})
    size_thresholds_payload = payload.get("size_thresholds", {})

    stalls: list[StallDefinition] = []
    for stall_payload in payload["stalls"]:
        stalls.append(
            StallDefinition(
                stall_id=stall_payload["id"],
                polygon=_as_point_list(stall_payload.get("polygon")),
                ground_polygon_m=_as_point_list(stall_payload.get("ground_polygon_m")),
                size_label=stall_payload.get("size"),
                occupancy_threshold=float(stall_payload.get("occupancy_threshold", 0.18)),
                metadata=stall_payload.get("metadata", {}),
            )
        )

    config = ParkingLotConfig(
        camera_id=payload["camera_id"],
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        stalls=stalls,
        calibration=CameraCalibration(
            image_points=_as_point_list(calibration_payload.get("image_points")) or [],
            ground_points_m=_as_point_list(calibration_payload.get("ground_points_m")) or [],
            pixels_per_meter=(
                float(calibration_payload["pixels_per_meter"])
                if calibration_payload.get("pixels_per_meter") is not None
                else None
            ),
        ),
        temporal_smoothing=TemporalSmoothingConfig(
            enabled=bool(smoothing_payload.get("enabled", True)),
            alpha=float(smoothing_payload.get("alpha", 0.7)),
        ),
        size_thresholds=SizeThresholds(
            compact_max_area_m2=float(size_thresholds_payload.get("compact_max_area_m2", 12.5)),
            standard_max_area_m2=float(size_thresholds_payload.get("standard_max_area_m2", 16.5)),
            large_max_area_m2=float(size_thresholds_payload.get("large_max_area_m2", 21.0)),
        ),
        reference_image_path=_resolve_optional_path(base_dir, payload.get("reference_image_path")),
        yolo_model_path=_resolve_optional_path(base_dir, payload.get("yolo_model_path")),
        detector_settings=payload.get("detector_settings", {}),
        config_path=config_path,
    )

    for stall in config.stalls:
        if stall.polygon is None and stall.ground_polygon_m is None:
            raise ValueError(f"Stall {stall.stall_id} must define either polygon or ground_polygon_m")

    return config
