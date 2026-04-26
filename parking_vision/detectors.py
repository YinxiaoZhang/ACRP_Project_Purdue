from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ParkingLotConfig
from .preprocess import coarse_object_mask, connected_components, diff_foreground_mask


VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle", "van"}


@dataclass(slots=True)
class DetectionMask:
    mask: np.ndarray
    confidence: float
    label: str
    source: str


class BaseVehicleDetector:
    name = "base"

    def detect(self, image: Image.Image, config: ParkingLotConfig) -> list[DetectionMask]:
        raise NotImplementedError


class HeuristicVehicleDetector(BaseVehicleDetector):
    name = "heuristic"

    def __init__(self) -> None:
        self._reference_cache: dict[Path, Image.Image] = {}

    def _load_reference(self, path: Path, size: tuple[int, int]) -> Image.Image:
        reference_image = self._reference_cache.get(path)
        if reference_image is None:
            reference_image = Image.open(path).convert("RGB")
            self._reference_cache[path] = reference_image
        if reference_image.size != size:
            return reference_image.resize(size, Image.Resampling.BILINEAR)
        return reference_image

    def detect(self, image: Image.Image, config: ParkingLotConfig) -> list[DetectionMask]:
        settings = {
            "foreground_threshold": 0.15,
            "min_component_area": 700,
            **config.detector_settings,
        }

        if config.reference_image_path and config.reference_image_path.exists():
            reference_image = self._load_reference(config.reference_image_path, image.size)
            vehicle_mask = diff_foreground_mask(
                image=image,
                reference_image=reference_image,
                threshold=float(settings["foreground_threshold"]),
            )
            source = "reference_difference"
        else:
            vehicle_mask = coarse_object_mask(image=image)
            source = "single_frame_heuristic"

        components = connected_components(vehicle_mask, min_area=int(settings["min_component_area"]))
        if not components and vehicle_mask.any():
            components = [vehicle_mask]

        detections: list[DetectionMask] = []
        image_area = max(1, image.size[0] * image.size[1])
        for component in components:
            confidence = min(0.98, float(component.sum()) / image_area * 14.0 + 0.35)
            detections.append(
                DetectionMask(
                    mask=component,
                    confidence=round(confidence, 3),
                    label="vehicle_cluster",
                    source=source,
                )
            )
        return detections


class YoloSegVehicleDetector(BaseVehicleDetector):
    name = "yolo_seg"

    def __init__(self, model_path: Path | None = None, confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO backend requested, but ultralytics is not installed. "
                "Install ultralytics and a compatible PyTorch runtime, then retry."
            ) from exc

        if model_path is None:
            raise ValueError("YOLO backend requires yolo_model_path in the config or request.")

        self._model = YOLO(str(model_path))
        self._confidence = confidence

    def detect(self, image: Image.Image, config: ParkingLotConfig) -> list[DetectionMask]:
        np_image = np.asarray(image.convert("RGB"))
        result = self._model.predict(source=np_image, verbose=False, conf=self._confidence)[0]

        if result.masks is None or result.boxes is None:
            return []

        class_name_lookup = result.names
        mask_arrays = result.masks.data.cpu().numpy()
        class_indices = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        detections: list[DetectionMask] = []
        for mask_array, class_index, confidence in zip(mask_arrays, class_indices, confidences, strict=True):
            class_name = str(class_name_lookup[int(class_index)]).lower()
            if class_name not in VEHICLE_CLASS_NAMES:
                continue
            detections.append(
                DetectionMask(
                    mask=mask_array > 0.5,
                    confidence=float(confidence),
                    label=class_name,
                    source="yolo_seg",
                )
            )
        return detections


def create_detector(config: ParkingLotConfig, backend: str) -> BaseVehicleDetector:
    normalized_backend = backend.lower().strip()
    if normalized_backend == "heuristic":
        return HeuristicVehicleDetector()
    if normalized_backend == "yolo_seg":
        return YoloSegVehicleDetector(
            model_path=config.yolo_model_path,
            confidence=float(config.detector_settings.get("yolo_confidence", 0.25)),
        )
    raise ValueError(f"Unsupported detector backend: {backend}")
