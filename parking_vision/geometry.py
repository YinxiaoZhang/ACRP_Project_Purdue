from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .config import CameraCalibration, Point, SizeThresholds, StallDefinition


def polygon_area(points: Iterable[Point]) -> float:
    polygon = np.asarray(list(points), dtype=float)
    if len(polygon) < 3:
        return 0.0
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return float(abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))) / 2.0)


def polygon_centroid(points: Iterable[Point]) -> Point:
    polygon = np.asarray(list(points), dtype=float)
    if len(polygon) == 0:
        return (0.0, 0.0)
    return (float(polygon[:, 0].mean()), float(polygon[:, 1].mean()))


def shrink_polygon(points: Iterable[Point], factor: float) -> list[Point]:
    cx, cy = polygon_centroid(points)
    shrunk: list[Point] = []
    for x_coord, y_coord in points:
        shrunk.append((cx + (x_coord - cx) * factor, cy + (y_coord - cy) * factor))
    return shrunk


def polygon_to_mask(size: tuple[int, int], points: Iterable[Point]) -> np.ndarray:
    width, height = size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(list(points), fill=255)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def solve_homography(image_points: list[Point], ground_points: list[Point]) -> np.ndarray:
    if len(image_points) < 4 or len(image_points) != len(ground_points):
        raise ValueError("Homography requires at least four matching image and ground points.")

    matrix_rows: list[list[float]] = []
    for (x_coord, y_coord), (ground_x, ground_y) in zip(image_points, ground_points, strict=True):
        matrix_rows.append([-x_coord, -y_coord, -1.0, 0.0, 0.0, 0.0, x_coord * ground_x, y_coord * ground_x, ground_x])
        matrix_rows.append([0.0, 0.0, 0.0, -x_coord, -y_coord, -1.0, x_coord * ground_y, y_coord * ground_y, ground_y])

    matrix = np.asarray(matrix_rows, dtype=float)
    _, _, vh = np.linalg.svd(matrix)
    homography = vh[-1].reshape((3, 3))
    return homography / homography[2, 2]


def transform_points(points: Iterable[Point], homography: np.ndarray) -> list[Point]:
    transformed: list[Point] = []
    for x_coord, y_coord in points:
        vector = np.array([x_coord, y_coord, 1.0], dtype=float)
        mapped = homography @ vector
        mapped /= mapped[2]
        transformed.append((float(mapped[0]), float(mapped[1])))
    return transformed


def get_image_polygon(stall: StallDefinition, calibration: CameraCalibration, inverse_homography: np.ndarray | None) -> list[Point]:
    if stall.polygon is not None:
        return list(stall.polygon)
    if stall.ground_polygon_m is None or inverse_homography is None:
        raise ValueError(f"Stall {stall.stall_id} cannot be projected without calibration.")
    return transform_points(stall.ground_polygon_m, inverse_homography)


def edge_lengths(points: Iterable[Point]) -> list[float]:
    polygon = list(points)
    if len(polygon) < 2:
        return []
    lengths: list[float] = []
    for index, current in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        lengths.append(float(np.hypot(next_point[0] - current[0], next_point[1] - current[1])))
    return lengths


def classify_stall_size(area_m2: float | None, thresholds: SizeThresholds) -> str:
    if area_m2 is None:
        return "unknown"
    if area_m2 <= thresholds.compact_max_area_m2:
        return "compact"
    if area_m2 <= thresholds.standard_max_area_m2:
        return "standard"
    if area_m2 <= thresholds.large_max_area_m2:
        return "large"
    return "oversized"


def stall_metrics(
    stall: StallDefinition,
    image_polygon: list[Point],
    calibration: CameraCalibration,
    homography: np.ndarray | None,
    thresholds: SizeThresholds,
) -> dict[str, float | str | None]:
    polygon_area_pixels = polygon_area(image_polygon)

    if stall.ground_polygon_m is not None:
        lengths = edge_lengths(stall.ground_polygon_m)
        area_m2 = polygon_area(stall.ground_polygon_m)
    elif homography is not None:
        projected = transform_points(image_polygon, homography)
        lengths = edge_lengths(projected)
        area_m2 = polygon_area(projected)
    elif calibration.pixels_per_meter:
        ppm = calibration.pixels_per_meter
        lengths = [length / ppm for length in edge_lengths(image_polygon)]
        area_m2 = polygon_area_pixels / (ppm * ppm)
    else:
        lengths = []
        area_m2 = None

    if lengths:
        sorted_lengths = sorted(lengths)
        width_m = float(np.mean(sorted_lengths[:2]))
        length_m = float(np.mean(sorted_lengths[-2:]))
    else:
        width_m = None
        length_m = None

    size_label = stall.size_label or classify_stall_size(area_m2, thresholds)

    return {
        "size": size_label,
        "polygon_area_pixels": round(polygon_area_pixels, 2),
        "polygon_area_m2": round(area_m2, 2) if area_m2 is not None else None,
        "length_m": round(length_m, 2) if length_m is not None else None,
        "width_m": round(width_m, 2) if width_m is not None else None,
    }
