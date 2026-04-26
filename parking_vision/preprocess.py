from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def normalize_lighting(image: Image.Image) -> np.ndarray:
    grayscale = ImageOps.autocontrast(image.convert("L"))
    grayscale_array = np.asarray(grayscale, dtype=np.float32)
    illumination = np.asarray(grayscale.filter(ImageFilter.GaussianBlur(radius=18)), dtype=np.float32) + 1.0
    normalized = np.clip((grayscale_array / illumination) * 128.0, 0.0, 255.0)
    return normalized


def saturation_channel(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    return hsv[:, :, 1]


def boolean_mask_cleanup(mask: np.ndarray, dilation_size: int = 3, erosion_size: int = 3) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if dilation_size > 1:
        mask_image = mask_image.filter(ImageFilter.MaxFilter(dilation_size))
    if erosion_size > 1:
        mask_image = mask_image.filter(ImageFilter.MinFilter(erosion_size))
    return np.asarray(mask_image, dtype=np.uint8) > 0


def diff_foreground_mask(
    image: Image.Image,
    reference_image: Image.Image,
    intensity_weight: float = 0.75,
    threshold: float = 0.15,
) -> np.ndarray:
    normalized_current = normalize_lighting(image)
    normalized_reference = normalize_lighting(reference_image)
    intensity_delta = np.abs(normalized_current - normalized_reference) / 255.0

    saturation_delta = np.abs(saturation_channel(image) - saturation_channel(reference_image)) / 255.0
    score = (intensity_delta * intensity_weight) + (saturation_delta * (1.0 - intensity_weight))
    mask = score >= threshold
    return boolean_mask_cleanup(mask, dilation_size=5, erosion_size=3)


def coarse_object_mask(image: Image.Image, darkness_bias: float = 0.56, saturation_bias: float = 0.2) -> np.ndarray:
    normalized = normalize_lighting(image)
    saturation = saturation_channel(image) / 255.0
    darkness = 1.0 - (normalized / 255.0)
    score = darkness * darkness_bias + saturation * saturation_bias
    adaptive_threshold = max(0.42, float(np.mean(score) + np.std(score) * 0.4))
    mask = score >= adaptive_threshold
    return boolean_mask_cleanup(mask, dilation_size=3, erosion_size=3)


def connected_components(mask: np.ndarray, min_area: int = 700) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue

            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            pixels: list[tuple[int, int]] = []

            while queue:
                y_coord, x_coord = queue.popleft()
                pixels.append((y_coord, x_coord))

                for neighbor_y, neighbor_x in (
                    (y_coord - 1, x_coord),
                    (y_coord + 1, x_coord),
                    (y_coord, x_coord - 1),
                    (y_coord, x_coord + 1),
                ):
                    if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
                        if mask[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                            visited[neighbor_y, neighbor_x] = True
                            queue.append((neighbor_y, neighbor_x))

            if len(pixels) < min_area:
                continue

            component = np.zeros_like(mask, dtype=bool)
            ys, xs = zip(*pixels)
            component[np.asarray(ys), np.asarray(xs)] = True
            components.append(component)

    return components
