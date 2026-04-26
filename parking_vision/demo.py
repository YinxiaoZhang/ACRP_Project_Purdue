from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import ParkingLotConfig, load_config
from .geometry import get_image_polygon, polygon_centroid, shrink_polygon, solve_homography


DEFAULT_OCCUPIED_STALLS = {"A2", "A3", "B2"}


def _as_int_polygon(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(int(round(x_coord)), int(round(y_coord))) for x_coord, y_coord in points]


def render_demo_image(
    config: ParkingLotConfig,
    occupied_stalls: set[str] | None = None,
) -> Image.Image:
    occupied_stalls = occupied_stalls or set()
    image = Image.new("RGB", (config.image_width, config.image_height), color=(56, 63, 69))
    draw = ImageDraw.Draw(image)

    homography = solve_homography(config.calibration.image_points, config.calibration.ground_points_m)
    inverse_homography = np.linalg.inv(homography)

    for y_coord in range(config.image_height):
        shade = int(48 + (y_coord / max(1, config.image_height - 1)) * 24)
        draw.line([(0, y_coord), (config.image_width, y_coord)], fill=(shade, shade + 4, shade + 6))

    lot_outline = _as_int_polygon(config.calibration.image_points)
    draw.polygon(lot_outline, outline=(226, 220, 120), width=3)

    for stall in config.stalls:
        image_polygon = get_image_polygon(stall, config.calibration, inverse_homography)
        polygon_int = _as_int_polygon(image_polygon)
        draw.polygon(polygon_int, outline=(240, 240, 240), width=3)

        centroid = polygon_centroid(image_polygon)
        label_x = int(centroid[0]) - 14
        label_y = int(centroid[1]) - 6
        draw.text((label_x, label_y), stall.stall_id, fill=(255, 255, 255))

        if stall.stall_id in occupied_stalls:
            car_polygon = shrink_polygon(image_polygon, factor=0.56)
            shadow_polygon = [(x_coord + 10.0, y_coord + 12.0) for x_coord, y_coord in car_polygon]
            draw.polygon(_as_int_polygon(shadow_polygon), fill=(18, 18, 18))

            car_color = (45, 112, 189) if stall.stall_id.endswith("2") else (186, 57, 52)
            draw.polygon(_as_int_polygon(car_polygon), fill=car_color, outline=(20, 20, 20), width=2)

            windshield = shrink_polygon(car_polygon, factor=0.62)
            draw.polygon(_as_int_polygon(windshield), fill=(179, 211, 235))

    return image.filter(ImageFilter.GaussianBlur(radius=0.4))


def write_demo_assets(
    config_path: str | Path,
    output_dir: str | Path,
    occupied_stalls: set[str] | None = None,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    empty_image = render_demo_image(config=config, occupied_stalls=set())
    occupied_image = render_demo_image(config=config, occupied_stalls=occupied_stalls or DEFAULT_OCCUPIED_STALLS)

    empty_path = output_dir / "demo_empty_lot.png"
    occupied_path = output_dir / "demo_current_lot.png"
    empty_image.save(empty_path)
    occupied_image.save(occupied_path)
    return empty_path, occupied_path
