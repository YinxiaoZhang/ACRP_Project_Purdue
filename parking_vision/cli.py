from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import serve
from .demo import DEFAULT_OCCUPIED_STALLS, write_demo_assets
from .service import ParkingVisionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parking space size and availability analyzer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one parking-lot image.")
    analyze_parser.add_argument("--image", required=True, help="Image path to analyze.")
    analyze_parser.add_argument("--config", required=True, help="Path to the parking-lot JSON config.")
    analyze_parser.add_argument("--backend", default="heuristic", help="Detector backend: heuristic or yolo_seg.")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP JSON API.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--default-config", help="Optional default config path for /analyze.")

    demo_parser = subparsers.add_parser("generate-demo", help="Create synthetic demo images.")
    demo_parser.add_argument("--config", required=True, help="Path to the parking-lot JSON config.")
    demo_parser.add_argument("--output-dir", required=True, help="Directory for generated demo images.")
    demo_parser.add_argument(
        "--occupied",
        nargs="*",
        default=sorted(DEFAULT_OCCUPIED_STALLS),
        help="List of stall IDs to mark occupied in the synthetic demo image.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        service = ParkingVisionService()
        result = service.analyze_path(image_path=args.image, config_path=args.config, backend=args.backend)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "serve":
        serve(host=args.host, port=args.port, default_config_path=args.default_config)
        return

    if args.command == "generate-demo":
        empty_path, occupied_path = write_demo_assets(
            config_path=args.config,
            output_dir=args.output_dir,
            occupied_stalls=set(args.occupied),
        )
        print(json.dumps({"empty_image": str(empty_path), "occupied_image": str(occupied_path)}, indent=2))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
