from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SIZE_ORDER = {
    "compact": 1,
    "standard": 2,
    "oversized": 3,
}


@dataclass(frozen=True)
class Stall:
    row: int
    col: int
    size: str
    occupied: bool


@dataclass(frozen=True)
class TrialResult:
    occupancy_ratio: float
    trial_index: int
    directed_distance: float
    sequential_distance: float
    directed_checks: int
    sequential_checks: int
    improvement_pct: float


@dataclass(frozen=True)
class ParkingGraph:
    adjacency: Dict[str, List[Tuple[str, int]]]
    entry_node: str
    stall_nodes: Dict[Tuple[int, int], str]
    search_route: List[str]
    inspection_nodes: Dict[str, List[Tuple[int, int]]]


def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    labels = list(weights.keys())
    probabilities = list(weights.values())
    return rng.choices(labels, weights=probabilities, k=1)[0]


def generate_random_weights(labels: Sequence[str], rng: random.Random) -> Dict[str, float]:
    raw_values = [rng.random() for _ in labels]
    total = sum(raw_values)
    return {
        label: raw_value / total
        for label, raw_value in zip(labels, raw_values)
    }


def can_fit(vehicle_size: str, stall_size: str) -> bool:
    return SIZE_ORDER[stall_size] >= SIZE_ORDER[vehicle_size]


def aisle_count(rows: int) -> int:
    return math.ceil(rows / 2)


def stall_rows_for_aisle(aisle_index: int, rows: int) -> List[int]:
    first_row = aisle_index * 2
    candidate_rows = [first_row, first_row + 1]
    return [row for row in candidate_rows if row < rows]


def aisle_index_from_level(level: int, rows: int) -> int:
    return aisle_count(rows) - 1 - level


def stall_rows_for_level(level: int, rows: int) -> List[int]:
    return stall_rows_for_aisle(aisle_index_from_level(level, rows), rows)


def center_node(level: int) -> str:
    return f"CENTER_{level}"


def edge_node(side: str, level: int) -> str:
    return f"{side.upper()}_EDGE_{level}"


def aisle_node(level: int, side: str, col: int) -> str:
    return f"{side.upper()}_AISLE_{level}_{col}"


def stall_node(row: int, col: int) -> str:
    return f"STALL_{row}_{col}"


def add_edge(adjacency: Dict[str, List[Tuple[str, int]]], source: str, target: str, cost: int = 1) -> None:
    adjacency.setdefault(source, []).append((target, cost))
    adjacency.setdefault(target, [])


def section_columns(side: str, cols: int) -> List[int]:
    if cols % 2 != 0:
        raise ValueError("cols must be even so the left and right parking sections are equally distributed.")

    half_cols = cols // 2
    if side == "left":
        return list(range(0, half_cols))
    if side == "right":
        return list(range(half_cols, cols))
    raise ValueError(f"Unsupported side: {side}")


def section_direction(level: int, side: str) -> str:
    # Search alternates around the central pathway:
    # even levels go right edge -> center -> left edge
    # odd levels go left edge -> center -> right edge
    if side == "left":
        return "center_to_edge" if level % 2 == 0 else "edge_to_center"
    if side == "right":
        return "edge_to_center" if level % 2 == 0 else "center_to_edge"
    raise ValueError(f"Unsupported side: {side}")


def ordered_section_columns(level: int, side: str, cols: int) -> List[int]:
    columns = section_columns(side, cols)
    direction = section_direction(level, side)

    if side == "left":
        return list(reversed(columns)) if direction == "center_to_edge" else columns
    return columns if direction == "center_to_edge" else list(reversed(columns))


def edge_cost(adjacency: Dict[str, List[Tuple[str, int]]], source: str, target: str) -> int:
    for neighbor, cost in adjacency.get(source, []):
        if neighbor == target:
            return cost
    raise KeyError(f"No edge found from {source} to {target}")


def build_lot(
    rows: int,
    cols: int,
    occupancy_ratio: float,
    stall_weights: Dict[str, float],
    rng: random.Random,
) -> List[Stall]:
    stalls: List[Stall] = []
    for row in range(rows):
        for col in range(cols):
            size = weighted_choice(rng, stall_weights)
            occupied = rng.random() < occupancy_ratio
            stalls.append(Stall(row=row, col=col, size=size, occupied=occupied))
    return stalls


def build_parking_graph(rows: int, cols: int) -> ParkingGraph:
    total_aisles = aisle_count(rows)
    adjacency: Dict[str, List[Tuple[str, int]]] = {}
    entry_node = "ENTRY"
    stall_nodes: Dict[Tuple[int, int], str] = {}
    search_route: List[str] = [entry_node, center_node(0)]
    inspection_nodes: Dict[str, List[Tuple[int, int]]] = {}

    half_cols = cols // 2
    add_edge(adjacency, entry_node, center_node(0))
    add_edge(adjacency, entry_node, edge_node("left", 0), cost=half_cols + 1)
    add_edge(adjacency, entry_node, edge_node("right", 0), cost=half_cols + 1)

    # The center pathway and outer connectors are modeled as two-way vertical routes.
    for level in range(total_aisles - 1):
        add_edge(adjacency, center_node(level), center_node(level + 1))
        add_edge(adjacency, center_node(level + 1), center_node(level))
        add_edge(adjacency, edge_node("left", level), edge_node("left", level + 1))
        add_edge(adjacency, edge_node("left", level + 1), edge_node("left", level))
        add_edge(adjacency, edge_node("right", level), edge_node("right", level + 1))
        add_edge(adjacency, edge_node("right", level + 1), edge_node("right", level))

    top_level = total_aisles - 1
    add_edge(adjacency, edge_node("left", top_level), edge_node("right", top_level), cost=cols + 2)
    add_edge(adjacency, edge_node("right", top_level), edge_node("left", top_level), cost=cols + 2)

    for level in range(total_aisles):
        aisle_rows = stall_rows_for_level(level, rows)

        for side in ("left", "right"):
            ordered_columns = ordered_section_columns(level, side, cols)
            direction = section_direction(level, side)
            start_anchor = center_node(level) if direction == "center_to_edge" else edge_node(side, level)
            end_anchor = edge_node(side, level) if direction == "center_to_edge" else center_node(level)

            first_col = ordered_columns[0]
            last_col = ordered_columns[-1]
            add_edge(adjacency, start_anchor, aisle_node(level, side, first_col))

            for col in section_columns(side, cols):
                current_node = aisle_node(level, side, col)
                inspection_nodes[current_node] = [(row, col) for row in aisle_rows]
                for row in aisle_rows:
                    current_stall_node = stall_node(row, col)
                    stall_nodes[(row, col)] = current_stall_node
                    add_edge(adjacency, current_node, current_stall_node)

            for current_col, next_col in zip(ordered_columns, ordered_columns[1:]):
                add_edge(adjacency, aisle_node(level, side, current_col), aisle_node(level, side, next_col))

            add_edge(adjacency, aisle_node(level, side, last_col), end_anchor)

    # Sequential-search baseline:
    # inspect the entire left section first, then transition and inspect the right section.
    for level in range(total_aisles):
        for col in ordered_section_columns(level, "left", cols):
            search_route.append(aisle_node(level, "left", col))

        left_end_node = edge_node("left", level) if section_direction(level, "left") == "center_to_edge" else center_node(level)
        if level < total_aisles - 1:
            next_node = edge_node("left", level + 1) if section_direction(level + 1, "left") == "edge_to_center" else center_node(level + 1)
            if left_end_node != next_node:
                search_route.append(left_end_node)
            search_route.append(next_node)
        elif left_end_node != edge_node("left", top_level):
            search_route.append(left_end_node)
            search_route.append(edge_node("left", top_level))

    if search_route[-1] != edge_node("left", top_level):
        search_route.append(edge_node("left", top_level))
    search_route.append(edge_node("right", top_level))

    for level in range(top_level, -1, -1):
        if level < top_level:
            start_node = edge_node("right", level) if section_direction(level, "right") == "edge_to_center" else center_node(level)
            search_route.append(start_node)

        for col in ordered_section_columns(level, "right", cols):
            search_route.append(aisle_node(level, "right", col))

        right_end_node = edge_node("right", level) if section_direction(level, "right") == "center_to_edge" else center_node(level)
        if level > 0:
            next_node = edge_node("right", level - 1) if section_direction(level - 1, "right") == "edge_to_center" else center_node(level - 1)
            if right_end_node != next_node:
                search_route.append(right_end_node)

    return ParkingGraph(
        adjacency=adjacency,
        entry_node=entry_node,
        stall_nodes=stall_nodes,
        search_route=search_route,
        inspection_nodes=inspection_nodes,
    )


def dijkstra_distances(adjacency: Dict[str, List[Tuple[str, int]]], start: str) -> Dict[str, float]:
    distances: Dict[str, float] = {start: 0.0}
    frontier: List[Tuple[float, str]] = [(0.0, start)]

    while frontier:
        current_distance, current_node = heapq.heappop(frontier)
        if current_distance > distances.get(current_node, float("inf")):
            continue

        for neighbor, weight in adjacency.get(current_node, []):
            next_distance = current_distance + weight
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                heapq.heappush(frontier, (next_distance, neighbor))

    return distances


def simulate_directed_assignment(
    vehicle_size: str,
    stalls: Sequence[Stall],
    parking_graph: ParkingGraph,
    entry_distances: Dict[str, float],
) -> Optional[Tuple[float, int]]:
    candidates = [
        stall for stall in stalls if (not stall.occupied and can_fit(vehicle_size, stall.size))
    ]
    if not candidates:
        return None

    best = min(
        candidates,
        key=lambda stall: (
            entry_distances.get(parking_graph.stall_nodes[(stall.row, stall.col)], float("inf")),
            stall.row,
            stall.col,
        ),
    )
    travel_distance = entry_distances.get(parking_graph.stall_nodes[(best.row, best.col)], float("inf"))
    if math.isinf(travel_distance):
        return None
    return travel_distance, 1


def simulate_sequential_search(
    vehicle_size: str,
    stalls_by_position: Dict[Tuple[int, int], Stall],
    parking_graph: ParkingGraph,
) -> Optional[Tuple[float, int]]:
    checks = 0
    traversed_distance = 0
    previous_node = parking_graph.search_route[0]

    for current_node in parking_graph.search_route[1:]:
        traversed_distance += edge_cost(parking_graph.adjacency, previous_node, current_node)
        previous_node = current_node

        stall_positions = parking_graph.inspection_nodes.get(previous_node, [])
        for row, col in stall_positions:
            checks += 1
            stall = stalls_by_position[(row, col)]
            if stall.occupied:
                continue
            if can_fit(vehicle_size, stall.size):
                traversed_distance += 1
                return float(traversed_distance), checks

    return None


def run_trial(
    rows: int,
    cols: int,
    occupancy_ratio: float,
    parking_graph: ParkingGraph,
    entry_distances: Dict[str, float],
    rng: random.Random,
) -> Optional[TrialResult]:
    size_labels = list(SIZE_ORDER.keys())
    vehicle_weights = generate_random_weights(size_labels, rng)
    stall_weights = generate_random_weights(size_labels, rng)
    vehicle_size = weighted_choice(rng, vehicle_weights)
    stalls = build_lot(rows, cols, occupancy_ratio, stall_weights, rng)

    if not any((not stall.occupied and can_fit(vehicle_size, stall.size)) for stall in stalls):
        return None

    directed = simulate_directed_assignment(vehicle_size, stalls, parking_graph, entry_distances)
    stalls_by_position = {(stall.row, stall.col): stall for stall in stalls}
    sequential = simulate_sequential_search(vehicle_size, stalls_by_position, parking_graph)

    if directed is None or sequential is None:
        return None

    directed_distance, directed_checks = directed
    sequential_distance, sequential_checks = sequential
    improvement_pct = ((sequential_distance - directed_distance) / sequential_distance) * 100.0

    return TrialResult(
        occupancy_ratio=occupancy_ratio,
        trial_index=-1,
        directed_distance=directed_distance,
        sequential_distance=sequential_distance,
        directed_checks=directed_checks,
        sequential_checks=sequential_checks,
        improvement_pct=improvement_pct,
    )


def run_condition(
    rows: int,
    cols: int,
    occupancy_ratio: float,
    trials: int,
    seed: int,
) -> List[TrialResult]:
    rng = random.Random(seed)
    parking_graph = build_parking_graph(rows, cols)
    entry_distances = dijkstra_distances(parking_graph.adjacency, parking_graph.entry_node)
    results: List[TrialResult] = []
    next_trial_index = 1

    while len(results) < trials:
        trial = run_trial(rows, cols, occupancy_ratio, parking_graph, entry_distances, rng)
        if trial is None:
            continue
        results.append(
            TrialResult(
                occupancy_ratio=trial.occupancy_ratio,
                trial_index=next_trial_index,
                directed_distance=trial.directed_distance,
                sequential_distance=trial.sequential_distance,
                directed_checks=trial.directed_checks,
                sequential_checks=trial.sequential_checks,
                improvement_pct=trial.improvement_pct,
            )
        )
        next_trial_index += 1

    return results


def summarize_results(results: Sequence[TrialResult]) -> List[Dict[str, float]]:
    grouped: Dict[float, List[TrialResult]] = {}
    for result in results:
        grouped.setdefault(result.occupancy_ratio, []).append(result)

    summary_rows: List[Dict[str, float]] = []
    for occupancy_ratio in sorted(grouped.keys()):
        group = grouped[occupancy_ratio]
        directed_distances = [item.directed_distance for item in group]
        sequential_distances = [item.sequential_distance for item in group]
        directed_checks = [item.directed_checks for item in group]
        sequential_checks = [item.sequential_checks for item in group]
        improvements = [item.improvement_pct for item in group]
        trial_count = len(group)
        sample_stddev = stdev(improvements) if trial_count > 1 else 0.0
        ci_half_width = 1.96 * sample_stddev / math.sqrt(trial_count) if trial_count > 0 else 0.0
        avg_improvement = mean(improvements)
        avg_directed_distance = mean(directed_distances)
        avg_sequential_distance = mean(sequential_distances)
        aggregate_improvement = (
            (avg_sequential_distance - avg_directed_distance) / avg_sequential_distance * 100.0
            if avg_sequential_distance > 0
            else 0.0
        )
        aggregate_ci_low, aggregate_ci_high = bootstrap_aggregate_improvement_ci(
            directed_distances,
            sequential_distances,
            seed=int(round(occupancy_ratio * 1_000_000)),
        )

        summary_rows.append(
            {
                "occupancy_pct": occupancy_ratio * 100.0,
                "trials": float(trial_count),
                "avg_directed_distance": avg_directed_distance,
                "avg_sequential_distance": avg_sequential_distance,
                "avg_directed_checks": mean(directed_checks),
                "avg_sequential_checks": mean(sequential_checks),
                "avg_improvement_pct": avg_improvement,
                "aggregate_improvement_pct": aggregate_improvement,
                "aggregate_ci_low": aggregate_ci_low,
                "aggregate_ci_high": aggregate_ci_high,
                "improvement_stddev": pstdev(improvements),
                "improvement_ci_low": avg_improvement - ci_half_width,
                "improvement_ci_high": avg_improvement + ci_half_width,
            }
        )

    return summary_rows


def write_trial_results(path: Path, results: Sequence[TrialResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "occupancy_pct",
                "trial_index",
                "directed_distance",
                "sequential_distance",
                "directed_checks",
                "sequential_checks",
                "improvement_pct",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    round(result.occupancy_ratio * 100.0, 2),
                    result.trial_index,
                    round(result.directed_distance, 4),
                    round(result.sequential_distance, 4),
                    result.directed_checks,
                    result.sequential_checks,
                    round(result.improvement_pct, 4),
                ]
            )


def write_summary(path: Path, summary_rows: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "occupancy_pct",
                "trials",
                "avg_directed_distance",
                "avg_sequential_distance",
                "avg_directed_checks",
                "avg_sequential_checks",
                "avg_improvement_pct",
                "aggregate_improvement_pct",
                "aggregate_ci_low",
                "aggregate_ci_high",
                "improvement_stddev",
                "improvement_ci_low",
                "improvement_ci_high",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    round(row["occupancy_pct"], 2),
                    int(row["trials"]),
                    round(row["avg_directed_distance"], 4),
                    round(row["avg_sequential_distance"], 4),
                    round(row["avg_directed_checks"], 4),
                    round(row["avg_sequential_checks"], 4),
                    round(row["avg_improvement_pct"], 4),
                    round(row["aggregate_improvement_pct"], 4),
                    round(row["aggregate_ci_low"], 4),
                    round(row["aggregate_ci_high"], 4),
                    round(row["improvement_stddev"], 4),
                    round(row["improvement_ci_low"], 4),
                    round(row["improvement_ci_high"], 4),
                ]
            )


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = rank - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def bootstrap_aggregate_improvement_ci(
    directed_distances: Sequence[float],
    sequential_distances: Sequence[float],
    seed: int,
    bootstrap_samples: int = 1000,
) -> Tuple[float, float]:
    if not directed_distances or not sequential_distances:
        return 0.0, 0.0

    sample_size = len(directed_distances)
    rng = random.Random(seed)
    aggregate_values: List[float] = []

    for _ in range(bootstrap_samples):
        directed_sum = 0.0
        sequential_sum = 0.0
        for _ in range(sample_size):
            index = rng.randrange(sample_size)
            directed_sum += directed_distances[index]
            sequential_sum += sequential_distances[index]
        aggregate_values.append(
            ((sequential_sum - directed_sum) / sequential_sum * 100.0) if sequential_sum > 0 else 0.0
        )

    aggregate_values.sort()
    return percentile(aggregate_values, 0.025), percentile(aggregate_values, 0.975)


def render_improvement_ci_plot(path: Path, summary_rows: Sequence[Dict[str, float]]) -> None:
    width = 920
    height = 540
    left = 90
    right = 40
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    occupancy_values = [row["occupancy_pct"] for row in summary_rows]
    improvement_values = [row["avg_improvement_pct"] for row in summary_rows]
    ci_lows = [row["improvement_ci_low"] for row in summary_rows]
    ci_highs = [row["improvement_ci_high"] for row in summary_rows]

    x_min = min(occupancy_values)
    x_max = max(occupancy_values)
    y_min = min(0.0, min(ci_lows))
    y_max = max(ci_highs) * 1.10
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    def x_position(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_position(value: float) -> float:
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    line_coordinates = [
        (x_position(row["occupancy_pct"]), y_position(row["avg_improvement_pct"]))
        for row in summary_rows
    ]
    upper_coordinates = [
        (x_position(row["occupancy_pct"]), y_position(row["improvement_ci_high"]))
        for row in summary_rows
    ]
    lower_coordinates = [
        (x_position(row["occupancy_pct"]), y_position(row["improvement_ci_low"]))
        for row in summary_rows
    ]
    line_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in line_coordinates)
    band_points = " ".join(
        [f"{x:.2f},{y:.2f}" for x, y in upper_coordinates]
        + [f"{x:.2f},{y:.2f}" for x, y in reversed(lower_coordinates)]
    )

    x_ticks = [int(row["occupancy_pct"]) for row in summary_rows]
    y_ticks = [y_min + (y_max - y_min) * step / 4.0 for step in range(5)]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe" />',
        f'<text x="{width / 2:.2f}" y="32" text-anchor="middle" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#102a43">'
        'Efficiency Improvement by Occupancy</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#d9e2ec" />',
    ]

    for tick in x_ticks:
        x = x_position(float(tick))
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick}%</text>'
        )

    for tick in y_ticks:
        y = y_position(tick)
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick:.1f}%</text>'
        )

    svg_parts.append(
        f'<polygon points="{band_points}" fill="#93c5fd" fill-opacity="0.45" stroke="none" />'
    )
    svg_parts.append(
        f'<polyline points="{line_points}" fill="none" stroke="#2f6fb0" stroke-width="3.5" />'
    )

    for x, y in line_coordinates:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#2f6fb0" />')

    svg_parts.append(
        f'<text x="{width / 2:.2f}" y="{height - 44}" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Occupancy Level (%)</text>'
    )
    svg_parts.append(
        f'<text x="24" y="{height / 2:.2f}" text-anchor="middle" transform="rotate(-90 24 {height / 2:.2f})" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Efficiency Improvement (%)</text>'
    )
    svg_parts.append(
        '<line x1="610" y1="92" x2="640" y2="92" stroke="#2f6fb0" stroke-width="3.5" />'
    )
    svg_parts.append(
        '<text x="650" y="96" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Mean improvement</text>'
    )
    svg_parts.append(
        '<rect x="610" y="108" width="30" height="18" fill="#93c5fd" fill-opacity="0.45" stroke="none" />'
    )
    svg_parts.append(
        '<text x="650" y="122" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#102a43">95% confidence band</text>'
    )
    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def scale_points(
    values: Sequence[Tuple[float, float]],
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> List[Tuple[float, float]]:
    xs = [point[0] for point in values]
    ys = [point[1] for point in values]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    plot_width = width - left - right
    plot_height = height - top - bottom

    scaled: List[Tuple[float, float]] = []
    for x_value, y_value in values:
        x = left + ((x_value - x_min) / (x_max - x_min)) * plot_width
        y = top + plot_height - ((y_value - y_min) / (y_max - y_min)) * plot_height
        scaled.append((x, y))
    return scaled


def render_method_comparison_plot(path: Path, summary_rows: Sequence[Dict[str, float]]) -> None:
    width = 920
    height = 540
    left = 90
    right = 40
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    occupancy_values = [row["occupancy_pct"] for row in summary_rows]
    directed_values = [(row["occupancy_pct"], row["avg_directed_distance"]) for row in summary_rows]
    sequential_values = [(row["occupancy_pct"], row["avg_sequential_distance"]) for row in summary_rows]
    all_values = directed_values + sequential_values

    points = scale_points(all_values, width, height, left, top, right, bottom)
    directed_points = points[: len(directed_values)]
    sequential_points = points[len(directed_values) :]

    x_min = min(occupancy_values)
    x_max = max(occupancy_values)
    y_max = max(
        max(row["avg_directed_distance"] for row in summary_rows),
        max(row["avg_sequential_distance"] for row in summary_rows),
    ) * 1.10
    if y_max <= 0:
        y_max = 1.0

    def polyline(points_to_join: Sequence[Tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points_to_join)

    def x_position(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_position(value: float) -> float:
        return top + plot_height - (value / y_max) * plot_height

    x_ticks = [int(row["occupancy_pct"]) for row in summary_rows]
    y_ticks = [y_max * step / 4.0 for step in range(5)]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe" />',
        f'<text x="{width / 2:.2f}" y="32" text-anchor="middle" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#102a43">'
        "Dijkstra Method vs. Traditional Method</text>",
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#d9e2ec" />',
    ]

    for tick in x_ticks:
        x = x_position(float(tick))
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick}%</text>'
        )

    for tick in y_ticks:
        y = y_position(tick)
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick:.1f}</text>'
        )

    svg_parts.append(
        f'<polyline points="{polyline(directed_points)}" fill="none" stroke="#2f6fb0" stroke-width="3.5" />'
    )
    svg_parts.append(
        f'<polyline points="{polyline(sequential_points)}" fill="none" stroke="#d97706" stroke-width="3.5" />'
    )

    for x, y in directed_points:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#2f6fb0" />')
    for x, y in sequential_points:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#d97706" />')

    svg_parts.append(
        f'<text x="{width / 2:.2f}" y="{height - 44}" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Occupancy Level (%)</text>'
    )
    svg_parts.append(
        f'<text x="26" y="{height / 2:.2f}" text-anchor="middle" transform="rotate(-90 26 {height / 2:.2f})" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Average Travel Distance (simulation units)</text>'
    )
    svg_parts.append(
        '<line x1="610" y1="92" x2="640" y2="92" stroke="#2f6fb0" stroke-width="3.5" />'
    )
    svg_parts.append(
        '<text x="650" y="96" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Dijkstra method</text>'
    )
    svg_parts.append(
        '<line x1="610" y1="118" x2="640" y2="118" stroke="#d97706" stroke-width="3.5" />'
    )
    svg_parts.append(
        '<text x="650" y="122" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Traditional method</text>'
    )
    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def render_aggregate_efficiency_plot(path: Path, summary_rows: Sequence[Dict[str, float]]) -> None:
    width = 920
    height = 540
    left = 90
    right = 40
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    occupancy_values = [row["occupancy_pct"] for row in summary_rows]
    improvement_values = [row["aggregate_improvement_pct"] for row in summary_rows]
    ci_lows = [row["aggregate_ci_low"] for row in summary_rows]
    ci_highs = [row["aggregate_ci_high"] for row in summary_rows]
    y_min = min(0.0, min(ci_lows)) if ci_lows else 0.0
    y_max = max(ci_highs) * 1.10 if ci_highs else 1.0
    if y_max <= 0:
        y_max = 1.0

    def x_position(value: float) -> float:
        return left + ((value - min(occupancy_values)) / (max(occupancy_values) - min(occupancy_values))) * plot_width

    def y_position(value: float) -> float:
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    line_coordinates = [
        (x_position(row["occupancy_pct"]), y_position(row["aggregate_improvement_pct"]))
        for row in summary_rows
    ]
    upper_coordinates = [
        (x_position(row["occupancy_pct"]), y_position(row["aggregate_ci_high"]))
        for row in summary_rows
    ]
    lower_coordinates = [
        (x_position(row["occupancy_pct"]), y_position(row["aggregate_ci_low"]))
        for row in summary_rows
    ]
    polyline_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in line_coordinates)
    band_points = " ".join(
        [f"{x:.2f},{y:.2f}" for x, y in upper_coordinates]
        + [f"{x:.2f},{y:.2f}" for x, y in reversed(lower_coordinates)]
    )
    x_ticks = [int(row["occupancy_pct"]) for row in summary_rows]
    y_ticks = [y_min + (y_max - y_min) * step / 4.0 for step in range(5)]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe" />',
        f'<text x="{width / 2:.2f}" y="32" text-anchor="middle" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#102a43">'
        'Efficiency Improvement by Occupancy</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#d9e2ec" />',
    ]

    for tick in x_ticks:
        x = x_position(float(tick))
        svg_parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eef2f6" />')
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick}%</text>'
        )

    for tick in y_ticks:
        y = y_position(tick)
        svg_parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#eef2f6" />')
        svg_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick:.1f}%</text>'
        )

    svg_parts.append(
        f'<polygon points="{band_points}" fill="#93c5fd" fill-opacity="0.45" stroke="none" />'
    )
    svg_parts.append(f'<polyline points="{polyline_points}" fill="none" stroke="#2f6fb0" stroke-width="3.5" />')
    for x, y in line_coordinates:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#2f6fb0" />')

    svg_parts.append(
        f'<text x="{width / 2:.2f}" y="{height - 44}" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Occupancy Level (%)</text>'
    )
    svg_parts.append(
        f'<text x="24" y="{height / 2:.2f}" text-anchor="middle" transform="rotate(-90 24 {height / 2:.2f})" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Efficiency Improvement (%)</text>'
    )
    svg_parts.append(
        '<line x1="610" y1="92" x2="640" y2="92" stroke="#2f6fb0" stroke-width="3.5" />'
    )
    svg_parts.append(
        '<text x="650" y="96" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Aggregate improvement</text>'
    )
    svg_parts.append(
        '<rect x="610" y="108" width="30" height="18" fill="#93c5fd" fill-opacity="0.45" stroke="none" />'
    )
    svg_parts.append(
        '<text x="650" y="122" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#102a43">95% confidence band</text>'
    )
    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def render_svg_plot(path: Path, summary_rows: Sequence[Dict[str, float]]) -> None:
    width = 900
    height = 560
    left = 80
    right = 40
    top = 60
    bottom = 80

    occupancy_values = [row["occupancy_pct"] for row in summary_rows]
    directed_values = [(row["occupancy_pct"], row["avg_directed_distance"]) for row in summary_rows]
    sequential_values = [(row["occupancy_pct"], row["avg_sequential_distance"]) for row in summary_rows]
    improvement_values = [(row["occupancy_pct"], row["avg_improvement_pct"]) for row in summary_rows]

    distance_values = directed_values + sequential_values
    distance_points = scale_points(distance_values, width, height, left, top, right, bottom + 140)
    directed_points = distance_points[: len(directed_values)]
    sequential_points = distance_points[len(directed_values) :]
    improvement_points = scale_points(improvement_values, width, height, left, top + 320, right, 40)

    plot_width = width - left - right
    distance_plot_height = height - top - (bottom + 140)

    x_ticks = [70, 75, 80, 85, 90, 95]
    all_distance_y = [row["avg_directed_distance"] for row in summary_rows] + [
        row["avg_sequential_distance"] for row in summary_rows
    ]
    distance_y_min = 0
    distance_y_max = max(all_distance_y) * 1.10
    if distance_y_max <= 0:
        distance_y_max = 1.0

    improvement_y_min = 0
    improvement_y_max = max(row["avg_improvement_pct"] for row in summary_rows) * 1.10
    if improvement_y_max <= 0:
        improvement_y_max = 1.0

    def polyline(points: Sequence[Tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    def x_position(value: float) -> float:
        return left + ((value - min(occupancy_values)) / (max(occupancy_values) - min(occupancy_values))) * plot_width

    def distance_y_position(value: float) -> float:
        return top + distance_plot_height - (value / distance_y_max) * distance_plot_height

    improvement_plot_top = top + 320
    improvement_plot_height = height - improvement_plot_top - 40

    def improvement_y_position(value: float) -> float:
        return improvement_plot_top + improvement_plot_height - (value / improvement_y_max) * improvement_plot_height

    distance_tick_values = [distance_y_max * step / 4.0 for step in range(5)]
    improvement_tick_values = [improvement_y_max * step / 4.0 for step in range(5)]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe" />',
        '<text x="40" y="32" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#102a43">'
        "Airport Parking Monte Carlo Simulation</text>",
        '<text x="40" y="52" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#486581">'
        "Directed stall assignment vs sequential driver search</text>",
    ]

    svg_parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{distance_plot_height}" fill="#ffffff" stroke="#d9e2ec" />')
    svg_parts.append(
        f'<rect x="{left}" y="{improvement_plot_top}" width="{plot_width}" height="{improvement_plot_height}" fill="#ffffff" stroke="#d9e2ec" />'
    )

    for tick in x_ticks:
        x = x_position(tick)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + distance_plot_height}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{improvement_plot_top}" x2="{x:.2f}" y2="{improvement_plot_top + improvement_plot_height}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{int(tick)}%</text>'
        )

    for tick in distance_tick_values:
        y = distance_y_position(tick)
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick:.0f}</text>'
        )

    for tick in improvement_tick_values:
        y = improvement_y_position(tick)
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#eef2f6" />'
        )
        svg_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#486581">{tick:.0f}</text>'
        )

    svg_parts.append(
        f'<polyline points="{polyline(directed_points)}" fill="none" stroke="#1f78b4" stroke-width="3" />'
    )
    svg_parts.append(
        f'<polyline points="{polyline(sequential_points)}" fill="none" stroke="#d95f02" stroke-width="3" />'
    )
    svg_parts.append(
        f'<polyline points="{polyline(improvement_points)}" fill="none" stroke="#2ca25f" stroke-width="3" />'
    )

    for x, y in directed_points:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1f78b4" />')
    for x, y in sequential_points:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#d95f02" />')
    for x, y in improvement_points:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2ca25f" />')

    svg_parts.append(
        f'<text x="{left}" y="{top - 14}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Average travel distance to assigned stall</text>'
    )
    svg_parts.append(
        f'<text x="{left}" y="{improvement_plot_top - 14}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Average efficiency improvement (%)</text>'
    )
    svg_parts.append(
        f'<text x="{width / 2:.2f}" y="{height - 44}" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#102a43">Occupancy level</text>'
    )

    legend_y = 92
    legend_x = width - 250
    legend_items = [
        ("#1f78b4", "Directed assignment distance"),
        ("#d95f02", "Sequential search distance"),
        ("#2ca25f", "Efficiency improvement"),
    ]
    for index, (color, label) in enumerate(legend_items):
        item_y = legend_y + index * 24
        svg_parts.append(f'<line x1="{legend_x}" y1="{item_y}" x2="{legend_x + 24}" y2="{item_y}" stroke="{color}" stroke-width="4" />')
        svg_parts.append(
            f'<text x="{legend_x + 34}" y="{item_y + 4}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#102a43">{label}</text>'
        )

    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation for airport parking efficiency.")
    parser.add_argument("--rows", type=int, default=50, help="Number of parking rows.")
    parser.add_argument("--cols", type=int, default=100, help="Number of parking columns.")
    parser.add_argument("--start-occupancy", type=float, default=0.70, help="Starting occupancy ratio.")
    parser.add_argument("--end-occupancy", type=float, default=0.95, help="Ending occupancy ratio.")
    parser.add_argument("--occupancy-step", type=float, default=0.05, help="Occupancy interval.")
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials per occupancy condition.")
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV and SVG artifacts.",
    )
    return parser.parse_args()


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    current = start
    while current <= stop + (step / 2.0):
        yield round(current, 10)
        current += step


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[TrialResult] = []
    for index, occupancy_ratio in enumerate(frange(args.start_occupancy, args.end_occupancy, args.occupancy_step)):
        seed = args.seed + index
        condition_results = run_condition(
            rows=args.rows,
            cols=args.cols,
            occupancy_ratio=occupancy_ratio,
            trials=args.trials,
            seed=seed,
        )
        all_results.extend(condition_results)

    summary_rows = summarize_results(all_results)

    trials_path = args.output_dir / "parking_efficiency_trials.csv"
    summary_path = args.output_dir / "parking_efficiency_summary.csv"
    plot_path = args.output_dir / "parking_efficiency_plot.svg"
    ci_plot_path = args.output_dir / "parking_efficiency_improvement_ci_plot.svg"
    method_plot_path = args.output_dir / "parking_efficiency_method_comparison_plot.svg"
    aggregate_plot_path = args.output_dir / "parking_efficiency_aggregate_improvement_plot.svg"

    write_trial_results(trials_path, all_results)
    write_summary(summary_path, summary_rows)
    render_svg_plot(plot_path, summary_rows)
    render_improvement_ci_plot(ci_plot_path, summary_rows)
    render_method_comparison_plot(method_plot_path, summary_rows)
    render_aggregate_efficiency_plot(aggregate_plot_path, summary_rows)

    print(f"Trials written to: {trials_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Plot written to: {plot_path}")
    print(f"CI plot written to: {ci_plot_path}")
    print(f"Method plot written to: {method_plot_path}")
    print(f"Aggregate efficiency plot written to: {aggregate_plot_path}")
    print("")
    print("Summary by occupancy:")
    for row in summary_rows:
        print(
            f"  {row['occupancy_pct']:.0f}% occupancy | "
            f"directed distance={row['avg_directed_distance']:.2f} | "
            f"sequential distance={row['avg_sequential_distance']:.2f} | "
            f"improvement={row['avg_improvement_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
