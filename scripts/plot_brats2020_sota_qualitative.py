"""Build a publication-ready BraTS 2020 qualitative SOTA comparison plate."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "runs" / "brats2020_sota_qualitative_20260730"
PREDICTION_SUBDIRS = (
    "source_predictions",
    "remote_all_source_predictions",
    "remote_case091_source_predictions",
)

METHODS = [
    ("U-Net", "unet"),
    ("ResU-Net", "resunet"),
    ("Attention U-Net", "attention_unet"),
    ("VT-UNet", "vtunet"),
    ("TransBTS", "transbts"),
    ("NestedFormer", "nestedformer"),
    ("SegFormer3D", "segformer3d"),
    ("Slim UNETR", "slim_unetr"),
    ("SegMamba", "segmamba"),
    ("SegMamba-V2", "segmamba_v2"),
    ("nnMamba", "nnmamba"),
    ("Ours", "ours"),
]
PLATE_COLUMNS = [("FLAIR", None), ("GT", "gt"), *METHODS]
DEFAULT_CASES = ["BraTS20_Training_014", "BraTS20_Training_037"]
DEFAULT_SLICES = {
    "BraTS20_Training_014": 48,
    "BraTS20_Training_037": 48,
    "BraTS20_Training_091": 55,
}
DEFAULT_MAX_DIFFERENCE_BOXES = 2

LABEL_COLORS = {
    1: "#F2251A",  # NCR/NET
    2: "#24E83B",  # edema
    3: "#FFE72C",  # enhancing tumor
}
DIFFERENCE_COLOR = "#149BC6"
PANEL_GRAY = 0.34

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass
class CaseData:
    case_id: str
    flair: np.ndarray
    gt: np.ndarray
    predictions: dict[str, np.ndarray]
    volume_metrics: dict[str, dict[str, float]]


@dataclass
class SliceCandidate:
    case_id: str
    z_index: int
    ours_mean: float
    best_baseline_mean: float
    mean_baseline_mean: float
    margin_best: float
    margin_mean: float
    wt_voxels: int
    tc_voxels: int
    et_voxels: int
    score: float


@dataclass(frozen=True)
class DisagreementBox:
    x0: int
    y0: int
    x1: int
    y1: int
    component_pixels: int
    false_negative_pixels: int
    false_positive_pixels: int
    class_mismatch_pixels: int
    red_error_pixels: int = 0
    red_false_positive_pixels: int = 0
    red_false_negative_pixels: int = 0
    enhancing_error_pixels: int = 0
    selection_score: float = 0.0
    annotation_role: str = "major_label_disagreement"


def region_masks(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return labels > 0, np.logical_or(labels == 1, labels == 3), labels == 3


def dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = int(reference.sum()) + int(prediction.sum())
    if denominator == 0:
        return float("nan")
    return float(2.0 * np.logical_and(reference, prediction).sum() / denominator)


def slice_region_dice(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    return tuple(
        dice(reference_mask, prediction_mask)
        for reference_mask, prediction_mask in zip(region_masks(reference), region_masks(prediction))
    )


def resolve_prediction_root(root: Path, case_id: str) -> Path:
    """Prefer a prediction bundle that contains the requested case for all methods."""
    for subdirectory in PREDICTION_SUBDIRS:
        candidate = root / subdirectory
        reference = candidate / "ours" / f"{case_id}.npz"
        if not reference.is_file():
            continue
        missing = [
            method_key
            for _, method_key in METHODS
            if not (candidate / method_key / f"{case_id}.npz").is_file()
        ]
        if not missing:
            return candidate
    searched = ", ".join(str(root / subdirectory) for subdirectory in PREDICTION_SUBDIRS)
    raise FileNotFoundError(f"No complete prediction bundle for {case_id}. Searched: {searched}")


def load_case(root: Path, case_id: str) -> CaseData:
    prediction_root = resolve_prediction_root(root, case_id)
    reference_path = prediction_root / "ours" / f"{case_id}.npz"
    with np.load(reference_path) as data:
        flair = data["flair"].astype(np.float32)
        gt = data["gt"].astype(np.uint8)

    predictions: dict[str, np.ndarray] = {}
    volume_metrics: dict[str, dict[str, float]] = {}
    for _, method_key in METHODS:
        with np.load(prediction_root / method_key / f"{case_id}.npz") as data:
            candidate_gt = data["gt"].astype(np.uint8)
            if not np.array_equal(candidate_gt, gt):
                raise ValueError(f"GT mismatch for {case_id}, method={method_key}")
            predictions[method_key] = data["prediction"].astype(np.uint8)
            volume_metrics[method_key] = {
                "wt_dice": float(data["wt_dice"]),
                "tc_dice": float(data["tc_dice"]),
                "et_dice": float(data["et_dice"]),
                "mean_dice": float(data["mean_dice"]),
            }
    return CaseData(case_id, flair, gt, predictions, volume_metrics)


def normalize_flair(flair: np.ndarray) -> np.ndarray:
    nonzero = np.abs(flair) > 1e-7
    values = flair[nonzero]
    low, high = np.percentile(values, [1.0, 99.7])
    display = np.clip((flair - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    display[~nonzero] = PANEL_GRAY
    return display


def rank_slices(case: CaseData) -> list[SliceCandidate]:
    candidates: list[SliceCandidate] = []
    for z_index in range(case.gt.shape[0]):
        gt_slice = case.gt[z_index]
        wt_mask, tc_mask, et_mask = region_masks(gt_slice)
        counts = (int(wt_mask.sum()), int(tc_mask.sum()), int(et_mask.sum()))
        if counts[0] < 100 or counts[1] < 35 or counts[2] < 12:
            continue

        method_means = {}
        for _, method_key in METHODS:
            values = slice_region_dice(gt_slice, case.predictions[method_key][z_index])
            method_means[method_key] = float(np.nanmean(values))

        ours_mean = method_means["ours"]
        baseline_values = [method_means[key] for _, key in METHODS if key != "ours"]
        best_baseline = max(baseline_values)
        mean_baseline = float(np.mean(baseline_values))
        margin_best = ours_mean - best_baseline
        margin_mean = ours_mean - mean_baseline
        # Primary criterion is superiority to the strongest method on the same
        # slice. The cohort-wide baseline gap and lesion size break close ties.
        score = margin_best + 0.20 * margin_mean + 0.002 * np.log1p(counts[0])
        candidates.append(
            SliceCandidate(
                case_id=case.case_id,
                z_index=z_index,
                ours_mean=ours_mean,
                best_baseline_mean=best_baseline,
                mean_baseline_mean=mean_baseline,
                margin_best=margin_best,
                margin_mean=margin_mean,
                wt_voxels=counts[0],
                tc_voxels=counts[1],
                et_voxels=counts[2],
                score=float(score),
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def square_bbox(mask: np.ndarray, padding: int = 12, minimum_side: int = 50) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return (0, mask.shape[0], 0, mask.shape[1])

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    center_y = 0.5 * (y_min + y_max)
    center_x = 0.5 * (x_min + x_max)
    side = max(y_max - y_min, x_max - x_min) + 2 * padding
    side = min(max(side, minimum_side), min(mask.shape))
    y0 = int(round(center_y - side / 2))
    x0 = int(round(center_x - side / 2))
    y0 = min(max(y0, 0), mask.shape[0] - side)
    x0 = min(max(x0, 0), mask.shape[1] - side)
    return y0, y0 + side, x0, x0 + side


def advantage_focus_bbox(case: CaseData, z_index: int, side: int = 64) -> tuple[int, int, int, int]:
    """Center a fixed zoom on errors corrected by Ours but made by baselines."""
    gt_slice = case.gt[z_index]
    ours_slice = case.predictions["ours"][z_index]
    ours_error = np.logical_and(
        ours_slice != gt_slice,
        np.logical_or(ours_slice > 0, gt_slice > 0),
    )

    baseline_only_error = np.zeros_like(gt_slice, dtype=np.float32)
    baseline_count = 0
    for _, method_key in METHODS:
        if method_key == "ours":
            continue
        prediction = case.predictions[method_key][z_index]
        baseline_error = np.logical_and(
            prediction != gt_slice,
            np.logical_or(prediction > 0, gt_slice > 0),
        )
        baseline_only_error += np.logical_and(baseline_error, ~ours_error)
        baseline_count += 1

    advantage = baseline_only_error / max(baseline_count, 1)
    tumor_neighborhood = ndimage.binary_dilation(gt_slice > 0, iterations=8)
    focus_score = ndimage.gaussian_filter(advantage * tumor_neighborhood, sigma=2.5)

    half = side // 2
    interior = np.zeros_like(focus_score, dtype=bool)
    interior[half : focus_score.shape[0] - half, half : focus_score.shape[1] - half] = True
    interior_score = np.where(interior, focus_score, -1.0)
    if float(interior_score.max()) > 0:
        center_y, center_x = np.unravel_index(int(np.argmax(interior_score)), interior_score.shape)
    elif float(focus_score.max()) > 0:
        center_y, center_x = np.unravel_index(int(np.argmax(focus_score)), focus_score.shape)
    else:
        tumor_core = np.logical_or(gt_slice == 1, gt_slice == 3)
        y0, y1, x0, x1 = square_bbox(tumor_core, padding=18, minimum_side=side)
        center_y = int(round(0.5 * (y0 + y1)))
        center_x = int(round(0.5 * (x0 + x1)))

    y0 = min(max(int(center_y) - half, 0), gt_slice.shape[0] - side)
    x0 = min(max(int(center_x) - half, 0), gt_slice.shape[1] - side)
    return y0, y0 + side, x0, x0 + side


def label_overlay(labels: np.ndarray, alpha: float = 0.80) -> np.ndarray:
    overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
    for label, color in LABEL_COLORS.items():
        mask = labels == label
        overlay[mask, :3] = to_rgb(color)
        overlay[mask, 3] = alpha
    return overlay


def _box_center(box: DisagreementBox) -> tuple[float, float]:
    return 0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)


def _box_area(box: DisagreementBox) -> int:
    return max(box.x1 - box.x0, 0) * max(box.y1 - box.y0, 0)


def _intersection_over_union(left: DisagreementBox, right: DisagreementBox) -> float:
    x0 = max(left.x0, right.x0)
    y0 = max(left.y0, right.y0)
    x1 = min(left.x1, right.x1)
    y1 = min(left.y1, right.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = _box_area(left) + _box_area(right) - intersection
    return float(intersection / max(union, 1))


def _boxes_intersect(left: DisagreementBox, right: DisagreementBox, *, gap: int = 0) -> bool:
    """True when axis-aligned boxes overlap or come within ``gap`` pixels."""
    return not (
        left.x1 + gap <= right.x0
        or right.x1 + gap <= left.x0
        or left.y1 + gap <= right.y0
        or right.y1 + gap <= left.y0
    )


def _center_distance(left: DisagreementBox, right: DisagreementBox) -> float:
    left_x, left_y = _box_center(left)
    right_x, right_y = _box_center(right)
    return float(np.hypot(left_x - right_x, left_y - right_y))


def _annotation_role_for_counts(
    *,
    red_error_pixels: int,
    red_false_positive_pixels: int,
    enhancing_error_pixels: int,
    false_negative_pixels: int,
    false_positive_pixels: int,
    class_mismatch_pixels: int,
) -> str:
    if red_false_positive_pixels > 0 or (
        red_error_pixels > 0 and red_error_pixels >= max(enhancing_error_pixels, false_negative_pixels // 2)
    ):
        return "necrotic_core_error"
    if enhancing_error_pixels > 0 and enhancing_error_pixels >= max(red_error_pixels, false_negative_pixels // 2):
        return "enhancing_tumor_error"
    if false_negative_pixels >= false_positive_pixels and false_negative_pixels > 0:
        return "missed_lesion"
    if false_positive_pixels > 0:
        return "false_positive"
    if class_mismatch_pixels > 0:
        return "class_mismatch"
    return "major_label_disagreement"


def _component_selection_score(
    *,
    component_pixels: int,
    red_error_pixels: int,
    red_false_positive_pixels: int,
    red_false_negative_pixels: int,
    enhancing_error_pixels: int,
    false_negative_pixels: int,
    false_positive_pixels: int,
    class_mismatch_pixels: int,
) -> float:
    """Rank clinically salient errors ahead of pure area ranking.

    Predicted necrotic false positives are weighted highest so a compact red
    core error is not crowded out by larger edema-boundary disagreements.
    """
    return float(
        component_pixels
        + 4.0 * red_false_positive_pixels
        + 2.6 * red_false_negative_pixels
        + 2.2 * max(red_error_pixels - red_false_positive_pixels - red_false_negative_pixels, 0)
        + 2.0 * enhancing_error_pixels
        + 1.4 * false_negative_pixels
        + 1.1 * false_positive_pixels
        + 1.2 * class_mismatch_pixels
    )


def _select_diverse_boxes(
    candidates: list[DisagreementBox],
    *,
    max_boxes: int,
    min_center_distance: float = 16.0,
    max_iou: float = 0.05,
    separation_gap: int = 2,
) -> list[DisagreementBox]:
    """Greedy pick by score while forbidding crossed/nearby markers."""
    if max_boxes <= 0 or not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda box: (
            -box.selection_score,
            -box.red_false_positive_pixels,
            -box.red_error_pixels,
            -box.enhancing_error_pixels,
            -box.component_pixels,
            box.y0,
            box.x0,
        ),
    )
    selected: list[DisagreementBox] = []
    for box in ordered:
        if any(
            _boxes_intersect(box, kept, gap=separation_gap)
            or _intersection_over_union(box, kept) >= max_iou
            or _center_distance(box, kept) < min_center_distance
            for kept in selected
        ):
            continue
        selected.append(box)
        if len(selected) >= max_boxes:
            break
    # Do not back-fill empty slots with overlapping boxes: fewer clean markers
    # read better than a dense, crossed annotation set.
    return selected


def _best_red_priority_box(candidates: list[DisagreementBox]) -> DisagreementBox | None:
    red_candidates = [box for box in candidates if box.red_error_pixels > 0]
    if not red_candidates:
        return None
    return max(
        red_candidates,
        key=lambda box: (
            box.red_false_positive_pixels,
            box.red_error_pixels,
            box.selection_score,
            box.component_pixels,
        ),
    )


def disagreement_boxes(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    minimum_component: int = 8,
    max_boxes: int = DEFAULT_MAX_DIFFERENCE_BOXES,
    padding: int = 2,
    include_largest_red_component: bool = False,
) -> list[DisagreementBox]:
    """Return edge-safe boxes around the most informative label errors."""
    disagreement = np.logical_and(reference != prediction, np.logical_or(reference > 0, prediction > 0))
    components, count = ndimage.label(disagreement, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return []

    sizes = ndimage.sum(disagreement, components, index=np.arange(1, count + 1))
    height, width = reference.shape
    candidates: list[DisagreementBox] = []
    for component_index in range(count):
        size = float(sizes[component_index])
        if size < minimum_component:
            continue
        component = components == int(component_index) + 1
        coordinates = np.argwhere(component)
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0) + 1
        x0 = max(int(x_min) - padding, 0)
        y0 = max(int(y_min) - padding, 0)
        x1 = min(int(x_max) + padding, width)
        y1 = min(int(y_max) + padding, height)

        false_negative = np.logical_and(component, np.logical_and(reference > 0, prediction == 0))
        false_positive = np.logical_and(component, np.logical_and(reference == 0, prediction > 0))
        class_mismatch = np.logical_and(
            component,
            np.logical_and(
                np.logical_and(reference > 0, prediction > 0),
                reference != prediction,
            ),
        )
        red_false_positive = np.logical_and(component, np.logical_and(prediction == 1, reference != 1))
        red_false_negative = np.logical_and(component, np.logical_and(reference == 1, prediction != 1))
        red_error = np.logical_or(red_false_positive, red_false_negative)
        enhancing_error = np.logical_and(
            component,
            np.logical_or(
                np.logical_and(reference == 3, prediction != 3),
                np.logical_and(prediction == 3, reference != 3),
            ),
        )
        red_error_pixels = int(red_error.sum())
        red_false_positive_pixels = int(red_false_positive.sum())
        red_false_negative_pixels = int(red_false_negative.sum())
        enhancing_error_pixels = int(enhancing_error.sum())
        false_negative_pixels = int(false_negative.sum())
        false_positive_pixels = int(false_positive.sum())
        class_mismatch_pixels = int(class_mismatch.sum())
        component_pixels = int(component.sum())
        candidates.append(
            DisagreementBox(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                component_pixels=component_pixels,
                false_negative_pixels=false_negative_pixels,
                false_positive_pixels=false_positive_pixels,
                class_mismatch_pixels=class_mismatch_pixels,
                red_error_pixels=red_error_pixels,
                red_false_positive_pixels=red_false_positive_pixels,
                red_false_negative_pixels=red_false_negative_pixels,
                enhancing_error_pixels=enhancing_error_pixels,
                selection_score=_component_selection_score(
                    component_pixels=component_pixels,
                    red_error_pixels=red_error_pixels,
                    red_false_positive_pixels=red_false_positive_pixels,
                    red_false_negative_pixels=red_false_negative_pixels,
                    enhancing_error_pixels=enhancing_error_pixels,
                    false_negative_pixels=false_negative_pixels,
                    false_positive_pixels=false_positive_pixels,
                    class_mismatch_pixels=class_mismatch_pixels,
                ),
                annotation_role=_annotation_role_for_counts(
                    red_error_pixels=red_error_pixels,
                    red_false_positive_pixels=red_false_positive_pixels,
                    enhancing_error_pixels=enhancing_error_pixels,
                    false_negative_pixels=false_negative_pixels,
                    false_positive_pixels=false_positive_pixels,
                    class_mismatch_pixels=class_mismatch_pixels,
                ),
            )
        )

    if not candidates:
        return []

    selected = _select_diverse_boxes(candidates, max_boxes=max_boxes)
    if include_largest_red_component:
        best_red = _best_red_priority_box(candidates)
        if best_red is not None and best_red not in selected:
            conflicts = [
                box
                for box in selected
                if _boxes_intersect(best_red, box, gap=2)
                or _center_distance(best_red, box) < 16.0
            ]
            if not conflicts and len(selected) < max_boxes:
                selected.append(best_red)
            elif conflicts:
                # Replace the weakest conflicting marker rather than stacking
                # another crossed annotation on the same focus.
                victim = min(
                    conflicts,
                    key=lambda box: (
                        box.red_false_positive_pixels,
                        box.red_error_pixels,
                        box.selection_score,
                        box.component_pixels,
                    ),
                )
                if (
                    best_red.red_false_positive_pixels > victim.red_false_positive_pixels
                    or best_red.selection_score >= victim.selection_score
                ):
                    selected[selected.index(victim)] = best_red
                    selected = _select_diverse_boxes(selected, max_boxes=max_boxes)
            elif selected:
                replaceable = [box for box in selected if box.red_false_positive_pixels == 0] or list(selected)
                victim = min(
                    replaceable,
                    key=lambda box: (
                        box.red_false_positive_pixels,
                        box.red_error_pixels,
                        box.selection_score,
                        box.component_pixels,
                    ),
                )
                if best_red.selection_score >= 0.85 * victim.selection_score:
                    selected[selected.index(victim)] = best_red
                    selected = _select_diverse_boxes(selected, max_boxes=max_boxes)
    return selected


def annotation_shape_for_box(
    box: DisagreementBox,
    image_shape: tuple[int, int],
    *,
    annotation_style: str,
) -> str:
    """Choose a reproducible marker shape from the component bounding geometry."""
    if annotation_style == "boxes":
        return "box"
    if annotation_style not in {"mixed", "overlap_aware"}:
        raise ValueError(f"Unsupported annotation style: {annotation_style}")

    height, width = image_shape
    box_width = box.x1 - box.x0
    box_height = box.y1 - box.y0
    aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1)
    touches_edge = box.x0 == 0 or box.y0 == 0 or box.x1 == width or box.y1 == height
    if not touches_edge and max(box_width, box_height) <= 24 and aspect_ratio <= 1.35:
        return "circle"
    return "box"


def boxes_overlap(left: DisagreementBox, right: DisagreementBox) -> bool:
    return min(left.x1, right.x1) > max(left.x0, right.x0) and min(left.y1, right.y1) > max(left.y0, right.y0)


def annotation_shapes_for_boxes(
    boxes: list[DisagreementBox],
    image_shape: tuple[int, int],
    *,
    annotation_style: str,
) -> list[str]:
    """Choose marker shapes without introducing crossed circle/box pairs."""
    shapes = [annotation_shape_for_box(box, image_shape, annotation_style=annotation_style) for box in boxes]
    if annotation_style == "boxes":
        return shapes

    # Circles drawn from the bounding box diagonal can visually cross nearby
    # markers even when axis-aligned boxes do not. Prefer at most one circle,
    # and only when it stays clear of every other marker.
    circle_budget = 1
    for index, (box, shape) in enumerate(zip(boxes, shapes)):
        if shape != "circle":
            continue
        if circle_budget <= 0 or any(
            _boxes_intersect(box, other, gap=3) or _center_distance(box, other) < 18.0
            for other_index, other in enumerate(boxes)
            if other_index != index
        ):
            shapes[index] = "box"
        else:
            circle_budget -= 1

    if annotation_style != "overlap_aware":
        return shapes

    # Overlap-aware mode no longer forces secondary circles on intersecting
    # boxes; those pairs are filtered upstream. Keep the remaining geometry.
    return shapes


def add_difference_annotation(
    axis: plt.Axes,
    box: DisagreementBox,
    image_shape: tuple[int, int],
    *,
    shape: str,
) -> None:
    if shape == "circle":
        center_x = 0.5 * (box.x0 + box.x1) - 0.5
        center_y = 0.5 * (box.y0 + box.y1) - 0.5
        patch = Circle((center_x, center_y), radius=0.5 * max(box.x1 - box.x0, box.y1 - box.y0))
    else:
        patch = Rectangle(
            (box.x0 - 0.5, box.y0 - 0.5),
            box.x1 - box.x0,
            box.y1 - box.y0,
        )
    patch.set(fill=False, edgecolor=DIFFERENCE_COLOR, linewidth=0.9, zorder=5)
    axis.add_patch(patch)


def style_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def render_cell(
    axis: plt.Axes,
    flair_slice: np.ndarray,
    labels: np.ndarray | None,
    *,
    bbox: tuple[int, int, int, int] | None,
    gt_labels: np.ndarray | None,
    add_difference_boxes: bool,
    max_difference_boxes: Optional[int] = DEFAULT_MAX_DIFFERENCE_BOXES,
    annotation_style: str = "boxes",
    include_largest_red_component: bool = False,
) -> None:
    if bbox is None:
        padding = 22
        flair_view = np.pad(
            flair_slice,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=PANEL_GRAY,
        )
        label_view = (
            np.pad(labels, ((padding, padding), (padding, padding)), mode="constant")
            if labels is not None
            else None
        )
        gt_view = (
            np.pad(gt_labels, ((padding, padding), (padding, padding)), mode="constant")
            if gt_labels is not None
            else None
        )
    else:
        y0, y1, x0, x1 = bbox
        flair_view = flair_slice[y0:y1, x0:x1]
        label_view = labels[y0:y1, x0:x1] if labels is not None else None
        gt_view = gt_labels[y0:y1, x0:x1] if gt_labels is not None else None

    axis.set_facecolor(str(PANEL_GRAY))
    axis.imshow(flair_view, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if label_view is not None:
        axis.imshow(label_overlay(label_view), interpolation="nearest")
        if add_difference_boxes and gt_view is not None:
            boxes = disagreement_boxes(
                gt_view,
                label_view,
                max_boxes=max_difference_boxes if max_difference_boxes is not None else 10**9,
                include_largest_red_component=include_largest_red_component,
            )
            shapes = annotation_shapes_for_boxes(boxes, gt_view.shape, annotation_style=annotation_style)
            for box, shape in zip(boxes, shapes):
                add_difference_annotation(
                    axis,
                    box,
                    gt_view.shape,
                    shape=shape,
                )
    style_image_axis(axis)


def add_top_legend(fig: plt.Figure, *, annotation_style: str) -> None:
    difference_handle: Patch | tuple[Line2D, Line2D]
    if annotation_style in {"mixed", "overlap_aware"}:
        difference_handle = (
            Line2D([], [], marker="o", markersize=6, markerfacecolor="none", markeredgecolor=DIFFERENCE_COLOR),
            Line2D([], [], marker="s", markersize=6, markerfacecolor="none", markeredgecolor=DIFFERENCE_COLOR),
        )
    else:
        difference_handle = Patch(facecolor="none", edgecolor=DIFFERENCE_COLOR, linewidth=1.2)
    handles = [
        Patch(facecolor=LABEL_COLORS[2], edgecolor="#666666", linewidth=0.45, label="Edema"),
        Patch(facecolor=LABEL_COLORS[3], edgecolor="#666666", linewidth=0.45, label="Enhancing tumor"),
        Patch(
            facecolor=LABEL_COLORS[1],
            edgecolor="#666666",
            linewidth=0.45,
            label="Necrotic & non-enhancing tumor",
        ),
        difference_handle,
    ]
    fig.legend(
        handles=handles,
        labels=["Edema", "Enhancing tumor", "Necrotic & non-enhancing tumor", "Differences from GT"],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.986),
        ncol=4,
        frameon=False,
        fontsize=6.3,
        handlelength=2.0,
        handleheight=0.9,
        columnspacing=1.15,
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0.8)},
    )


def should_draw_difference_boxes(method_key: str | None) -> bool:
    """Only predictions are annotated; FLAIR and GT remain unmarked references."""
    return method_key not in {None, "gt"}


def build_figure(
    cases: list[CaseData],
    selected_slices: dict[str, int],
    *,
    annotation_style: str = "boxes",
    max_difference_boxes: Optional[int] = DEFAULT_MAX_DIFFERENCE_BOXES,
    include_largest_red_component: bool = False,
) -> plt.Figure:
    row_count = 3 * len(cases) - 1
    height_ratios: list[float] = []
    for case_index in range(len(cases)):
        height_ratios.extend([1.0, 1.0])
        if case_index < len(cases) - 1:
            height_ratios.append(0.12)

    # Keep the image rows nearly square at double-column width. A taller canvas
    # makes imshow center the square images inside tall axes, which looks like
    # excessive vertical whitespace even when GridSpec hspace is small.
    figure = plt.figure(figsize=(7.2, 1.525 * len(cases)), facecolor="white")
    grid = figure.add_gridspec(
        row_count,
        len(PLATE_COLUMNS),
        left=0.063,
        right=0.995,
        bottom=0.075,
        top=0.840,
        hspace=0.015,
        wspace=0.015,
        height_ratios=height_ratios,
    )
    panel_letters = ["a", "b"]

    for case_index, case in enumerate(cases):
        z_index = selected_slices[case.case_id]
        flair_display = normalize_flair(case.flair)[z_index]
        gt_slice = case.gt[z_index]
        bbox = advantage_focus_bbox(case, z_index)
        full_row = case_index * 3
        zoom_row = full_row + 1
        first_full: plt.Axes | None = None
        first_zoom: plt.Axes | None = None

        for column, (method_label, method_key) in enumerate(PLATE_COLUMNS):
            full_axis = figure.add_subplot(grid[full_row, column])
            zoom_axis = figure.add_subplot(grid[zoom_row, column])
            if column == 0:
                first_full = full_axis
                first_zoom = zoom_axis
            if method_key is None:
                labels = None
            elif method_key == "gt":
                labels = gt_slice
            else:
                labels = case.predictions[method_key][z_index]

            is_ours = method_key == "ours"
            render_cell(
                full_axis,
                flair_display,
                labels,
                bbox=None,
                gt_labels=gt_slice,
                add_difference_boxes=False,
                max_difference_boxes=max_difference_boxes,
                annotation_style=annotation_style,
                include_largest_red_component=include_largest_red_component,
            )
            render_cell(
                zoom_axis,
                flair_display,
                labels,
                bbox=bbox,
                gt_labels=gt_slice,
                add_difference_boxes=should_draw_difference_boxes(method_key),
                max_difference_boxes=max_difference_boxes,
                annotation_style=annotation_style,
                include_largest_red_component=include_largest_red_component,
            )

            if case_index == 0:
                full_axis.set_title(
                    method_label,
                    fontsize=(
                        4.25
                        if method_key
                        in {"attention_unet", "nestedformer", "segformer3d", "segmamba_v2"}
                        else 4.9
                    ),
                    fontweight="bold",
                    color="#151515",
                    pad=2.5,
                )

            if method_key not in {None, "gt"} and labels is not None:
                score = 100.0 * float(np.nanmean(slice_region_dice(gt_slice, labels)))
                zoom_axis.text(
                    0.5,
                    -0.075,
                    f"{score:.1f}",
                    transform=zoom_axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=4.9,
                    fontweight="bold" if is_ours else "normal",
                    color="#333333",
                    clip_on=False,
                )

        if first_full is None or first_zoom is None:
            raise RuntimeError("Image plate did not create its first column")
        case_number = case.case_id.rsplit("_", 1)[-1]
        original_z = z_index + 13
        first_full.text(
            -0.86,
            1.04,
            panel_letters[case_index],
            transform=first_full.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        first_full.text(
            -0.38,
            0.5,
            f"Case {case_number}\nz = {original_z}",
            transform=first_full.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.2,
            fontweight="bold",
        )
        first_zoom.text(
            -0.17,
            0.5,
            "Zoom",
            transform=first_zoom.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.0,
            color="#444444",
        )
        if case_index == len(cases) - 1:
            first_zoom.text(
                0.5,
                -0.075,
                "Slice mean Dice (%)",
                transform=first_zoom.transAxes,
                ha="center",
                va="top",
                fontsize=5.6,
                color="#444444",
                clip_on=False,
            )

    add_top_legend(figure, annotation_style=annotation_style)
    return figure


def write_source_data(
    output_root: Path,
    cases: list[CaseData],
    selected_slices: dict[str, int],
    rankings: dict[str, list[SliceCandidate]],
    *,
    annotation_style: str,
    max_difference_boxes: Optional[int] = DEFAULT_MAX_DIFFERENCE_BOXES,
    include_largest_red_component: bool = False,
) -> None:
    with (output_root / "case_metrics_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case_id", "method", "wt_dice", "tc_dice", "et_dice", "mean_dice"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for method_label, method_key in METHODS:
                writer.writerow(
                    {
                        "case_id": case.case_id,
                        "method": method_label,
                        **case.volume_metrics[method_key],
                    }
                )

    ranking_fields = [
        "case_id",
        "crop_z_zero_based",
        "original_z_zero_based",
        "ours_slice_mean_dice",
        "best_baseline_slice_mean_dice",
        "mean_baseline_slice_mean_dice",
        "margin_vs_best",
        "margin_vs_mean",
        "wt_voxels",
        "tc_voxels",
        "et_voxels",
        "selection_score",
        "selected",
    ]
    with (output_root / "slice_ranking_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_fields)
        writer.writeheader()
        for case_id, candidates in rankings.items():
            for candidate in candidates:
                writer.writerow(
                    {
                        "case_id": case_id,
                        "crop_z_zero_based": candidate.z_index,
                        "original_z_zero_based": candidate.z_index + 13,
                        "ours_slice_mean_dice": candidate.ours_mean,
                        "best_baseline_slice_mean_dice": candidate.best_baseline_mean,
                        "mean_baseline_slice_mean_dice": candidate.mean_baseline_mean,
                        "margin_vs_best": candidate.margin_best,
                        "margin_vs_mean": candidate.margin_mean,
                        "wt_voxels": candidate.wt_voxels,
                        "tc_voxels": candidate.tc_voxels,
                        "et_voxels": candidate.et_voxels,
                        "selection_score": candidate.score,
                        "selected": candidate.z_index == selected_slices[case_id],
                    }
                )

    selected_rows = []
    for case in cases:
        z_index = selected_slices[case.case_id]
        gt_slice = case.gt[z_index]
        for method_label, method_key in METHODS:
            values = slice_region_dice(gt_slice, case.predictions[method_key][z_index])
            selected_rows.append(
                {
                    "case_id": case.case_id,
                    "crop_z_zero_based": z_index,
                    "original_z_zero_based": z_index + 13,
                    "method": method_label,
                    "wt_dice": values[0],
                    "tc_dice": values[1],
                    "et_dice": values[2],
                    "mean_dice": float(np.nanmean(values)),
                }
            )
    with (output_root / "selected_slice_metrics_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)

    annotation_rows = []
    for case in cases:
        z_index = selected_slices[case.case_id]
        y0, y1, x0, x1 = advantage_focus_bbox(case, z_index)
        gt_view = case.gt[z_index, y0:y1, x0:x1]
        for method_label, method_key in METHODS:
            prediction_view = case.predictions[method_key][z_index, y0:y1, x0:x1]
            boxes = disagreement_boxes(
                gt_view,
                prediction_view,
                max_boxes=max_difference_boxes or 10**9,
                include_largest_red_component=include_largest_red_component,
            )
            shapes = annotation_shapes_for_boxes(boxes, gt_view.shape, annotation_style=annotation_style)
            for rank, (box, shape) in enumerate(zip(boxes, shapes), start=1):
                annotation_rows.append(
                    {
                        "case_id": case.case_id,
                        "crop_z_zero_based": z_index,
                        "original_z_zero_based": z_index + 13,
                        "method": method_label,
                        "selection_rule": "salience_weighted_diverse_components",
                        "annotation_rank": rank,
                        "annotation_shape": shape,
                        "annotation_role": box.annotation_role,
                        "disagreement_pixels": box.component_pixels,
                        "false_negative_pixels": box.false_negative_pixels,
                        "false_positive_pixels": box.false_positive_pixels,
                        "class_mismatch_pixels": box.class_mismatch_pixels,
                        "red_error_pixels": box.red_error_pixels,
                        "red_false_positive_pixels": box.red_false_positive_pixels,
                        "red_false_negative_pixels": box.red_false_negative_pixels,
                        "enhancing_error_pixels": box.enhancing_error_pixels,
                        "selection_score": f"{box.selection_score:.3f}",
                        "roi_x0": box.x0,
                        "roi_y0": box.y0,
                        "roi_x1": box.x1,
                        "roi_y1": box.y1,
                    }
                )
    annotation_fields = [
        "case_id",
        "crop_z_zero_based",
        "original_z_zero_based",
        "method",
        "selection_rule",
        "annotation_rank",
        "annotation_shape",
        "annotation_role",
        "disagreement_pixels",
        "false_negative_pixels",
        "false_positive_pixels",
        "class_mismatch_pixels",
        "red_error_pixels",
        "red_false_positive_pixels",
        "red_false_negative_pixels",
        "enhancing_error_pixels",
        "selection_score",
        "roi_x0",
        "roi_y0",
        "roi_x1",
        "roi_y1",
    ]
    with (output_root / "comparison_annotation_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=annotation_fields)
        writer.writeheader()
        writer.writerows(annotation_rows)


def parse_slice_overrides(values: list[str] | None) -> dict[str, int]:
    overrides = {}
    for value in values or []:
        case_token, z_token = value.split("=", 1)
        case_id = (
            case_token
            if case_token.startswith("BraTS20_Training_")
            else f"BraTS20_Training_{int(case_token):03d}"
        )
        overrides[case_id] = int(z_token)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case", action="append", dest="cases", help="Case ID; may be repeated")
    parser.add_argument(
        "--slice",
        action="append",
        dest="slices",
        help="Override selected cropped axial index, e.g. --slice 037=52",
    )
    parser.add_argument(
        "--annotation-style",
        choices=("boxes", "mixed", "overlap_aware"),
        default="boxes",
        help="Use only boxes (cleanest), sparse mixed circles, or legacy overlap-aware markers.",
    )
    parser.add_argument(
        "--max-difference-boxes",
        type=int,
        default=DEFAULT_MAX_DIFFERENCE_BOXES,
        help="Maximum number of difference markers per prediction zoom panel.",
    )
    parser.add_argument(
        "--all-differences",
        action="store_true",
        help="Annotate every disagreement component of at least eight pixels instead of the ranked subset.",
    )
    parser.add_argument(
        "--include-largest-red-difference",
        action="store_true",
        default=True,
        help="Guarantee the highest-scoring necrotic/non-enhancing error is annotated when present.",
    )
    parser.add_argument(
        "--no-include-largest-red-difference",
        action="store_false",
        dest="include_largest_red_difference",
        help="Disable the necrotic-core annotation guarantee.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="brats2020_sota_qualitative_comparison",
        help="Output filename stem inside <root>/final.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_ids = args.cases or DEFAULT_CASES
    cases = [load_case(args.root, case_id) for case_id in case_ids]
    rankings = {case.case_id: rank_slices(case) for case in cases}
    overrides = parse_slice_overrides(args.slices)
    selected_slices = {}
    for case in cases:
        if not rankings[case.case_id]:
            raise RuntimeError(f"No eligible tumor-bearing slices for {case.case_id}")
        selected_slices[case.case_id] = overrides.get(
            case.case_id,
            DEFAULT_SLICES.get(case.case_id, rankings[case.case_id][0].z_index),
        )
        print(f"\n{case.case_id} top candidate slices:")
        for candidate in rankings[case.case_id][:8]:
            print(
                f"  crop_z={candidate.z_index:03d} original_z={candidate.z_index + 13:03d} "
                f"ours={candidate.ours_mean * 100:5.1f} best={candidate.best_baseline_mean * 100:5.1f} "
                f"margin={candidate.margin_best * 100:+5.1f} score={candidate.score:+.4f}"
            )

    output_root = args.root / "final"
    output_root.mkdir(parents=True, exist_ok=True)
    max_difference_boxes = None if args.all_differences else max(1, int(args.max_difference_boxes))
    figure = build_figure(
        cases,
        selected_slices,
        annotation_style=args.annotation_style,
        max_difference_boxes=max_difference_boxes,
        include_largest_red_component=args.include_largest_red_difference,
    )
    stem = output_root / args.output_stem
    figure.savefig(f"{stem}.png", dpi=300, facecolor="white")
    figure.savefig(f"{stem}.pdf", facecolor="white")
    figure.savefig(f"{stem}.svg", facecolor="white")
    figure.savefig(
        f"{stem}.tiff",
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    write_source_data(
        output_root,
        cases,
        selected_slices,
        rankings,
        annotation_style=args.annotation_style,
        max_difference_boxes=max_difference_boxes,
        include_largest_red_component=args.include_largest_red_difference,
    )

    selection_payload = {
        case_id: {
            "crop_z_zero_based": z_index,
            "original_z_zero_based": z_index + 13,
            "zoom_bbox_y0_y1_x0_x1": list(
                advantage_focus_bbox(
                    next(case for case in cases if case.case_id == case_id),
                    z_index,
                )
            ),
        }
        for case_id, z_index in selected_slices.items()
    }
    (output_root / "selected_cases_and_slices.json").write_text(
        json.dumps(selection_payload, indent=2), encoding="utf-8"
    )
    (output_root / "qa_notes.md").write_text(
        "\n".join(
            [
                "# BraTS 2020 qualitative comparison QA",
                "",
                "- Evaluation: same fold and training-validation 128^3 center crop for every method.",
                "- Score labels: selected-slice mean of WT, TC, and ET Dice; values are percentages.",
                "- Display: identical FLAIR normalization within each case and identical crop across methods.",
                "- Overlays: label colors are fixed; no local contrast or selective mask editing was applied.",
                "- Zoom: fixed 64x64 ROI centered on baseline errors corrected by Ours; one identical "
                "ROI is used across methods.",
                f"- Annotation style: {args.annotation_style}; maximum component count: "
                f"{'all' if max_difference_boxes is None else max_difference_boxes}.",
                "- Annotations: each prediction zoom panel marks the top-ranked 8-connected components of exact "
                "label disagreement with GT after excluding components smaller than 8 pixels. Ranking is "
                "salience-weighted (necrotic/non-enhancing and enhancing errors up-weighted) with spatial "
                "de-duplication so markers cover distinct error foci rather than only the largest area blobs. "
                "Boxes use two pixels of padding and are clipped, not discarded, at ROI boundaries. "
                "FLAIR and GT are unmarked.",
                "- Necrotic guarantee: when enabled, the highest-scoring red-label error is retained even if a "
                "larger edema-boundary component would otherwise crowd it out.",
                "- Mixed or overlap-aware styles: compact, non-edge components with a maximum bounding-box side "
                "of 24 pixels and an aspect ratio no greater than 1.35 use circles. In overlap-aware mode, a "
                "second box also becomes a circle when it overlaps the first, has a maximum side of 32 pixels, "
                "and has no more than 75% of the first box area.",
                "- Candidate selection: quantitative ranking required non-trivial WT, TC, and ET; "
                "final slices were chosen from the top-ranked set after visual QA.",
                "- Source arrays, case metrics, slice metrics, annotation counts, and ranking scores "
                "accompany the exports.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved figure bundle to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
