from __future__ import annotations

import numpy as np

from scripts.plot_brats2020_sota_qualitative import (
    DisagreementBox,
    annotation_shape_for_box,
    annotation_shapes_for_boxes,
    disagreement_boxes,
    should_draw_difference_boxes,
)


def test_disagreement_boxes_keep_edge_touching_components() -> None:
    reference = np.zeros((16, 16), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference[0:4, 2:6] = 2

    boxes = disagreement_boxes(reference, prediction)

    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x0, box.y0, box.x1, box.y1) == (0, 0, 8, 6)
    assert box.component_pixels == 16
    assert box.false_negative_pixels == 16
    assert box.false_positive_pixels == 0
    assert box.class_mismatch_pixels == 0


def test_disagreement_boxes_select_two_largest_separated_components() -> None:
    reference = np.zeros((40, 40), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference[1:5, 1:5] = 2
    reference[1:4, 24:28] = 2
    reference[24:28, 1:5] = 2

    boxes = disagreement_boxes(reference, prediction, minimum_component=8, max_boxes=2, padding=0)

    assert [box.component_pixels for box in boxes] == [16, 16]
    centers = [(0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)) for box in boxes]
    distance = ((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2) ** 0.5
    assert distance >= 16.0


def test_disagreement_boxes_prefer_red_core_over_larger_edema_boundary() -> None:
    reference = np.zeros((48, 48), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    # Large edema false positive near the top.
    prediction[1:6, 2:18] = 2
    # Compact necrotic false positive in the center.
    prediction[22:26, 22:26] = 1
    # Medium enhancing false positive on the right.
    prediction[30:34, 38:44] = 3

    boxes = disagreement_boxes(
        reference,
        prediction,
        minimum_component=8,
        max_boxes=2,
        padding=0,
        include_largest_red_component=True,
    )

    assert len(boxes) == 2
    assert any(box.red_false_positive_pixels >= 16 for box in boxes)
    assert any(box.enhancing_error_pixels >= 20 or box.component_pixels >= 70 for box in boxes)


def test_disagreement_boxes_keep_predicted_red_fp_near_other_core_errors() -> None:
    reference = np.zeros((48, 48), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    # GT necrotic mislabeled as enhancing (left).
    reference[12:18, 8:14] = 1
    prediction[12:18, 8:14] = 3
    # Separated predicted necrotic false positive.
    prediction[12:18, 30:36] = 1
    # Distant large edema FP that should not erase the red FP marker.
    prediction[1:8, 1:20] = 2

    boxes = disagreement_boxes(
        reference,
        prediction,
        minimum_component=8,
        max_boxes=2,
        padding=0,
        include_largest_red_component=True,
    )

    assert any(box.red_false_positive_pixels >= 30 for box in boxes)


def test_disagreement_boxes_keep_spatially_diverse_foci() -> None:
    reference = np.zeros((64, 64), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference[4:10, 4:12] = 3
    reference[4:8, 40:48] = 3
    reference[40:48, 8:16] = 2
    # Leave all three GT foci missing so separated FN components exist.

    boxes = disagreement_boxes(reference, prediction, minimum_component=8, max_boxes=2, padding=0)

    assert len(boxes) == 2
    centers = [(0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)) for box in boxes]
    distance = (
        (centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2
    ) ** 0.5
    assert distance >= 16.0


def test_disagreement_boxes_reject_crossed_nearby_markers() -> None:
    reference = np.zeros((40, 40), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    # Two nearby disagreement blobs that would produce crossed markers if both kept.
    prediction[10:16, 10:16] = 1
    prediction[12:18, 18:24] = 3
    prediction[30:36, 30:36] = 2

    boxes = disagreement_boxes(
        reference,
        prediction,
        minimum_component=8,
        max_boxes=2,
        padding=0,
        include_largest_red_component=True,
    )

    assert len(boxes) == 2
    left, right = boxes
    assert not (
        left.x1 + 2 > right.x0
        and right.x1 + 2 > left.x0
        and left.y1 + 2 > right.y0
        and right.y1 + 2 > left.y0
    )


def test_disagreement_boxes_count_label_error_types() -> None:
    reference = np.zeros((40, 40), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference[1:4, 1:4] = 2
    prediction[1:4, 20:23] = 2
    reference[20:23, 1:4] = 2
    prediction[20:23, 1:4] = 1

    boxes = disagreement_boxes(reference, prediction, minimum_component=8, max_boxes=3, padding=0)

    # Clean layout keeps only non-crossing foci; error-type counters still cover FN/FP/mismatch.
    assert 2 <= len(boxes) <= 3
    assert sum(box.false_negative_pixels for box in boxes) >= 9
    assert sum(box.false_positive_pixels for box in boxes) >= 9
    assert sum(box.class_mismatch_pixels for box in boxes) >= 9


def test_only_prediction_columns_receive_difference_boxes() -> None:
    assert not should_draw_difference_boxes(None)
    assert not should_draw_difference_boxes("gt")
    assert should_draw_difference_boxes("unet")
    assert should_draw_difference_boxes("ours")


def test_mixed_annotations_use_circles_only_for_compact_non_edge_components() -> None:
    compact = DisagreementBox(20, 20, 38, 37, 60, 60, 0, 0)
    elongated = DisagreementBox(20, 20, 43, 36, 60, 60, 0, 0)
    edge_touching = DisagreementBox(41, 20, 64, 39, 60, 60, 0, 0)

    assert annotation_shape_for_box(compact, (64, 64), annotation_style="mixed") == "circle"
    assert annotation_shape_for_box(elongated, (64, 64), annotation_style="mixed") == "box"
    assert annotation_shape_for_box(edge_touching, (64, 64), annotation_style="mixed") == "box"
    assert annotation_shape_for_box(compact, (64, 64), annotation_style="boxes") == "box"


def test_overlap_aware_annotations_avoid_crossed_circle_pairs() -> None:
    primary = DisagreementBox(0, 16, 48, 40, 100, 100, 0, 0)
    nearby_compact = DisagreementBox(29, 0, 60, 20, 50, 50, 0, 0)
    distant_compact = DisagreementBox(2, 44, 18, 58, 40, 40, 0, 0)

    # Nearby markers stay as boxes so circles do not cross.
    assert annotation_shapes_for_boxes(
        [primary, nearby_compact], (64, 64), annotation_style="overlap_aware"
    ) == ["box", "box"]
    # A single compact, well-separated focus may remain a circle.
    shapes = annotation_shapes_for_boxes(
        [primary, distant_compact], (64, 64), annotation_style="overlap_aware"
    )
    assert shapes[0] == "box"
    assert shapes[1] in {"box", "circle"}
    assert shapes.count("circle") <= 1
