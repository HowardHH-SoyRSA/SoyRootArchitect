from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import soyrootbio.visualize as visualize
from soyrootbio.types import RootPath
from soyrootbio.visualize import (
    _ANGLE_ARROW_MUTATION_SCALE,
    _ANGLE_VECTOR_SCALE_FRACTION,
    _ORDER1_ANGLE_MIN_LENGTH_MESH_UNITS,
    _ORDER2_ANGLE_MIN_LENGTH_MESH_UNITS,
    _adaptive_angle_label_font_size,
    _angle_label_route,
    _angle_vector_legend_spec,
    _annotate_angle_tip,
    _display_root_name,
    _draw_angle_vectors,
    _format_angle_tip_label,
    _ordered_angle_label_layout,
    _show_angle_for_lateral,
)


class _RecordingAxis:
    def __init__(self) -> None:
        self.annotations: list[dict] = []
        self.lines: list[dict] = []
        self.texts: list[dict] = []

    def annotate(self, *args, **kwargs) -> None:
        self.annotations.append({"args": args, "kwargs": kwargs})

    def plot(self, *args, **kwargs) -> None:
        self.lines.append({"args": args, "kwargs": kwargs})

    def text(self, *args, **kwargs) -> None:
        self.texts.append({"args": args, "kwargs": kwargs})


@pytest.fixture
def vector_fixture() -> tuple[RootPath, np.ndarray, dict[str, float]]:
    lateral = RootPath(
        root_id="order1_001",
        points=np.array(
            [
                [0.0, 0.0, 0.6],
                [0.2, 0.0, 0.5],
                [0.4, 0.0, 0.4],
            ]
        ),
    )
    primary = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    traits = {
        "tip_vector_dx": 1.0,
        "tip_vector_dy": 0.0,
        "tip_vector_dz": 0.0,
        "tip_start_vector_dx": 1.0,
        "tip_start_vector_dy": 0.0,
        "tip_start_vector_dz": 0.0,
        "base_vector_dx": 1.0,
        "base_vector_dy": 0.0,
        "base_vector_dz": -1.0,
        "primary_vector_dx": 0.0,
        "primary_vector_dy": 0.0,
        "primary_vector_dz": -1.0,
    }
    return lateral, primary, traits


@pytest.mark.parametrize("mode", ["tip_gravity", "tip_start_gravity"])
def test_gravity_angle_views_draw_only_the_measured_lateral_vector(
    mode: str,
    vector_fixture: tuple[RootPath, np.ndarray, dict[str, float]],
) -> None:
    lateral, primary, traits = vector_fixture
    axis = _RecordingAxis()

    _draw_angle_vectors(
        axis,
        lateral,
        primary,
        trait_row=traits,
        mode=mode,
        scale=0.1,
    )

    assert len(axis.annotations) == 1
    arrow = axis.annotations[0]["kwargs"]
    assert arrow["arrowprops"]["color"] == "#b00020"
    assert arrow["arrowprops"]["mutation_scale"] == pytest.approx(
        _ANGLE_ARROW_MUTATION_SCALE
    )
    tip_xz = lateral.points[-1, [0, 2]]
    np.testing.assert_allclose(np.asarray(arrow["xy"]) - tip_xz, [0.1, 0.0])


def test_primary_angle_view_uses_lateral_start_and_primary_vectors_at_insertion(
    vector_fixture: tuple[RootPath, np.ndarray, dict[str, float]],
) -> None:
    lateral, primary, traits = vector_fixture
    axis = _RecordingAxis()

    _draw_angle_vectors(
        axis,
        lateral,
        primary,
        trait_row=traits,
        mode="tip_primary",
        scale=0.1,
    )

    assert [item["kwargs"]["arrowprops"]["color"] for item in axis.annotations] == [
        "#b00020",
        "#14833b",
    ]
    assert all(
        item["kwargs"]["arrowprops"]["mutation_scale"]
        == pytest.approx(_ANGLE_ARROW_MUTATION_SCALE)
        for item in axis.annotations
    )
    insertion_xz = lateral.points[0, [0, 2]]
    red_arrow = axis.annotations[0]["kwargs"]
    green_arrow = axis.annotations[1]["kwargs"]
    np.testing.assert_allclose(red_arrow["xytext"], insertion_xz)
    np.testing.assert_allclose(green_arrow["xytext"], insertion_xz)
    np.testing.assert_allclose(
        np.asarray(red_arrow["xy"]) - insertion_xz,
        np.array([1.0, -1.0]) / np.sqrt(2.0) * 0.1,
    )
    np.testing.assert_allclose(
        np.asarray(green_arrow["xy"]) - insertion_xz,
        [0.0, -0.1],
    )


def test_angle_vector_legends_are_compact_and_match_visible_vectors() -> None:
    assert _ANGLE_VECTOR_SCALE_FRACTION == pytest.approx(0.022)
    assert _ANGLE_ARROW_MUTATION_SCALE == pytest.approx(4.5)
    assert _angle_vector_legend_spec("tip_gravity") == [("#b00020", "Tip")]
    assert _angle_vector_legend_spec("tip_start_gravity") == [("#b00020", "S–T")]
    assert _angle_vector_legend_spec("tip_primary") == [
        ("#b00020", "Start"),
        ("#14833b", "Primary"),
    ]


@pytest.mark.parametrize(
    ("order", "length", "expected"),
    [
        (1, 3.499, False),
        (1, 3.5, True),
        (1, 8.0, True),
        (2, 1.299, False),
        (2, 1.3, True),
        (2, 4.0, True),
        (3, 0.25, True),
    ],
)
def test_short_order1_and_order2_roots_hide_angles_below_mesh_unit_limits(
    order: int,
    length: float,
    expected: bool,
) -> None:
    lateral = RootPath(
        root_id=f"root-o{order}-001",
        points=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        order=order,
    )

    assert _ORDER1_ANGLE_MIN_LENGTH_MESH_UNITS == pytest.approx(3.5)
    assert _ORDER2_ANGLE_MIN_LENGTH_MESH_UNITS == pytest.approx(1.3)
    assert _show_angle_for_lateral(
        lateral,
        {"length": length, "length_unit": "mesh_unit"},
    ) is expected


def test_angle_visibility_uses_reported_mesh_length_not_normalized_path_length() -> None:
    lateral = RootPath(
        root_id="root-o1-001",
        points=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        order=1,
    )

    assert lateral.length < _ORDER1_ANGLE_MIN_LENGTH_MESH_UNITS
    assert _show_angle_for_lateral(
        lateral,
        {"length": 3.5, "length_unit": "mesh_unit"},
    )
    assert _show_angle_for_lateral(lateral, {})  # Preserve legacy fallback.


@pytest.mark.parametrize(
    "mode",
    ["tip_gravity", "tip_start_gravity", "tip_primary"],
)
def test_angle_front_views_skip_short_order1_and_order2_annotations_and_vectors(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.figure

    laterals = [
        RootPath(
            "root-o1-001",
            np.array([[-0.6, 0.0, 0.8], [-0.5, 0.0, 0.5]]),
            order=1,
        ),
        RootPath(
            "root-o1-002",
            np.array([[0.0, 0.0, 0.8], [0.2, 0.0, 0.4]]),
            order=1,
        ),
        RootPath(
            "root-o2-001",
            np.array([[0.4, 0.0, 0.7], [0.7, 0.0, 0.2]]),
            order=2,
        ),
        RootPath(
            "root-o2-002",
            np.array([[0.6, 0.0, 0.7], [0.9, 0.0, 0.1]]),
            order=2,
        ),
    ]
    points = np.vstack(
        [
            np.array([[-0.1, 0.0, 1.0], [-0.1, 0.0, 0.0]]),
            *(lateral.points for lateral in laterals),
        ]
    )
    primary_mask = np.zeros(len(points), dtype=bool)
    primary_mask[:2] = True
    lateral_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    traits = pd.DataFrame(
        [
            {"root_id": "root-o1-001", "length": 3.49, "angle": 10.0},
            {"root_id": "root-o1-002", "length": 3.5, "angle": 20.0},
            {"root_id": "root-o2-001", "length": 1.29, "angle": 30.0},
            {"root_id": "root-o2-002", "length": 1.3, "angle": 40.0},
        ]
    )
    vector_calls: list[str] = []
    annotation_calls: list[str] = []

    monkeypatch.setattr(
        visualize,
        "_draw_angle_vectors",
        lambda _axis, lateral, *_args, **_kwargs: vector_calls.append(
            lateral.root_id
        ),
    )
    monkeypatch.setattr(
        visualize,
        "_annotate_angle_tip",
        lambda _axis, lateral, *_args, **_kwargs: annotation_calls.append(
            lateral.root_id
        ),
    )
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda *_args, **_kwargs: None,
    )

    visualize._save_one_angle_front_view(
        Path(f"unused-{mode}.png"),
        points,
        primary_mask,
        lateral_labels,
        points[:2],
        laterals,
        traits,
        trait_column="angle",
        title="Test",
        mode=mode,
        gravity=np.array([0.0, 0.0, -1.0]),
        max_points=100,
    )

    expected = {"root-o1-002", "root-o2-002"}
    assert set(vector_calls) == expected
    assert set(annotation_calls) == expected
    assert "root-o1-001" not in vector_calls
    assert "root-o1-001" not in annotation_calls
    assert "root-o2-001" not in vector_calls
    assert "root-o2-001" not in annotation_calls


@pytest.mark.parametrize(
    ("root_id", "expected"),
    [
        ("root-o1-003", "o1-003"),
        ("o2-014", "o2-014"),
        ("order3_007", "o3-007"),
    ],
)
def test_display_root_name_uses_order_number_pattern(
    root_id: str,
    expected: str,
) -> None:
    assert _display_root_name(root_id) == expected


def test_tip_label_combines_compact_root_name_and_angle() -> None:
    assert _format_angle_tip_label("root-o2-004", 47.26) == "o2-004 47.3°"
    assert _format_angle_tip_label("root-o2-004", np.nan) == "o2-004 —"


def test_tip_label_uses_restored_side_column_indicatrix(
    vector_fixture: tuple[RootPath, np.ndarray, dict[str, float]],
) -> None:
    lateral, _, _ = vector_fixture
    lateral.root_id = "root-o1-003"
    tip = lateral.points[-1]
    axis = _RecordingAxis()

    _annotate_angle_tip(
        axis,
        lateral,
        tip,
        35.0,
        label_position=(1.2, 0.6),
        entry_x=0.8,
        outer_x=1.05,
        label_line_end_x=1.18,
        outward=1,
        font_size=4.2,
    )

    assert len(axis.lines) == 1
    line = axis.lines[0]
    np.testing.assert_allclose(line["args"][0], [tip[0], 0.8, 1.05, 1.18])
    np.testing.assert_allclose(
        line["args"][1],
        [tip[2], tip[2], 0.6, 0.6],
    )
    assert line["kwargs"]["color"] == "#4774bf"
    assert line["kwargs"]["solid_joinstyle"] == "round"

    assert len(axis.texts) == 1
    text = axis.texts[0]
    assert text["args"] == (1.2, 0.6, "o1-003 35.0°")
    assert text["kwargs"]["fontsize"] == pytest.approx(4.2)


def test_angle_label_route_uses_a_shared_outward_corridor() -> None:
    lower_tip = np.array([0.3, 0.0, 0.2])
    upper_tip = np.array([0.6, 0.0, 0.8])
    lower = _angle_label_route(
        lower_tip,
        entry_x=1.0,
        outer_x=1.3,
        label_line_end_x=1.38,
        label_position=(1.4, 0.1),
    )
    upper = _angle_label_route(
        upper_tip,
        entry_x=1.0,
        outer_x=1.3,
        label_line_end_x=1.38,
        label_position=(1.4, 0.9),
    )

    np.testing.assert_allclose(
        lower,
        [[0.3, 0.2], [1.0, 0.2], [1.3, 0.1], [1.38, 0.1]],
    )
    np.testing.assert_allclose(
        upper,
        [[0.6, 0.8], [1.0, 0.8], [1.3, 0.9], [1.38, 0.9]],
    )
    # Transfer segments preserve vertical order at both rails, so they cannot
    # cross. The horizontal entry and label segments occupy separate X bands.
    assert lower[1, 1] < upper[1, 1]
    assert lower[2, 1] < upper[2, 1]


def test_angle_label_layout_is_deterministic_and_preserves_tip_order() -> None:
    items = [
        (
            RootPath("root-o1-003", np.array([[0.0, 0.0, 0.0]])),
            np.array([0.0, 0.0, 0.8]),
            30.0,
        ),
        (
            RootPath("root-o1-001", np.array([[0.0, 0.0, 0.0]])),
            np.array([0.0, 0.0, 0.2]),
            10.0,
        ),
        (
            RootPath("root-o1-002", np.array([[0.0, 0.0, 0.0]])),
            np.array([0.0, 0.0, 0.5]),
            20.0,
        ),
    ]
    layout = _ordered_angle_label_layout(
        items,
        z_min=0.0,
        z_max=1.0,
        z_padding=0.05,
    )

    assert [item[0].root_id for item in layout] == [
        "root-o1-001",
        "root-o1-002",
        "root-o1-003",
    ]
    assert [item[3] for item in layout] == pytest.approx([0.05, 0.5, 0.95])


def test_angle_label_font_shrinks_to_preserve_vertical_spacing() -> None:
    assert _adaptive_angle_label_font_size(
        1,
        axes_height_points=600.0,
    ) == pytest.approx(5.4)
    dense_size = _adaptive_angle_label_font_size(
        160,
        axes_height_points=600.0,
        usable_height_fraction=0.9,
    )
    pitch = 0.9 * 600.0 / 159.0
    assert dense_size < 5.4
    assert dense_size <= 0.72 * pitch + 1e-12
    counts = [1, 20, 80, 160]
    sizes = [
        _adaptive_angle_label_font_size(
            count,
            axes_height_points=600.0,
            usable_height_fraction=0.9,
        )
        for count in counts
    ]
    assert sizes == sorted(sizes, reverse=True)
    assert _adaptive_angle_label_font_size(
        160,
        axes_height_points=400.0,
        usable_height_fraction=0.9,
    ) < dense_size
