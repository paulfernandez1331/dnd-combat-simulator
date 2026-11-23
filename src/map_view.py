"""A labeled 5-foot grid. Plotly selection is the click."""

from __future__ import annotations

import plotly.graph_objects as go

EMPTY = "#f4f0e6"
REACHABLE = "#cfe3c4"
PARTY = "#7eafd4"
ENEMY = "#e09a8a"
AIM = "#f3e2a8"
TARGET_LINE = "#c8960a"
PREVIEW = "rgba(224, 168, 74, 0.4)"
PREVIEW_LINE = "#e0a84a"


def cell_label(creature) -> str:
    prefix = "P" if creature.side == "party" else "E"
    return f"{prefix} {creature.name}<br>{creature.hp}/{creature.max_hp}"


def board_figure(
    encounter,
    highlight: set[tuple[int, int]] | None = None,
    current_id: str | None = None,
    aim: set[tuple[int, int]] | None = None,
    targets: set[tuple[int, int]] | None = None,
    preview: set[tuple[int, int]] | None = None,
):
    """One square per cell. Text is the name and hit points. Axes are feet, north up.

    Green squares are movement. Gold empty squares are legal aim points. Gold
    outlines are legal targets. Amber rectangles are the area about to resolve.
    Occupied squares keep the party or enemy color so the label stays readable.
    """
    highlight = highlight or set()
    aim = aim or set()
    targets = targets or set()
    preview = preview or set()
    occupants: dict[tuple[int, int], object] = {}
    for creature in encounter.combatants:
        for square in creature.squares():
            occupants.setdefault(square, creature)
    colors = [
        [0.0, EMPTY],
        [0.19, EMPTY],
        [0.20, REACHABLE],
        [0.39, REACHABLE],
        [0.40, PARTY],
        [0.59, PARTY],
        [0.60, ENEMY],
        [0.79, ENEMY],
        [0.80, AIM],
        [1.0, AIM],
    ]
    z_rows = []
    text_rows = []
    custom = []
    current = None
    for y in range(encounter.height):
        z_row = []
        text_row = []
        custom_row = []
        for x in range(encounter.width):
            creature = occupants.get((x, y))
            if creature is None:
                if (x, y) in aim:
                    z_row.append(4)
                else:
                    z_row.append(1 if (x, y) in highlight else 0)
                text_row.append("")
            else:
                z_row.append(2 if creature.side == "party" else 3)
                if (x, y) == tuple(creature.position):
                    text_row.append(cell_label(creature))
                else:
                    text_row.append("")
                if creature.id == current_id:
                    current = creature
            custom_row.append([x, y])
        z_rows.append(z_row)
        text_rows.append(text_row)
        custom.append(custom_row)
    figure = go.Figure(
        data=go.Heatmap(
            x=[5 * column + 2.5 for column in range(encounter.width)],
            y=[5 * row + 2.5 for row in range(encounter.height)],
            z=z_rows,
            text=text_rows,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#1c1915"},
            customdata=custom,
            colorscale=colors,
            zmin=0,
            zmax=4,
            showscale=False,
            xgap=4,
            ygap=4,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    for square in preview:
        _cell_shape(figure, square, line=PREVIEW_LINE, width=2, fill=PREVIEW)
    for square in targets:
        _cell_shape(figure, square, line=TARGET_LINE, width=3, fill="rgba(0,0,0,0)")
    if current is not None:
        for square in sorted(current.squares()):
            _cell_shape(figure, square, line="#16130f", width=4, fill="rgba(0,0,0,0)")
    figure.update_xaxes(
        title="feet east",
        range=[-1, encounter.width * 5 + 1],
        dtick=10,
        showgrid=False,
        zeroline=False,
    )
    figure.update_yaxes(
        title="feet north",
        range=[-1, encounter.height * 5 + 1],
        dtick=10,
        scaleanchor="x",
        showgrid=False,
        zeroline=False,
    )
    figure.update_layout(
        height=720,
        margin={"l": 48, "r": 16, "t": 16, "b": 48},
        plot_bgcolor="#d9d3c5",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _cell_shape(figure, square: tuple[int, int], *, line: str, width: int, fill: str) -> None:
    left = square[0] * 5
    bottom = square[1] * 5
    figure.add_shape(
        type="rect",
        x0=left,
        x1=left + 5,
        y0=bottom,
        y1=bottom + 5,
        line={"color": line, "width": width},
        fillcolor=fill,
    )


def selected_square(event) -> tuple[int, int] | None:
    """Square from a Plotly chart selection, or None when the click is empty."""
    if event is None:
        return None
    selection = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if selection is None:
        return None
    points = selection.get("points") if isinstance(selection, dict) else getattr(selection, "points", None)
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if isinstance(custom, (list, tuple)) and len(custom) >= 2:
        return int(custom[0]), int(custom[1])
    x_value = point.get("x") if isinstance(point, dict) else getattr(point, "x", None)
    y_value = point.get("y") if isinstance(point, dict) else getattr(point, "y", None)
    if x_value is None or y_value is None:
        return None
    return int(x_value // 5), int(y_value // 5)
