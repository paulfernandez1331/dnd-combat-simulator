"""Five-foot squares. Diagonals cost the same as orthogonal steps (Chebyshev)."""

from __future__ import annotations

from collections import deque

FACINGS: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "NE": (1, 1),
    "E": (1, 0),
    "SE": (1, -1),
    "S": (0, -1),
    "SW": (-1, -1),
    "W": (-1, 0),
    "NW": (-1, 1),
}

Square = tuple[int, int]


def chebyshev(a: Square, b: Square) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def feet_between(a: Square, b: Square) -> int:
    return chebyshev(a, b) * 5


def step_path(origin: Square, dest: Square) -> list[Square]:
    """Chebyshev steps from ``origin`` to ``dest``, excluding the start."""
    current = (int(origin[0]), int(origin[1]))
    goal = (int(dest[0]), int(dest[1]))
    squares: list[Square] = []
    guard = 0
    while current != goal and guard < 64:
        sx, sy = current
        dx, dy = goal
        step_x = 0 if sx == dx else (1 if dx > sx else -1)
        step_y = 0 if sy == dy else (1 if dy > sy else -1)
        current = (sx + step_x, sy + step_y)
        squares.append(current)
        guard += 1
    return squares


def in_bounds(square: Square, width: int, height: int) -> bool:
    return 0 <= square[0] < width and 0 <= square[1] < height


def neighbors(square: Square) -> list[Square]:
    x, y = square
    return [
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    ]


SIZE_SPAN = {
    "tiny": 1,
    "small": 1,
    "medium": 1,
    "large": 2,
    "huge": 3,
    "gargantuan": 4,
}


def span_for(size: str | None) -> int:
    """Squares on a side. An unknown size is one square."""
    return SIZE_SPAN.get(str(size or "medium").strip().lower(), 1)


def footprint(anchor: Square, span: int) -> set[Square]:
    """Squares occupied when ``anchor`` is the southwest corner."""
    x, y = int(anchor[0]), int(anchor[1])
    side = max(1, int(span))
    return {(x + dx, y + dy) for dx in range(side) for dy in range(side)}


def _axis_gap(start: int, span: int, other: int, other_span: int) -> int:
    end = start + max(1, span) - 1
    other_end = other + max(1, other_span) - 1
    if start <= other_end and other <= end:
        return 0
    if other > end:
        return other - end
    return start - other_end


def space_gap(origin: Square, origin_span: int, other: Square, other_span: int) -> int:
    """Chebyshev steps between two rectangles. Zero when they share a square."""
    return max(
        _axis_gap(int(origin[0]), origin_span, int(other[0]), other_span),
        _axis_gap(int(origin[1]), origin_span, int(other[1]), other_span),
    )


def _body_clear(anchor: Square, span: int, width: int, height: int, blocked: set[Square]) -> bool:
    for cell in footprint(anchor, span):
        if not in_bounds(cell, width, height) or cell in blocked:
            return False
    return True


def reachable_anchors(
    origin: Square,
    span: int,
    speed_feet: int,
    *,
    width: int,
    height: int,
    blocked: set[Square],
    passable: set[Square],
    step_cost=None,
) -> set[Square]:
    """Corners a creature can end on.

    Each step moves the southwest corner one square, including diagonally,
    and costs 5 feet. ``blocked`` squares cannot be entered. ``passable``
    squares can be crossed and cannot be stopped on. The whole body has to
    stay on the board. ``step_cost(anchor)`` is the cost of that step in
    units of 5 feet. It is 1 when the ground is clear.
    """
    if step_cost is not None:
        return _reachable_weighted(
            origin,
            span,
            speed_feet,
            width=width,
            height=height,
            blocked=blocked,
            passable=passable,
            step_cost=step_cost,
        )
    budget = max(0, speed_feet) // 5
    if budget == 0:
        return set()
    side = max(1, int(span))
    seen: dict[Square, int] = {origin: 0}
    queue: deque[tuple[Square, int]] = deque([(origin, 0)])
    ends: set[Square] = set()
    while queue:
        anchor, cost = queue.popleft()
        if cost > 0 and _body_clear(anchor, side, width, height, blocked | passable):
            ends.add(anchor)
        if cost == budget:
            continue
        for nxt in neighbors(anchor):
            if not _body_clear(nxt, side, width, height, blocked):
                continue
            ncost = cost + 1
            if seen.get(nxt, budget + 1) <= ncost:
                continue
            seen[nxt] = ncost
            queue.append((nxt, ncost))
    return ends


def _reachable_weighted(
    origin: Square,
    span: int,
    speed_feet: int,
    *,
    width: int,
    height: int,
    blocked: set[Square],
    passable: set[Square],
    step_cost,
) -> set[Square]:
    """Same search as ``reachable_anchors`` when a step can cost more than 5 feet."""
    import heapq

    budget = max(0, speed_feet) // 5
    if budget == 0:
        return set()
    side = max(1, int(span))
    best: dict[Square, int] = {origin: 0}
    heap: list[tuple[int, Square]] = [(0, origin)]
    ends: set[Square] = set()
    while heap:
        cost, anchor = heapq.heappop(heap)
        if cost != best.get(anchor):
            continue
        if cost > 0 and _body_clear(anchor, side, width, height, blocked | passable):
            ends.add(anchor)
        if cost >= budget:
            continue
        for nxt in neighbors(anchor):
            if not _body_clear(nxt, side, width, height, blocked):
                continue
            ncost = cost + max(1, int(step_cost(nxt)))
            if ncost > budget or best.get(nxt, budget + 1) <= ncost:
                continue
            best[nxt] = ncost
            heapq.heappush(heap, (ncost, nxt))
    return ends


def reachable_squares(
    origin: Square,
    speed_feet: int,
    *,
    width: int,
    height: int,
    blocked: set[Square],
    passable: set[Square],
) -> set[Square]:
    """Squares a creature can end on.

    ``blocked`` cannot be entered (enemies). ``passable`` can be crossed but
    not ended on (allies). Each step, including a diagonal, costs 5 feet.
    """
    budget = max(0, speed_feet) // 5
    if budget == 0:
        return set()
    seen: dict[Square, int] = {origin: 0}
    queue: deque[tuple[Square, int]] = deque([(origin, 0)])
    ends: set[Square] = set()
    while queue:
        square, cost = queue.popleft()
        if cost > 0 and square not in passable and square not in blocked:
            ends.add(square)
        if cost == budget:
            continue
        for nxt in neighbors(square):
            if not in_bounds(nxt, width, height) or nxt in blocked:
                continue
            ncost = cost + 1
            if seen.get(nxt, budget + 1) <= ncost:
                continue
            seen[nxt] = ncost
            queue.append((nxt, ncost))
    return ends


def _ball(center: Square, radius: int) -> set[Square]:
    cx, cy = center
    return {
        (cx + dx, cy + dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if max(abs(dx), abs(dy)) <= radius
    }


def sphere(center: Square, size_feet: int) -> set[Square]:
    """Squares within ``size_feet`` of ``center``, including the center."""
    return _ball(center, max(0, size_feet) // 5)


def cylinder(center: Square, size_feet: int) -> set[Square]:
    """Same footprint as a sphere. The grid has no height."""
    return sphere(center, size_feet)


def _forward_lateral(dx: int, dy: int, facing: Square) -> tuple[float, float]:
    fx, fy = facing
    magnitude = fx * fx + fy * fy
    px, py = -fy, fx
    forward = (dx * fx + dy * fy) / magnitude
    lateral = abs(dx * px + dy * py) / magnitude
    return forward, lateral


def cone(origin: Square, size_feet: int, facing: Square) -> set[Square]:
    """A widening template. A square is inside when it is in front of the
    origin, within the length, and its sideways offset is no larger than its
    forward distance. The origin square itself is excluded.
    """
    length = max(1, size_feet // 5)
    ox, oy = origin
    cells: set[Square] = set()
    for dx in range(-length, length + 1):
        for dy in range(-length, length + 1):
            if (dx, dy) == (0, 0) or max(abs(dx), abs(dy)) > length:
                continue
            forward, lateral = _forward_lateral(dx, dy, facing)
            if forward > 0 and lateral <= forward + 1e-9:
                cells.add((ox + dx, oy + dy))
    return cells


def line(origin: Square, size_feet: int, facing: Square) -> set[Square]:
    """A 5-foot-wide line. Each step follows ``facing`` and excludes the origin."""
    length = max(1, size_feet // 5)
    fx, fy = facing
    return {
        (origin[0] + fx * step, origin[1] + fy * step)
        for step in range(1, length + 1)
    }


def cube(origin: Square, size_feet: int, facing: Square) -> set[Square]:
    """An axis-aligned cube whose near face is centered on ``origin``.

    ``origin`` is included. Lateral offsets are centered, and an even side
    length extends one extra square toward the positive perpendicular.
    """
    side = max(1, size_feet // 5)
    fx, fy = facing
    px, py = -fy, fx
    start = -(side // 2)
    cells: set[Square] = set()
    for depth in range(side):
        for lat in range(start, start + side):
            cells.add((origin[0] + fx * depth + px * lat, origin[1] + fy * depth + py * lat))
    return cells


def area_squares(
    origin: Square,
    area_type: str,
    size_feet: int,
    facing: Square | None = None,
) -> set[Square]:
    kind = area_type.lower()
    if kind == "sphere":
        return sphere(origin, size_feet)
    if kind == "cylinder":
        return cylinder(origin, size_feet)
    if kind == "cone":
        if facing is None:
            raise ValueError("A cone needs a facing.")
        return cone(origin, size_feet, facing)
    if kind == "line":
        if facing is None:
            raise ValueError("A line needs a facing.")
        return line(origin, size_feet, facing)
    if kind == "cube":
        if facing is None:
            raise ValueError("A cube needs a facing.")
        return cube(origin, size_feet, facing)
    raise ValueError(f"Unsupported area type: {area_type}")
