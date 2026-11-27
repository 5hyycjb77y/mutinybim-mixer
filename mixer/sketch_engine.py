"""Self-supervised Napkin Sketch -> MutinyBIM data engine.

Generates deterministic, topologically diverse parametric MutinyBIM floor plans
paired with exact runnable `@permitzip/bim-script` ground truth, plus organic
hand-drawn sketch renderings (pure-stdlib SVG and PIL/NumPy textured PNGs).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKETCH_TOPOLOGIES: tuple[str, ...] = (
    "office_suite",
    "corridor_double_loaded",
    "l_shaped_studio",
    "boutique_cafe",
    "apartment_unit",
    "two_story_compact",
)

SKETCH_STYLES: tuple[str, ...] = (
    "black_sharpie_napkin",
    "blue_ballpoint_graph",
    "ipad_procreate_white",
    "phone_photo_blueprint",
)


@dataclass(frozen=True)
class RoomSpec:
    name: str
    number: str
    x0: float
    y0: float
    x1: float
    y1: float
    floor: int = 1


@dataclass(frozen=True)
class WallSegment:
    x0: float
    y0: float
    x1: float
    y1: float
    thickness_in: float
    is_exterior: bool
    floor: int = 1

    @property
    def length_ft(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass(frozen=True)
class OpeningSpec:
    kind: str  # "door" or "window"
    wall_group: str
    segment_index: int  # 1-based MutinyBIM element(n)
    along_ft: float
    width_ft: float
    floor: int
    x: float
    y: float
    ux: float
    uy: float


@dataclass(frozen=True)
class SketchPlanSample:
    sample_id: str
    seed: int
    topology: str
    width_ft: float
    depth_ft: float
    floors: int
    wall_height_ft: float
    ceiling_height_ft: float
    rooms: list[RoomSpec]
    walls: list[WallSegment]
    openings: list[OpeningSpec]
    ts_source: str
    user_prompt: str
    expected_kinds: dict[str, int]


def _clamp_along(target: float, seg_len: float, width_ft: float) -> float | None:
    margin = max(2.2, width_ft * 0.6 + 1.0)
    if seg_len < margin * 2.0:
        return None
    return round(max(margin, min(seg_len - margin, target)), 2)


def _make_opening(
    kind: str,
    wall_group: str,
    segment_index: int,
    seg: WallSegment,
    target_along: float,
    width_ft: float,
) -> OpeningSpec | None:
    seg_len = seg.length_ft
    along = _clamp_along(target_along, seg_len, width_ft)
    if along is None or seg_len < 1e-6:
        return None
    ux = (seg.x1 - seg.x0) / seg_len
    uy = (seg.y1 - seg.y0) / seg_len
    return OpeningSpec(
        kind=kind,
        wall_group=wall_group,
        segment_index=segment_index,
        along_ft=along,
        width_ft=width_ft,
        floor=seg.floor,
        x=seg.x0 + ux * along,
        y=seg.y0 + uy * along,
        ux=ux,
        uy=uy,
    )


def _poly_segments(pts: list[tuple[float, float]], thickness_in: float, is_exterior: bool, floor: int) -> list[WallSegment]:
    segs: list[WallSegment] = []
    for idx in range(len(pts)):
        x0, y0 = pts[idx]
        x1, y1 = pts[(idx + 1) % len(pts)]
        segs.append(WallSegment(x0, y0, x1, y1, thickness_in, is_exterior, floor))
    return segs


def _fmt_num(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _emit_ts_single_floor(
    *,
    title: str,
    wall_height_ft: float,
    ceiling_height_ft: float,
    exterior_pts: list[tuple[float, float]],
    interior_chains: list[tuple[str, list[tuple[float, float]]]],
    openings: list[OpeningSpec],
    rooms: list[RoomSpec],
    sheet_num: str = "A-101",
) -> str:
    ext_pts_str = ", ".join(f"at({_fmt_num(x)}, {_fmt_num(y)})" for x, y in exterior_pts)
    lines = [
        f"// Auto-generated MutinyBIM floor plan: {title}",
        'import { defineScript, units, building, documents } from "@permitzip/bim-script";',
        "",
        "const ft = units.ft;",
        "const at = (x: number, y: number) => ({ x: ft(x), y: ft(y) });",
        "const inset = (x0: number, y0: number, x1: number, y1: number) => [at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1)];",
        "",
        "export default defineScript({",
        '  kind: "layout",',
        "  reads: { levels: true },",
        "  parameters: {},",
        "  run(ctx) {",
        f'    ctx.label({title!r});',
        '    const level = building.level(ctx, { name: "Level 1", elevation: ft(0) });',
        f"    const outside = building.walls(ctx, {{ level, points: [{ext_pts_str}], closed: true, thickness: units.in(8), height: ft({_fmt_num(wall_height_ft)}) }});",
    ]
    for var_name, chain in interior_chains:
        pts_str = ", ".join(f"at({_fmt_num(x)}, {_fmt_num(y)})" for x, y in chain)
        lines.append(
            f"    const {var_name} = building.walls(ctx, {{ level, points: [{pts_str}], thickness: units.in(5), height: ft({_fmt_num(wall_height_ft)}) }});"
        )

    for op in openings:
        if op.kind == "door":
            lines.append(
                f"    building.door(ctx, {{ wall: {op.wall_group}.element({op.segment_index}), along: ft({_fmt_num(op.along_ft)}), width: units.in({int(round(op.width_ft * 12))}) }});"
            )
        else:
            lines.append(
                f"    building.window(ctx, {{ wall: {op.wall_group}.element({op.segment_index}), along: ft({_fmt_num(op.along_ft)}), width: ft({_fmt_num(op.width_ft)}), height: ft(5) }});"
            )

    for idx, rm in enumerate(rooms):
        cx = 0.5 * (rm.x0 + rm.x1)
        cy = 0.5 * (rm.y0 + rm.y1)
        var = f"room{idx + 1}"
        lines.append(
            f'    const {var} = building.spaceInWalls(ctx, {{ level, inside: at({_fmt_num(cx)}, {_fmt_num(cy)}), number: {rm.number!r}, name: {rm.name!r} }});'
        )
        lines.append(
            f"    building.ceiling(ctx, {{ level, space: {var}, outline: inset({_fmt_num(rm.x0)}, {_fmt_num(rm.y0)}, {_fmt_num(rm.x1)}, {_fmt_num(rm.y1)}), height: ft({_fmt_num(ceiling_height_ft)}) }});"
        )

    lines.extend([
        f'    const sheet = documents.sheet(ctx, {{ number: {sheet_num!r}, name: "Level 1 floor plan" }});',
        '    const plan = documents.planView(ctx, { level, name: "Level 1 - Plan", scale: 48 });',
        "    documents.place(ctx, sheet, plan, { x: units.mm(300), y: units.mm(280) });",
        f'    ctx.report("Rooms", [{", ".join(repr(r.name) for r in rooms)}]);',
        "  },",
        "});",
    ])
    return "\n".join(lines)


def generate_sketch_sample(seed: int, topology: str | None = None) -> SketchPlanSample:
    """Deterministically generate a paired architectural sketch layout and valid MutinyBIM script."""
    rng = random.Random(seed)
    topo = topology or SKETCH_TOPOLOGIES[seed % len(SKETCH_TOPOLOGIES)]
    if topo not in SKETCH_TOPOLOGIES:
        raise ValueError(f"Unknown topology '{topo}'. Expected one of {SKETCH_TOPOLOGIES}")

    wall_h = float(rng.choice([10, 11, 12]))
    ceil_h = float(wall_h - 2.0)

    walls: list[WallSegment] = []
    openings: list[OpeningSpec] = []
    rooms: list[RoomSpec] = []

    if topo == "office_suite":
        W = float(rng.randrange(36, 50, 2))
        D = float(rng.randrange(24, 36, 2))
        sx = float(rng.randrange(int(W * 0.45), int(W * 0.65), 2))
        sy = float(rng.randrange(11, int(D - 10), 1))

        ext_pts = [(0.0, 0.0), (W, 0.0), (W, D), (0.0, D)]
        ext_segs = _poly_segments(ext_pts, 8.0, True, 1)
        split_seg = WallSegment(sx, 0.0, sx, D, 5.0, False, 1)
        cross_seg = WallSegment(sx, sy, W, sy, 5.0, False, 1)
        walls = [*ext_segs, split_seg, cross_seg]

        for op in (
            _make_opening("door", "outside", 1, ext_segs[0], sx * 0.5, 3.0),
            _make_opening("door", "split", 1, split_seg, sy * 0.5, 3.0),
            _make_opening("door", "split", 1, split_seg, sy + (D - sy) * 0.5, 3.0),
            _make_opening("door", "cross", 1, cross_seg, (W - sx) * 0.5, 3.0),
            _make_opening("window", "outside", 3, ext_segs[2], W * 0.25, 4.0),
            _make_opening("window", "outside", 3, ext_segs[2], W * 0.72, 4.0),
        ):
            if op:
                openings.append(op)

        rooms = [
            RoomSpec("Open Office", "101", 0.0, 0.0, sx, D, 1),
            RoomSpec("Conference", "102", sx, 0.0, W, sy, 1),
            RoomSpec("Private Office", "103", sx, sy, W, D, 1),
        ]
        ts_source = _emit_ts_single_floor(
            title=f"Office suite ({int(W)} ft x {int(D)} ft)",
            wall_height_ft=wall_h,
            ceiling_height_ft=ceil_h,
            exterior_pts=ext_pts,
            interior_chains=[("split", [(sx, 0.0), (sx, D)]), ("cross", [(sx, sy), (W, sy)])],
            openings=openings,
            rooms=rooms,
        )
        floors = 1

    elif topo == "corridor_double_loaded":
        W = float(rng.randrange(40, 54, 2))
        D = float(rng.randrange(26, 34, 2))
        cy0 = float(int(D // 2) - 3)
        cy1 = cy0 + 6.0
        nx = float(rng.randrange(16, int(W - 14), 2))
        sx = float(rng.randrange(14, int(W - 16), 2))

        ext_pts = [(0.0, 0.0), (W, 0.0), (W, D), (0.0, D)]
        ext_segs = _poly_segments(ext_pts, 8.0, True, 1)
        c_south = WallSegment(0.0, cy0, W, cy0, 5.0, False, 1)
        c_north = WallSegment(0.0, cy1, W, cy1, 5.0, False, 1)
        s_div = WallSegment(sx, 0.0, sx, cy0, 5.0, False, 1)
        n_div = WallSegment(nx, cy1, nx, D, 5.0, False, 1)
        walls = [*ext_segs, c_south, c_north, s_div, n_div]

        for op in (
            _make_opening("door", "outside", 4, ext_segs[3], D * 0.5, 3.0),
            _make_opening("door", "cSouth", 1, c_south, sx * 0.5, 3.0),
            _make_opening("door", "cSouth", 1, c_south, sx + (W - sx) * 0.5, 3.0),
            _make_opening("door", "cNorth", 1, c_north, nx * 0.5, 3.0),
            _make_opening("door", "cNorth", 1, c_north, nx + (W - nx) * 0.5, 3.0),
            _make_opening("window", "outside", 1, ext_segs[0], sx * 0.5, 4.0),
            _make_opening("window", "outside", 3, ext_segs[2], W - nx * 0.5, 4.0),
        ):
            if op:
                openings.append(op)

        rooms = [
            RoomSpec("Corridor", "101", 0.0, cy0, W, cy1, 1),
            RoomSpec("Reception", "102", 0.0, 0.0, sx, cy0, 1),
            RoomSpec("Conf Room", "103", sx, 0.0, W, cy0, 1),
            RoomSpec("Studio A", "104", 0.0, cy1, nx, D, 1),
            RoomSpec("Studio B", "105", nx, cy1, W, D, 1),
        ]
        ts_source = _emit_ts_single_floor(
            title=f"Corridor office ({int(W)} ft x {int(D)} ft)",
            wall_height_ft=wall_h,
            ceiling_height_ft=ceil_h,
            exterior_pts=ext_pts,
            interior_chains=[
                ("cSouth", [(0.0, cy0), (W, cy0)]),
                ("cNorth", [(0.0, cy1), (W, cy1)]),
                ("sDiv", [(sx, 0.0), (sx, cy0)]),
                ("nDiv", [(nx, cy1), (nx, D)]),
            ],
            openings=openings,
            rooms=rooms,
        )
        floors = 1

    elif topo == "l_shaped_studio":
        W = float(rng.randrange(38, 50, 2))
        W1 = float(rng.randrange(20, int(W - 14), 2))
        D = float(rng.randrange(30, 40, 2))
        D1 = float(rng.randrange(12, int(D - 14), 1))
        # Keep the horizontal partition off the notch line y=D1 so it tees cleanly
        # into the vertical exterior wall instead of meeting the inside L-corner collinearly.
        north_y = float(D1 + rng.choice([2, 3, 4]))

        ext_pts = [(0.0, 0.0), (W, 0.0), (W, D1), (W1, D1), (W1, D), (0.0, D)]
        ext_segs = _poly_segments(ext_pts, 8.0, True, 1)
        east_div = WallSegment(W1, 0.0, W1, D1, 5.0, False, 1)
        north_div = WallSegment(0.0, north_y, W1, north_y, 5.0, False, 1)
        walls = [*ext_segs, east_div, north_div]

        for op in (
            _make_opening("door", "outside", 1, ext_segs[0], W1 * 0.5, 3.0),
            _make_opening("door", "eastDiv", 1, east_div, D1 * 0.5, 3.0),
            _make_opening("door", "northDiv", 1, north_div, W1 * 0.5, 3.0),
            _make_opening("window", "outside", 2, ext_segs[1], D1 * 0.5, 4.0),
            _make_opening("window", "outside", 5, ext_segs[4], W1 * 0.5, 4.0),
        ):
            if op:
                openings.append(op)

        rooms = [
            RoomSpec("Main Studio", "101", 0.0, 0.0, W1, north_y, 1),
            RoomSpec("Workshop", "102", W1, 0.0, W, D1, 1),
            RoomSpec("Gallery", "103", 0.0, north_y, W1, D, 1),
        ]
        ts_source = _emit_ts_single_floor(
            title=f"L-shaped studio ({int(W)} ft x {int(D)} ft)",
            wall_height_ft=wall_h,
            ceiling_height_ft=ceil_h,
            exterior_pts=ext_pts,
            interior_chains=[
                ("eastDiv", [(W1, 0.0), (W1, D1)]),
                ("northDiv", [(0.0, north_y), (W1, north_y)]),
            ],
            openings=openings,
            rooms=rooms,
        )
        floors = 1

    elif topo == "boutique_cafe":
        W = float(rng.randrange(32, 44, 2))
        D = float(rng.randrange(22, 30, 2))
        bx = float(rng.randrange(int(W * 0.55), int(W - 10), 2))
        by = float(rng.randrange(10, int(D - 10), 1))

        ext_pts = [(0.0, 0.0), (W, 0.0), (W, D), (0.0, D)]
        ext_segs = _poly_segments(ext_pts, 8.0, True, 1)
        boh = WallSegment(bx, 0.0, bx, D, 5.0, False, 1)
        wc_div = WallSegment(bx, by, W, by, 5.0, False, 1)
        walls = [*ext_segs, boh, wc_div]

        for op in (
            _make_opening("door", "outside", 1, ext_segs[0], bx * 0.35, 3.0),
            _make_opening("door", "boh", 1, boh, by * 0.5, 3.0),
            _make_opening("door", "boh", 1, boh, by + (D - by) * 0.5, 3.0),
            _make_opening("window", "outside", 1, ext_segs[0], bx * 0.72, 4.0),
            _make_opening("window", "outside", 4, ext_segs[3], D * 0.5, 4.0),
        ):
            if op:
                openings.append(op)

        rooms = [
            RoomSpec("Cafe Seating", "101", 0.0, 0.0, bx, D, 1),
            RoomSpec("Roastery Prep", "102", bx, 0.0, W, by, 1),
            RoomSpec("Restroom", "103", bx, by, W, D, 1),
        ]
        ts_source = _emit_ts_single_floor(
            title=f"Boutique cafe ({int(W)} ft x {int(D)} ft)",
            wall_height_ft=wall_h,
            ceiling_height_ft=ceil_h,
            exterior_pts=ext_pts,
            interior_chains=[("boh", [(bx, 0.0), (bx, D)]), ("wcDiv", [(bx, by), (W, by)])],
            openings=openings,
            rooms=rooms,
        )
        floors = 1

    elif topo == "apartment_unit":
        W = float(rng.randrange(30, 40, 2))
        D = float(rng.randrange(18, 24, 2))
        bx = float(rng.randrange(13, int(W - 14), 1))
        bath_y = float(rng.randrange(8, int(D - 8), 1))

        ext_pts = [(0.0, 0.0), (W, 0.0), (W, D), (0.0, D)]
        ext_segs = _poly_segments(ext_pts, 8.0, True, 1)
        mid = WallSegment(bx, 0.0, bx, D, 5.0, False, 1)
        bath = WallSegment(0.0, bath_y, bx, bath_y, 5.0, False, 1)
        walls = [*ext_segs, mid, bath]

        for op in (
            _make_opening("door", "outside", 1, ext_segs[0], bx + (W - bx) * 0.5, 3.0),
            _make_opening("door", "mid", 1, mid, bath_y * 0.5, 3.0),
            _make_opening("door", "bath", 1, bath, bx * 0.5, 3.0),
            _make_opening("window", "outside", 3, ext_segs[2], (W - bx) * 0.5, 4.0),
            _make_opening("window", "outside", 1, ext_segs[0], bx * 0.5, 4.0),
        ):
            if op:
                openings.append(op)

        rooms = [
            RoomSpec("Bedroom", "101", 0.0, 0.0, bx, bath_y, 1),
            RoomSpec("Bath", "102", 0.0, bath_y, bx, D, 1),
            RoomSpec("Living Studio", "103", bx, 0.0, W, D, 1),
        ]
        ts_source = _emit_ts_single_floor(
            title=f"Apartment unit ({int(W)} ft x {int(D)} ft)",
            wall_height_ft=wall_h,
            ceiling_height_ft=ceil_h,
            exterior_pts=ext_pts,
            interior_chains=[("mid", [(bx, 0.0), (bx, D)]), ("bath", [(0.0, bath_y), (bx, bath_y)])],
            openings=openings,
            rooms=rooms,
        )
        floors = 1

    else:  # "two_story_compact"
        W = float(rng.randrange(34, 46, 2))
        D = float(rng.randrange(24, 32, 2))
        sx = float(rng.randrange(int(W * 0.48), int(W * 0.65), 2))
        sy = float(rng.randrange(11, int(D - 10), 1))
        floors = 2

        ext_pts = [(0.0, 0.0), (W, 0.0), (W, D), (0.0, D)]
        ext_segs = _poly_segments(ext_pts, 8.0, True, 1)
        split_seg = WallSegment(sx, 0.0, sx, D, 5.0, False, 1)
        cross_seg = WallSegment(sx, sy, W, sy, 5.0, False, 1)
        walls = [*ext_segs, split_seg, cross_seg]

        for op in (
            _make_opening("door", "outside", 1, ext_segs[0], sx * 0.45, 3.0),
            _make_opening("door", "split", 1, split_seg, sy + (D - sy) * 0.5, 3.0),
            _make_opening("door", "cross", 1, cross_seg, (W - sx) * 0.5, 3.0),
            _make_opening("window", "outside", 3, ext_segs[2], W * 0.28, 4.0),
            _make_opening("window", "outside", 3, ext_segs[2], W * 0.72, 4.0),
        ):
            if op:
                openings.append(op)

        for fl in (1, 2):
            base_num = fl * 100
            rooms.extend([
                RoomSpec("Open Studio", str(base_num + 1), 0.0, 0.0, sx, D, fl),
                RoomSpec("Meeting", str(base_num + 2), sx, 0.0, W, sy, fl),
                RoomSpec("Office", str(base_num + 3), sx, sy, W, D, fl),
            ])

        ts_source = "\n".join([
            f"// Auto-generated 2-story MutinyBIM floor plan ({int(W)} ft x {int(D)} ft)",
            'import { defineScript, units, building, documents } from "@permitzip/bim-script";',
            "",
            "const ft = units.ft;",
            "const at = (x: number, y: number) => ({ x: ft(x), y: ft(y) });",
            "const inset = (x0: number, y0: number, x1: number, y1: number) => [at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1)];",
            "",
            "export default defineScript({",
            '  kind: "layout",',
            "  reads: { levels: true },",
            "  parameters: {},",
            "  run(ctx) {",
            f'    ctx.label("Two-story building ({int(W)} ft x {int(D)} ft)");',
            "    const floors = [",
            '      { name: "Level 1", elevation: ft(0), sheet: "A-101" },',
            '      { name: "Level 2", elevation: ft(12), sheet: "A-102" },',
            "    ];",
            "    for (const floor of floors) {",
            "      const level = building.level(ctx, { name: floor.name, elevation: floor.elevation });",
            f"      const outside = building.walls(ctx, {{ level, points: [at(0, 0), at({_fmt_num(W)}, 0), at({_fmt_num(W)}, {_fmt_num(D)}), at(0, {_fmt_num(D)})], closed: true, thickness: units.in(8), height: ft({_fmt_num(wall_h)}) }});",
            f"      const split = building.walls(ctx, {{ level, points: [at({_fmt_num(sx)}, 0), at({_fmt_num(sx)}, {_fmt_num(D)})], thickness: units.in(5), height: ft({_fmt_num(wall_h)}) }});",
            f"      const cross = building.walls(ctx, {{ level, points: [at({_fmt_num(sx)}, {_fmt_num(sy)}), at({_fmt_num(W)}, {_fmt_num(sy)})], thickness: units.in(5), height: ft({_fmt_num(wall_h)}) }});",
            f'      if (floor.name === "Level 1") building.door(ctx, {{ wall: outside.element(1), along: ft({_fmt_num(sx * 0.45)}), width: units.in(36) }});',
            f"      building.door(ctx, {{ wall: split.element(1), along: ft({_fmt_num(sy + (D - sy) * 0.5)}), width: units.in(36) }});",
            f"      building.door(ctx, {{ wall: cross.element(1), along: ft({_fmt_num((W - sx) * 0.5)}), width: units.in(36) }});",
            f"      for (const along of [{_fmt_num(W * 0.28)}, {_fmt_num(W * 0.72)}]) building.window(ctx, {{ wall: outside.element(3), along: ft(along), width: ft(4), height: ft(5) }});",
            '      const n = floor.name === "Level 1" ? 100 : 200;',
            f'      const r1 = building.spaceInWalls(ctx, {{ level, inside: at({_fmt_num(sx * 0.5)}, {_fmt_num(D * 0.5)}), number: `${{n + 1}}`, name: "Open Studio" }});',
            f'      const r2 = building.spaceInWalls(ctx, {{ level, inside: at({_fmt_num(sx + (W - sx) * 0.5)}, {_fmt_num(sy * 0.5)}), number: `${{n + 2}}`, name: "Meeting" }});',
            f'      const r3 = building.spaceInWalls(ctx, {{ level, inside: at({_fmt_num(sx + (W - sx) * 0.5)}, {_fmt_num(sy + (D - sy) * 0.5)}), number: `${{n + 3}}`, name: "Office" }});',
            f"      building.ceiling(ctx, {{ level, space: r1, outline: inset(0, 0, {_fmt_num(sx)}, {_fmt_num(D)}), height: ft({_fmt_num(ceil_h)}) }});",
            f"      building.ceiling(ctx, {{ level, space: r2, outline: inset({_fmt_num(sx)}, 0, {_fmt_num(W)}, {_fmt_num(sy)}), height: ft({_fmt_num(ceil_h)}) }});",
            f"      building.ceiling(ctx, {{ level, space: r3, outline: inset({_fmt_num(sx)}, {_fmt_num(sy)}, {_fmt_num(W)}, {_fmt_num(D)}), height: ft({_fmt_num(ceil_h)}) }});",
            "      const sheet = documents.sheet(ctx, { number: floor.sheet, name: `${floor.name} floor plan` });",
            "      const plan = documents.planView(ctx, { level, name: `${floor.name} - Plan`, scale: 48 });",
            "      documents.place(ctx, sheet, plan, { x: units.mm(300), y: units.mm(280) });",
            "    }",
            "  },",
            "});",
        ])

    floor1_rooms = [r for r in rooms if r.floor == 1]
    room_names = ", ".join(r.name for r in floor1_rooms)
    floor_phrase = "two-story " if floors == 2 else ""
    user_prompt = (
        f"Convert this hand-drawn architectural floor plan sketch (~{int(W)} ft x {int(D)} ft {floor_phrase}"
        f"with {room_names}) into a complete parametric MutinyBIM TypeScript script with exterior/interior "
        "walls, doors, windows, traced rooms, ceilings, and floor plan sheets."
    )
    expected_kinds = {
        "level": floors,
        "wall": len(walls) * floors,
        "space": len(rooms),
        "ceiling": len(rooms),
        "sheet": floors,
    }

    return SketchPlanSample(
        sample_id=f"sketch_{topo}_{seed}",
        seed=seed,
        topology=topo,
        width_ft=W,
        depth_ft=D,
        floors=floors,
        wall_height_ft=wall_h,
        ceiling_height_ft=ceil_h,
        rooms=rooms,
        walls=walls,
        openings=openings,
        ts_source=ts_source,
        user_prompt=user_prompt,
        expected_kinds=expected_kinds,
    )


def generate_sketch_dataset_specs(
    count: int,
    *,
    start_seed: int = 1000,
    topologies: tuple[str, ...] | None = None,
) -> list[SketchPlanSample]:
    """Generate a balanced list of ``count`` self-supervised sketch-to-BIM samples."""
    pool = topologies or SKETCH_TOPOLOGIES
    samples: list[SketchPlanSample] = []
    for idx in range(count):
        topo = pool[idx % len(pool)]
        samples.append(generate_sketch_sample(start_seed + idx, topology=topo))
    return samples


def _style_palette(style: str) -> dict[str, Any]:
    if style == "blue_ballpoint_graph":
        return {"bg": "#f7fbff", "ink": "#1a4e9b", "grid": "#cfe3fa", "stroke": 3.0, "label": "#133b76"}
    if style == "ipad_procreate_white":
        return {"bg": "#ffffff", "ink": "#1f2328", "grid": None, "stroke": 3.2, "label": "#2f363d"}
    if style == "phone_photo_blueprint":
        return {"bg": "#efeae0", "ink": "#222428", "grid": "#e2dacb", "stroke": 3.4, "label": "#2c2e33"}
    # default: black_sharpie_napkin
    return {"bg": "#faf6ee", "ink": "#161618", "grid": None, "stroke": 4.2, "label": "#222226"}


def _wobbly_svg_path(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rng: random.Random,
    *,
    wobble_px: float = 3.0,
    overshoot_px: float = 7.0,
) -> str:
    dx = x1 - x0
    dy = y1 - y0
    dist = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    sx = x0 - ux * rng.uniform(0.0, overshoot_px)
    sy = y0 - uy * rng.uniform(0.0, overshoot_px)
    ex = x1 + ux * rng.uniform(0.0, overshoot_px)
    ey = y1 + uy * rng.uniform(0.0, overshoot_px)
    mx = 0.5 * (sx + ex) + nx * rng.uniform(-wobble_px, wobble_px)
    my = 0.5 * (sy + ey) + ny * rng.uniform(-wobble_px, wobble_px)
    return f"M {sx:.1f} {sy:.1f} Q {mx:.1f} {my:.1f} {ex:.1f} {ey:.1f}"


def render_parametric_sketch_svg(
    sample: SketchPlanSample,
    *,
    style: str = "black_sharpie_napkin",
    width: int = 896,
    height: int = 672,
    seed: int | None = None,
) -> str:
    """Render an organic hand-drawn SVG sketch of ``sample`` (pure Python stdlib)."""
    rng = random.Random(sample.seed if seed is None else seed)
    pal = _style_palette(style)

    pad = 88.0
    scale = min((width - 2 * pad) / max(sample.width_ft, 1.0), (height - 2 * pad) / max(sample.depth_ft, 1.0))
    ox = (width - sample.width_ft * scale) * 0.5
    oy = (height - sample.depth_ft * scale) * 0.5

    def to_px(x_ft: float, y_ft: float) -> tuple[float, float]:
        # Flip Y so north (+y) appears at top of sketch.
        return ox + x_ft * scale, height - (oy + y_ft * scale)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{pal["bg"]}"/>',
    ]

    if pal["grid"]:
        step = 24
        for gx in range(0, width, step):
            parts.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{height}" stroke="{pal["grid"]}" stroke-width="1"/>')
        for gy in range(0, height, step):
            parts.append(f'<line x1="0" y1="{gy}" x2="{width}" y2="{gy}" stroke="{pal["grid"]}" stroke-width="1"/>')

    # Floor 1 walls with corner overshoots & hand wobble.
    for w in sample.walls:
        if w.floor != 1:
            continue
        px0, py0 = to_px(w.x0, w.y0)
        px1, py1 = to_px(w.x1, w.y1)
        sw = pal["stroke"] * (1.25 if w.is_exterior else 0.9)
        d_main = _wobbly_svg_path(px0, py0, px1, py1, rng, wobble_px=2.6, overshoot_px=6.5)
        parts.append(
            f'<path d="{d_main}" fill="none" stroke="{pal["ink"]}" stroke-width="{sw:.1f}" stroke-linecap="round"/>'
        )

    # Doors (swing arc + leaf) and windows (double-line mark).
    for op in sample.openings:
        if op.floor != 1:
            continue
        half_w = 0.5 * op.width_ft
        hx0, hy0 = to_px(op.x - op.ux * half_w, op.y - op.uy * half_w)
        hx1, hy1 = to_px(op.x + op.ux * half_w, op.y + op.uy * half_w)
        # Normal in pixel coords (flip y).
        nx_px = -op.uy
        ny_px = -op.ux
        r_px = op.width_ft * scale
        if op.kind == "door":
            leaf_x = hx0 + nx_px * r_px
            leaf_y = hy0 + ny_px * r_px
            parts.append(
                f'<line x1="{hx0:.1f}" y1="{hy0:.1f}" x2="{leaf_x:.1f}" y2="{leaf_y:.1f}" '
                f'stroke="{pal["ink"]}" stroke-width="2.2" stroke-linecap="round"/>'
            )
            parts.append(
                f'<path d="M {leaf_x:.1f} {leaf_y:.1f} Q {(leaf_x + hx1) * 0.5 + nx_px * r_px * 0.3:.1f} '
                f'{(leaf_y + hy1) * 0.5 + ny_px * r_px * 0.3:.1f} {hx1:.1f} {hy1:.1f}" '
                f'fill="none" stroke="{pal["ink"]}" stroke-width="1.6" stroke-dasharray="4 3"/>'
            )
        else:
            off = 4.0
            parts.append(
                f'<line x1="{hx0 + nx_px * off:.1f}" y1="{hy0 + ny_px * off:.1f}" '
                f'x2="{hx1 + nx_px * off:.1f}" y2="{hy1 + ny_px * off:.1f}" '
                f'stroke="{pal["bg"]}" stroke-width="6"/>'
            )
            parts.append(
                f'<line x1="{hx0 - nx_px * off:.1f}" y1="{hy0 - ny_px * off:.1f}" '
                f'x2="{hx1 - nx_px * off:.1f}" y2="{hy1 - ny_px * off:.1f}" '
                f'stroke="{pal["ink"]}" stroke-width="1.8"/>'
            )
            parts.append(
                f'<line x1="{hx0 + nx_px * off:.1f}" y1="{hy0 + ny_px * off:.1f}" '
                f'x2="{hx1 + nx_px * off:.1f}" y2="{hy1 + ny_px * off:.1f}" '
                f'stroke="{pal["ink"]}" stroke-width="1.8"/>'
            )

    # Room labels with slight organic tilt.
    for rm in sample.rooms:
        if rm.floor != 1:
            continue
        cx, cy = to_px(0.5 * (rm.x0 + rm.x1), 0.5 * (rm.y0 + rm.y1))
        rot = rng.uniform(-3.0, 3.0)
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" fill="{pal["label"]}" font-family="Comic Sans MS, Chalkboard SE, sans-serif" '
            f'font-size="18" font-weight="600" text-anchor="middle" transform="rotate({rot:.1f} {cx:.1f} {cy:.1f})">'
            f"{rm.name.upper()}</text>"
        )

    # Overall rough width & depth dimension notes.
    bot_l = to_px(0.0, 0.0)
    bot_r = to_px(sample.width_ft, 0.0)
    dim_y = bot_l[1] + 34.0
    parts.append(
        f'<path d="{_wobbly_svg_path(bot_l[0], dim_y, bot_r[0], dim_y, rng, wobble_px=1.5, overshoot_px=2.0)}" '
        f'fill="none" stroke="{pal["label"]}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{0.5 * (bot_l[0] + bot_r[0]):.1f}" y="{dim_y + 20:.1f}" fill="{pal["label"]}" '
        f'font-family="monospace" font-size="16" text-anchor="middle">~{int(sample.width_ft)}\'-0"</text>'
    )
    parts.append(
        f'<text x="{bot_l[0] - 42:.1f}" y="{height * 0.5:.1f}" fill="{pal["label"]}" '
        f'font-family="monospace" font-size="16" text-anchor="middle" '
        f'transform="rotate(-90 {bot_l[0] - 42:.1f} {height * 0.5:.1f})">~{int(sample.depth_ft)}\'-0"</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_parametric_sketch_png(
    sample: SketchPlanSample,
    out_path: str | Path,
    *,
    style: str = "black_sharpie_napkin",
    width: int = 896,
    height: int = 672,
    seed: int | None = None,
) -> str:
    """Render a textured organic hand-drawn PNG sketch (lazy-imports PIL & NumPy)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(sample.seed if seed is None else seed)
    np_rng = np.random.default_rng(sample.seed if seed is None else seed)
    pal = _style_palette(style)

    img = Image.new("RGB", (width, height), pal["bg"])
    draw = ImageDraw.Draw(img)

    if pal["grid"]:
        for gx in range(0, width, 24):
            draw.line([(gx, 0), (gx, height)], fill=pal["grid"], width=1)
        for gy in range(0, height, 24):
            draw.line([(0, gy), (width, gy)], fill=pal["grid"], width=1)

    pad = 88.0
    scale = min((width - 2 * pad) / max(sample.width_ft, 1.0), (height - 2 * pad) / max(sample.depth_ft, 1.0))
    ox = (width - sample.width_ft * scale) * 0.5
    oy = (height - sample.depth_ft * scale) * 0.5

    def to_px(x_ft: float, y_ft: float) -> tuple[float, float]:
        return ox + x_ft * scale, height - (oy + y_ft * scale)

    def draw_wobbly_line(x0: float, y0: float, x1: float, y1: float, stroke_w: int, overshoot: float = 6.0) -> None:
        dx, dy = x1 - x0, y1 - y0
        dist = max(math.hypot(dx, dy), 1e-6)
        ux, uy = dx / dist, dy / dist
        nx, ny = -uy, ux
        sx = x0 - ux * rng.uniform(0.0, overshoot)
        sy = y0 - uy * rng.uniform(0.0, overshoot)
        ex = x1 + ux * rng.uniform(0.0, overshoot)
        ey = y1 + uy * rng.uniform(0.0, overshoot)
        steps = max(6, int(dist / 22))
        pts = []
        for s in range(steps + 1):
            t = s / steps
            wob = math.sin(t * math.pi) * rng.uniform(-2.2, 2.2)
            pts.append((sx + (ex - sx) * t + nx * wob, sy + (ey - sy) * t + ny * wob))
        draw.line(pts, fill=pal["ink"], width=stroke_w, joint="curve")

    for w in sample.walls:
        if w.floor != 1:
            continue
        px0, py0 = to_px(w.x0, w.y0)
        px1, py1 = to_px(w.x1, w.y1)
        sw = max(2, int(round(pal["stroke"] * (1.25 if w.is_exterior else 0.9))))
        draw_wobbly_line(px0, py0, px1, py1, sw, overshoot=6.5)

    for op in sample.openings:
        if op.floor != 1:
            continue
        half_w = 0.5 * op.width_ft
        hx0, hy0 = to_px(op.x - op.ux * half_w, op.y - op.uy * half_w)
        hx1, hy1 = to_px(op.x + op.ux * half_w, op.y + op.uy * half_w)
        nx_px, ny_px = -op.uy, -op.ux
        r_px = op.width_ft * scale
        if op.kind == "door":
            leaf_x = hx0 + nx_px * r_px
            leaf_y = hy0 + ny_px * r_px
            draw.line([(hx0, hy0), (leaf_x, leaf_y)], fill=pal["ink"], width=2)
            arc_pts = []
            for s in range(10):
                t = s / 9.0
                ax = (1 - t) ** 2 * leaf_x + 2 * (1 - t) * t * ((leaf_x + hx1) * 0.5 + nx_px * r_px * 0.25) + t**2 * hx1
                ay = (1 - t) ** 2 * leaf_y + 2 * (1 - t) * t * ((leaf_y + hy1) * 0.5 + ny_px * r_px * 0.25) + t**2 * hy1
                arc_pts.append((ax, ay))
            draw.line(arc_pts, fill=pal["ink"], width=2)
        else:
            off = 4.0
            draw.line([(hx0, hy0), (hx1, hy1)], fill=pal["bg"], width=7)
            draw.line([(hx0 - nx_px * off, hy0 - ny_px * off), (hx1 - nx_px * off, hy1 - ny_px * off)], fill=pal["ink"], width=2)
            draw.line([(hx0 + nx_px * off, hy0 + ny_px * off), (hx1 + nx_px * off, hy1 + ny_px * off)], fill=pal["ink"], width=2)

    for rm in sample.rooms:
        if rm.floor != 1:
            continue
        cx, cy = to_px(0.5 * (rm.x0 + rm.x1), 0.5 * (rm.y0 + rm.y1))
        draw.text((cx - len(rm.name) * 3.5, cy - 6), rm.name.upper(), fill=pal["label"])

    bot_l = to_px(0.0, 0.0)
    bot_r = to_px(sample.width_ft, 0.0)
    draw.line([(bot_l[0], bot_l[1] + 28), (bot_r[0], bot_r[1] + 28)], fill=pal["label"], width=2)
    draw.text((0.5 * (bot_l[0] + bot_r[0]) - 22, bot_l[1] + 34), f"~{int(sample.width_ft)}'-0\"", fill=pal["label"])
    draw.text((max(8.0, bot_l[0] - 58), height * 0.5), f"~{int(sample.depth_ft)}'-0\"", fill=pal["label"])

    if style == "phone_photo_blueprint":
        img = img.rotate(rng.uniform(-3.0, 3.0), resample=Image.Resampling.BILINEAR, fillcolor=pal["bg"])

    arr = np.asarray(img, dtype=np.float32)
    noise = np_rng.normal(0.0, 4.5, size=(height, width, 1)).astype(np.float32)
    if style in {"black_sharpie_napkin", "phone_photo_blueprint"}:
        # Subtle warm desk-lighting gradient across the sheet.
        yy, xx = np.mgrid[0:height, 0:width]
        grad = 1.0 - 0.08 * (((xx - width * 0.45) / width) ** 2 + ((yy - height * 0.45) / height) ** 2)
        arr = arr * grad[..., None]
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(arr, mode="RGB").filter(ImageFilter.SMOOTH_MORE if style == "black_sharpie_napkin" else ImageFilter.DETAIL)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_file)
    return str(out_file)


def sketchify_image(
    input_path: str | Path,
    out_path: str | Path,
    *,
    style: str = "phone_photo_blueprint",
    seed: int = 42,
) -> str:
    """Apply realistic phone-photo tilt, paper grain, and lighting to a MutinyBIM SVG->PNG export."""
    import numpy as np
    from PIL import Image, ImageFilter

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    pal = _style_palette(style)

    img = Image.open(input_path).convert("RGB")
    img = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.Resampling.BILINEAR, fillcolor=pal["bg"])
    img = img.filter(ImageFilter.GaussianBlur(radius=0.45))

    arr = np.asarray(img, dtype=np.float32)
    # Slightly warm paper tint + realistic camera sensor grain.
    tint = np.array([1.0, 0.985, 0.95], dtype=np.float32)
    noise = np_rng.normal(0.0, 5.0, size=(arr.shape[0], arr.shape[1], 1)).astype(np.float32)
    arr = np.clip(arr * tint + noise, 0, 255).astype(np.uint8)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_file)
    return str(out_file)
