"""Passive Solar Dome, Sun-Dial HUD & Direct Sun Hours Engine for MutinyBIM Studio.

Coordinate & physical assumptions (v0.3):
- Z-up right-handed architectural world in metres:
  ``+X = Plan East``, ``+Y = Plan North``, ``+Z = Up``.
- Azimuth convention: ``0° = North``, ``90° = East``, ``180° = South``, ``270° = West``
  (clockwise from True North, ``[0°, 360°)``).
- ``trueNorthDeg`` follows MutinyBIM ``packages/bim-kernel/src/site.ts``:
  degrees clockwise from Plan North (``+Y``) to True North (``trueNorthInPlan = (sin θ, cos θ)``).
- Solar angles use approximate Cooper declination / hour-angle math (~±1°) in **local solar time**
  (not civil clock time). Output is passive-solar direct-sun hours on the workplane, not code or
  energy compliance; glazing frames, mullions, and glass transmittance are not modelled in v0.3.
- MutinyBIM's ``exportIfc`` writes geometry relative to the internal origin ``(0, 0, 0)``
  (storing ``basePoint`` in ``IfcMapConversion``), so IfcOpenShell world metres map directly to
  kernel plan millimetres (``* 1000``).
- Top-level imports stay 100% Python stdlib so unit tests and CLI tools import cleanly
  without ``numpy``, ``torch``, ``matplotlib``, ``PIL``, ``trimesh``, or ``ifcopenshell``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any


CITY_PRESETS: dict[str, dict[str, Any]] = {
    "san_francisco": {"label": "San Francisco, US (37.77°N)", "lat_deg": 37.7749, "lon_deg": -122.4194},
    "new_york": {"label": "New York, US (40.71°N)", "lat_deg": 40.7128, "lon_deg": -74.0060},
    "london": {"label": "London, UK (51.51°N)", "lat_deg": 51.5074, "lon_deg": -0.1278},
    "madrid": {"label": "Madrid, ES (40.42°N)", "lat_deg": 40.4168, "lon_deg": -3.7038},
    "tokyo": {"label": "Tokyo, JP (35.68°N)", "lat_deg": 35.6762, "lon_deg": 139.6503},
    "sydney": {"label": "Sydney, AU (33.87°S)", "lat_deg": -33.8688, "lon_deg": 151.2093},
    "singapore": {"label": "Singapore, SG (1.35°N)", "lat_deg": 1.3521, "lon_deg": 103.8198},
    "cairo": {"label": "Cairo, EG (30.04°N)", "lat_deg": 30.0444, "lon_deg": 31.2357},
    "reykjavik": {"label": "Reykjavik, IS (64.15°N)", "lat_deg": 64.1466, "lon_deg": -21.9426},
    "mexico_city": {"label": "Mexico City, MX (19.43°N)", "lat_deg": 19.4326, "lon_deg": -99.1332},
}


KEY_DATES: dict[str, dict[str, Any]] = {
    "summer_solstice": {"label": "Summer Solstice (Jun 21)", "day_of_year": 172},
    "spring_equinox": {"label": "Spring Equinox (Mar 20)", "day_of_year": 80},
    "autumn_equinox": {"label": "Autumn Equinox (Sep 22)", "day_of_year": 266},
    "winter_solstice": {"label": "Winter Solstice (Dec 21)", "day_of_year": 355},
}


@dataclass(frozen=True)
class SolarPosition:
    """Instantaneous solar geometry in Z-up East/North/Up metres."""

    hour_solar: float
    altitude_deg: float
    azimuth_deg: float
    shadow_len_1m: float | None
    sun_vec: tuple[float, float, float]


def _solar_declination_rad(day_of_year: int) -> float:
    """Cooper's equation returning solar declination in radians."""
    doy = max(1, min(366, int(day_of_year)))
    return math.radians(23.45) * math.sin(2.0 * math.pi * (284 + doy) / 365.0)


def solar_position(
    lat_deg: float,
    day_of_year: int,
    hour_solar: float,
    *,
    true_north_deg: float = 0.0,
) -> SolarPosition:
    """Compute solar altitude, clockwise-from-North azimuth, 1 m pole shadow, and unit vector."""
    lat = math.radians(max(-89.9, min(89.9, float(lat_deg))))
    decl = _solar_declination_rad(day_of_year)
    # Solar hour angle: negative morning, 0 solar noon, positive afternoon.
    ha = math.radians(15.0 * (float(hour_solar) - 12.0))

    sin_alt = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)
    alt_deg = math.degrees(alt)

    cos_alt = math.cos(alt)
    if cos_alt < 1e-7:
        az_true = 180.0 if lat >= decl else 0.0
    else:
        cos_az = (math.sin(decl) - math.sin(lat) * sin_alt) / (math.cos(lat) * cos_alt)
        cos_az = max(-1.0, min(1.0, cos_az))
        az_rad = math.acos(cos_az)
        az_true = math.degrees(az_rad)
        if math.sin(ha) > 0.0:
            az_true = 360.0 - az_true

    az_model_deg = (az_true + float(true_north_deg)) % 360.0
    az_model_rad = math.radians(az_model_deg)
    sun_x = cos_alt * math.sin(az_model_rad)  # Plan +X (East when true_north_deg == 0)
    sun_y = cos_alt * math.cos(az_model_rad)  # Plan +Y (North when true_north_deg == 0)
    sun_z = sin_alt  # Up

    shadow_len: float | None = None
    if alt_deg > 0.5:
        shadow_len = round(1.0 / math.tan(alt), 2)

    return SolarPosition(
        hour_solar=float(hour_solar),
        altitude_deg=round(alt_deg, 2),
        azimuth_deg=round(az_true % 360.0, 2),
        shadow_len_1m=shadow_len,
        sun_vec=(sun_x, sun_y, sun_z),
    )


def _sunrise_sunset_hours(lat_deg: float, day_of_year: int) -> tuple[float, float]:
    """Approximate local solar sunrise and sunset hours (handle polar day/night safely)."""
    lat = math.radians(max(-89.9, min(89.9, float(lat_deg))))
    decl = _solar_declination_rad(day_of_year)
    cos_w0 = -math.tan(lat) * math.tan(decl)
    if cos_w0 <= -1.0:
        return 0.0, 24.0
    if cos_w0 >= 1.0:
        return 12.0, 12.0
    w0_deg = math.degrees(math.acos(cos_w0))
    half_day_h = w0_deg / 15.0
    return max(0.0, 12.0 - half_day_h), min(24.0, 12.0 + half_day_h)


def solar_day_arc(
    lat_deg: float,
    day_of_year: int,
    *,
    step_minutes: int = 30,
    true_north_deg: float = 0.0,
) -> tuple[list[SolarPosition], float, float]:
    """Sample daylight ``SolarPosition`` points across the given day plus ``(sunrise, sunset)``."""
    sunrise_h, sunset_h = _sunrise_sunset_hours(lat_deg, day_of_year)
    step_h = max(5, int(step_minutes)) / 60.0
    positions: list[SolarPosition] = []
    if sunset_h > sunrise_h:
        h = sunrise_h + 0.5 * step_h
        while h < sunset_h:
            pos = solar_position(lat_deg, day_of_year, h, true_north_deg=true_north_deg)
            if pos.altitude_deg > 0.5:
                positions.append(pos)
            h += step_h
    return positions, sunrise_h, sunset_h


def build_scalar_heatmap_overlay(
    cells: list[dict[str, Any]],
    *,
    min_value: float = 0.0,
    max_value: float = 13.0,
    threshold: float | None = 4.0,
    palette: str = "thermal",
    opacity: float = 0.78,
) -> dict[str, Any]:
    """Serialize cell results into MutinyBIM's exact ``ScalarHeatmapOverlay`` JSON schema."""
    return {
        "cells": cells,
        "minValue": float(min_value),
        "maxValue": float(max_value),
        **({"threshold": float(threshold)} if threshold is not None else {}),
        "palette": palette,
        "opacity": float(opacity),
    }


def _fmt_hhmm(hour_val: float) -> str:
    h = int(hour_val) % 24
    m = int(round((hour_val - int(hour_val)) * 60.0))
    if m == 60:
        h = (h + 1) % 24
        m = 0
    return f"{h:02d}:{m:02d}"


def render_sun_dial_svg(
    lat_deg: float,
    day_of_year: int,
    hour_solar: float,
    *,
    city_label: str = "",
    date_label: str = "",
    true_north_deg: float = 0.0,
) -> str:
    """Render a compact dark architectural Sun-Dial HUD SVG matching MutinyBIM's compass widget."""
    positions, sunrise_h, sunset_h = solar_day_arc(
        lat_deg,
        day_of_year,
        step_minutes=20,
        true_north_deg=true_north_deg,
    )
    now_pos = solar_position(lat_deg, day_of_year, hour_solar, true_north_deg=true_north_deg)

    cx, cy, r_outer = 150.0, 120.0, 74.0

    def _proj(pos: SolarPosition) -> tuple[float, float]:
        # Stereographic-like radial projection in True North frame (top = True North).
        r = r_outer * max(0.0, min(1.0, (90.0 - max(0.0, pos.altitude_deg)) / 90.0))
        az = math.radians(pos.azimuth_deg % 360.0)
        return cx + r * math.sin(az), cy - r * math.cos(az)

    arc_pts = " ".join(f"{_proj(p)[0]:.1f},{_proj(p)[1]:.1f}" for p in positions)
    sx, sy = _proj(now_pos) if now_pos.altitude_deg > 0 else (cx, cy + r_outer)
    shadow_txt = f"{now_pos.shadow_len_1m:.2f} m" if now_pos.shadow_len_1m is not None else "Horizon / Night"
    sub_hdr = f"{city_label} · {date_label}".strip(" ·")

    plan_north_svg = ""
    if abs(float(true_north_deg)) > 0.1:
        pn_rad = math.radians(-float(true_north_deg))
        pn_x = cx + r_outer * math.sin(pn_rad)
        pn_y = cy - r_outer * math.cos(pn_rad)
        plan_north_svg = (
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{pn_x:.1f}" y2="{pn_y:.1f}" '
            f'stroke="#38bdf8" stroke-width="1.4" stroke-dasharray="4,3"/>'
            f'<text x="{pn_x:.1f}" y="{pn_y - 4:.1f}" text-anchor="middle" font-size="9" fill="#38bdf8">Plan+Y</text>'
        )

    arc_svg = (
        f'<polyline points="{arc_pts}" fill="none" stroke="#f4b942" stroke-width="2.6" stroke-linecap="round"/>'
        if arc_pts
        else ""
    )
    sun_svg = (
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6.5" fill="#ffd15c" stroke="#fff6d6" stroke-width="1.5"/>'
        if now_pos.altitude_deg > 0
        else f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="#64748b"/>'
    )

    hour_ticks = ""
    for hr in (8, 12, 16):
        hp = solar_position(lat_deg, day_of_year, float(hr), true_north_deg=true_north_deg)
        if hp.altitude_deg > 0.5:
            hx, hy = _proj(hp)
            hour_ticks += (
                f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="2.2" fill="#e2e8f0"/>'
                f'<text x="{hx:.1f}" y="{hy - 7:.1f}" text-anchor="middle" font-size="10" fill="#cbd5e1">{hr:02d}:00</text>'
            )

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="310" viewBox="0 0 300 310" '
        'style="background:#14171c;border-radius:12px;font-family:Inter,system-ui,sans-serif;">'
        f'<text x="150" y="20" text-anchor="middle" font-size="11" fill="#94a3b8">{sub_hdr}</text>'
        f'<circle cx="{cx}" cy="{cy}" r="{r_outer}" fill="#1b2028" stroke="#334155" stroke-width="1.4"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r_outer * 0.66:.1f}" fill="none" stroke="#262f3d" stroke-dasharray="3,3"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r_outer * 0.33:.1f}" fill="none" stroke="#262f3d" stroke-dasharray="3,3"/>'
        f'<text x="{cx}" y="{cy - r_outer - 7}" text-anchor="middle" font-size="12" font-weight="600" fill="#e2e8f0">N</text>'
        f'<text x="{cx}" y="{cy + r_outer + 15}" text-anchor="middle" font-size="12" font-weight="600" fill="#94a3b8">S</text>'
        f'<text x="{cx + r_outer + 11}" y="{cy + 4}" text-anchor="middle" font-size="12" font-weight="600" fill="#94a3b8">E</text>'
        f'<text x="{cx - r_outer - 11}" y="{cy + 4}" text-anchor="middle" font-size="12" font-weight="600" fill="#94a3b8">W</text>'
        f"{plan_north_svg}"
        f"{arc_svg}"
        f"{hour_ticks}"
        f"{sun_svg}"
        f'<text x="150" y="232" text-anchor="middle" font-size="11.5" fill="#e2e8f0">'
        f"Alt {now_pos.altitude_deg:.1f}° · Az {now_pos.azimuth_deg:.1f}° ({_fmt_hhmm(hour_solar)} solar)</text>"
        f'<text x="150" y="252" text-anchor="middle" font-size="11" fill="#cbd5e1">'
        f"1 m pole shadow: {shadow_txt}</text>"
        f'<text x="150" y="272" text-anchor="middle" font-size="11" fill="#94a3b8">'
        f"Sunrise ≈ {_fmt_hhmm(sunrise_h)} · Sunset ≈ {_fmt_hhmm(sunset_h)}</text>"
        '<text x="150" y="292" text-anchor="middle" font-size="9.5" fill="#64748b">'
        f"Solar time · approx. ±1° · True North {float(true_north_deg):+.1f}°</text>"
        "</svg>"
    )


def read_ifc_true_north_deg(ifc_path_or_text: str | Path) -> float:
    """Parse ``TrueNorth`` from an IFC's ``IfcGeometricRepresentationContext`` (degrees CW from Plan +Y)."""
    if isinstance(ifc_path_or_text, Path):
        text = ifc_path_or_text.read_text(encoding="utf-8", errors="replace")
    elif "\n" in ifc_path_or_text or "IFC" in ifc_path_or_text.upper():
        text = ifc_path_or_text
    else:
        p = Path(ifc_path_or_text)
        text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else str(ifc_path_or_text)

    directions: dict[str, tuple[float, float]] = {}
    for m in re.finditer(r"#(\d+)\s*=\s*IFCDIRECTION\(\(\s*([^\)]+?)\s*\)\)", text, re.IGNORECASE):
        parts = [p.strip() for p in m.group(2).split(",")]
        if len(parts) >= 2:
            try:
                directions[m.group(1)] = (float(parts[0]), float(parts[1]))
            except ValueError:
                pass

    model_ref: str | None = None
    fallback_ref: str | None = None
    for m in re.finditer(r"#\d+\s*=\s*IFCGEOMETRICREPRESENTATIONCONTEXT\(([^;]+)\)\s*;", text, re.IGNORECASE):
        args = [a.strip() for a in m.group(1).split(",")]
        if len(args) >= 6 and args[5].startswith("#"):
            ref_id = args[5][1:]
            ctx_type = args[1].strip("'\" ").lower()
            if ctx_type == "model":
                model_ref = ref_id
                break
            if fallback_ref is None:
                fallback_ref = ref_id

    chosen = model_ref or fallback_ref
    if chosen and chosen in directions:
        dx, dy = directions[chosen]
        if math.hypot(dx, dy) > 1e-8:
            deg = (math.degrees(math.atan2(dx, dy)) % 360.0)
            if deg > 180.0:
                deg -= 360.0
            if abs(deg + 180.0) < 1e-6:
                deg = 180.0
            return 0.0 if abs(deg) < 1e-6 else round(deg, 4)
    return 0.0


def compute_passive_solar_reward(
    winter_mean_hours: float,
    summer_mean_hours: float,
    *,
    verify_ok: bool,
    spec_match: float,
    has_closed_envelope: bool,
    max_facade_wwr: float,
    max_wwr_cap: float = 0.40,
    target_winter_hours: float = 4.0,
) -> float:
    """Fail-closed passive-solar exposure reward in ``[0.0, 1.0]``.

    All four structural/lint gates are required keyword arguments; missing or invalid
    inputs either raise ``TypeError`` (when omitted) or fail closed with ``0.0``.
    """
    if verify_ok is not True or has_closed_envelope is not True:
        return 0.0
    if spec_match is None or max_facade_wwr is None:
        return 0.0
    try:
        spec_val = float(spec_match)
        wwr_val = float(max_facade_wwr)
        cap_val = float(max_wwr_cap)
    except (TypeError, ValueError):
        return 0.0
    if (
        not math.isfinite(spec_val)
        or not math.isfinite(wwr_val)
        or not math.isfinite(cap_val)
        or spec_val < 1.0 - 1e-6
        or wwr_val < 0.0
        or wwr_val > cap_val
    ):
        return 0.0

    w = max(0.0, float(winter_mean_hours))
    s = max(0.0, float(summer_mean_hours))
    tgt = max(0.5, float(target_winter_hours))
    winter_score = min(1.0, w / tgt)
    summer_penalty = max(0.0, (s - (tgt * 1.4)) / (tgt * 2.5))
    return round(max(0.0, min(1.0, winter_score - 0.35 * summer_penalty)), 4)


def _split_step_args(arg_str: str) -> list[str]:
    """Split top-level IFC STEP arguments respecting nested parentheses and single quotes."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    in_str = False
    i = 0
    n = len(arg_str)
    while i < n:
        ch = arg_str[i]
        if ch == "'":
            if in_str and i + 1 < n and arg_str[i + 1] == "'":
                cur.append("''")
                i += 2
                continue
            in_str = not in_str
            cur.append(ch)
        elif not in_str:
            if ch == "(":
                depth += 1
                cur.append(ch)
            elif ch == ")":
                depth = max(0, depth - 1)
                cur.append(ch)
            elif ch == "," and depth == 0:
                out.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        else:
            cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail or out:
        out.append(tail)
    return out


def _parse_step_tuple_floats(token: str) -> tuple[float, ...]:
    s = token.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    vals: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p or p == "$":
            continue
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return tuple(vals)


def _parse_step_ref_list(token: str) -> list[str]:
    s = token.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return [s[1:]] if s.startswith("#") else []
    return [p.strip()[1:] for p in s[1:-1].split(",") if p.strip().startswith("#")]


def _unquote_step(token: str) -> str:
    s = token.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    return ""


def _step_ref(token: str) -> str | None:
    s = token.strip()
    return s[1:] if s.startswith("#") and s[1:].isdigit() else None


def _point_in_poly_2d(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / ((y2 - y1) if abs(y2 - y1) > 1e-12 else 1e-12) + x1):
            inside = not inside
    return inside


def _signed_area_2d(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _dist_point_to_seg_2d(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    """Return `(distance, t_clamped)` from point P to segment AB."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    projx, projy = ax + t * dx, ay + t * dy
    return math.hypot(px - projx, py - projy), t


def _split_segments_at_junctions(
    segments: list[tuple[tuple[float, float], tuple[float, float], int]],
    snap_tol: float = 0.30,
) -> list[tuple[tuple[float, float], tuple[float, float], int]]:
    """Snap vertices and split wall segments at T-junctions / crossings so outer loops trace cleanly."""
    # 1. Cluster endpoints within snap_tol.
    reps: list[tuple[float, float]] = []

    def _snap(pt: tuple[float, float]) -> tuple[float, float]:
        for r in reps:
            if math.hypot(pt[0] - r[0], pt[1] - r[1]) <= snap_tol:
                return r
        reps.append(pt)
        return pt

    snapped = [(_snap(a), _snap(b), wid) for a, b, wid in segments]
    # Collect all unique endpoints + segment-segment crossing points.
    split_pts: list[tuple[float, float]] = list(reps)
    for i in range(len(snapped)):
        a1, b1, _ = snapped[i]
        for j in range(i + 1, len(snapped)):
            a2, b2, _ = snapped[j]
            denom = (b1[0] - a1[0]) * (b2[1] - a2[1]) - (b1[1] - a1[1]) * (b2[0] - a2[0])
            if abs(denom) < 1e-9:
                continue
            t = ((a2[0] - a1[0]) * (b2[1] - a2[1]) - (a2[1] - a1[1]) * (b2[0] - a2[0])) / denom
            u = ((a2[0] - a1[0]) * (b1[1] - a1[1]) - (a2[1] - a1[1]) * (b1[0] - a1[0])) / denom
            if 1e-3 < t < 1.0 - 1e-3 and 1e-3 < u < 1.0 - 1e-3:
                split_pts.append(_snap((a1[0] + t * (b1[0] - a1[0]), a1[1] + t * (b1[1] - a1[1]))))

    out: list[tuple[tuple[float, float], tuple[float, float], int]] = []
    for a, b, wid in snapped:
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 0.05:
            continue
        ts: list[tuple[float, tuple[float, float]]] = [(0.0, a), (1.0, b)]
        for p in split_pts:
            if math.hypot(p[0] - a[0], p[1] - a[1]) <= snap_tol or math.hypot(p[0] - b[0], p[1] - b[1]) <= snap_tol:
                continue
            d, t = _dist_point_to_seg_2d(p[0], p[1], a[0], a[1], b[0], b[1])
            if d <= snap_tol and 0.01 < t < 0.99:
                ts.append((t, p))
        ts.sort(key=lambda item: item[0])
        for k in range(len(ts) - 1):
            p0, p1 = ts[k][1], ts[k + 1][1]
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= 0.05:
                out.append((p0, p1, wid))
    return out


def _trace_outer_loop(
    segments: list[tuple[tuple[float, float], tuple[float, float], int]],
) -> tuple[bool, list[tuple[float, float]], set[int]]:
    """Trace the CCW outer boundary of a 2D wall graph via planar half-edge face traversal.

    Returns ``(is_clean_closed_loop, outer_polygon_ccw, boundary_wall_ids)``.
    Any missing exterior wall or dangling exterior spur causes at least one bridge
    half-edge pair ``(u -> v)`` and ``(v -> u)`` to both appear in the outer walk,
    failing the closed-loop check even when an interior room loop exists inside.
    """
    if len(segments) < 3:
        return False, [], set()

    adj: dict[tuple[float, float], set[tuple[float, float]]] = {}
    edge_wids: dict[tuple[tuple[float, float], tuple[float, float]], set[int]] = {}
    for a, b, wid in segments:
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        edge_wids.setdefault((a, b), set()).add(wid)
        edge_wids.setdefault((b, a), set()).add(wid)

    if not adj:
        return False, [], set()

    # Start at bottom-most (then left-most) vertex so exterior sits on the right of min-angle outgoing edge.
    v0 = min(adj.keys(), key=lambda p: (p[1], p[0]))
    nbrs0 = list(adj[v0])
    if not nbrs0:
        return False, [], set()
    v1 = min(nbrs0, key=lambda q: math.atan2(q[1] - v0[1], q[0] - v0[0]) % (2.0 * math.pi))

    walk_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    poly: list[tuple[float, float]] = [v0]
    curr_u, curr_v = v0, v1
    max_steps = max(16, len(segments) * 4 + 4)

    for _ in range(max_steps):
        walk_edges.append((curr_u, curr_v))
        poly.append(curr_v)
        nbrs = list(adj.get(curr_v, ()))
        if not nbrs:
            return False, [], set()
        theta_back = math.atan2(curr_u[1] - curr_v[1], curr_u[0] - curr_v[0])
        best_next = curr_u
        best_delta = 10.0 * math.pi
        for nxt in nbrs:
            if nxt == curr_u and len(nbrs) > 1:
                continue
            theta_out = math.atan2(nxt[1] - curr_v[1], nxt[0] - curr_v[0])
            delta = (theta_out - theta_back) % (2.0 * math.pi)
            if delta < 1e-7:
                delta = 2.0 * math.pi
            if delta < best_delta:
                best_delta = delta
                best_next = nxt
        curr_u, curr_v = curr_v, best_next
        if curr_u == v0 and curr_v == v1:
            break
    else:
        return False, [], set()

    poly = poly[:-1]
    directed = set(walk_edges)
    # Any reversed pair means the outer walk crossed a bridge or open wall segment.
    has_bridge = any((b, a) in directed for a, b in directed)
    area = _signed_area_2d(poly)
    if has_bridge or area <= 1.0:
        return False, [], set()

    b_wids: set[int] = set()
    for a, b in walk_edges:
        b_wids.update(edge_wids.get((a, b), ()))
    return True, poly, b_wids


def _facade_bucket_from_true_bearing(bearing_deg: float) -> str:
    b = bearing_deg % 360.0
    if b < 45.0 or b >= 315.0:
        return "N"
    if b < 135.0:
        return "E"
    if b < 225.0:
        return "S"
    return "W"


def _extract_model_elements_from_ifc(ifc_text: str) -> dict[str, Any]:
    """Pure-Python IFC 4.3 STEP extractor for envelope walls, windows, slabs, roofs, and true north."""
    entities: dict[str, tuple[str, list[str]]] = {}
    for m in re.finditer(r"#(\d+)\s*=\s*([A-Z0-9_]+)\((.*)\)\s*;", ifc_text, re.IGNORECASE):
        entities[m.group(1)] = (m.group(2).upper(), _split_step_args(m.group(3)))

    # Unit scale to metres (MutinyBIM writes millimetres via IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.)).
    unit_scale = 1.0
    for _, (etype, args) in entities.items():
        if etype == "IFCSIUNIT" and len(args) >= 4 and ".LENGTHUNIT." in args[1].upper():
            if ".MILLI." in args[2].upper():
                unit_scale = 0.001
            break

    true_north_deg = read_ifc_true_north_deg(ifc_text)

    def _point3(ref_id: str | None) -> tuple[float, float, float]:
        if not ref_id or ref_id not in entities:
            return (0.0, 0.0, 0.0)
        et, args = entities[ref_id]
        if et == "IFCCARTESIANPOINT" and args:
            vals = _parse_step_tuple_floats(args[0])
            x = (vals[0] if len(vals) > 0 else 0.0) * unit_scale
            y = (vals[1] if len(vals) > 1 else 0.0) * unit_scale
            z = (vals[2] if len(vals) > 2 else 0.0) * unit_scale
            return (x, y, z)
        return (0.0, 0.0, 0.0)

    def _dir2(ref_id: str | None) -> tuple[float, float]:
        if not ref_id or ref_id not in entities:
            return (1.0, 0.0)
        et, args = entities[ref_id]
        if et == "IFCDIRECTION" and args:
            vals = _parse_step_tuple_floats(args[0])
            dx = vals[0] if len(vals) > 0 else 1.0
            dy = vals[1] if len(vals) > 1 else 0.0
            mag = math.hypot(dx, dy)
            if mag > 1e-8:
                return (dx / mag, dy / mag)
        return (1.0, 0.0)

    def _axis2_3d(ref_id: str | None) -> tuple[tuple[float, float, float], tuple[float, float]]:
        if not ref_id or ref_id not in entities:
            return (0.0, 0.0, 0.0), (1.0, 0.0)
        et, args = entities[ref_id]
        if et == "IFCAXIS2PLACEMENT3D":
            loc = _point3(_step_ref(args[0])) if len(args) > 0 else (0.0, 0.0, 0.0)
            xdir = _dir2(_step_ref(args[2])) if len(args) > 2 else (1.0, 0.0)
            return loc, xdir
        return (0.0, 0.0, 0.0), (1.0, 0.0)

    placement_cache: dict[str, tuple[tuple[float, float, float], tuple[float, float]]] = {}

    def _world_placement(ref_id: str | None) -> tuple[tuple[float, float, float], tuple[float, float]]:
        if not ref_id or ref_id not in entities:
            return (0.0, 0.0, 0.0), (1.0, 0.0)
        if ref_id in placement_cache:
            return placement_cache[ref_id]
        et, args = entities[ref_id]
        if et != "IFCLOCALPLACEMENT":
            return (0.0, 0.0, 0.0), (1.0, 0.0)
        parent_ref = _step_ref(args[0]) if len(args) > 0 else None
        rel_ref = _step_ref(args[1]) if len(args) > 1 else None
        (px, py, pz), (pdx, pdy) = _world_placement(parent_ref) if parent_ref else ((0.0, 0.0, 0.0), (1.0, 0.0))
        (lx, ly, lz), (rdx, rdy) = _axis2_3d(rel_ref)
        wx = px + lx * pdx - ly * pdy
        wy = py + lx * pdy + ly * pdx
        wz = pz + lz
        wdx = pdx * rdx - pdy * rdy
        wdy = pdx * rdy + pdy * rdx
        res = ((wx, wy, wz), (wdx, wdy))
        placement_cache[ref_id] = res
        return res

    # Collect external wall hints from IFCRELSPACEBOUNDARY and Pset_WallCommon IsExternal.
    external_hints: set[str] = set()
    opening_to_host: dict[str, str] = {}
    window_to_opening: dict[str, str] = {}
    pset_external: set[str] = set()

    for _, (et, args) in entities.items():
        if et == "IFCRELSPACEBOUNDARY" and len(args) >= 9:
            el_ref = _step_ref(args[5])
            if el_ref and ".EXTERNAL." in args[8].upper():
                external_hints.add(el_ref)
        elif et == "IFCRELVOIDSELEMENT" and len(args) >= 6:
            host_ref, op_ref = _step_ref(args[4]), _step_ref(args[5])
            if host_ref and op_ref:
                opening_to_host[op_ref] = host_ref
        elif et == "IFCRELFILLSELEMENT" and len(args) >= 6:
            op_ref, fill_ref = _step_ref(args[4]), _step_ref(args[5])
            if op_ref and fill_ref:
                window_to_opening[fill_ref] = op_ref
        elif et == "IFCPROPERTYSET" and len(args) >= 5:
            for p_ref in _parse_step_ref_list(args[4]):
                p_ent = entities.get(p_ref)
                if p_ent and p_ent[0] == "IFCPROPERTYSINGLEVALUE" and len(p_ent[1]) >= 3:
                    if _unquote_step(p_ent[1][0]).lower() == "isexternal" and ".T." in p_ent[1][2].upper():
                        pset_external.add(entities[_][1][0] if False else _)

    if pset_external:
        for _, (et, args) in entities.items():
            if et == "IFCRELDEFINESBYPROPERTIES" and len(args) >= 6:
                prop_ref = _step_ref(args[5])
                if prop_ref in pset_external:
                    for obj_ref in _parse_step_ref_list(args[4]):
                        external_hints.add(obj_ref)

    def _extruded_solids(rep_ref: str | None) -> list[tuple[str, list[str], tuple[float, float, float], float]]:
        out: list[tuple[str, list[str], tuple[float, float, float], float]] = []
        if not rep_ref or rep_ref not in entities:
            return out
        et, args = entities[rep_ref]
        if et != "IFCPRODUCTDEFINITIONSHAPE" or len(args) < 3:
            return out
        for srep_id in _parse_step_ref_list(args[2]):
            srep = entities.get(srep_id)
            if not srep or srep[0] != "IFCSHAPEREPRESENTATION" or len(srep[1]) < 4:
                continue
            for item_id in _parse_step_ref_list(srep[1][3]):
                item = entities.get(item_id)
                if item and item[0] == "IFCEXTRUDEDAREASOLID" and len(item[1]) >= 4:
                    prof_ref = _step_ref(item[1][0])
                    pos_ref = _step_ref(item[1][1])
                    try:
                        depth = float(item[1][3]) * unit_scale
                    except ValueError:
                        depth = 0.0
                    pos_loc, _ = _axis2_3d(pos_ref)
                    if prof_ref and prof_ref in entities:
                        out.append((entities[prof_ref][0], entities[prof_ref][1], pos_loc, depth))
        return out

    walls: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    roofs: list[dict[str, Any]] = []
    slabs: list[dict[str, Any]] = []

    for eid, (et, args) in entities.items():
        if et in ("IFCWALL", "IFCWALLSTANDARDCASE", "IFCCURTAINWALL") and len(args) >= 7:
            (wx, wy, wz), (wdx, wdy) = _world_placement(_step_ref(args[5]))
            solids = _extruded_solids(_step_ref(args[6]))
            length_m = 0.0
            height_m = 3.0
            cx_local = 0.0
            cy_local = 0.0
            for prof_et, prof_args, pos_loc, depth in solids:
                if depth > 0.0:
                    height_m = max(height_m if len(solids) > 1 else 0.0, depth)
                if prof_et == "IFCRECTANGLEPROFILEDEF" and len(prof_args) >= 5:
                    try:
                        xdim = float(prof_args[3]) * unit_scale
                        length_m = max(length_m, xdim)
                        pos2d = entities.get(_step_ref(prof_args[2]) or "")
                        if pos2d and pos2d[0] == "IFCAXIS2PLACEMENT2D" and len(pos2d[1]) >= 1:
                            c = _point3(_step_ref(pos2d[1][0]))
                            cx_local, cy_local = c[0], c[1]
                    except ValueError:
                        pass
            if length_m <= 0.05:
                continue
            x0_loc = cx_local - 0.5 * length_m
            x1_loc = cx_local + 0.5 * length_m
            start = (wx + x0_loc * wdx - cy_local * wdy, wy + x0_loc * wdy + cy_local * wdx)
            end = (wx + x1_loc * wdx - cy_local * wdy, wy + x1_loc * wdy + cy_local * wdx)
            label_text = (_unquote_step(args[2]) + " " + _unquote_step(args[4])).lower()
            is_ext = eid in external_hints or et == "IFCCURTAINWALL" or "exterior" in label_text or "curtain" in label_text
            walls.append(
                {
                    "id": eid,
                    "start": start,
                    "end": end,
                    "base_z_m": wz,
                    "height_m": height_m,
                    "top_z_m": wz + height_m,
                    "length_m": length_m,
                    "is_external": is_ext,
                }
            )
        elif et == "IFCWINDOW" and len(args) >= 10:
            (wx, wy, wz), _ = _world_placement(_step_ref(args[5]))
            w_h = 0.0
            w_w = 0.0
            try:
                if args[8] != "$":
                    w_h = float(args[8]) * unit_scale
                if args[9] != "$":
                    w_w = float(args[9]) * unit_scale
            except ValueError:
                pass
            if w_w <= 0.0 or w_h <= 0.0:
                for prof_et, prof_args, _, depth in _extruded_solids(_step_ref(args[6])):
                    if prof_et == "IFCRECTANGLEPROFILEDEF" and len(prof_args) >= 5:
                        try:
                            w_w = w_w or float(prof_args[3]) * unit_scale
                            w_h = w_h or depth
                        except ValueError:
                            pass
            op_id = window_to_opening.get(eid)
            host_wall_id = opening_to_host.get(op_id) if op_id else None
            windows.append(
                {
                    "id": eid,
                    "pos": (wx, wy),
                    "base_z_m": wz,
                    "width_m": max(0.0, w_w),
                    "height_m": max(0.0, w_h),
                    "area_m2": max(0.0, w_w * w_h),
                    "host_wall_id": host_wall_id,
                }
            )
        elif et == "IFCROOF":
            (_, _, rz), _ = _world_placement(_step_ref(args[5]) if len(args) > 5 else None)
            roofs.append({"id": eid, "top_z_m": rz})
        elif et == "IFCSLAB" and len(args) >= 7:
            (_, _, sz), _ = _world_placement(_step_ref(args[5]))
            pt = args[8].upper() if len(args) > 8 else ""
            max_top = sz
            for _, _, pos_loc, depth in _extruded_solids(_step_ref(args[6])):
                max_top = max(max_top, sz + pos_loc[2] + depth)
            slabs.append({"id": eid, "base_z_m": sz, "top_z_m": max_top, "is_roof_type": ".ROOF." in pt})

    return {
        "walls": walls,
        "windows": windows,
        "roofs": roofs,
        "slabs": slabs,
        "true_north_deg": true_north_deg,
    }


def envelope_gates(
    model_or_ifc: dict[str, Any] | str | Path,
    *,
    true_north_deg: float | None = None,
) -> dict[str, Any]:
    """Compute ``has_closed_envelope`` and True-North-aware ``max_facade_wwr`` from a model or IFC."""
    if isinstance(model_or_ifc, dict):
        data = model_or_ifc
        resolved_tn = float(data.get("true_north_deg", 0.0) if true_north_deg is None else true_north_deg)
    else:
        if isinstance(model_or_ifc, Path):
            ifc_text = model_or_ifc.read_text(encoding="utf-8", errors="replace")
        elif "\n" in model_or_ifc or "IFC" in model_or_ifc.upper():
            ifc_text = model_or_ifc
        else:
            p = Path(model_or_ifc)
            ifc_text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else str(model_or_ifc)
        data = _extract_model_elements_from_ifc(ifc_text)
        resolved_tn = float(data["true_north_deg"] if true_north_deg is None else true_north_deg)

    raw_walls = data.get("walls") or []
    raw_windows = data.get("windows") or []
    raw_roofs = data.get("roofs") or []
    raw_slabs = data.get("slabs") or []

    walls: list[dict[str, Any]] = []
    wall_by_id: dict[Any, dict[str, Any]] = {}
    for idx, w in enumerate(raw_walls):
        s = (float(w["start"][0]), float(w["start"][1]))
        e = (float(w["end"][0]), float(w["end"][1]))
        length_m = float(w.get("length_m") or math.hypot(e[0] - s[0], e[1] - s[1]))
        if length_m < 0.05:
            continue
        base_z = float(w.get("base_z_m", 0.0))
        height_m = float(w.get("height_m", 3.0))
        wid = w.get("id", f"w_{idx}")
        entry = {
            "idx": idx,
            "id": wid,
            "start": s,
            "end": e,
            "length_m": length_m,
            "base_z_m": base_z,
            "height_m": height_m,
            "top_z_m": float(w.get("top_z_m", base_z + height_m)),
            "is_external": bool(w.get("is_external", False)),
            "storey_key": round(base_z * 2.0) / 2.0,
        }
        walls.append(entry)
        wall_by_id[wid] = entry

    if not walls:
        return {
            "has_closed_envelope": False,
            "storeys_closed": False,
            "has_roof_or_top_slab": bool(raw_roofs),
            "max_facade_wwr": 0.0,
            "facade_wwr": {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0},
            "true_north_deg": resolved_tn,
            "exterior_wall_count": 0,
            "window_count": len(raw_windows),
            "storey_count": 0,
        }

    storeys: dict[float, list[dict[str, Any]]] = {}
    for w in walls:
        storeys.setdefault(w["storey_key"], []).append(w)

    storeys_closed = True
    exterior_wall_idxs: set[int] = set()
    outer_poly_by_storey: dict[float, list[tuple[float, float]]] = {}

    for s_key, s_walls in storeys.items():
        hinted = [w for w in s_walls if w["is_external"]]
        candidate_walls = hinted if len(hinted) >= 3 else s_walls
        segs = [(w["start"], w["end"], w["idx"]) for w in candidate_walls]
        split_segs = _split_segments_at_junctions(segs)
        closed, poly, b_idxs = _trace_outer_loop(split_segs)
        if not closed:
            storeys_closed = False
        else:
            outer_poly_by_storey[s_key] = poly
            exterior_wall_idxs.update(b_idxs)

    max_wall_top_z = max(w["top_z_m"] for w in walls)
    top_storey_base_z = max(storeys.keys())
    has_roof_or_top_slab = bool(raw_roofs) or any(bool(s.get("is_roof_type", False)) for s in raw_slabs)
    if not has_roof_or_top_slab:
        for s in raw_slabs:
            top_z = float(s.get("top_z_m", s.get("base_z_m", 0.0)))
            if top_z >= top_storey_base_z + 1.0 and abs(top_z - max_wall_top_z) <= 0.65:
                has_roof_or_top_slab = True
                break

    has_closed_envelope = bool(storeys_closed and has_roof_or_top_slab)

    # Determine outward normal & True North cardinal facade for every exterior wall.
    wall_facade: dict[int, str] = {}
    facade_wall_area: dict[str, float] = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
    facade_win_area: dict[str, float] = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}

    for idx in sorted(exterior_wall_idxs):
        w = walls[idx]
        s, e = w["start"], w["end"]
        dx, dy = e[0] - s[0], e[1] - s[1]
        mag = max(math.hypot(dx, dy), 1e-8)
        ux, uy = dx / mag, dy / mag
        n_left = (-uy, ux)
        n_right = (uy, -ux)
        mid = (0.5 * (s[0] + e[0]), 0.5 * (s[1] + e[1]))
        poly = outer_poly_by_storey.get(w["storey_key"])
        if poly:
            in_left = _point_in_poly_2d(mid[0] + 0.25 * n_left[0], mid[1] + 0.25 * n_left[1], poly)
            out_n = n_right if in_left else n_left
        else:
            s_walls = storeys[w["storey_key"]]
            cx = sum(0.5 * (sw["start"][0] + sw["end"][0]) for sw in s_walls) / len(s_walls)
            cy = sum(0.5 * (sw["start"][1] + sw["end"][1]) for sw in s_walls) / len(s_walls)
            to_wall = (mid[0] - cx, mid[1] - cy)
            out_n = n_left if (n_left[0] * to_wall[0] + n_left[1] * to_wall[1]) >= 0.0 else n_right

        plan_bearing = math.degrees(math.atan2(out_n[0], out_n[1])) % 360.0
        true_bearing = (plan_bearing - resolved_tn) % 360.0
        bucket = _facade_bucket_from_true_bearing(true_bearing)
        wall_facade[idx] = bucket
        facade_wall_area[bucket] += w["length_m"] * w["height_m"]

    # Map windows onto host exterior wall facades.
    for win in raw_windows:
        area = float(win.get("area_m2") or (float(win.get("width_m", 0.0)) * float(win.get("height_m", 0.0))))
        if area <= 0.0:
            continue
        target_idx: int | None = None
        host_id = win.get("host_wall_id")
        if host_id is not None and host_id in wall_by_id:
            cand_idx = wall_by_id[host_id]["idx"]
            if cand_idx in wall_facade:
                target_idx = cand_idx
        elif win.get("pos") is not None:
            wx, wy = float(win["pos"][0]), float(win["pos"][1])
            wz = float(win.get("base_z_m", 0.0))
            best_d = 0.65
            for idx in wall_facade:
                w = walls[idx]
                if abs(w["storey_key"] - round(wz * 2.0) / 2.0) > 1.0 and wz > 0.0:
                    continue
                d, _ = _dist_point_to_seg_2d(wx, wy, w["start"][0], w["start"][1], w["end"][0], w["end"][1])
                if d < best_d:
                    best_d = d
                    target_idx = idx
        if target_idx is not None and target_idx in wall_facade:
            facade_win_area[wall_facade[target_idx]] += area

    facade_wwr = {
        k: round(min(1.0, facade_win_area[k] / facade_wall_area[k]), 4) if facade_wall_area[k] > 1e-6 else 0.0
        for k in ("N", "E", "S", "W")
    }
    max_facade_wwr = max(facade_wwr.values()) if facade_wwr else 0.0

    return {
        "has_closed_envelope": has_closed_envelope,
        "storeys_closed": storeys_closed,
        "has_roof_or_top_slab": has_roof_or_top_slab,
        "max_facade_wwr": max_facade_wwr,
        "facade_wwr": facade_wwr,
        "true_north_deg": resolved_tn,
        "exterior_wall_count": len(exterior_wall_idxs),
        "window_count": len(raw_windows),
        "storey_count": len(storeys),
    }


def _sub3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _ray_triangle_any_hit_pure(
    origins: list[tuple[float, float, float]],
    direction: tuple[float, float, float],
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    min_t: float = 0.02,
) -> list[bool]:
    """Zero-dependency pure-Python Möller–Trumbore ray-triangle occlusion test."""
    prepped = []
    for p0, p1, p2 in triangles:
        e1 = _sub3(p1, p0)
        e2 = _sub3(p2, p0)
        pvec = _cross3(direction, e2)
        det = _dot3(e1, pvec)
        if abs(det) > 1e-7:
            prepped.append((p0, e1, e2, pvec, 1.0 / det))

    out: list[bool] = []
    for ori in origins:
        hit = False
        for p0, e1, e2, pvec, inv_det in prepped:
            tvec = _sub3(ori, p0)
            u = _dot3(tvec, pvec) * inv_det
            if u < 0.0 or u > 1.0:
                continue
            qvec = _cross3(tvec, e1)
            v = _dot3(qvec, direction) * inv_det
            if v < 0.0 or u + v > 1.0 + 1e-5:
                continue
            t = _dot3(qvec, e2) * inv_det
            if t > min_t:
                hit = True
                break
        out.append(hit)
    return out


def compute_direct_sun_hours_for_points(
    points: list[tuple[float, float, float]],
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    *,
    lat_deg: float,
    day_of_year: int,
    true_north_deg: float = 0.0,
    step_minutes: int = 30,
) -> list[float]:
    """Compute direct sun hours at explicit 3D points against triangle soup (GPU/NumPy or stdlib)."""
    positions, _, _ = solar_day_arc(
        lat_deg,
        day_of_year,
        step_minutes=step_minutes,
        true_north_deg=true_north_deg,
    )
    hours_per_step = max(5, int(step_minutes)) / 60.0
    totals = [0.0 for _ in points]
    if not points or not positions:
        return totals

    try:
        import numpy as np  # type: ignore

        ori_arr = np.asarray(points, dtype=np.float32)
        if triangles:
            tri_arr = np.asarray(triangles, dtype=np.float32)
            v0 = tri_arr[:, 0, :]
            e1 = tri_arr[:, 1, :] - v0
            e2 = tri_arr[:, 2, :] - v0
        else:
            v0 = e1 = e2 = np.zeros((0, 3), dtype=np.float32)

        for pos in positions:
            blocked = _ray_triangle_any_hit(ori_arr, pos.sun_vec, v0, e1, e2)
            for idx, is_blk in enumerate(blocked):
                if not bool(is_blk):
                    totals[idx] += hours_per_step
        return [round(val, 2) for val in totals]
    except ImportError:
        for pos in positions:
            blocked = _ray_triangle_any_hit_pure(points, pos.sun_vec, triangles)
            for idx, is_blk in enumerate(blocked):
                if not is_blk:
                    totals[idx] += hours_per_step
        return [round(val, 2) for val in totals]


def _ray_triangle_any_hit_numpy(
    origins: Any,
    direction: Any,
    v0: Any,
    e1: Any,
    e2: Any,
    min_t: float = 0.02,
) -> Any:
    """Vectorized NumPy Möller–Trumbore test: returns bool `(N_rays,)` indicating blocked rays."""
    import numpy as np

    d = np.asarray(direction, dtype=np.float32)
    n_rays = len(origins)
    n_tris = len(v0)
    if n_rays == 0 or n_tris == 0:
        return np.zeros((n_rays,), dtype=bool)

    pvec = np.cross(d[None, :], e2)  # (T, 3)
    det = np.einsum("ij,ij->i", e1, pvec)  # (T,)
    valid_tri = np.abs(det) > 1e-7
    if not np.any(valid_tri):
        return np.zeros((n_rays,), dtype=bool)

    v0_v = v0[valid_tri]
    e1_v = e1[valid_tri]
    e2_v = e2[valid_tri]
    pvec_v = pvec[valid_tri]
    inv_det = (1.0 / det[valid_tri])[:, None]  # (T_v, 1)

    qvec_part = np.cross(e1_v, pvec_v)  # unused placeholder avoided below
    del qvec_part

    blocked = np.zeros((n_rays,), dtype=bool)
    chunk = 256
    for start in range(0, n_rays, chunk):
        end = min(start + chunk, n_rays)
        ori = origins[start:end]  # (C, 3)
        tvec = ori[None, :, :] - v0_v[:, None, :]  # (T_v, C, 3)
        u = np.einsum("tck,tk->tc", tvec, pvec_v) * inv_det
        in_u = (u >= 0.0) & (u <= 1.0)
        qvec = np.cross(tvec, e1_v[:, None, :])  # (T_v, C, 3)
        v = np.einsum("tck,k->tc", qvec, d) * inv_det
        in_tri = in_u & (v >= 0.0) & (u + v <= 1.0 + 1e-5)
        t = np.einsum("tck,tk->tc", qvec, e2_v) * inv_det
        hit = in_tri & (t > min_t)
        blocked[start:end] = np.any(hit, axis=0)
    return blocked


def _ray_triangle_any_hit(
    origins: Any,
    direction: tuple[float, float, float],
    v0: Any,
    e1: Any,
    e2: Any,
    min_t: float = 0.02,
) -> Any:
    """Use CUDA PyTorch when available, otherwise NumPy vectorized raycaster."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and len(v0) > 0:
            device = torch.device("cuda")
            o_t = torch.as_tensor(origins, dtype=torch.float32, device=device)
            d_t = torch.as_tensor(direction, dtype=torch.float32, device=device)
            v0_t = torch.as_tensor(v0, dtype=torch.float32, device=device)
            e1_t = torch.as_tensor(e1, dtype=torch.float32, device=device)
            e2_t = torch.as_tensor(e2, dtype=torch.float32, device=device)
            pvec = torch.linalg.cross(d_t.unsqueeze(0).expand_as(e2_t), e2_t, dim=-1)
            det = (e1_t * pvec).sum(dim=-1)
            mask = det.abs() > 1e-7
            if not bool(mask.any()):
                return [False] * len(origins)
            v0_t = v0_t[mask]
            e1_t = e1_t[mask]
            e2_t = e2_t[mask]
            pvec = pvec[mask]
            inv_det = (1.0 / det[mask]).unsqueeze(1)
            blocked = torch.zeros((o_t.shape[0],), dtype=torch.bool, device=device)
            for start in range(0, o_t.shape[0], 512):
                ori = o_t[start : start + 512]
                tvec = ori.unsqueeze(0) - v0_t.unsqueeze(1)
                u = (tvec * pvec.unsqueeze(1)).sum(dim=-1) * inv_det
                qvec = torch.linalg.cross(tvec, e1_t.unsqueeze(1), dim=-1)
                v = (qvec * d_t.view(1, 1, 3)).sum(dim=-1) * inv_det
                t = (qvec * e2_t.unsqueeze(1)).sum(dim=-1) * inv_det
                hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0 + 1e-5) & (t > min_t)
                blocked[start : start + 512] = hit.any(dim=0)
            return blocked.cpu().numpy()
    except Exception:
        pass
    return _ray_triangle_any_hit_numpy(origins, direction, v0, e1, e2, min_t=min_t)


def compute_direct_sun_hours(
    ifc_path: str | Path,
    *,
    lat_deg: float,
    day_of_year: int,
    true_north_deg: float | None = None,
    step_minutes: int = 30,
    grid_spacing_m: float = 1.0,
    pad_m: float = 4.0,
    workplane_offset_m: float = 0.85,
    storey_index: int = 0,
) -> dict[str, Any]:
    """Raytrace direct sun hours across a workplane grid for an IFC 4.3 model.

    If ``true_north_deg`` is ``None``, reads ``TrueNorth`` from the IFC geometric context.
    Note: glazing, frames, mullions, and transmittance are not modelled in v0.3.
    """
    import numpy as np

    from .studio_render import _load_ifc_scene_and_meshes

    resolved_tn = read_ifc_true_north_deg(ifc_path) if true_north_deg is None else float(true_north_deg)
    _, labeled_meshes, storey_elevations_m = _load_ifc_scene_and_meshes(ifc_path)
    opaque = [
        mesh
        for ifc_cls, mesh, _ in labeled_meshes
        if not ifc_cls.startswith("IfcWindow")
    ]
    if not opaque:
        opaque = [mesh for _, mesh, _ in labeled_meshes]
    if not opaque:
        raise ValueError(f"No IFC geometry found in {ifc_path}")

    all_verts = np.vstack([m.vertices for m in opaque])
    vmin = all_verts.min(axis=0)
    vmax = all_verts.max(axis=0)

    if storey_elevations_m:
        s_idx = min(max(0, storey_index), len(storey_elevations_m) - 1)
        workplane_z = float(storey_elevations_m[s_idx]) + float(workplane_offset_m)
    else:
        workplane_z = float(vmin[2]) + float(workplane_offset_m)

    xs = np.arange(float(vmin[0]) - pad_m, float(vmax[0]) + pad_m + 1e-4, float(grid_spacing_m))
    ys = np.arange(float(vmin[1]) - pad_m, float(vmax[1]) + pad_m + 1e-4, float(grid_spacing_m))
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    pts_x = gx.ravel()
    pts_y = gy.ravel()
    origins = np.stack([pts_x, pts_y, np.full_like(pts_x, workplane_z)], axis=-1).astype(np.float32)

    tri_v0_list = []
    tri_e1_list = []
    tri_e2_list = []
    for m in opaque:
        verts = np.asarray(m.vertices, dtype=np.float32)
        faces = np.asarray(m.faces, dtype=np.int64)
        if len(faces) == 0:
            continue
        p0 = verts[faces[:, 0]]
        tri_v0_list.append(p0)
        tri_e1_list.append(verts[faces[:, 1]] - p0)
        tri_e2_list.append(verts[faces[:, 2]] - p0)

    v0 = np.vstack(tri_v0_list)
    e1 = np.vstack(tri_e1_list)
    e2 = np.vstack(tri_e2_list)

    positions, sunrise_h, sunset_h = solar_day_arc(
        lat_deg,
        day_of_year,
        step_minutes=step_minutes,
        true_north_deg=resolved_tn,
    )
    hours_per_step = max(5, int(step_minutes)) / 60.0
    sun_hours = np.zeros((len(origins),), dtype=np.float32)

    for pos in positions:
        blocked = np.asarray(_ray_triangle_any_hit(origins, pos.sun_vec, v0, e1, e2), dtype=bool)
        sun_hours[~blocked] += hours_per_step

    cell_mm = int(round(float(grid_spacing_m) * 1000.0))
    z_mm = int(round(workplane_z * 1000.0))
    cells: list[dict[str, Any]] = []
    for px, py, val in zip(pts_x.tolist(), pts_y.tolist(), sun_hours.tolist()):
        v_round = round(float(val), 2)
        cells.append(
            {
                "xMm": int(round(px * 1000.0)),
                "yMm": int(round(py * 1000.0)),
                "zMm": z_mm,
                "widthMm": cell_mm,
                "depthMm": cell_mm,
                "value": v_round,
                "fail": v_round < 2.0,
            }
        )

    max_possible = max(round(len(positions) * hours_per_step, 2), 1.0)
    overlay = build_scalar_heatmap_overlay(
        cells,
        min_value=0.0,
        max_value=max_possible,
        threshold=4.0,
        palette="thermal",
        opacity=0.78,
    )

    return {
        "cells": cells,
        "grid_shape": (len(ys), len(xs)),
        "grid_xs": xs.tolist(),
        "grid_ys": ys.tolist(),
        "sun_hours": sun_hours.reshape(len(ys), len(xs)).tolist(),
        "sun_positions": positions,
        "sunrise_hour": sunrise_h,
        "sunset_hour": sunset_h,
        "true_north_deg": resolved_tn,
        "min_hours": round(float(sun_hours.min()), 2),
        "max_hours": round(float(sun_hours.max()), 2),
        "mean_hours": round(float(sun_hours.mean()), 2),
        "pct_over_4h": round(100.0 * float(np.mean(sun_hours >= 4.0)), 1),
        "workplane_z_m": round(workplane_z, 3),
        "building_bounds_m": (vmin.tolist(), vmax.tolist()),
        "labeled_meshes": labeled_meshes,
        "overlay": overlay,
        "assumptions": (
            "Direct sun reaching the floor/workplane through openings (glazing, frames, mullions, and transmittance "
            "not modelled); approximate solar geometry (±1°) in local solar time. Heatmap JSON matches MutinyBIM "
            "ScalarHeatmapOverlay format (requires an editor contribution calling setScalarHeatmap to display)."
        ),
    }


def render_solar_dome_png(
    solar_result: dict[str, Any],
    out_png_path: str | Path,
    *,
    lat_deg: float,
    day_of_year: int,
    true_north_deg: float | None = None,
    title: str = "Direct Sun Hours",
) -> str:
    """Render a Ladybug-style 3D Solar Dome + Direct Sun Hours thermal heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    fig = plt.figure(figsize=(10.5, 7.8), dpi=140, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    vmin, vmax = solar_result["building_bounds_m"]
    vmin_a = np.asarray(vmin, dtype=np.float64)
    vmax_a = np.asarray(vmax, dtype=np.float64)
    cx = 0.5 * (vmin_a[0] + vmax_a[0])
    cy = 0.5 * (vmin_a[1] + vmax_a[1])
    cz = float(vmin_a[2])
    span = float(np.linalg.norm(vmax_a[:2] - vmin_a[:2]))
    dome_r = max(14.0, 0.95 * span + 5.0)

    # 1. Thermal heatmap grid on workplane.
    xs = np.asarray(solar_result["grid_xs"], dtype=np.float64)
    ys = np.asarray(solar_result["grid_ys"], dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    hours = np.asarray(solar_result["sun_hours"], dtype=np.float64)
    gz = np.full_like(gx, float(solar_result["workplane_z_m"]))
    cmap = LinearSegmentedColormap.from_list(
        "ladybug",
        ["#1d4ed8", "#7c3aed", "#dc2626", "#f59e0b", "#fef08a"],
    )
    max_h = max(float(solar_result["overlay"]["maxValue"]), 1.0)
    norm = plt.Normalize(vmin=0.0, vmax=max_h)
    facecolors = cmap(norm(hours[:-1, :-1])) if hours.shape[0] > 1 and hours.shape[1] > 1 else None
    if facecolors is not None:
        ax.plot_surface(
            gx,
            gy,
            gz,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            shade=False,
            alpha=0.88,
            zorder=1,
        )

    # 2. Architectural massing triangles from IFC geometry.
    for ifc_cls, mesh, rgba in solar_result["labeled_meshes"]:
        if ifc_cls.startswith("IfcRoof"):
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0:
            continue
        tris = verts[faces]
        alpha = 0.35 if ifc_cls.startswith("IfcWindow") else 0.92
        col = (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, alpha)
        poly = Poly3DCollection(tris, facecolors=[col], edgecolors=[(0.25, 0.25, 0.28, 0.4)], linewidths=0.3)
        ax.add_collection3d(poly)

    # 3. Compass base ring + cardinal/intercardinal labels.
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    ring_x = cx + dome_r * np.sin(angles)
    ring_y = cy + dome_r * np.cos(angles)
    ring_z = np.full_like(angles, cz)
    ax.plot(ring_x, ring_y, ring_z, color="#7f1d1d", linewidth=1.2)
    ax.plot(
        cx + (dome_r * 1.05) * np.sin(angles),
        cy + (dome_r * 1.05) * np.cos(angles),
        ring_z,
        color="#991b1b",
        linewidth=0.8,
    )

    resolved_tn = float(solar_result.get("true_north_deg", 0.0) if true_north_deg is None else true_north_deg)

    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    for i, label in enumerate(dirs):
        ang = math.radians((i * 22.5 + resolved_tn) % 360.0)
        lx = cx + (dome_r * 1.13) * math.sin(ang)
        ly = cy + (dome_r * 1.13) * math.cos(ang)
        weight = "bold" if len(label) == 1 else "normal"
        fsize = 11 if len(label) == 1 else 7.5
        ax.text(lx, ly, cz, label, color="#1e293b", fontsize=fsize, fontweight=weight, ha="center", va="center")

    # 4. Ladybug 3D sky dome arcs for equinox & solstices + active day stars.
    for doy, lw, ls in ((172, 1.3, "-"), (80, 1.0, "--"), (355, 1.0, ":")):
        arc_pos, _, _ = solar_day_arc(lat_deg, doy, step_minutes=20, true_north_deg=resolved_tn)
        if not arc_pos:
            continue
        ax.plot(
            [cx + dome_r * p.sun_vec[0] for p in arc_pos],
            [cy + dome_r * p.sun_vec[1] for p in arc_pos],
            [cz + dome_r * max(0.0, p.sun_vec[2]) for p in arc_pos],
            color="#991b1b",
            linewidth=lw,
            linestyle=ls,
            alpha=0.82,
        )

    active_pos = solar_result["sun_positions"]
    if active_pos:
        ax.scatter(
            [cx + dome_r * p.sun_vec[0] for p in active_pos[::2]],
            [cy + dome_r * p.sun_vec[1] for p in active_pos[::2]],
            [cz + dome_r * max(0.0, p.sun_vec[2]) for p in active_pos[::2]],
            color="#dc2626",
            marker="*",
            s=36,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.58, pad=0.04, aspect=18)
    cbar.set_label("hours", fontsize=10, color="#1e293b")

    ax.set_xlim(cx - dome_r * 1.15, cx + dome_r * 1.15)
    ax.set_ylim(cy - dome_r * 1.15, cy + dome_r * 1.15)
    ax.set_zlim(cz, cz + dome_r * 1.02)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.view_init(elev=36, azim=-52)
    ax.set_axis_off()
    ax.set_title(
        f"{title}  ·  mean {solar_result['mean_hours']:.1f}h  ·  {solar_result['pct_over_4h']:.0f}% >= 4h",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
        pad=8,
    )

    out_file = Path(out_png_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=140, facecolor="white")
    plt.close(fig)
    return str(out_file)


def render_solstice_shadow_gif(
    ifc_path: str | Path,
    out_gif_path: str | Path,
    *,
    lat_deg: float,
    day_of_year: int,
    true_north_deg: float | None = None,
    width: int = 512,
    height: int = 384,
    duration_ms: int = 240,
) -> str:
    """Render an animated hourly shadow-sweep GIF across daylight hours for the selected city & date."""
    from PIL import ImageDraw

    from .studio_render import _filter_meshes_for_camera, _load_ifc_scene_and_meshes, _rasterize_camera_views

    resolved_tn = read_ifc_true_north_deg(ifc_path) if true_north_deg is None else float(true_north_deg)
    _, labeled_meshes, storey_elevations_m = _load_ifc_scene_and_meshes(ifc_path)
    meshes = _filter_meshes_for_camera(labeled_meshes, "isometric_cutaway")
    positions, _, _ = solar_day_arc(
        lat_deg,
        day_of_year,
        step_minutes=60,
        true_north_deg=resolved_tn,
    )
    if not positions:
        positions = [solar_position(lat_deg, day_of_year, 12.0, true_north_deg=resolved_tn)]

    frames = []
    for pos in positions:
        _, clay_pil = _rasterize_camera_views(
            meshes,
            camera_preset="isometric_cutaway",
            width=width,
            height=height,
            storey_elevations_m=storey_elevations_m,
            light_dir_override=pos.sun_vec,
        )
        draw = ImageDraw.Draw(clay_pil)
        draw.rectangle([10, 10, 265, 36], fill=(20, 23, 28))
        draw.text(
            (18, 17),
            f"{_fmt_hhmm(pos.hour_solar)} solar  |  Alt {pos.altitude_deg:.0f}° Az {pos.azimuth_deg:.0f}°",
            fill=(255, 224, 130),
        )
        frames.append(clay_pil)

    out_file = Path(out_gif_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return str(out_file)
