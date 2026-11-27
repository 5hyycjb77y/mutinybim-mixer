"""Zero-dependency Stdio JSON-RPC 2.0 MCP server for MutinyBIM Mixer.

Allows AI agents (Antigravity, Claude Code, Cursor, Gemini CLI, Windsurf) to
inspect the pinned spec, synthesize parametric sketches, compute solar sun-dials
and envelope/WWR gates, verify adapter compatibility, and optionally run live
``@permitzip/bim-script`` verification/IFC export via ``verifier/server.mjs``.

Run directly:
    python3 -m mixer.mcp_server
    python3 -m mixer.mcp_server --verifier-url http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import urllib.parse

from .lora_ops import check_adapter_compatibility, load_spec
from .sketch_engine import (
    SKETCH_STYLES,
    SKETCH_TOPOLOGIES,
    generate_sketch_sample,
    render_parametric_sketch_svg,
)
from .solar_engine import (
    CITY_PRESETS,
    KEY_DATES,
    build_cumulative_sky_matrix,
    compute_direct_sun_hours_for_points,
    compute_incident_radiation_for_points,
    compute_passive_solar_reward,
    describe_solar_lighting_for_camera,
    envelope_gates,
    render_sun_dial_svg,
    solar_day_arc,
    solar_position,
)
from .studio_render import (
    ART_DIRECTION_PRESETS,
    CAMERA_PRESETS,
    MIXTAPE_PRESETS,
    build_render_prompt,
    compute_camera_pose,
)
from verifier.verify_client import MutinyVerifierClient

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = Path(os.environ.get("MIXER_WORKSPACE") or DEFAULT_REPO_ROOT).resolve()
DEFAULT_VERIFIER_URL = os.environ.get("MIXER_VERIFIER_URL", "http://127.0.0.1:8787")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _validate_verifier_url(url: str, *, allow_remote: bool = False) -> str:
    parsed = urllib.parse.urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid verifier URL: {url!r}")
    if not allow_remote and parsed.hostname.lower() not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"Refusing non-loopback verifier host '{parsed.hostname}'; pass --allow-remote-verifier to permit remote URLs."
        )
    return str(url).strip()


def _inside_workspace(raw: str, *, workspace: Path, must_exist: bool) -> Path:
    ws = workspace.resolve()
    raw_path = Path(str(raw))
    candidate = raw_path if raw_path.is_absolute() else (ws / raw_path)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ws):
        raise ValueError(f"Paths must be inside the workspace ({ws}).")
    if must_exist and not resolved.exists():
        raise ValueError("That file doesn't exist in the workspace.")
    return resolved


TOOLS: list[dict[str, Any]] = [
    {
        "name": "mixer_info",
        "description": "Return pinned spec.json contract, registry.json adapter count, and available preset keys.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "generate_sketch",
        "description": "Synthesize a parametric MutinyBIM floor plan, TypeScript script, and napkin-sketch SVG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topology": {
                    "type": "string",
                    "description": f"One of {list(SKETCH_TOPOLOGIES)}; optional (cycles by seed if omitted).",
                },
                "seed": {"type": "integer", "default": 42},
                "style": {
                    "type": "string",
                    "default": SKETCH_STYLES[0],
                    "description": f"One of {list(SKETCH_STYLES)}.",
                },
            },
        },
    },
    {
        "name": "solar_position_and_dial",
        "description": "Compute solar altitude, clockwise-from-North azimuth (local solar time, ±1° approx.), sunrise/sunset, and SVG Sun-Dial HUD.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": f"Optional preset key in {list(CITY_PRESETS)}."},
                "lat_deg": {"type": "number", "description": "Latitude in degrees (-89.9..89.9); overrides city preset."},
                "date": {"type": "string", "description": f"Optional preset key in {list(KEY_DATES)}."},
                "day_of_year": {"type": "integer", "description": "Day of year (1..366); overrides date preset."},
                "hour_solar": {"type": "number", "default": 12.0, "description": "Local solar hour (0..24)."},
                "true_north_deg": {"type": "number", "default": 0.0, "description": "Degrees clockwise from Plan +Y to True North."},
                "include_svg": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "check_envelope_gates",
        "description": "Check closed perimeter per storey, roof/top slab presence, and True-North-aware facade WWR on an IFC file, STEP text, or model dict; optionally scores compute_passive_solar_reward.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ifc_path": {"type": "string", "description": "Path inside the mixer workspace to an .ifc file."},
                "ifc_text": {"type": "string", "description": "Raw IFC STEP text."},
                "model": {"type": "object", "description": "Structured dict with walls, windows, roofs, slabs."},
                "true_north_deg": {"type": "number", "description": "Optional override; auto-read from IFC when omitted."},
                "winter_mean_hours": {"type": "number"},
                "summer_mean_hours": {"type": "number"},
                "verify_ok": {"type": "boolean"},
                "spec_match": {"type": "number"},
            },
        },
    },
    {
        "name": "solar_direct_sun_hours",
        "description": "Compute direct solar exposure hours for explicit 3D points (zero-dep) or a full IFC workplane grid + optional 3D Solar Dome PNG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "triangles": {"type": "array", "description": "List of 3-vertex triangles [[x,y,z],[x,y,z],[x,y,z]] for explicit point mode."},
                "ifc_path": {"type": "string", "description": "Path inside the mixer workspace to an .ifc file for full grid + heatmap mode (needs numpy/ifcopenshell)."},
                "out_dir": {"type": "string", "description": "Optional output directory inside the mixer workspace for solar_dome.png and solar_heatmap.json."},
                "city": {"type": "string"},
                "lat_deg": {"type": "number"},
                "date": {"type": "string"},
                "day_of_year": {"type": "integer"},
                "true_north_deg": {"type": "number"},
                "step_minutes": {"type": "integer", "default": 30},
                "grid_spacing_m": {"type": "number", "default": 1.0},
            },
        },
    },
    {
        "name": "solar_incident_radiation",
        "description": "Compute Ladybug/Ecotect Tregenza 145-patch cumulative incident solar radiation (kWh/m²) and Sky View Factor for explicit points/normals (zero-dep) or full IFC model surfaces + 3D PNG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "normals": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "triangles": {"type": "array", "description": "List of 3-vertex triangles [[x,y,z],[x,y,z],[x,y,z]] for explicit point mode."},
                "ifc_path": {"type": "string", "description": "Path inside the mixer workspace to an .ifc file for full surface radiation mode."},
                "out_dir": {"type": "string", "description": "Optional output directory inside the mixer workspace for solar_radiation.png and radiation_heatmap.json."},
                "epw_path": {"type": "string", "description": "Optional EnergyPlus .epw weather file inside workspace."},
                "epw_text": {"type": "string", "description": "Optional inline EnergyPlus .epw weather text."},
                "city": {"type": "string"},
                "lat_deg": {"type": "number"},
                "start_doy": {"type": "integer", "default": 1},
                "end_doy": {"type": "integer", "default": 365},
                "start_hour": {"type": "number", "default": 0.0},
                "end_hour": {"type": "number", "default": 24.0},
                "day_step": {"type": "integer", "default": 1},
                "true_north_deg": {"type": "number"},
                "ground_reflectance": {"type": "number", "default": 0.2},
                "max_edge_m": {"type": "number", "default": 1.5},
            },
        },
    },
    {
        "name": "build_studio_prompt",
        "description": "Build depth-conditioned photorealistic positive and negative prompts from user vibe, art direction, camera preset, and optional solar time/location.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_vibe": {"type": "string", "default": ""},
                "style_key": {"type": "string", "default": "ArchDaily Golden Hour Timber"},
                "camera_key": {"type": "string", "default": "isometric_cutaway"},
                "hour_solar": {"type": "number", "description": "Optional solar hour (0..24) to inject camera-relative sun/shadow cues."},
                "city": {"type": "string"},
                "lat_deg": {"type": "number"},
                "date": {"type": "string"},
                "day_of_year": {"type": "integer"},
                "true_north_deg": {"type": "number", "default": 0.0},
            },
        },
    },
    {
        "name": "check_adapter_compatibility",
        "description": "Validate LoRA adapter_config.json files or dicts against the pinned spec.json subspace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapters": {
                    "type": "array",
                    "description": "List of adapter_config.json paths or adapter directories inside the mixer workspace, or inline config dicts.",
                    "items": {"type": ["string", "object"]},
                },
            },
            "required": ["adapters"],
        },
    },
    {
        "name": "verify_script",
        "description": "Verify a @permitzip/bim-script snippet against the server's configured verifier/server.mjs HTTP process.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ts_source": {"type": "string"},
                "project_name": {"type": "string", "default": "Agent Verification"},
                "expected_kinds": {"type": "object"},
                "min_elements": {"type": "integer", "default": 1},
            },
            "required": ["ts_source"],
        },
    },
    {
        "name": "export_bundle",
        "description": "Verify + export IFC 4.3, SVG sheets, PDF, and envelope_gates via the server's configured verifier/server.mjs process.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ts_source": {"type": "string"},
                "project_name": {"type": "string", "default": "Agent Export"},
                "out_dir": {"type": "string", "description": "Optional directory inside the mixer workspace to write model.ifc, sheets.pdf, and SVGs."},
            },
            "required": ["ts_source"],
        },
    },
]


def _resolve_lat_doy(args: dict[str, Any]) -> tuple[float, int, str, str]:
    city_key = str(args.get("city") or "san_francisco")
    city = CITY_PRESETS.get(city_key) or next(iter(CITY_PRESETS.values()))
    lat_deg = float(args["lat_deg"]) if args.get("lat_deg") is not None else float(city["lat_deg"])
    city_label = city["label"] if args.get("lat_deg") is None and city_key in CITY_PRESETS else f"Lat {lat_deg:.2f}°"

    date_key = str(args.get("date") or "summer_solstice")
    dt = KEY_DATES.get(date_key) or next(iter(KEY_DATES.values()))
    doy = int(args["day_of_year"]) if args.get("day_of_year") is not None else int(dt["day_of_year"])
    date_label = dt["label"] if args.get("day_of_year") is None and date_key in KEY_DATES else f"Day {doy}"
    return lat_deg, doy, city_label, date_label


def _tool_mixer_info(_args: dict[str, Any], *, repo_root: Path, verifier_url: str, workspace: Path, **_kw: Any) -> dict[str, Any]:
    spec = load_spec(repo_root / "spec.json")
    reg_path = repo_root / "registry.json"
    registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {"adapters": []}
    return {
        "spec_version": spec.get("spec_version"),
        "base_model": spec.get("base_model"),
        "lora": spec.get("lora"),
        "mutinybim": spec.get("mutinybim"),
        "trades": sorted((spec.get("trades") or {}).keys()),
        "registered_adapters": len(registry.get("adapters") or []),
        "default_verifier_url": verifier_url,
        "workspace": str(workspace),
        "presets": {
            "art_directions": list(ART_DIRECTION_PRESETS.keys()),
            "cameras": list(CAMERA_PRESETS.keys()),
            "mixtapes": list(MIXTAPE_PRESETS.keys()),
            "sketch_topologies": list(SKETCH_TOPOLOGIES),
            "sketch_styles": list(SKETCH_STYLES),
            "cities": list(CITY_PRESETS.keys()),
            "solar_dates": list(KEY_DATES.keys()),
        },
    }


def _tool_generate_sketch(args: dict[str, Any], **_kw: Any) -> dict[str, Any]:
    seed = int(args.get("seed", 42))
    topology = args.get("topology")
    if topology is not None and str(topology) not in SKETCH_TOPOLOGIES:
        raise ValueError(f"Unknown topology '{topology}'; expected one of {list(SKETCH_TOPOLOGIES)}")
    style = str(args.get("style") or SKETCH_STYLES[0])
    if style not in SKETCH_STYLES:
        raise ValueError(f"Unknown style '{style}'; expected one of {list(SKETCH_STYLES)}")
    sample = generate_sketch_sample(seed, topology=str(topology) if topology is not None else None)
    svg = render_parametric_sketch_svg(sample, style=style)
    return {
        "sample_id": sample.sample_id,
        "topology": sample.topology,
        "floors": sample.floors,
        "user_prompt": sample.user_prompt,
        "expected_kinds": sample.expected_kinds,
        "min_elements": sum(sample.expected_kinds.values()),
        "room_count": len(sample.rooms),
        "wall_count": len(sample.walls),
        "opening_count": len(sample.openings),
        "ts_source": sample.ts_source,
        "svg": svg,
    }


def _tool_solar_position_and_dial(args: dict[str, Any], **_kw: Any) -> dict[str, Any]:
    lat_deg, doy, city_label, date_label = _resolve_lat_doy(args)
    hour_solar = float(args.get("hour_solar", 12.0))
    tn = float(args.get("true_north_deg", 0.0))
    pos = solar_position(lat_deg, doy, hour_solar, true_north_deg=tn)
    positions, sunrise_h, sunset_h = solar_day_arc(lat_deg, doy, step_minutes=30, true_north_deg=tn)
    out: dict[str, Any] = {
        "lat_deg": lat_deg,
        "day_of_year": doy,
        "hour_solar": pos.hour_solar,
        "true_north_deg": tn,
        "altitude_deg": pos.altitude_deg,
        "azimuth_deg": pos.azimuth_deg,
        "shadow_len_1m": pos.shadow_len_1m,
        "sun_vec": list(pos.sun_vec),
        "sunrise_hour": round(sunrise_h, 3),
        "sunset_hour": round(sunset_h, 3),
        "daylight_sample_count": len(positions),
    }
    if bool(args.get("include_svg", True)):
        out["svg"] = render_sun_dial_svg(
            lat_deg,
            doy,
            hour_solar,
            city_label=city_label,
            date_label=date_label,
            true_north_deg=tn,
        )
    return out


def _tool_check_envelope_gates(args: dict[str, Any], *, workspace: Path, **_kw: Any) -> dict[str, Any]:
    tn = float(args["true_north_deg"]) if args.get("true_north_deg") is not None else None
    if args.get("model") is not None:
        source: Any = args["model"]
    elif args.get("ifc_text"):
        source = str(args["ifc_text"])
    elif args.get("ifc_path"):
        source = _inside_workspace(str(args["ifc_path"]), workspace=workspace, must_exist=True)
    else:
        raise ValueError("Provide one of 'ifc_path', 'ifc_text', or 'model'.")

    gates = envelope_gates(source, true_north_deg=tn)
    if all(args.get(k) is not None for k in ("winter_mean_hours", "summer_mean_hours", "verify_ok", "spec_match")):
        gates["passive_solar_reward"] = compute_passive_solar_reward(
            float(args["winter_mean_hours"]),
            float(args["summer_mean_hours"]),
            verify_ok=bool(args["verify_ok"]),
            spec_match=float(args["spec_match"]),
            has_closed_envelope=bool(gates["has_closed_envelope"]),
            max_facade_wwr=float(gates["max_facade_wwr"]),
        )
    return gates


def _tool_solar_direct_sun_hours(args: dict[str, Any], *, workspace: Path, **_kw: Any) -> dict[str, Any]:
    lat_deg, doy, _, date_label = _resolve_lat_doy(args)
    tn = float(args["true_north_deg"]) if args.get("true_north_deg") is not None else None
    step_min = int(args.get("step_minutes", 30))

    if args.get("points") is not None:
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in args["points"]]
        tris = [
            (
                (float(t[0][0]), float(t[0][1]), float(t[0][2])),
                (float(t[1][0]), float(t[1][1]), float(t[1][2])),
                (float(t[2][0]), float(t[2][1]), float(t[2][2])),
            )
            for t in (args.get("triangles") or [])
        ]
        hours = compute_direct_sun_hours_for_points(
            pts,
            tris,
            lat_deg=lat_deg,
            day_of_year=doy,
            true_north_deg=0.0 if tn is None else tn,
            step_minutes=step_min,
        )
        return {
            "lat_deg": lat_deg,
            "day_of_year": doy,
            "true_north_deg": 0.0 if tn is None else tn,
            "point_hours": hours,
        }

    if not args.get("ifc_path"):
        raise ValueError("Provide either 'points' or 'ifc_path'.")

    ifc_file = _inside_workspace(str(args["ifc_path"]), workspace=workspace, must_exist=True)
    out_dir = _inside_workspace(str(args["out_dir"]), workspace=workspace, must_exist=False) if args.get("out_dir") else None

    from .solar_engine import compute_direct_sun_hours, render_solar_dome_png

    res = compute_direct_sun_hours(
        str(ifc_file),
        lat_deg=lat_deg,
        day_of_year=doy,
        true_north_deg=tn,
        step_minutes=step_min,
        grid_spacing_m=float(args.get("grid_spacing_m", 1.0)),
    )
    out: dict[str, Any] = {
        "lat_deg": lat_deg,
        "day_of_year": doy,
        "true_north_deg": res["true_north_deg"],
        "min_hours": res["min_hours"],
        "max_hours": res["max_hours"],
        "mean_hours": res["mean_hours"],
        "pct_over_4h": res["pct_over_4h"],
        "workplane_z_m": res["workplane_z_m"],
        "cell_count": len(res["cells"]),
        "assumptions": res["assumptions"],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        dome_path = render_solar_dome_png(
            res,
            out_dir / "solar_dome.png",
            lat_deg=lat_deg,
            day_of_year=doy,
            true_north_deg=res["true_north_deg"],
            title=f"Direct Sun Hours — {date_label}",
        )
        heatmap_path = out_dir / "solar_heatmap.json"
        heatmap_path.write_text(json.dumps(res["overlay"], indent=2), encoding="utf-8")
        out["solar_dome_png"] = dome_path
        out["solar_heatmap_json"] = str(heatmap_path)
    return out


def _tool_solar_incident_radiation(args: dict[str, Any], *, workspace: Path, **_kw: Any) -> dict[str, Any]:
    lat_deg, _, _, _ = _resolve_lat_doy(args)
    start_doy = int(args.get("start_doy", 1))
    end_doy = int(args.get("end_doy", 365))
    start_hour = float(args.get("start_hour", 0.0))
    end_hour = float(args.get("end_hour", 24.0))
    day_step = max(1, int(args.get("day_step", 1)))
    tn = float(args["true_north_deg"]) if args.get("true_north_deg") is not None else None
    ground_reflectance = float(args.get("ground_reflectance", 0.2))

    epw_source: str | Path | None = None
    if args.get("epw_text"):
        epw_source = str(args["epw_text"])
    elif args.get("epw_path"):
        epw_source = _inside_workspace(str(args["epw_path"]), workspace=workspace, must_exist=True)

    if args.get("points") is not None:
        n_points = len(args["points"])
        n_tris = len(args.get("triangles") or [])
        # Cost is points x triangles x 145 sky patches; keep one call to seconds, not minutes.
        if n_points > 5000 or n_tris > 20000 or n_points * max(1, n_tris) > 2_000_000:
            raise ValueError(
                f"Too much work for one call ({n_points} points x {n_tris} triangles): "
                "use at most 5000 points, 20000 triangles and 2,000,000 point-triangle pairs."
            )
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in args["points"]]
        raw_normals = args.get("normals")
        if raw_normals is None:
            nrms = [(0.0, 0.0, 1.0) for _ in pts]
        else:
            nrms = [(float(n[0]), float(n[1]), float(n[2])) for n in raw_normals]
        tris = [
            (
                (float(t[0][0]), float(t[0][1]), float(t[0][2])),
                (float(t[1][0]), float(t[1][1]), float(t[1][2])),
                (float(t[2][0]), float(t[2][1]), float(t[2][2])),
            )
            for t in (args.get("triangles") or [])
        ]
        resolved_tn = 0.0 if tn is None else tn
        sky = build_cumulative_sky_matrix(
            lat_deg=None if (epw_source is not None and args.get("lat_deg") is None and args.get("city") is None) else lat_deg,
            start_doy=start_doy,
            end_doy=end_doy,
            start_hour=start_hour,
            end_hour=end_hour,
            true_north_deg=resolved_tn,
            epw_path_or_text=epw_source,
            day_step=day_step,
        )
        res = compute_incident_radiation_for_points(
            pts,
            nrms,
            tris,
            sky,
            ground_reflectance=ground_reflectance,
        )
        return {
            "sky_source": sky["sky_source"],
            "lat_deg": sky["lat_deg"],
            "start_doy": sky["start_doy"],
            "end_doy": sky["end_doy"],
            "true_north_deg": resolved_tn,
            "total_kwh_m2": res["total_kwh_m2"],
            "direct_kwh_m2": res["direct_kwh_m2"],
            "diffuse_kwh_m2": res["diffuse_kwh_m2"],
            "sky_view_factor": res["sky_view_factor"],
        }

    if not args.get("ifc_path"):
        raise ValueError("Provide either 'points' or 'ifc_path'.")

    ifc_file = _inside_workspace(str(args["ifc_path"]), workspace=workspace, must_exist=True)
    out_dir = _inside_workspace(str(args["out_dir"]), workspace=workspace, must_exist=False) if args.get("out_dir") else None

    from .solar_engine import compute_incident_solar_radiation, render_incident_radiation_png

    rad = compute_incident_solar_radiation(
        str(ifc_file),
        lat_deg=None if (epw_source is not None and args.get("lat_deg") is None and args.get("city") is None) else lat_deg,
        start_doy=start_doy,
        end_doy=end_doy,
        start_hour=start_hour,
        end_hour=end_hour,
        true_north_deg=tn,
        epw_path_or_text=epw_source,
        max_edge_m=float(args.get("max_edge_m", 1.5)),
        ground_reflectance=ground_reflectance,
        day_step=day_step,
    )
    out_rad: dict[str, Any] = {
        "sky_source": rad["sky_source"],
        "lat_deg": rad["lat_deg"],
        "start_doy": rad["start_doy"],
        "end_doy": rad["end_doy"],
        "true_north_deg": rad["true_north_deg"],
        "min_kwh_m2": rad["min_kwh_m2"],
        "max_kwh_m2": rad["max_kwh_m2"],
        "mean_kwh_m2": rad["mean_kwh_m2"],
        "total_surface_area_m2": rad["total_surface_area_m2"],
        "total_incident_kwh": rad["total_incident_kwh"],
        "by_class": rad["by_class"],
        "surface_count": len(rad["total_kwh_m2"]),
        "assumptions": rad["assumptions"],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        rad_png = render_incident_radiation_png(
            rad,
            out_dir / "solar_radiation.png",
            title=f"Incident Solar Radiation ({rad['start_doy']}–{rad['end_doy']})",
        )
        heatmap_path = out_dir / "radiation_heatmap.json"
        heatmap_path.write_text(json.dumps(rad["overlay"], indent=2), encoding="utf-8")
        out_rad["solar_radiation_png"] = rad_png
        out_rad["radiation_heatmap_json"] = str(heatmap_path)
    return out_rad


def _tool_build_studio_prompt(args: dict[str, Any], **_kw: Any) -> dict[str, Any]:
    style_key = str(args.get("style_key") or "ArchDaily Golden Hour Timber")
    camera_key = str(args.get("camera_key") or "isometric_cutaway")
    solar_lighting: dict[str, Any] | None = None
    solar_cue: str | None = None
    solar_neg: str | None = None
    if args.get("hour_solar") is not None:
        lat_deg, doy, _, _ = _resolve_lat_doy(args)
        tn = float(args.get("true_north_deg") or 0.0)
        pos = solar_position(lat_deg, doy, float(args["hour_solar"]), true_north_deg=tn)
        pose = compute_camera_pose((0.0, 0.0, 0.0), (12.0, 8.0, 3.5), camera_key)
        solar_lighting = describe_solar_lighting_for_camera(pos, pose)
        solar_cue = solar_lighting["prompt_cue"]
        solar_neg = solar_lighting["negative_cue"]

    pos_p, neg_p = build_render_prompt(
        str(args.get("user_vibe") or ""),
        style_key,
        camera_key,
        solar_lighting_cue=solar_cue,
        solar_negative_cue=solar_neg,
    )
    out: dict[str, Any] = {
        "style_key": style_key,
        "camera_key": camera_key,
        "positive_prompt": pos_p,
        "negative_prompt": neg_p,
    }
    if solar_lighting is not None:
        out["solar_lighting"] = solar_lighting
    return out


def _tool_check_adapter_compatibility(args: dict[str, Any], *, repo_root: Path, workspace: Path, **_kw: Any) -> dict[str, Any]:
    spec = load_spec(repo_root / "spec.json")
    results: list[dict[str, Any]] = []
    for item in args.get("adapters") or []:
        if isinstance(item, dict):
            cfg = item
            label = str(item.get("base_model_name_or_path") or "inline_config")
        else:
            p = _inside_workspace(str(item), workspace=workspace, must_exist=True)
            if p.is_dir():
                p = _inside_workspace(str(p / "adapter_config.json"), workspace=workspace, must_exist=True)
            cfg = json.loads(p.read_text(encoding="utf-8"))
            label = str(p)
        issues = check_adapter_compatibility(cfg, spec)
        results.append({"adapter": label, "compatible": len(issues) == 0, "issues": issues})
    return {
        "all_compatible": all(r["compatible"] for r in results),
        "results": results,
    }


def _connect_verifier(url: str, *, allow_remote: bool = False) -> MutinyVerifierClient:
    safe_url = _validate_verifier_url(url, allow_remote=allow_remote)
    client = MutinyVerifierClient(safe_url, timeout=60.0)
    try:
        client.health()
    except Exception as exc:
        raise RuntimeError(
            f"MutinyBIM verifier not reachable at {safe_url}: {exc}. "
            "Start it with: node verifier/server.mjs --mutinybim /path/to/mutinybim --port 8787"
        ) from exc
    return client


def _tool_verify_script(
    args: dict[str, Any],
    *,
    verifier_url: str,
    allow_remote_verifier: bool = False,
    **_kw: Any,
) -> dict[str, Any]:
    # Ignore any caller-supplied verifier_url; use server configuration only.
    client = _connect_verifier(verifier_url, allow_remote=allow_remote_verifier)
    res = client.verify(
        str(args["ts_source"]),
        expected_kinds=args.get("expected_kinds"),
        min_elements=int(args.get("min_elements", 1)),
        project_name=str(args.get("project_name") or "Agent Verification"),
    )
    return {
        "ok": res.ok,
        "reward": res.reward,
        "breakdown": res.breakdown,
        "stage_failed": res.stage_failed,
        "error": res.error,
        "kinds": res.kinds,
        "lint_summary": res.lint_summary,
    }


def _tool_export_bundle(
    args: dict[str, Any],
    *,
    verifier_url: str,
    workspace: Path,
    allow_remote_verifier: bool = False,
    **_kw: Any,
) -> dict[str, Any]:
    out_dir = _inside_workspace(str(args["out_dir"]), workspace=workspace, must_exist=False) if args.get("out_dir") else None
    client = _connect_verifier(verifier_url, allow_remote=allow_remote_verifier)
    bundle = client.export_bundle(
        str(args["ts_source"]),
        project_name=str(args.get("project_name") or "Agent Export"),
    )
    out: dict[str, Any] = {
        "ok": bundle.verify.ok,
        "reward": bundle.verify.reward,
        "error": bundle.verify.error,
        "kinds": bundle.verify.kinds,
        "lint_summary": bundle.verify.lint_summary,
        "sheet_count": len(bundle.sheets),
        "has_ifc": bool(bundle.ifc_text),
        "has_pdf": bool(bundle.pdf_base64),
    }
    if bundle.ifc_text:
        out["envelope_gates"] = envelope_gates(bundle.ifc_text)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        if bundle.ifc_text:
            ifc_f = out_dir / "model.ifc"
            ifc_f.write_text(bundle.ifc_text, encoding="utf-8")
            written.append(str(ifc_f))
        if bundle.pdf_base64:
            pdf_f = out_dir / "sheets.pdf"
            pdf_f.write_bytes(base64.b64decode(bundle.pdf_base64))
            written.append(str(pdf_f))
        for sh in bundle.sheets:
            safe_num = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sh.get("number") or "sheet")).strip("._") or "sheet"
            svg_f = (out_dir / f"{safe_num}.svg").resolve()
            if not svg_f.is_relative_to(out_dir):
                raise ValueError(f"Unsafe sheet output path: {svg_f}")
            svg_f.write_text(str(sh.get("svg") or ""), encoding="utf-8")
            written.append(str(svg_f))
        out["written_files"] = written
    return out


_DISPATCH = {
    "mixer_info": _tool_mixer_info,
    "generate_sketch": _tool_generate_sketch,
    "solar_position_and_dial": _tool_solar_position_and_dial,
    "check_envelope_gates": _tool_check_envelope_gates,
    "solar_direct_sun_hours": _tool_solar_direct_sun_hours,
    "solar_incident_radiation": _tool_solar_incident_radiation,
    "build_studio_prompt": _tool_build_studio_prompt,
    "check_adapter_compatibility": _tool_check_adapter_compatibility,
    "verify_script": _tool_verify_script,
    "export_bundle": _tool_export_bundle,
}


def handle_jsonrpc_message(
    msg: dict[str, Any],
    *,
    verifier_url: str = DEFAULT_VERIFIER_URL,
    repo_root: Path = DEFAULT_REPO_ROOT,
    workspace: Path | None = None,
    allow_remote_verifier: bool = False,
) -> dict[str, Any] | None:
    """Handle one JSON-RPC 2.0 MCP message and return the response dict (or ``None`` for notifications)."""
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    ws = (workspace or repo_root).resolve()

    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    if msg_id is None:
        return None

    if method == "initialize":
        proto = str(params.get("protocolVersion") or "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mutinybim-mixer", "version": "0.1.0"},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        fn = _DISPATCH.get(name)
        if fn is None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        try:
            payload = fn(
                arguments,
                repo_root=repo_root,
                verifier_url=verifier_url,
                workspace=ws,
                allow_remote_verifier=allow_remote_verifier,
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error in {name}: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stdio JSON-RPC 2.0 MCP server for mutinybim-mixer")
    parser.add_argument("--verifier-url", default=DEFAULT_VERIFIER_URL, help="Optional running verifier/server.mjs URL (loopback only by default)")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Path to mutinybim-mixer repository root")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Allowed workspace directory for file reads/writes")
    parser.add_argument("--allow-remote-verifier", action="store_true", help="Permit non-loopback verifier URLs")
    ns = parser.parse_args(argv)
    repo_root = Path(ns.repo_root).resolve()
    workspace = Path(ns.workspace).resolve()
    try:
        verifier_url = _validate_verifier_url(ns.verifier_url, allow_remote=bool(ns.allow_remote_verifier))
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_jsonrpc_message(
            msg,
            verifier_url=verifier_url,
            repo_root=repo_root,
            workspace=workspace,
            allow_remote_verifier=bool(ns.allow_remote_verifier),
        )
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
