"""Visual studio helpers for Vibe Architect Studio (`03_vibe_architect_studio.ipynb`).

Pure-stdlib presets and prompt builders are always importable locally.
Heavy 3D/diffusion dependencies (``ifcopenshell``, ``trimesh``, ``numpy``,
``PIL``, ``diffusers``, ``cairosvg``) are imported lazily inside rendering
functions so tests and lightweight tools work without GPU packages.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

ART_DIRECTION_PRESETS: dict[str, dict[str, str]] = {
    "ArchDaily Golden Hour Timber": {
        "prompt": (
            "architectural photography, ArchDaily editorial feature, warm Japanese hinoki timber, "
            "honed limestone floor, matte black architectural hardware, soft golden hour sunlight "
            "casting long crisp shadows through tall glazing, lush indoor olive tree, 35mm tilt-shift lens, "
            "photorealistic, ultra detailed materials"
        ),
        "negative": "cartoon, blurry, distorted geometry, warped walls, watermark, text overlay, low quality",
    },
    "Brutalist Concrete & Diffused Daylight": {
        "prompt": (
            "architectural monograph photography, board-formed architectural concrete walls, "
            "terrazzo floor, brushed stainless steel accents, soft overcast museum skylighting, "
            "calm serene spatial volume, sharp edges, tilt-shift architectural lens, photorealistic"
        ),
        "negative": "clutter, neon colors, warped perspective, cartoon, low resolution, watermark",
    },
    "Moody Hospitality & Exposed MEP": {
        "prompt": (
            "interior architectural photography, high-craft boutique hospitality space, warm walnut millwork, "
            "exposed matte-black spiral HVAC ductwork and copper conduit in tall ceiling plenum, "
            "warm 2700K indirect cove lighting, atmospheric evening glow, editorial photography"
        ),
        "negative": "oversaturated, distorted ducts, messy geometry, cartoon, lowres, text",
    },
    "Scandinavian Minimalist Studio": {
        "prompt": (
            "Nordic architectural photography, pale white oak floors, limewash plaster walls, "
            "sleek aluminum-framed glazing, crisp bright diffused north daylight, minimal bespoke furniture, "
            "clean lines, photorealistic, high dynamic range"
        ),
        "negative": "dark, cluttered, ornate, warped walls, blurry, watermark",
    },
    "Architectural Watercolor & Ink Cutaway": {
        "prompt": (
            "hand-rendered architectural competition illustration, crisp archival ink linework, "
            "delicate translucent watercolor washes on cold-press cotton paper, subtle ambient occlusion, "
            "warm sunlit shadows, museum-grade architectural presentation drawing"
        ),
        "negative": "photographic grain, dark muddy colors, messy lines, 3d plastic render, text",
    },
}

CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "isometric_cutaway": {
        "label": "Isometric Dollhouse Cutaway",
        "projection": "ortho",
        "elevation_deg": 35.0,
        "azimuth_deg": 42.0,
        "hide_cutaway_classes": ("IfcCovering", "IfcRoof"),
        "prompt_hint": "isometric architectural cutaway diorama view looking down into furnished interior rooms",
    },
    "eye_level_interior": {
        "label": "Eye-Level Interior Perspective",
        "projection": "perspective",
        "elevation_deg": 5.0,
        "azimuth_deg": 28.0,
        "hide_cutaway_classes": (),
        "prompt_hint": "eye-level wide-angle interior architectural perspective looking across the space",
    },
    "opposite_interior": {
        "label": "Opposite Interior Angle",
        "projection": "perspective",
        "elevation_deg": 6.0,
        "azimuth_deg": -145.0,
        "hide_cutaway_classes": (),
        "prompt_hint": "reverse eye-level interior architectural view looking back across the room toward daylight windows",
    },
    "corner_axonometric": {
        "label": "Exterior Corner Axonometric",
        "projection": "ortho",
        "elevation_deg": 28.0,
        "azimuth_deg": -38.0,
        "hide_cutaway_classes": (),
        "prompt_hint": "clean architectural three-quarter exterior axonometric massing view",
    },
}

VIBE_INSPIRATION_CHIPS: list[dict[str, str]] = [
    {
        "title": "Timber Coffee Lab",
        "prompt": (
            "Design a 36 ft x 24 ft warm Japanese timber coffee shop and roastery on Level 1 with 14 ft walls, "
            "an open espresso bar, a quiet tasting room separated by an interior partition with doors, "
            "large north windows, 10 ft partial ceiling with exposed supply ductwork at 11.5 ft, and sheet A-101."
        ),
    },
    {
        "title": "Boutique Design Studio",
        "prompt": (
            "Create a 44 ft x 28 ft open architectural studio with a glass-walled conference room, "
            "private director office, collaborative pin-up gallery, north clerestory windows, 9.5 ft ceilings, "
            "exposed rooftop HVAC supply/return ducts at 11.5 ft, and floor plans on A-101 and M-101."
        ),
    },
    {
        "title": "2-Story Courtyard Clinic",
        "prompt": (
            "Build a 2-story 40 ft x 30 ft boutique wellness clinic (Level 1 at 0 ft, Level 2 at 12 ft) with "
            "reception, consultation rooms, doors and north windows, 9 ft ceilings, a 2-floor restroom sanitary "
            "plumbing riser sized by code, and architectural + plumbing sheets A-101, A-102, and P-101."
        ),
    },
    {
        "title": "Loft Apartment + Power",
        "prompt": (
            "Model a 34 ft x 16 ft bright minimalist loft apartment (Bedroom 101, Living Studio 102) with "
            "9 ft walls, doors, windows, NEC 2023 code-spaced receptacles from panel LP-1, and sheets A-101 and E-101."
        ),
    },
]

MIXTAPE_PRESETS: dict[str, dict[str, Any]] = {
    "Timber / Brutalist / Watercolor Trio": {
        "mode": "style_trio",
        "description": "Same exact IFC geometry rendered across 3 distinct architectural art directions.",
        "tracks": [
            {"label": "Track A: Golden Hour Timber", "style": "ArchDaily Golden Hour Timber"},
            {"label": "Track B: Brutalist Daylight", "style": "Brutalist Concrete & Diffused Daylight"},
            {"label": "Track C: Competition Watercolor", "style": "Architectural Watercolor & Ink Cutaway"},
        ],
    },
    "Daylight / Exposed MEP / Golden Hour": {
        "mode": "style_trio",
        "description": "Compare Scandinavian studio daylight, moody exposed MEP, and warm timber.",
        "tracks": [
            {"label": "Track A: Scandinavian Studio", "style": "Scandinavian Minimalist Studio"},
            {"label": "Track B: Moody Exposed MEP", "style": "Moody Hospitality & Exposed MEP"},
            {"label": "Track C: Golden Hour Timber", "style": "ArchDaily Golden Hour Timber"},
        ],
    },
    "Weight-Space LoRA Fader Sweep": {
        "mode": "weight_sweep",
        "description": "Blend community trade LoRAs at 3 fader settings to visualize weight-space programming.",
        "tracks": [
            {"label": "Track A: Minimal Arch (1.2 / 0.2 / 0.0)", "faders": (1.2, 0.2, 0.0)},
            {"label": "Track B: Balanced Multi-Trade (0.9 / 0.8 / 0.3)", "faders": (0.9, 0.8, 0.3)},
            {"label": "Track C: Max MEP + ClashNeg (0.8 / 1.1 / 0.6)", "faders": (0.8, 1.1, 0.6)},
        ],
    },
}


def build_render_prompt(user_vibe: str, style_key: str, camera_key: str) -> tuple[str, str]:
    """Build `(positive_prompt, negative_prompt)` combining user vibe, art direction, and camera framing."""
    style = ART_DIRECTION_PRESETS.get(style_key) or next(iter(ART_DIRECTION_PRESETS.values()))
    cam = CAMERA_PRESETS.get(camera_key) or CAMERA_PRESETS["isometric_cutaway"]
    vibe_clean = " ".join((user_vibe or "").split())[:220]
    positive = f"{cam['prompt_hint']}, {style['prompt']}"
    if vibe_clean:
        positive = f"{positive}, inspired by: {vibe_clean}"
    return positive, style["negative"]


_IFC_CLASS_COLORS: dict[str, tuple[int, int, int, int]] = {
    "IfcWall": (235, 232, 226, 255),
    "IfcWallStandardCase": (235, 232, 226, 255),
    "IfcSlab": (185, 188, 194, 255),
    "IfcCovering": (244, 244, 240, 255),
    "IfcRoof": (160, 165, 172, 255),
    "IfcWindow": (145, 205, 235, 170),
    "IfcDoor": (176, 128, 88, 255),
    "IfcDuctSegment": (52, 58, 66, 255),
    "IfcDuctFitting": (42, 48, 56, 255),
    "IfcPipeSegment": (48, 122, 178, 255),
    "IfcPipeFitting": (40, 102, 152, 255),
    "IfcFlowTerminal": (210, 214, 220, 255),
    "IfcDistributionElement": (90, 96, 108, 255),
    "IfcColumn": (140, 142, 148, 255),
    "IfcBeam": (130, 132, 138, 255),
}


def _color_for_ifc_class(ifc_class: str) -> tuple[int, int, int, int]:
    for key, rgba in _IFC_CLASS_COLORS.items():
        if ifc_class.startswith(key):
            return rgba
    return (210, 208, 204, 255)


def _clip_near(
    poly: list[tuple[float, float, float]],
    near: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Sutherland-Hodgman polygon clip against view-space plane ``z >= near``."""
    out: list[tuple[float, float, float]] = []
    n = len(poly)
    for i, a in enumerate(poly):
        b = poly[(i + 1) % n]
        a_in = a[2] >= near
        b_in = b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            dz = b[2] - a[2]
            t = (near - a[2]) / dz if abs(dz) > 1e-12 else 0.0
            out.append(
                (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    near,
                )
            )
    return out


def _vec_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_norm(a: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if mag < 1e-8:
        return (0.0, 0.0, 0.0)
    return (a[0] / mag, a[1] / mag, a[2] / mag)


_SKIP_IFC_CLASSES: tuple[str, ...] = (
    "IfcOpeningElement",
    "IfcOpeningStandardCase",
    "IfcSpace",
    "IfcVirtualElement",
    "IfcAnnotation",
    "IfcGrid",
)


def _should_skip_ifc_class(ifc_cls: str) -> bool:
    """Skip void/opening solids, space volumes, grids, and 2D annotations in 3D render/raycast meshes.

    Exact class names only: a prefix match on "IfcSpace" would also hide IfcSpaceHeater.
    """
    return ifc_cls in _SKIP_IFC_CLASSES


def compute_camera_pose(
    vmin: tuple[float, float, float] | list[float],
    vmax: tuple[float, float, float] | list[float],
    camera_preset: str,
    *,
    storey_elevations_m: list[float] | None = None,
    storey_index: int = 0,
    space_boxes: list[dict[str, float]] | None = None,
    width: int = 1024,
    height: int = 768,
    azimuth_override: float | None = None,
    elevation_override: float | None = None,
) -> dict[str, Any]:
    """Compute camera basis, eye/target coordinates, and constant ortho scale."""
    cam = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["isometric_cutaway"])
    vmin_t = (float(vmin[0]), float(vmin[1]), float(vmin[2]))
    vmax_t = (float(vmax[0]), float(vmax[1]), float(vmax[2]))
    center = (
        0.5 * (vmin_t[0] + vmax_t[0]),
        0.5 * (vmin_t[1] + vmax_t[1]),
        0.5 * (vmin_t[2] + vmax_t[2]),
    )
    dx = vmax_t[0] - vmin_t[0]
    dy = vmax_t[1] - vmin_t[1]
    dz = vmax_t[2] - vmin_t[2]
    extent = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0)
    radius = 0.5 * extent

    az_deg = cam["azimuth_deg"] if azimuth_override is None else azimuth_override
    el_deg = cam["elevation_deg"] if elevation_override is None else elevation_override
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cam_dir = (
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    )

    if cam["projection"] == "perspective":
        if storey_elevations_m:
            s_idx = min(max(0, storey_index), len(storey_elevations_m) - 1)
            storey_z = float(storey_elevations_m[s_idx])
        else:
            storey_z = vmin_t[2]
        eye_z = storey_z + 1.6

        chosen_space: dict[str, float] | None = None
        if space_boxes:
            on_storey = [sb for sb in space_boxes if abs(float(sb["zmin"]) - storey_z) < 0.6]
            candidates = on_storey or list(space_boxes)
            best_sb = max(
                candidates,
                key=lambda sb: max(0.0, float(sb["xmax"]) - float(sb["xmin"]))
                * max(0.0, float(sb["ymax"]) - float(sb["ymin"])),
            )
            if (float(best_sb["xmax"]) - float(best_sb["xmin"])) > 1.2 and (
                float(best_sb["ymax"]) - float(best_sb["ymin"])
            ) > 1.2:
                chosen_space = best_sb

        if chosen_space is not None:
            sx_min = float(chosen_space["xmin"])
            sy_min = float(chosen_space["ymin"])
            sx_max = float(chosen_space["xmax"])
            sy_max = float(chosen_space["ymax"])
            eye_z = float(chosen_space["zmin"]) + 1.6
            mx = min(0.8, 0.16 * (sx_max - sx_min))
            my = min(0.8, 0.16 * (sy_max - sy_min))
            corner_a = (sx_min + mx, sy_min + my, eye_z)
            corner_b = (sx_max - mx, sy_max - my, eye_z)
            is_opp = camera_preset == "opposite_interior" or (
                camera_preset != "eye_level_interior" and (az_deg % 360.0) >= 180.0
            )
            eye, target = (corner_b, corner_a) if is_opp else (corner_a, corner_b)
        else:
            eye = (
                center[0] - 0.18 * extent * math.cos(az),
                center[1] - 0.18 * extent * math.sin(az),
                eye_z,
            )
            target = (
                center[0] + 0.25 * extent * math.cos(az),
                center[1] + 0.25 * extent * math.sin(az),
                eye_z,
            )
    else:
        eye = (
            center[0] + cam_dir[0] * (1.8 * extent),
            center[1] + cam_dir[1] * (1.8 * extent),
            center[2] + cam_dir[2] * (1.8 * extent),
        )
        target = center

    forward = _vec_norm(_vec_sub(target, eye))
    world_up = (0.0, 0.0, 1.0)
    right = _vec_cross(forward, world_up)
    if math.sqrt(_vec_dot(right, right)) < 1e-6:
        right = (1.0, 0.0, 0.0)
    else:
        right = _vec_norm(right)
    up = _vec_norm(_vec_cross(right, forward))

    # Bounding-sphere constant ortho scale avoids 15% zoom pumping during turntables.
    px_scale = 0.86 * min(width, height) / (2.0 * radius)
    rel_center = _vec_sub(center, eye)
    cx_world = _vec_dot(rel_center, right)
    cy_world = _vec_dot(rel_center, up)
    fov = math.radians(68.0)
    focal = (0.5 * height) / math.tan(0.5 * fov)

    return {
        "projection": cam["projection"],
        "eye": eye,
        "target": target,
        "center": center,
        "extent": extent,
        "radius": radius,
        "forward": forward,
        "right": right,
        "up": up,
        "px_scale": px_scale,
        "cx_world": cx_world,
        "cy_world": cy_world,
        "focal": focal,
    }


def _rasterize_camera_views(
    meshes: list[tuple[Any, tuple[int, int, int, int]]],
    *,
    camera_preset: str,
    width: int,
    height: int,
    azimuth_override: float | None = None,
    elevation_override: float | None = None,
    storey_elevations_m: list[float] | None = None,
    storey_index: int = 0,
    space_boxes: list[dict[str, float]] | None = None,
    light_dir_override: tuple[float, float, float] | list[float] | None = None,
) -> tuple[Any, Any]:
    """Pure-NumPy headless software z-buffer rasterizer producing `(depth_pil, clay_pil)`."""
    import numpy as np
    from PIL import Image

    if not meshes:
        blank_d = Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")
        blank_c = Image.fromarray(np.full((height, width, 3), 245, dtype=np.uint8), mode="RGB")
        return blank_d, blank_c

    all_verts = np.vstack([m.vertices for m, _ in meshes])
    vmin = all_verts.min(axis=0)
    vmax = all_verts.max(axis=0)
    pose = compute_camera_pose(
        vmin.tolist(),
        vmax.tolist(),
        camera_preset,
        storey_elevations_m=storey_elevations_m,
        storey_index=storey_index,
        space_boxes=space_boxes,
        width=width,
        height=height,
        azimuth_override=azimuth_override,
        elevation_override=elevation_override,
    )
    eye = np.array(pose["eye"], dtype=np.float64)
    forward = np.array(pose["forward"], dtype=np.float64)
    right = np.array(pose["right"], dtype=np.float64)
    up = np.array(pose["up"], dtype=np.float64)

    if light_dir_override is not None:
        light_dir = np.array(light_dir_override, dtype=np.float64)
    else:
        light_dir = np.array([0.45, 0.35, 0.82], dtype=np.float64)
    light_norm = float(np.linalg.norm(light_dir))
    light_dir = light_dir / max(light_norm, 1e-8)

    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    cbuf = np.full((height, width, 3), 244, dtype=np.uint8)

    is_ortho = pose["projection"] == "ortho"
    px_scale = float(pose["px_scale"])
    cx_world = float(pose["cx_world"])
    cy_world = float(pose["cy_world"])
    focal = float(pose["focal"])
    near_plane = 0.05

    def _raster_tri_2d(
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        shaded: Any,
        perspective: bool,
    ) -> None:
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-5:
            return
        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
        if max_x < min_x or max_y < min_y:
            return

        gx = np.arange(min_x, max_x + 1, dtype=np.float32)[None, :] + 0.5
        gy = np.arange(min_y, max_y + 1, dtype=np.float32)[:, None] + 0.5
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            return

        if perspective:
            inv_z = w0 * (1.0 / z0) + w1 * (1.0 / z1) + w2 * (1.0 / z2)
            pz = 1.0 / np.maximum(inv_z, 1e-6)
        else:
            pz = w0 * z0 + w1 * z1 + w2 * z2

        patch_z = zbuf[min_y : max_y + 1, min_x : max_x + 1]
        vis = inside & (pz < patch_z)
        if not np.any(vis):
            return
        patch_z[vis] = pz[vis]
        patch_c = cbuf[min_y : max_y + 1, min_x : max_x + 1]
        patch_c[vis] = shaded

    for mesh, rgba in meshes:
        verts = mesh.vertices
        faces = mesh.faces
        if len(faces) == 0:
            continue
        rel = verts - eye
        vx = rel @ right
        vy = rel @ up
        vz = rel @ forward

        if is_ortho:
            sx = (vx - cx_world) * px_scale + width * 0.5
            sy = height * 0.5 - (vy - cy_world) * px_scale

        tri_normals = mesh.face_normals
        base_rgb = np.array(rgba[:3], dtype=np.float32)

        for fi, face in enumerate(faces):
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            if light_dir_override is not None:
                ndotl = max(0.0, float(np.dot(tri_normals[fi], light_dir)))
                shade = 0.25 + 0.75 * ndotl
            else:
                ndotl = abs(float(np.dot(tri_normals[fi], light_dir)))
                shade = 0.42 + 0.58 * ndotl
            shaded = np.clip(base_rgb * shade, 0, 255).astype(np.uint8)

            if is_ortho:
                _raster_tri_2d(
                    float(sx[i0]),
                    float(sy[i0]),
                    float(vz[i0]),
                    float(sx[i1]),
                    float(sy[i1]),
                    float(vz[i1]),
                    float(sx[i2]),
                    float(sy[i2]),
                    float(vz[i2]),
                    shaded,
                    perspective=False,
                )
            else:
                poly_vs = [
                    (float(vx[i0]), float(vy[i0]), float(vz[i0])),
                    (float(vx[i1]), float(vy[i1]), float(vz[i1])),
                    (float(vx[i2]), float(vy[i2]), float(vz[i2])),
                ]
                clipped = _clip_near(poly_vs, near=near_plane)
                if len(clipped) < 3:
                    continue
                proj = [
                    (
                        (p[0] / p[2]) * focal + width * 0.5,
                        height * 0.5 - (p[1] / p[2]) * focal,
                        p[2],
                    )
                    for p in clipped
                ]
                p0 = proj[0]
                for k in range(1, len(proj) - 1):
                    p1, p2 = proj[k], proj[k + 1]
                    _raster_tri_2d(
                        p0[0],
                        p0[1],
                        p0[2],
                        p1[0],
                        p1[1],
                        p1[2],
                        p2[0],
                        p2[1],
                        p2[2],
                        shaded,
                        perspective=True,
                    )

    valid = np.isfinite(zbuf)
    depth_img = np.zeros((height, width), dtype=np.uint8)
    if np.any(valid):
        zmin = float(zbuf[valid].min())
        zmax = float(zbuf[valid].max())
        span = max(zmax - zmin, 1e-4)
        # Closer = brighter (255), background = 0 (ControlNet depth convention).
        norm = np.clip(1.0 - (zbuf[valid] - zmin) / span, 0.0, 1.0)
        depth_img[valid] = (40.0 + 215.0 * norm).astype(np.uint8)

    return Image.fromarray(depth_img, mode="L"), Image.fromarray(cbuf, mode="RGB")


def _load_ifc_scene_and_metadata(
    ifc_path: str | Path,
) -> tuple[Any, list[tuple[str, Any, tuple[int, int, int, int]]], list[float], list[dict[str, float]]]:
    """Parse an IFC 4.3 file once into `(scene, labeled_meshes, storey_elevations_m, space_boxes)`.

    Skips opening voids, space volumes, virtual boundaries, grids, and 2D annotations so
    boolean opening cuts on walls stay open in both renders and solar raycasts.
    """
    import ifcopenshell  # type: ignore
    import ifcopenshell.geom  # type: ignore
    import ifcopenshell.util.unit  # type: ignore
    import numpy as np
    import trimesh  # type: ignore

    ifc_file = ifcopenshell.open(str(ifc_path))
    unit_scale = float(ifcopenshell.util.unit.calculate_unit_scale(ifc_file) or 1.0)
    storey_elevations_m = sorted(
        float(st.Elevation) * unit_scale
        for st in ifc_file.by_type("IfcBuildingStorey")
        if getattr(st, "Elevation", None) is not None
    )

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    scene = trimesh.Scene()
    labeled_meshes: list[tuple[str, Any, tuple[int, int, int, int]]] = []
    slab_top_zs: list[float] = []
    space_boxes: list[dict[str, float]] = []

    for product in ifc_file.by_type("IfcProduct"):
        if product.Representation is None:
            continue
        ifc_cls = product.is_a()
        is_space = ifc_cls == "IfcSpace"  # exact: IfcSpaceHeater is equipment, not a room
        if not is_space and _should_skip_ifc_class(ifc_cls):
            continue

        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
        except Exception:
            continue

        verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        faces = np.array(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        if len(verts) == 0 or len(faces) == 0:
            continue

        if is_space:
            space_boxes.append(
                {
                    "xmin": float(verts[:, 0].min()),
                    "ymin": float(verts[:, 1].min()),
                    "xmax": float(verts[:, 0].max()),
                    "ymax": float(verts[:, 1].max()),
                    "zmin": float(verts[:, 2].min()),
                }
            )
            continue

        if ifc_cls.startswith("IfcSlab"):
            slab_top_zs.append(float(verts[:, 2].max()))
        rgba = _color_for_ifc_class(ifc_cls)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.face_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(faces), 1))
        scene.add_geometry(mesh, node_name=f"{ifc_cls}_{product.id()}")
        labeled_meshes.append((ifc_cls, mesh, rgba))

    if not storey_elevations_m and slab_top_zs:
        storey_elevations_m = [min(slab_top_zs)]

    return scene, labeled_meshes, storey_elevations_m, space_boxes


def _load_ifc_scene_and_meshes(
    ifc_path: str | Path,
) -> tuple[Any, list[tuple[str, Any, tuple[int, int, int, int]]], list[float]]:
    """Backward-compatible loader returning `(trimesh.Scene, labeled_meshes, storey_elevations_m)`."""
    scene, labeled_meshes, storey_elevations_m, _ = _load_ifc_scene_and_metadata(ifc_path)
    return scene, labeled_meshes, storey_elevations_m


def _build_synthetic_floor_meshes(
    labeled_meshes: list[tuple[str, Any, tuple[int, int, int, int]]],
    storey_elevations_m: list[float],
    space_boxes: list[dict[str, float]] | None = None,
) -> list[tuple[Any, tuple[int, int, int, int]]]:
    """Synthesize flat render/depth-only floor quads per storey lacking an ``IfcSlab``."""
    import numpy as np
    import trimesh  # type: ignore

    if not labeled_meshes:
        return []

    slab_zs = [
        float(m.vertices[:, 2].max())
        for ifc_cls, m, _ in labeled_meshes
        if ifc_cls.startswith("IfcSlab") and len(m.vertices) > 0
    ]
    all_verts = np.vstack([m.vertices for _, m, _ in labeled_meshes])
    all_xmin, all_ymin = float(all_verts[:, 0].min()), float(all_verts[:, 1].min())
    all_xmax, all_ymax = float(all_verts[:, 0].max()), float(all_verts[:, 1].max())
    target_elevs = storey_elevations_m or [float(all_verts[:, 2].min())]

    floor_rgba = (205, 202, 196, 255)
    floors: list[tuple[Any, tuple[int, int, int, int]]] = []

    def _quad_mesh(xmin: float, ymin: float, xmax: float, ymax: float, z: float) -> Any:
        verts = np.array(
            [
                [xmin, ymin, z],
                [xmax, ymin, z],
                [xmax, ymax, z],
                [xmin, ymax, z],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.face_colors = np.tile(np.array(floor_rgba, dtype=np.uint8), (2, 1))
        return mesh

    for elev in target_elevs:
        if any(abs(sz - elev) <= 0.35 for sz in slab_zs):
            continue
        storey_spaces = [sb for sb in (space_boxes or []) if abs(float(sb["zmin"]) - elev) < 0.6]
        if storey_spaces:
            for sb in storey_spaces:
                floors.append(
                    (
                        _quad_mesh(
                            float(sb["xmin"]),
                            float(sb["ymin"]),
                            float(sb["xmax"]),
                            float(sb["ymax"]),
                            elev,
                        ),
                        floor_rgba,
                    )
                )
        else:
            st_verts = [
                m.vertices
                for _, m, _ in labeled_meshes
                if len(m.vertices) > 0 and elev - 0.6 <= float(m.vertices[:, 2].min()) <= elev + 1.0
            ]
            if st_verts:
                sv = np.vstack(st_verts)
                xmin, ymin = float(sv[:, 0].min()), float(sv[:, 1].min())
                xmax, ymax = float(sv[:, 0].max()), float(sv[:, 1].max())
            else:
                xmin, ymin, xmax, ymax = all_xmin, all_ymin, all_xmax, all_ymax
            floors.append((_quad_mesh(xmin, ymin, xmax, ymax, elev), floor_rgba))

    return floors


def _filter_meshes_for_camera(
    labeled_meshes: list[tuple[str, Any, tuple[int, int, int, int]]],
    camera_preset: str,
) -> list[tuple[Any, tuple[int, int, int, int]]]:
    cam = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["isometric_cutaway"])
    hidden_prefixes = tuple(cam.get("hide_cutaway_classes") or ())
    kept = [
        (mesh, rgba)
        for ifc_cls, mesh, rgba in labeled_meshes
        if not (hidden_prefixes and any(ifc_cls.startswith(p) for p in hidden_prefixes))
    ]
    if not kept:
        kept = [(mesh, rgba) for _, mesh, rgba in labeled_meshes]
    return kept


def ifc_to_glb_and_control(
    ifc_path: str | Path,
    out_dir: str | Path,
    *,
    camera_preset: str = "isometric_cutaway",
    width: int = 1024,
    height: int = 768,
) -> dict[str, str | None]:
    """Convert an IFC 4.3 file into an interactive ``model.glb`` + headless ``depth.png`` and ``clay.png``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scene, labeled_meshes, storey_elevations_m, space_boxes = _load_ifc_scene_and_metadata(ifc_path)
    glb_path = out_path / "model.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    control_meshes = _filter_meshes_for_camera(labeled_meshes, camera_preset)
    control_meshes = [*control_meshes, *_build_synthetic_floor_meshes(labeled_meshes, storey_elevations_m, space_boxes)]
    depth_pil, clay_pil = _rasterize_camera_views(
        control_meshes,
        camera_preset=camera_preset,
        width=width,
        height=height,
        storey_elevations_m=storey_elevations_m,
        space_boxes=space_boxes,
    )
    depth_path = out_path / "depth.png"
    clay_path = out_path / "clay.png"
    depth_pil.save(depth_path)
    clay_pil.save(clay_path)

    return {
        "glb_path": str(glb_path),
        "depth_path": str(depth_path),
        "clay_path": str(clay_path),
    }


def compose_grid(
    image_paths: list[str | Path],
    captions: list[str],
    out_path: str | Path,
    *,
    cols: int = 2,
    pad: int = 16,
) -> str:
    """Stitch multiple PNGs into a clean white presentation board with captions."""
    from PIL import Image, ImageDraw, ImageFont

    if not image_paths:
        raise ValueError("image_paths must be non-empty")

    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)
    caption_h = 44
    rows = math.ceil(len(imgs) / max(1, cols))

    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    board_w = pad + cols * (cell_w + pad)
    board_h = pad + rows * (cell_h + caption_h + pad)
    board = Image.new("RGB", (board_w, board_h), (255, 255, 255))
    draw = ImageDraw.Draw(board)

    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + caption_h + pad)
        resized = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        board.paste(resized, (x, y))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(220, 222, 226), width=2)
        cap = captions[idx] if idx < len(captions) else f"View {idx + 1}"
        draw.text((x + 10, y + cell_h + 10), cap, fill=(32, 34, 38), font=font)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_file)
    return str(out_file)


def render_ifc_turntable_gif(
    ifc_path: str | Path,
    out_gif_path: str | Path,
    *,
    num_frames: int = 18,
    width: int = 512,
    height: int = 384,
    elevation_deg: float = 34.0,
    duration_ms: int = 110,
) -> str:
    """Render a smooth 360-degree isometric cutaway turntable GIF directly from IFC geometry."""
    _, labeled_meshes, storey_elevations_m, space_boxes = _load_ifc_scene_and_metadata(ifc_path)
    meshes = [
        *_filter_meshes_for_camera(labeled_meshes, "isometric_cutaway"),
        *_build_synthetic_floor_meshes(labeled_meshes, storey_elevations_m, space_boxes),
    ]

    frames = []
    for frame_idx in range(max(4, num_frames)):
        az = (360.0 * frame_idx) / max(4, num_frames)
        _, clay_pil = _rasterize_camera_views(
            meshes,
            camera_preset="isometric_cutaway",
            width=width,
            height=height,
            azimuth_override=az,
            elevation_override=elevation_deg,
            storey_elevations_m=storey_elevations_m,
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


def render_storyboard_4up(
    ifc_path: str | Path,
    out_dir: str | Path,
    user_vibe: str,
    style_key: str,
    *,
    render_photoreal_shots: bool = True,
    backend: str = "sdxl_depth",
    steps: int = 18,
    seed: int = 42,
    width: int = 768,
    height: int = 576,
) -> dict[str, Any]:
    """Render a 4-angle presentation board (2x2) locked to a single IFC model."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _, labeled_meshes, storey_elevations_m, space_boxes = _load_ifc_scene_and_metadata(ifc_path)
    synth_floors = _build_synthetic_floor_meshes(labeled_meshes, storey_elevations_m, space_boxes)

    cam_keys = ["isometric_cutaway", "eye_level_interior", "opposite_interior", "corner_axonometric"]
    clay_paths: list[str] = []
    depth_paths: list[str] = []
    shots: list[str] = []
    captions: list[str] = []

    for cam_key in cam_keys:
        meshes = [*_filter_meshes_for_camera(labeled_meshes, cam_key), *synth_floors]
        depth_pil, clay_pil = _rasterize_camera_views(
            meshes,
            camera_preset=cam_key,
            width=width,
            height=height,
            storey_elevations_m=storey_elevations_m,
            space_boxes=space_boxes,
        )
        d_path = out_path / f"depth_{cam_key}.png"
        c_path = out_path / f"clay_{cam_key}.png"
        depth_pil.save(d_path)
        clay_pil.save(c_path)
        depth_paths.append(str(d_path))
        clay_paths.append(str(c_path))
        captions.append(CAMERA_PRESETS[cam_key]["label"])

        if render_photoreal_shots:
            r_path = render_photoreal(
                d_path,
                user_vibe,
                style_key,
                cam_key,
                out_path / f"render_{cam_key}.png",
                backend=backend,
                steps=steps,
                seed=seed,
            )
            shots.append(r_path)
        else:
            shots.append(str(c_path))

    board_path = compose_grid(shots, captions, out_path / "storyboard_2x2.png", cols=2)
    return {
        "board_path": board_path,
        "shots": shots,
        "clay_paths": clay_paths,
        "depth_paths": depth_paths,
    }


_PIPELINE_CACHE: dict[str, Any] = {}


def render_photoreal(
    control_image_path: str | Path,
    user_vibe: str,
    style_key: str,
    camera_key: str,
    out_path: str | Path,
    *,
    backend: str = "sdxl_depth",
    steps: int = 24,
    seed: int = 42,
) -> str:
    """Render a geometry-locked photorealistic image from ``depth.png``."""
    import torch  # type: ignore
    from PIL import Image

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    positive, negative = build_render_prompt(user_vibe, style_key, camera_key)
    control_img = Image.open(control_image_path).convert("RGB")
    gen = torch.Generator(device="cpu").manual_seed(seed)

    if backend == "flux_depth":
        from diffusers import FluxControlPipeline  # type: ignore

        pipe = _PIPELINE_CACHE.get("flux_depth")
        if pipe is None:
            pipe = FluxControlPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Depth-dev",
                torch_dtype=torch.bfloat16,
            )
            pipe.enable_model_cpu_offload()
            _PIPELINE_CACHE["flux_depth"] = pipe
        image = pipe(
            prompt=positive,
            control_image=control_img,
            height=control_img.height,
            width=control_img.width,
            num_inference_steps=steps,
            guidance_scale=10.0,
            generator=gen,
        ).images[0]
    else:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline  # type: ignore

        pipe = _PIPELINE_CACHE.get("sdxl_depth")
        if pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=torch.float16,
            )
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe.enable_model_cpu_offload()
            _PIPELINE_CACHE["sdxl_depth"] = pipe

        image = pipe(
            prompt=positive,
            negative_prompt=negative,
            image=control_img,
            num_inference_steps=steps,
            controlnet_conditioning_scale=0.65,
            generator=gen,
        ).images[0]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)
    return str(out_file)


def svg_to_png(svg_text: str, out_path: str | Path, *, output_width: int = 1280) -> str:
    """Convert a MutinyBIM SVG sheet into a PNG image for Gemma 4 Vision critique."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(out_file), output_width=output_width)
        return str(out_file)
    except Exception:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (output_width, int(output_width * 0.72)), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, img.width - 20, img.height - 20], outline=(40, 40, 40), width=3)
        draw.text((40, 40), "MutinyBIM Sheet Preview (install cairosvg for raster export)", fill=(30, 30, 30))
        img.save(out_file)
        return str(out_file)
