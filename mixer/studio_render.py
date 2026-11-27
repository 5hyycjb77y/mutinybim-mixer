"""Visual studio helpers for Vibe Architect Studio (`03_vibe_architect_studio.ipynb`).

Pure-stdlib presets and prompt builders are always importable locally.
Heavy 3D/diffusion dependencies (``ifcopenshell``, ``trimesh``, ``numpy``,
``PIL``, ``diffusers``, ``cairosvg``) are imported lazily inside rendering
functions so tests and lightweight tools work without GPU packages.
"""

from __future__ import annotations

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


def _rasterize_camera_views(
    meshes: list[tuple[Any, tuple[int, int, int, int]]],
    *,
    camera_preset: str,
    width: int,
    height: int,
) -> tuple[Any, Any]:
    """Pure-NumPy headless software z-buffer rasterizer producing `(depth_pil, clay_pil)`."""
    import math
    import numpy as np
    from PIL import Image

    cam = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["isometric_cutaway"])
    if not meshes:
        blank_d = Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")
        blank_c = Image.fromarray(np.full((height, width, 3), 245, dtype=np.uint8), mode="RGB")
        return blank_d, blank_c

    all_verts = np.vstack([m.vertices for m, _ in meshes])
    vmin = all_verts.min(axis=0)
    vmax = all_verts.max(axis=0)
    center = (vmin + vmax) * 0.5
    extent = float(np.linalg.norm(vmax - vmin))
    extent = max(extent, 1.0)

    az = math.radians(cam["azimuth_deg"])
    el = math.radians(cam["elevation_deg"])
    # Camera direction pointing from target toward eye (Z-up architectural frame).
    cam_dir = np.array(
        [
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ],
        dtype=np.float64,
    )
    if cam["projection"] == "perspective":
        # Place camera inside/near center looking outward across the room.
        eye = center + np.array([-0.22 * extent * math.cos(az), -0.22 * extent * math.sin(az), 0.05 * extent])
        target = center + np.array([0.25 * extent * math.cos(az), 0.25 * extent * math.sin(az), -0.02 * extent])
    else:
        eye = center + cam_dir * (1.8 * extent)
        target = center

    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    light_dir = np.array([0.45, 0.35, 0.82], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    cbuf = np.full((height, width, 3), 244, dtype=np.uint8)

    # Pre-project to compute orthographic scale cleanly if ortho.
    if cam["projection"] == "ortho":
        rel_all = all_verts - eye
        rx = rel_all @ right
        ry = rel_all @ up
        span_x = max(float(rx.max() - rx.min()), 1e-3)
        span_y = max(float(ry.max() - ry.min()), 1e-3)
        px_scale = 0.86 * min(width / span_x, height / span_y)
        cx_world = 0.5 * float(rx.min() + rx.max())
        cy_world = 0.5 * float(ry.min() + ry.max())
    else:
        fov = math.radians(68.0)
        focal = (0.5 * height) / math.tan(0.5 * fov)

    for mesh, rgba in meshes:
        verts = mesh.vertices
        faces = mesh.faces
        if len(faces) == 0:
            continue
        rel = verts - eye
        vx = rel @ right
        vy = rel @ up
        vz = rel @ forward

        if cam["projection"] == "ortho":
            sx = (vx - cx_world) * px_scale + width * 0.5
            sy = height * 0.5 - (vy - cy_world) * px_scale
        else:
            safe_z = np.maximum(vz, 0.05)
            sx = (vx / safe_z) * focal + width * 0.5
            sy = height * 0.5 - (vy / safe_z) * focal

        tri_normals = mesh.face_normals
        base_rgb = np.array(rgba[:3], dtype=np.float32)

        for fi, face in enumerate(faces):
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            z0, z1, z2 = float(vz[i0]), float(vz[i1]), float(vz[i2])
            if cam["projection"] == "perspective" and (z0 <= 0.05 or z1 <= 0.05 or z2 <= 0.05):
                continue

            x0, y0 = float(sx[i0]), float(sy[i0])
            x1, y1 = float(sx[i1]), float(sy[i1])
            x2, y2 = float(sx[i2]), float(sy[i2])

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-5:
                continue

            min_x = max(0, int(math.floor(min(x0, x1, x2))))
            max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
            min_y = max(0, int(math.floor(min(y0, y1, y2))))
            max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
            if max_x < min_x or max_y < min_y:
                continue

            gx = np.arange(min_x, max_x + 1, dtype=np.float32)[None, :] + 0.5
            gy = np.arange(min_y, max_y + 1, dtype=np.float32)[:, None] + 0.5
            w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
            w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside):
                continue

            pz = w0 * z0 + w1 * z1 + w2 * z2
            patch_z = zbuf[min_y : max_y + 1, min_x : max_x + 1]
            vis = inside & (pz < patch_z)
            if not np.any(vis):
                continue

            patch_z[vis] = pz[vis]
            ndotl = abs(float(np.dot(tri_normals[fi], light_dir)))
            shade = 0.42 + 0.58 * ndotl
            shaded = np.clip(base_rgb * shade, 0, 255).astype(np.uint8)
            patch_c = cbuf[min_y : max_y + 1, min_x : max_x + 1]
            patch_c[vis] = shaded

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


def ifc_to_glb_and_control(
    ifc_path: str | Path,
    out_dir: str | Path,
    *,
    camera_preset: str = "isometric_cutaway",
    width: int = 1024,
    height: int = 768,
) -> dict[str, str | None]:
    """Convert an IFC 4.3 file into an interactive ``model.glb`` + headless ``depth.png`` and ``clay.png``."""
    import ifcopenshell  # type: ignore
    import ifcopenshell.geom  # type: ignore
    import numpy as np
    import trimesh  # type: ignore

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ifc_file = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    scene = trimesh.Scene()
    cam = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["isometric_cutaway"])
    hidden_prefixes = tuple(cam.get("hide_cutaway_classes") or ())
    control_meshes: list[tuple[Any, tuple[int, int, int, int]]] = []

    for product in ifc_file.by_type("IfcProduct"):
        if product.Representation is None:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
        except Exception:
            continue

        verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        faces = np.array(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        if len(verts) == 0 or len(faces) == 0:
            continue

        ifc_cls = product.is_a()
        rgba = _color_for_ifc_class(ifc_cls)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.face_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(faces), 1))
        scene.add_geometry(mesh, node_name=f"{ifc_cls}_{product.id()}")

        if not (hidden_prefixes and any(ifc_cls.startswith(p) for p in hidden_prefixes)):
            control_meshes.append((mesh, rgba))

    glb_path = out_path / "model.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    if not control_meshes:
        # Fallback if model only had roof/coverings.
        control_meshes = [(geom, (215, 215, 215, 255)) for geom in scene.geometry.values()]

    depth_pil, clay_pil = _rasterize_camera_views(
        control_meshes,
        camera_preset=camera_preset,
        width=width,
        height=height,
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
