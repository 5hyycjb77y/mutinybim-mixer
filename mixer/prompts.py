"""Standardized single-trade training prompts and held-out multi-trade coordination benchmarks."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_TEMPLATE = """You are an expert AEC software engineer writing deterministic MutinyBIM TypeScript scripts.

Rules:
1. Import only from "@permitzip/bim-script" (e.g. defineScript, units, building, routes, parts, equipment, documents, drafting, checks).
2. Always export default defineScript({ kind, reads, parameters, run(ctx) { ... } }).
3. Use typed unit helpers (units.ft, units.in, units.mm) for every dimension — never bare numbers for lengths.
4. Keep MEP routing, ceiling heights, and equipment clearances clash-free so bim-lint reports 0 errors.
5. Output one clean ```ts fenced code block containing the complete runnable script.
"""

TRADE_TASKS: dict[str, list[dict[str, Any]]] = {
    "architecture": [
        {
            "id": "arch_two_room_suite",
            "trade": "architecture",
            "start": "empty",
            "prompt": "Create a 1-story 36 ft x 24 ft architectural office suite on Level 1 with 8 in exterior walls (11 ft tall), an interior 5 in partition dividing it into an open office and conference room, interior and exterior doors, three north windows, traced spaces, 9 ft ceilings, and an A-101 floor plan sheet.",
            "expected_kinds": {"level": 1, "wall": 5, "door": 2, "window": 3, "space": 2, "ceiling": 2, "sheet": 1},
            "min_elements": 15,
        },
        {
            "id": "arch_two_story_clinic",
            "trade": "architecture",
            "start": "empty",
            "prompt": "Build a 2-story 44 ft x 30 ft medical clinic (Level 1 at 0 ft, Level 2 at 12 ft). Split each floor into Reception/Waiting and two Exam Rooms using 5 in interior walls, add doors and south-wall windows, trace spaces from walls, place 9 ft ceilings, and lay out sheets A-101 and A-102.",
            "expected_kinds": {"level": 2, "wall": 8, "space": 6, "ceiling": 6, "sheet": 2},
            "min_elements": 24,
        },
        {
            "id": "arch_corridor_rooms",
            "trade": "architecture",
            "start": "empty",
            "prompt": "Model a single-level 50 ft x 28 ft building with a central 6 ft corridor and four private offices along the north and south sides, doors into every office, traced rooms, 9 ft ceilings, and a 1/4\" = 1'-0\" plan on sheet A-101.",
            "expected_kinds": {"level": 1, "wall": 6, "door": 4, "space": 5, "sheet": 1},
            "min_elements": 18,
        },
    ],
    "mechanical": [
        {
            "id": "mech_rtu_two_zones",
            "trade": "mechanical",
            "start": "empty",
            "prompt": "Create a 60 ft x 30 ft single-story building (14 ft walls, 10 ft ceilings) divided into two 30 ft x 30 ft classrooms. Size 4000 CFM supply and return rectangular ducts (12 in deep) via mechanical.duct_size, define and place a 10-ton RTU on the roof at 14 ft, route supply and return trunks at 11 ft above the ceilings, tag ducts, and place M-101 plan + M-601 equipment schedule sheets.",
            "expected_kinds": {"level": 1, "wall": 5, "space": 2, "run_segment": 2, "sheet": 2},
            "min_elements": 12,
        },
        {
            "id": "mech_split_ahu_office",
            "trade": "mechanical",
            "start": "empty",
            "prompt": "Build a 48 ft x 28 ft office with 13 ft walls and a 9 ft ceiling, define a custom 6-ton rooftop air handler part type with supply and return bottom connectors sized via mechanical.duct_size for 2400 CFM, place it at 13 ft, route clash-free supply and return ducts at 10.5 ft, auto-tag ducts, and put the mechanical plan on M-101.",
            "expected_kinds": {"level": 1, "wall": 4, "run_segment": 2, "sheet": 1},
            "min_elements": 10,
        },
    ],
    "plumbing": [
        {
            "id": "plumb_two_story_restroom_riser",
            "trade": "plumbing",
            "start": "empty",
            "prompt": "Model a 2-story plumbing riser (Level 1 at 0 ft, Level 2 at 12 ft) serving two water closets and two lavatories per floor. Size horizontal branches, stack, and building drain with plumbing.size_drain (2021 edition, 1/4 in/ft slope), compute Hunter curve water demand, route PVC sanitary piping connecting all fixtures, and document on sheet P-101.",
            "expected_kinds": {"level": 2, "part": 8, "run_segment": 3, "sheet": 1},
            "min_elements": 14,
        },
        {
            "id": "plumb_single_floor_core",
            "trade": "plumbing",
            "start": "empty",
            "prompt": "Create a single-level restroom bank with three water closets and two lavatories along a wet wall, size the horizontal sanitary branch and building drain via plumbing.size_drain at 1/4 in/ft slope, connect every fixture into the sloped PVC drain below the slab, and tag the drain on sheet P-101.",
            "expected_kinds": {"level": 1, "part": 5, "run_segment": 1, "sheet": 1},
            "min_elements": 8,
        },
    ],
    "electrical": [
        {
            "id": "elec_one_bed_apartment",
            "trade": "electrical",
            "start": "empty",
            "prompt": "Model a 32 ft x 14 ft apartment (Bedroom 101 and Living room 102) with 9 ft walls, place panel LP-1, propose and accept NEC 2023 210.52 receptacles via electrical.receptacles, circuit each room from a saved circuit_template, compute electrical.loads, and output panel schedule E-601 plus power plan E-101.",
            "expected_kinds": {"level": 1, "wall": 5, "space": 2, "part": 4, "sheet": 2},
            "min_elements": 15,
        },
        {
            "id": "elec_small_office_power",
            "trade": "electrical",
            "start": "empty",
            "prompt": "Create a 40 ft x 20 ft two-room office suite on Level 1, place a 125A 208V panelboard PP-1, run electrical.receptacles under NEC 2023 commercial/dwelling rules, apply a circuit template per room, evaluate electrical.loads, and generate sheets E-101 and E-601.",
            "expected_kinds": {"level": 1, "wall": 5, "space": 2, "sheet": 2},
            "min_elements": 12,
        },
    ],
    "drafting": [
        {
            "id": "draft_rtu_curb_detail",
            "trade": "drafting",
            "start": "empty",
            "prompt": "Draft a 3/4\" = 1'-0\" mechanical roof curb detail view using the drafting CAD API: sloped metal deck + rigid insulation regions with hatches, break lines, mirrored roof curb rails, RTU cabinet with supply/return drops, dimensions, left/right callout stacks, fitCrop, drafting.check QA, and place onto sheet M-501.",
            "expected_kinds": {"view": 1, "sheet": 1, "viewport": 1},
            "min_elements": 10,
        },
        {
            "id": "draft_exhaust_wall_cap",
            "trade": "drafting",
            "start": "empty",
            "prompt": "Create a 3/4\" = 1'-0\" ceiling exhaust fan and 6\" hooded exterior wall cap drafting detail with masonry + gypsum hatched regions, masking region across the wall penetration, flow arrow, dimensions, callout stacks, drafting.check QA, and place on sheet M-502.",
            "expected_kinds": {"view": 1, "sheet": 1, "viewport": 1},
            "min_elements": 8,
        },
    ],
    "structural": [
        {
            "id": "struct_steel_bay",
            "trade": "structural",
            "start": "empty",
            "prompt": "Lay out a single-story 40 ft x 30 ft structural enclosure on Level 1 with perimeter walls, interior bearing partition, traced rooms, and structural framing/grid documentation on sheet S-101.",
            "expected_kinds": {"level": 1, "wall": 4, "sheet": 1},
            "min_elements": 6,
        },
    ],
    "fire_protection": [
        {
            "id": "fp_office_zones",
            "trade": "fire_protection",
            "start": "empty",
            "prompt": "Model a 40 ft x 30 ft office floor on Level 1 with three enclosed rooms, 9 ft ceilings, and a fire-protection life-safety plan on sheet FP-101.",
            "expected_kinds": {"level": 1, "wall": 5, "space": 3, "ceiling": 3, "sheet": 1},
            "min_elements": 12,
        },
    ],
}

MULTI_TRADE_BENCHMARK: list[dict[str, Any]] = [
    {
        "id": "mt_office_arch_mech",
        "trades": ["architecture", "mechanical"],
        "start": "empty",
        "prompt": "Design a 50 ft x 30 ft 1-story office (Open Office, Conference, Private Office) with 14 ft exterior walls, 9 ft ceilings, doors and windows, PLUS a 10-ton RTU mounted at 14 ft routing sized supply and return ducts in the ceiling plenum at 11 ft with zero bim-lint clashes. Output architectural sheet A-101 and mechanical sheet M-101.",
        "expected_kinds": {"level": 1, "wall": 6, "door": 3, "space": 3, "ceiling": 3, "run_segment": 2, "sheet": 2},
        "min_elements": 20,
    },
    {
        "id": "mt_apartment_arch_elec",
        "trades": ["architecture", "electrical"],
        "start": "empty",
        "prompt": "Build a 36 ft x 18 ft studio + bedroom unit with exterior/interior walls, doors, windows, traced rooms, 9 ft ceilings, AND a full NEC 2023 electrical layout (panel LP-1, proposed receptacles, room circuit template, load calculation, panel schedule on E-601, power plan on E-101, architectural plan on A-101).",
        "expected_kinds": {"level": 1, "wall": 5, "space": 2, "part": 4, "sheet": 3},
        "min_elements": 18,
    },
    {
        "id": "mt_restroom_arch_plumb",
        "trades": ["architecture", "plumbing"],
        "start": "empty",
        "prompt": "Model two stacked 24 ft x 16 ft restroom suites (Level 1 at 0 ft, Level 2 at 12 ft) with enclosing walls, entry doors, rooms, AND a code-sized PVC sanitary riser + sloped branches/building drain serving two WCs and two lavatories per floor, documented on A-101 and P-101 with zero clashes.",
        "expected_kinds": {"level": 2, "wall": 8, "space": 2, "part": 8, "run_segment": 3, "sheet": 2},
        "min_elements": 24,
    },
    {
        "id": "mt_clinic_arch_mech_elec",
        "trades": ["architecture", "mechanical", "electrical"],
        "start": "empty",
        "prompt": "Create a 48 ft x 28 ft single-story clinic (Waiting 101, Exam 102) with 14 ft exterior walls, 9 ft ceilings, doors, windows, rooftop HVAC unit + plenum ducts at 11 ft sized for 3200 CFM, and NEC 2023 panel LP-1 + room-circuited receptacles. Place plans on A-101, M-101, and E-101 with zero clashes.",
        "expected_kinds": {"level": 1, "wall": 5, "space": 2, "ceiling": 2, "run_segment": 2, "part": 4, "sheet": 3},
        "min_elements": 20,
    },
    {
        "id": "mt_mech_plan_and_detail",
        "trades": ["mechanical", "drafting"],
        "start": "empty",
        "prompt": "Model a 40 ft x 24 ft workshop with a rooftop unit + supply duct at 11 ft (14 ft walls, 9 ft ceiling) on M-101, AND draft a 3/4\" = 1'-0\" RTU roof curb detail with hatches, dimensions, and callouts on sheet M-501.",
        "expected_kinds": {"level": 1, "wall": 4, "run_segment": 1, "sheet": 2},
        "min_elements": 12,
    },
    {
        "id": "mt_two_floor_all_trades",
        "trades": ["architecture", "mechanical", "plumbing", "electrical"],
        "start": "empty",
        "prompt": "Create a compact 2-story 40 ft x 24 ft commercial building (Level 1 at 0 ft, Level 2 at 12 ft, 12 ft floor-to-floor, 9 ft ceilings): walls, doors, and two rooms per floor; a 2-floor restroom sanitary stack sized via plumbing.size_drain; a rooftop HVAC unit feeding Level 2 plenum ducts; and panel LP-1 with NEC 2023 receptacles on Level 1. Keep everything clash-free.",
        "expected_kinds": {"level": 2, "wall": 8, "space": 4, "part": 6, "sheet": 1},
        "min_elements": 22,
    },
]


def format_task_prompt(task: dict[str, Any], kit_reference_snippet: str = "") -> list[dict[str, str]]:
    """Build Gemma 4 chat messages for training, GRPO, or benchmark evaluation."""
    sys_text = SYSTEM_PROMPT_TEMPLATE
    if kit_reference_snippet:
        sys_text = f"{sys_text}\n## MutinyBIM Script Kit Reference\n{kit_reference_snippet.strip()}\n"
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": str(task["prompt"])},
    ]
