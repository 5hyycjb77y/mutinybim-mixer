#!/usr/bin/env node
/**
 * Persistent warm HTTP verifier for MutinyBIM scripts (`@permitzip/bim-script`,
 * `@permitzip/bim-kernel`, `@permitzip/bim-lint`, `@permitzip/bim-first-party`).
 *
 * Keeps the TypeScript loader, first-party trade packs, and lint engine warm
 * in a single Node (>=22.13) process so GRPO rollouts and CMA-ES EvoMerge
 * evaluations run in milliseconds instead of paying cold Node startup per script.
 *
 * Usage:
 *   node verifier/server.mjs --mutinybim /path/to/mutinybim --port 8787
 */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import { stripTypeScriptTypes } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");

function parseCliArgs(argv) {
  let mutinybimDir = process.env.MUTINYBIM_DIR || path.resolve(REPO_ROOT, "mutinybim");
  let port = Number(process.env.VERIFIER_PORT || 8787);
  let host = process.env.VERIFIER_HOST || "127.0.0.1";
  let specPath = process.env.MUTINYBIM_SPEC || path.resolve(REPO_ROOT, "spec.json");

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--mutinybim" && argv[i + 1]) mutinybimDir = path.resolve(argv[++i]);
    else if (arg === "--port" && argv[i + 1]) port = Number(argv[++i]);
    else if (arg === "--host" && argv[i + 1]) host = argv[++i];
    else if (arg === "--spec" && argv[i + 1]) specPath = path.resolve(argv[++i]);
    else if (arg === "--help" || arg === "-h") {
      process.stdout.write("usage: node verifier/server.mjs [--mutinybim <dir>] [--port 8787] [--host 127.0.0.1] [--spec spec.json]\n");
      process.exit(0);
    }
  }
  return { mutinybimDir, port, host, specPath };
}

const args = parseCliArgs(process.argv.slice(2));
if (!existsSync(args.mutinybimDir)) {
  process.stderr.write(`[verifier] MutinyBIM directory not found: ${args.mutinybimDir}\n`);
  process.exit(1);
}

// Register tsx ESM loader from the cloned MutinyBIM repo so .ts workspace sources resolve cleanly.
const tsxApiPath = path.join(args.mutinybimDir, "node_modules/tsx/dist/esm/api/index.mjs");
if (existsSync(tsxApiPath)) {
  const tsxApi = await import(pathToFileURL(tsxApiPath).href);
  tsxApi.register();
} else {
  try {
    const tsxApi = await import("tsx/esm/api");
    tsxApi.register();
  } catch (err) {
    process.stderr.write(`[verifier] Could not load tsx/esm/api. Did you run npm install in ${args.mutinybimDir}?\n`);
    throw err;
  }
}

const pkgUrl = (rel) => pathToFileURL(path.join(args.mutinybimDir, rel)).href;
const kitUrl = pkgUrl("packages/bim-script/src/index.ts");

const [{ installFirstPartyAddOns }, kernel, bimScript, bimLint, bimIfc, bimSheets, bimProtocol] = await Promise.all([
  import(pkgUrl("packages/bim-first-party/src/index.ts")),
  import(pkgUrl("packages/bim-kernel/src/index.ts")),
  import(kitUrl),
  import(pkgUrl("packages/bim-lint/src/index.ts")),
  import(pkgUrl("packages/bim-ifc/src/index.ts")),
  import(pkgUrl("packages/bim-sheets/src/index.ts")),
  import(pkgUrl("packages/bim-sync/src/protocol-version.ts")),
]);

installFirstPartyAddOns();
// A disposable verifier plays both sides at the pinned release. Use its actual
// server feature set, not legacy defaults or an allow-all upgrade policy.
const FEATURES = bimProtocol.serverProtocol().features;

const { DEFAULT_SETTINGS, buildSheetSceneFor, elementValues, executeCommand, sceneToSvg } = kernel;
const {
  KIT_EXAMPLES,
  checkScriptOutput,
  kitDeclarations,
  kitUseWarnings,
  lintScript,
  readsWarnings,
  referenceMarkdown,
  resolveValues,
  runScript,
  sliceModel,
} = bimScript;
const { DEFAULT_LINT_CONFIG, LintEngine } = bimLint;
const { exportIfc } = bimIfc;
const { mergeSheetPdfs, modelPrintOptions, renderSheetPdf } = bimSheets;

const DEFAULT_REWARD_WEIGHTS = {
  format_and_lint: 0.15,
  compile_and_load: 0.15,
  executes_cleanly: 0.25,
  kernel_check_ok: 0.15,
  bim_lint_clean: 0.2,
  spec_match: 0.1,
};

function loadRewardWeights(specPath) {
  try {
    if (existsSync(specPath)) {
      const parsed = JSON.parse(readFileSync(specPath, "utf8"));
      if (parsed && typeof parsed.reward_weights === "object") {
        return { ...DEFAULT_REWARD_WEIGHTS, ...parsed.reward_weights };
      }
    }
  } catch {
    // Fallback to defaults if spec.json cannot be read.
  }
  return DEFAULT_REWARD_WEIGHTS;
}

const REWARD_WEIGHTS = loadRewardWeights(args.specPath);

let mutinybimCommit = "unknown";
try {
  mutinybimCommit = execFileSync("git", ["-C", args.mutinybimDir, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
} catch {
  // Ignore when mutinybimDir is not a git repo.
}

/**
 * Strip Gemma 4 `<|channel>thought ... <channel|>` blocks and extract the
 * primary TypeScript fenced block if present.
 */
function extractSource(raw) {
  if (typeof raw !== "string") return "";
  let text = raw.replace(/<\|think\|>/g, "");
  text = text.replace(/<\|channel>thought[\s\S]*?<channel\|>/g, "");
  // Handle unclosed thought block gracefully.
  const openThought = text.indexOf("<|channel>thought");
  if (openThought >= 0) {
    const closeThought = text.indexOf("<channel|>", openThought);
    if (closeThought < 0) {
      const fencedAfter = text.slice(openThought).match(/```(?:ts|typescript|js|javascript)?\s*\n([\s\S]*?)```/i);
      text = fencedAfter ? fencedAfter[0] : text.slice(0, openThought);
    }
  }

  const blocks = [...text.matchAll(/```(?:ts|typescript|js|javascript)?\s*\n([\s\S]*?)```/gi)];
  if (blocks.length > 0) {
    // Prefer the longest fenced block containing defineScript, or longest overall.
    const withDefine = blocks.filter((b) => /\bdefineScript\b/.test(b[1] || ""));
    const pool = withDefine.length > 0 ? withDefine : blocks;
    pool.sort((a, b) => (b[1] || "").length - (a[1] || "").length);
    return (pool[0][1] || "").trim();
  }
  return text.trim();
}

function makeBaseModel(start = "empty", customState = null) {
  if (customState && typeof customState === "object") return customState;
  const empty = {
    modelId: "eval",
    revision: 0,
    elements: {},
    connections: [],
    settings: DEFAULT_SETTINGS,
  };
  if (start !== "starter") return empty;
  const made = executeCommand(empty, {
    type: "create_level",
    levelId: "level-1",
    name: "Level 1",
    elevationMm: 0,
  });
  if (!made.ok) throw new Error(`Failed to create starter model: ${made.error.message}`);
  return made.model;
}

const KIT_SPECIFIER_RE = new RegExp("([\"'])" + "@permitzip/bim-script".replace("/", "\\/") + "\\1", "g");

async function loadScriptFromSource(source) {
  const js = stripTypeScriptTypes(source).replace(KIT_SPECIFIER_RE, JSON.stringify(kitUrl));
  const mod = await import("data:text/javascript;base64," + Buffer.from(js, "utf8").toString("base64"));
  if (!mod || !mod.default) throw new Error("Script has no default export.");
  const def = mod.default;
  if (!def.manifest || typeof def.runWith !== "function") {
    throw new Error("Default export is not a defineScript(...) definition.");
  }
  return def;
}

function computeSpecMatch(kinds, expectedKinds, minElements) {
  const totalElements = Object.values(kinds).reduce((a, b) => a + b, 0);
  if (expectedKinds && typeof expectedKinds === "object" && Object.keys(expectedKinds).length > 0) {
    const entries = Object.entries(expectedKinds);
    let met = 0;
    for (const [kind, minCount] of entries) {
      const actual = Number(kinds[kind] || 0);
      const target = Number(minCount || 1);
      if (target <= 0) met += 1;
      else met += Math.min(1, actual / target);
    }
    return met / entries.length;
  }
  if (typeof minElements === "number" && minElements > 0) {
    return Math.min(1, totalElements / minElements);
  }
  return totalElements > 0 ? 1.0 : 0.0;
}

function weightedReward(breakdown) {
  let score = 0;
  for (const [key, weight] of Object.entries(REWARD_WEIGHTS)) {
    score += Number(weight) * Number(breakdown[key] || 0);
  }
  return Math.max(0, Math.min(1, Number(score.toFixed(6))));
}

async function evaluateOne(req, options = {}) {
  const started = performance.now();
  const breakdown = {
    format_and_lint: 0,
    compile_and_load: 0,
    executes_cleanly: 0,
    kernel_check_ok: 0,
    bim_lint_clean: 0,
    spec_match: 0,
  };

  const source = extractSource(req?.source ?? "");
  if (!source) {
    return {
      ok: false,
      reward: 0,
      breakdown,
      stage_failed: "empty_source",
      error: "No TypeScript source found in completion.",
      steps: 0,
      label: "",
      kinds: {},
      lint_summary: { errors: 0, warnings: 0, infos: 0, findings: [] },
      elapsed_ms: Number((performance.now() - started).toFixed(2)),
    };
  }

  // Stage 1: Static lint & security gate (never execute unsafe source).
  const lintFindings = lintScript(source);
  const kitWarnings = kitUseWarnings(source);
  const sourceUnsafe = lintFindings.some((f) => f.rule === "source");
  const shapeBad = lintFindings.some((f) => f.rule === "shape");

  if (sourceUnsafe) {
    return {
      ok: false,
      reward: 0,
      breakdown,
      stage_failed: "security_lint",
      error: lintFindings.map((f) => f.message).join(" | "),
      steps: 0,
      label: "",
      kinds: {},
      lint_summary: { errors: 0, warnings: 0, infos: 0, findings: [] },
      elapsed_ms: Number((performance.now() - started).toFixed(2)),
    };
  }

  const staticPenalty = Math.min(0.6, lintFindings.length * 0.25 + kitWarnings.length * 0.15);
  breakdown.format_and_lint = shapeBad ? 0.25 : Math.max(0, 1.0 - staticPenalty);

  // Stage 2: Strip TypeScript types & load ESM module.
  let definition;
  try {
    definition = await loadScriptFromSource(source);
    breakdown.compile_and_load = 1.0;
  } catch (err) {
    return {
      ok: false,
      reward: weightedReward(breakdown),
      breakdown,
      stage_failed: "compile_and_load",
      error: err instanceof Error ? err.message : String(err),
      steps: 0,
      label: "",
      kinds: {},
      lint_summary: { errors: 0, warnings: 0, infos: 0, findings: [] },
      elapsed_ms: Number((performance.now() - started).toFixed(2)),
    };
  }

  const readIssues = readsWarnings(source, definition.manifest.reads);
  if (readIssues.length > 0) {
    breakdown.format_and_lint = Math.max(0, breakdown.format_and_lint - 0.1 * readIssues.length);
  }

  // Stage 3: Execute script against base model slice.
  let baseModel;
  let output;
  let scriptHash = "";
  try {
    baseModel = makeBaseModel(req?.start || "empty", req?.model_state || null);
    const slice = sliceModel(baseModel, definition.manifest.reads, [], FEATURES);
    const values = resolveValues(definition.manifest, req?.params || {});
    scriptHash = createHash("sha256").update(source, "utf8").digest("hex");
    output = await runScript(definition, { slice, values, scriptHash });
    breakdown.executes_cleanly = 1.0;
  } catch (err) {
    return {
      ok: false,
      reward: weightedReward(breakdown),
      breakdown,
      stage_failed: "run_script",
      error: err instanceof Error ? err.message : String(err),
      steps: 0,
      label: "",
      kinds: {},
      lint_summary: { errors: 0, warnings: 0, infos: 0, findings: [] },
      elapsed_ms: Number((performance.now() - started).toFixed(2)),
    };
  }

  // Stage 4: Kernel preflight + bim-lint coordination checks.
  let finalModel = baseModel;
  let lintFindingsOut = [];
  let checkResult;
  try {
    checkResult = checkScriptOutput(baseModel, output, {
      features: FEATURES,
      pictures: false,
      onAfterEdits: (after) => {
        finalModel = after;
        const engine = new LintEngine();
        engine.build(after);
        lintFindingsOut = engine.check({ config: DEFAULT_LINT_CONFIG });
      },
    });
  } catch (err) {
    return {
      ok: false,
      reward: weightedReward(breakdown),
      breakdown,
      stage_failed: "check_output",
      error: err instanceof Error ? err.message : String(err),
      steps: Number(output?.steps || 0),
      label: String(output?.label || ""),
      kinds: {},
      lint_summary: { errors: 0, warnings: 0, infos: 0, findings: [] },
      elapsed_ms: Number((performance.now() - started).toFixed(2)),
    };
  }

  if (checkResult.ok) {
    breakdown.kernel_check_ok = 1.0;
  } else {
    const refusedMsg =
      checkResult.kind === "edits"
        ? checkResult.batches.find((b) => b.status === "refused")?.message || "Edit batch refused."
        : checkResult.types.flatMap((t) => t.problems).join("; ") || "Part check failed.";
    return {
      ok: false,
      reward: weightedReward(breakdown),
      breakdown,
      stage_failed: "kernel_check",
      error: refusedMsg,
      steps: Number(output?.steps || 0),
      label: String(output?.label || ""),
      kinds: {},
      lint_summary: { errors: 0, warnings: 0, infos: 0, findings: [] },
      elapsed_ms: Number((performance.now() - started).toFixed(2)),
    };
  }

  const kinds = {};
  for (const el of elementValues(finalModel.elements)) {
    kinds[el.kind] = (kinds[el.kind] || 0) + 1;
  }

  let errors = 0;
  let warnings = 0;
  let infos = 0;
  for (const f of lintFindingsOut) {
    if (f.severity === "error") errors += 1;
    else if (f.severity === "warning") warnings += 1;
    else infos += 1;
  }

  // Smooth clash/coordination reward: 1.0 when zero errors & warnings, decaying cleanly.
  const lintPenalty = errors * 0.35 + warnings * 0.1;
  breakdown.bim_lint_clean = Math.max(0, 1.0 - Math.min(1.0, lintPenalty));
  breakdown.spec_match = computeSpecMatch(kinds, req?.expected_kinds, req?.min_elements);

  const ok = checkResult.ok && errors === 0;
  const result = {
    ok,
    reward: weightedReward(breakdown),
    breakdown,
    stage_failed: errors > 0 ? "bim_lint_errors" : null,
    error: errors > 0 ? lintFindingsOut.filter((f) => f.severity === "error").slice(0, 3).map((f) => f.message).join(" | ") : null,
    steps: Number(output?.steps || 0),
    label: String(output?.label || ""),
    kinds,
    lint_summary: {
      errors,
      warnings,
      infos,
      findings: lintFindingsOut.slice(0, 10).map((f) => ({
        id: f.id,
        rule: f.rule,
        severity: f.severity,
        message: f.message,
        at: f.at,
      })),
    },
    elapsed_ms: Number((performance.now() - started).toFixed(2)),
  };
  if (options.includeFinalModel) {
    result._finalModel = finalModel;
    result._scriptHash = scriptHash;
  }
  return result;
}

function slugify(name) {
  const cleaned = String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return cleaned || "mutinybim-model";
}

async function exportBundle(req) {
  const evalResult = await evaluateOne(req, { includeFinalModel: true });
  const { _finalModel: finalModel, _scriptHash: scriptHash, ...publicEval } = evalResult;
  if (!finalModel) {
    return {
      ...publicEval,
      ifc_text: null,
      ifc_report: null,
      sheets: [],
      pdf_base64: null,
    };
  }

  const projectName = String(req?.project_name || "Vibe Architect Studio");
  const timeStamp = typeof req?.time_stamp === "string" && req.time_stamp ? req.time_stamp : new Date().toISOString();
  const fileName = typeof req?.file_name === "string" && req.file_name ? req.file_name : `${slugify(projectName)}.ifc`;
  let ifcText = null;
  let ifcReport = null;
  if (req?.include_ifc !== false) {
    try {
      const ifc = exportIfc({
        model: finalModel,
        project: { id: "vibe-studio", name: projectName },
        timeStamp,
        fileName,
      });
      ifcText = ifc.text;
      ifcReport = ifc.report;
    } catch (err) {
      ifcReport = { error: err instanceof Error ? err.message : String(err) };
    }
  }

  const sheets = [];
  const renderedPdfs = [];
  const wantSvg = req?.include_svg !== false;
  const wantPdf = req?.include_pdf !== false;
  if (wantSvg || wantPdf) {
    const sheetElements = elementValues(finalModel.elements)
      .filter((el) => el.kind === "sheet" && !el.deleted)
      .sort((a, b) => String(a.number || a.id).localeCompare(String(b.number || b.id)));

    for (const sheet of sheetElements) {
      try {
        const scene = buildSheetSceneFor(finalModel, sheet.id, { projectName });
        const number = String(sheet.number || sheet.id);
        const title = String(sheet.title || sheet.name || number);
        if (wantSvg) {
          sheets.push({ id: sheet.id, number, title, svg: sceneToSvg(scene) });
        }
        if (wantPdf) {
          try {
            const pdf = await renderSheetPdf(
              scene,
              {
                title: `${number} ${title}`,
                modelRevision: finalModel.revision,
                snapshotSha256: scriptHash || "0".repeat(64),
                renderer: "mutinybim-mixer-v0.1",
              },
              undefined,
              null,
              modelPrintOptions(finalModel),
            );
            renderedPdfs.push({ number, title, bytes: pdf.bytes });
          } catch {
            // Skip PDF if optional title-block font assets are absent on host.
          }
        }
      } catch {
        // Continue exporting remaining sheets if one sheet fails to build.
      }
    }
  }

  let pdfBase64 = null;
  if (renderedPdfs.length === 1) {
    pdfBase64 = Buffer.from(renderedPdfs[0].bytes).toString("base64");
  } else if (renderedPdfs.length > 1) {
    try {
      const merged = await mergeSheetPdfs(renderedPdfs, {
        title: projectName,
        modelRevision: finalModel.revision,
        snapshotSha256: scriptHash || "0".repeat(64),
        renderer: "mutinybim-mixer-v0.1",
      });
      pdfBase64 = Buffer.from(merged.bytes).toString("base64");
    } catch {
      pdfBase64 = Buffer.from(renderedPdfs[0].bytes).toString("base64");
    }
  }

  return {
    ...publicEval,
    ifc_text: ifcText,
    ifc_report: ifcReport,
    sheets,
    pdf_base64: pdfBase64,
  };
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    const limit = 8 * 1024 * 1024;
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error("Request body too large."));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf8");
        resolve(raw ? JSON.parse(raw) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || args.host}`);

    if (req.method === "GET" && url.pathname === "/health") {
      return sendJson(res, 200, {
        status: "ok",
        mutinybim_dir: args.mutinybimDir,
        mutinybim_commit: mutinybimCommit,
        node_version: process.version,
        reward_weights: REWARD_WEIGHTS,
      });
    }

    if (req.method === "GET" && url.pathname === "/reference") {
      const compact = url.searchParams.get("compact") === "1";
      const md = referenceMarkdown();
      return sendJson(res, 200, {
        markdown: compact ? md.slice(0, 12000) : md,
        declarations: compact ? "" : kitDeclarations(),
        examples: KIT_EXAMPLES.map((ex) => ({
          name: ex.name,
          title: ex.title,
          builds: ex.builds,
          topics: ex.topics,
          source: ex.source,
        })),
      });
    }

    if (req.method === "POST" && url.pathname === "/verify") {
      const body = await readJsonBody(req);
      const result = await evaluateOne(body);
      return sendJson(res, 200, result);
    }

    if (req.method === "POST" && url.pathname === "/export") {
      const body = await readJsonBody(req);
      const result = await exportBundle(body);
      return sendJson(res, 200, result);
    }

    if (req.method === "POST" && url.pathname === "/verify_batch") {
      const body = await readJsonBody(req);
      const items = Array.isArray(body?.items) ? body.items : [];
      const results = [];
      for (const item of items) {
        results.push(await evaluateOne(item));
      }
      return sendJson(res, 200, { results });
    }

    return sendJson(res, 404, { error: "Not found" });
  } catch (err) {
    return sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
  }
});

server.listen(args.port, args.host, () => {
  process.stdout.write(
    `[mutinybim-verifier] listening on http://${args.host}:${args.port} (mutinybim=${mutinybimCommit.slice(0, 12)})\n`,
  );
});
