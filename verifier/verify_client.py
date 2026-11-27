"""Zero-dependency Python client for the warm MutinyBIM HTTP verifier (`verifier/server.mjs`)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

_THOUGHT_BLOCK_RE = re.compile(r"<\|channel>thought\s*[\s\S]*?<channel\|>", re.IGNORECASE)
_FENCED_CODE_RE = re.compile(
    r"```(?:ts|typescript|js|javascript)?\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)


def strip_gemma_thinking(text: str) -> tuple[str, str]:
    """Split a Gemma 4 completion into ``(thought_trace, final_answer)``.

    Handles closed ``<|channel>thought ... <channel|>`` blocks as well as
    unclosed thought streams cleanly.
    """
    if not text:
        return "", ""

    cleaned = text.replace("<|think|>", "")
    thoughts: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        inner = match.group(0)
        inner = re.sub(r"^<\|channel>thought\s*", "", inner, flags=re.IGNORECASE)
        inner = re.sub(r"<channel\|>$", "", inner, flags=re.IGNORECASE)
        thoughts.append(inner.strip())
        return ""

    final = _THOUGHT_BLOCK_RE.sub(_collect, cleaned)

    # Gracefully handle an unclosed <|channel>thought block if the model hit max_tokens.
    open_idx = final.find("<|channel>thought")
    if open_idx >= 0:
        tail = final[open_idx + len("<|channel>thought") :]
        fenced_in_tail = _FENCED_CODE_RE.search(tail)
        if fenced_in_tail:
            thoughts.append(tail[: fenced_in_tail.start()].strip())
            final = final[:open_idx] + tail[fenced_in_tail.start() :]
        else:
            thoughts.append(tail.strip())
            final = final[:open_idx]

    return "\n\n".join(t for t in thoughts if t), final.strip()


def extract_typescript(text: str) -> str:
    """Extract TypeScript source from a Gemma 4 completion."""
    _, answer = strip_gemma_thinking(text)
    target = answer or text
    matches = [m.group(1).strip() for m in _FENCED_CODE_RE.finditer(target) if m.group(1).strip()]
    if matches:
        with_define = [m for m in matches if "defineScript" in m]
        pool = with_define if with_define else matches
        return max(pool, key=len)

    # Fallback: slice from the first `import` or `export default` statement.
    m = re.search(r"^(?:import\s|export\s+default\s)", target, flags=re.MULTILINE)
    if m:
        return target[m.start() :].strip()
    return target.strip()


@dataclass
class VerifyResult:
    ok: bool
    reward: float
    breakdown: dict[str, float]
    stage_failed: str | None
    error: str | None
    steps: int
    label: str
    kinds: dict[str, int]
    lint_summary: dict[str, Any]
    elapsed_ms: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifyResult:
        return cls(
            ok=bool(data.get("ok", False)),
            reward=float(data.get("reward", 0.0)),
            breakdown={k: float(v) for k, v in (data.get("breakdown") or {}).items()},
            stage_failed=data.get("stage_failed"),
            error=data.get("error"),
            steps=int(data.get("steps", 0)),
            label=str(data.get("label", "")),
            kinds={k: int(v) for k, v in (data.get("kinds") or {}).items()},
            lint_summary=dict(data.get("lint_summary") or {}),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
        )


@dataclass
class ExportBundle:
    verify: VerifyResult
    ifc_text: str | None
    ifc_report: dict[str, Any] | None
    sheets: list[dict[str, str]]
    pdf_base64: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportBundle:
        return cls(
            verify=VerifyResult.from_dict(data),
            ifc_text=data.get("ifc_text"),
            ifc_report=data.get("ifc_report"),
            sheets=list(data.get("sheets") or []),
            pdf_base64=data.get("pdf_base64"),
        )


class MutinyVerifierClient:
    """HTTP client talking to `verifier/server.mjs`."""

    def __init__(self, base_url: str = "http://127.0.0.1:8787", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        host = urllib.parse.urlparse(url).hostname or ""
        opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if host in {"127.0.0.1", "localhost", "::1"}
            else urllib.request.build_opener()
        )
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Verifier HTTP {exc.code} at {path}: {body}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def reference(self, compact: bool = False) -> dict[str, Any]:
        suffix = "?compact=1" if compact else ""
        return self._request("GET", f"/reference{suffix}")

    def verify(
        self,
        source: str,
        *,
        start: str = "empty",
        params: dict[str, Any] | None = None,
        expected_kinds: dict[str, int] | None = None,
        min_elements: int | None = None,
        model_state: dict[str, Any] | None = None,
    ) -> VerifyResult:
        payload: dict[str, Any] = {"source": source, "start": start}
        if params is not None:
            payload["params"] = params
        if expected_kinds is not None:
            payload["expected_kinds"] = expected_kinds
        if min_elements is not None:
            payload["min_elements"] = min_elements
        if model_state is not None:
            payload["model_state"] = model_state
        return VerifyResult.from_dict(self._request("POST", "/verify", payload))

    def verify_batch(self, items: list[dict[str, Any]]) -> list[VerifyResult]:
        if not items:
            return []
        resp = self._request("POST", "/verify_batch", {"items": items})
        return [VerifyResult.from_dict(r) for r in resp.get("results", [])]

    def export_bundle(
        self,
        source: str,
        *,
        project_name: str = "Vibe Architect Studio",
        start: str = "empty",
        params: dict[str, Any] | None = None,
        expected_kinds: dict[str, int] | None = None,
        min_elements: int | None = None,
        include_ifc: bool = True,
        include_svg: bool = True,
        include_pdf: bool = True,
    ) -> ExportBundle:
        payload: dict[str, Any] = {
            "source": source,
            "project_name": project_name,
            "start": start,
            "include_ifc": include_ifc,
            "include_svg": include_svg,
            "include_pdf": include_pdf,
        }
        if params is not None:
            payload["params"] = params
        if expected_kinds is not None:
            payload["expected_kinds"] = expected_kinds
        if min_elements is not None:
            payload["min_elements"] = min_elements
        return ExportBundle.from_dict(self._request("POST", "/export", payload))


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def make_grpo_reward_func(
    client: MutinyVerifierClient,
    *,
    default_start: str = "empty",
    invert_for_clash_adapter: bool = False,
) -> Callable[..., list[float]]:
    """Return a TRL ``GRPOTrainer``-compatible reward function backed by MutinyBIM.

    When ``invert_for_clash_adapter=True`` (for training ``clash_negation``),
    scripts that execute cleanly *and* trigger ``bim-lint`` errors receive high
    reward so the resulting LoRA can later be negated in weight space.
    """

    def reward_func(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        starts = kwargs.get("start")
        expected_list = kwargs.get("expected_kinds")
        min_elems_list = kwargs.get("min_elements")

        batch: list[dict[str, Any]] = []
        for idx, comp in enumerate(completions):
            raw = _completion_to_text(comp)
            ts = extract_typescript(raw)
            item: dict[str, Any] = {
                "source": ts,
                "start": starts[idx] if isinstance(starts, list) and idx < len(starts) else default_start,
            }
            if isinstance(expected_list, list) and idx < len(expected_list) and expected_list[idx]:
                item["expected_kinds"] = expected_list[idx]
            if isinstance(min_elems_list, list) and idx < len(min_elems_list) and min_elems_list[idx] is not None:
                item["min_elements"] = min_elems_list[idx]
            batch.append(item)

        results = client.verify_batch(batch)
        if not invert_for_clash_adapter:
            return [r.reward for r in results]

        rewards: list[float] = []
        for r in results:
            if r.breakdown.get("kernel_check_ok", 0.0) < 1.0:
                rewards.append(0.0)
            else:
                errors = int(r.lint_summary.get("errors", 0))
                rewards.append(min(1.0, 0.2 + 0.4 * errors) if errors > 0 else 0.05)
        return rewards

    return reward_func
