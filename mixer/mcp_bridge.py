"""Live bridge between a merged Gemma 4 OpenAI-compatible server and `mutinybim-mcp --http`.

Connects an inference endpoint (Colab tunnel or local vLLM/llama.cpp server)
to a running MutinyBIM MCP server (`npm run mcp:http` / `npm run demo`) so
the weight-space merged brain can co-design models interactively in `bim-editor`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from verifier.verify_client import extract_typescript


class McpHttpClient:
    """Minimal Streamable HTTP JSON-RPC client for `@permitzip/bim-mcp`."""

    def __init__(self, url: str = "http://127.0.0.1:3333/mcp", timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout
        self.session_id: str | None = None
        self._req_id = 0

    def _rpc(self, method: str, params: dict[str, Any] | None = None, *, is_notification: bool = False) -> dict[str, Any] | None:
        self._req_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not is_notification:
            payload["id"] = self._req_id
        if params is not None:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        host = urllib.parse.urlparse(self.url).hostname or ""
        opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if host in {"127.0.0.1", "localhost", "::1"}
            else urllib.request.build_opener()
        )
        with opener.open(req, timeout=self.timeout) as resp:
            sid = resp.headers.get("mcp-session-id")
            if sid:
                self.session_id = sid
            raw = resp.read().decode("utf-8", errors="replace").strip()
            ctype = resp.headers.get("content-type", "")

        if is_notification or not raw:
            return None

        if "text/event-stream" in ctype or raw.startswith("event:") or raw.startswith("data:"):
            # Extract last JSON-RPC data payload from SSE stream.
            last_obj: dict[str, Any] | None = None
            for line in raw.splitlines():
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk:
                        last_obj = json.loads(chunk)
            if last_obj is None:
                raise RuntimeError(f"Empty SSE response from MCP server: {raw[:200]}")
            return last_obj

        return json.loads(raw)

    def initialize(self) -> dict[str, Any]:
        resp = self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mutinybim-mixer-bridge", "version": "0.1.0"},
            },
        )
        self._rpc("notifications/initialized", {}, is_notification=True)
        return resp or {}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resp = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if not resp:
            return {}
        if "error" in resp:
            raise RuntimeError(f"MCP error on {name}: {resp['error']}")
        result = resp.get("result", {})
        content = result.get("content", [])
        if content and isinstance(content, list) and "text" in content[0]:
            text = content[0]["text"]
            if text.lstrip().startswith("{"):
                return json.loads(text)
            return {"message": text}
        return result


def call_openai_chat(
    llm_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str = "EMPTY",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    endpoint = llm_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def run_bridge_turn(
    mcp: McpHttpClient,
    *,
    llm_url: str,
    model_name: str,
    model_id: str,
    instruction: str,
    reference_text: str = "",
    auto_approve: bool = False,
    api_key: str = "EMPTY",
) -> dict[str, Any]:
    """Generate a MutinyBIM script, preflight via MCP, self-repair once on errors, and apply."""
    system = (
        "You are a merged multi-trade MutinyBIM Gemma 4 brain. Write one clean TypeScript script using "
        "@permitzip/bim-script (`export default defineScript(...)`) to fulfill the user's request.\n"
        + (f"\nReference:\n{reference_text[:8000]}" if reference_text else "")
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]

    for attempt in range(2):
        completion = call_openai_chat(llm_url, model_name, messages, api_key=api_key)
        ts_source = extract_typescript(completion)
        script_obj = {"source": ts_source, "reads": {"levels": True}}

        checked = mcp.call_tool("bim_check_output", {"script": script_obj})
        findings = checked.get("findings") or []
        if findings:
            if attempt == 0:
                messages.append({"role": "assistant", "content": completion})
                messages.append(
                    {
                        "role": "user",
                        "content": f"bim_check_output reported findings: {json.dumps(findings)}. Fix the script and return one valid ```ts block.",
                    }
                )
                continue
            return {"status": "check_failed", "findings": findings, "source": ts_source}

        preview = mcp.call_tool("bim_run_script", {"model_id": model_id, "script": script_obj})
        if preview.get("status") != "previewed":
            if attempt == 0:
                messages.append({"role": "assistant", "content": completion})
                messages.append(
                    {
                        "role": "user",
                        "content": f"bim_run_script failed: {json.dumps(preview)}. Fix the error and return one valid ```ts block.",
                    }
                )
                continue
            return {"status": "run_failed", "preview": preview, "source": ts_source}

        label = preview.get("label", "(no label)")
        steps = preview.get("steps", 0)
        print(f"\n[Preview] {label} ({steps} steps)")

        if not auto_approve:
            ans = input("Apply these edits to the live MutinyBIM model? [y/N]: ").strip().lower()
            if ans not in {"y", "yes"}:
                return {"status": "cancelled", "preview": preview, "source": ts_source}

        apply_args = (preview.get("apply_with") or {}).get("args")
        if not apply_args:
            return {"status": "error", "message": "Missing apply_with.args in preview", "preview": preview}

        applied = mcp.call_tool("bim_apply_output", apply_args)
        lint = mcp.call_tool("bim_lint_check", {"model_id": model_id})
        return {"status": "applied", "applied": applied, "lint": lint, "source": ts_source}

    return {"status": "failed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Connect merged Gemma 4 endpoint to MutinyBIM MCP server.")
    parser.add_argument("--llm-url", required=True, help="OpenAI-compatible base URL (e.g. https://xxxx.trycloudflare.com/v1)")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:3333/mcp", help="MutinyBIM MCP Streamable HTTP URL")
    parser.add_argument("--model", default="mutinybim-gemma4-merged", help="Model name served by the endpoint")
    parser.add_argument("--model-id", default="demo", help="Target MutinyBIM model id")
    parser.add_argument("--api-key", default="EMPTY", help="Optional Bearer API key")
    parser.add_argument("--auto-approve", action="store_true", help="Apply previewed script edits without confirmation")
    args = parser.parse_args(argv)

    mcp = McpHttpClient(args.mcp_url)
    mcp.initialize()
    ref_resp = mcp.call_tool("bim_script_reference", {})
    ref_text = str(ref_resp.get("markdown") or ref_resp.get("message") or "")
    print(f"Connected to MutinyBIM MCP at {args.mcp_url} (model_id={args.model_id}). Type 'quit' to exit.")

    while True:
        try:
            prompt = input("\nDesign prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in {"quit", "exit"}:
            break
        try:
            res = run_bridge_turn(
                mcp,
                llm_url=args.llm_url,
                model_name=args.model,
                model_id=args.model_id,
                instruction=prompt,
                reference_text=ref_text,
                auto_approve=args.auto_approve,
                api_key=args.api_key,
            )
            print(json.dumps({k: v for k, v in res.items() if k != "source"}, indent=2)[:1500])
        except urllib.error.URLError as exc:
            print(f"Network error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
