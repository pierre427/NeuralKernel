#!/usr/bin/env python3
"""kernel/web_tools.py — bridge the harness web tools (harness/tool-bridge/tiered-web.mjs: MCP + DuckDuckGo
fallback) into the Python kernel as the syscall-tier tools op:web_search / op:fetch_page. Used by the escalation
ladder's RESEARCH tier so the kernel can look up + build a solution it can't synthesize cold. Network I/O ->
syscall tier (capability-gated, NOT 'proven'). Calls node via subprocess (the tool itself lives in JS)."""
from __future__ import annotations
import os, json, shutil, subprocess


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "harness", "tool-bridge")):
            return d
        d = os.path.dirname(d)
    return "/Users/pierrelamy/Desktop/Test Harness"


_ROOT = _repo_root()
_TIERED = os.path.join(_ROOT, "harness", "tool-bridge", "tiered-web.mjs")
_SENT = "__WEBTOOL__"                                   # printable sentinel prefixing the node result line


def web_available() -> bool:
    """True iff the web tools can actually run (node present + the tiered-web module exists)."""
    return bool(shutil.which("node")) and os.path.isfile(_TIERED)


def _call_node(tool: str, args: dict, timeout: float = 30.0) -> dict:
    if not web_available():
        return {"ok": False, "error": "web tools unavailable (node or tiered-web.mjs missing)"}
    script = (f"import({json.dumps(_TIERED)}).then(async m => {{"
              f" const r = await m.executeTieredWebTool({json.dumps(tool)}, {json.dumps(args)});"
              f" process.stdout.write('\\n{_SENT}'+JSON.stringify(r)); }})"
              f".catch(e => {{ process.stdout.write('\\n{_SENT}'+JSON.stringify({{ok:false,error:String(e&&e.message||e)}})); }});")
    try:
        r = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=timeout, cwd=_ROOT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "web tool timeout"}
    except Exception as e:
        return {"ok": False, "error": f"web tool spawn failed: {type(e).__name__}: {e}"}
    out = r.stdout or ""
    # The genuine result is NEWLINE-prefixed ('\n'+_SENT); JSON.stringify can never emit a raw newline, so embedded
    # page/snippet text containing the literal sentinel cannot spoof or collide with the real result line.
    mark = "\n" + _SENT
    i = out.rfind(mark)
    if i < 0:
        return {"ok": False, "error": "no web-tool output", "stderr": (r.stderr or "")[-300:]}
    try:
        return json.loads(out[i + len(mark):])
    except Exception:
        return {"ok": False, "error": "unparseable web-tool output"}


def web_search(query: str, max_results: int = 5) -> dict:
    return _call_node("web_search", {"query": query, "max_results": int(max_results)})


def fetch_page(url: str) -> dict:
    return _call_node("fetch_page", {"url": url})


# ── syscall-tier tool wiring (register on a Scheduler as op:web_search / op:fetch_page) ──
def web_search_handler(sched, inputs):
    q = (inputs or {}).get("query")
    return web_search(q, (inputs or {}).get("max_results", 5)) if q else {"error": "web_search requires a query"}


def fetch_page_handler(sched, inputs):
    u = (inputs or {}).get("url")
    return fetch_page(u) if u else {"error": "fetch_page requires a url"}


def web_tool_specs():
    try:
        from .tools import ToolSpec
    except ImportError:
        from tools import ToolSpec
    return [
        ToolSpec(name="web_search", handler=web_search_handler,
                 description="Search the web (titles, URLs, snippets). Use to research how to solve a problem you "
                             "can't synthesize cold; follow up with fetch_page on a promising URL.",
                 signature="web_search(query:str, max_results=5)"),
        ToolSpec(name="fetch_page", handler=fetch_page_handler,
                 description="Fetch + extract the text of a web page (give it a URL from web_search).",
                 signature="fetch_page(url:str)"),
    ]


def register_web_tools(sched, grant: bool = True) -> list:
    try:
        from .tools import register_tool
    except ImportError:
        from tools import register_tool
    return [register_tool(sched, s, grant=grant) for s in web_tool_specs()]


__all__ = ["web_available", "web_search", "fetch_page", "web_tool_specs", "register_web_tools"]
