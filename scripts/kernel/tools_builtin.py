#!/usr/bin/env python3
"""kernel/tools_builtin.py — native SYSCALL-tier tool handlers: datetime + sysinfo.

handler signature is handler(sched, inputs: dict) -> result (the ABI register_tool wraps). stdlib-only with a
psutil fast-path where available; every host probe degrades gracefully so the tool never raises.

LEAST-EXPOSURE by default (user directive): sysinfo's HOST facet reports free disk for ONLY the running partition
and no paths/hostnames/usernames; broader/all-mounts detail is gated behind expose_all. The SCHEDULER facet
REDACTS secret-like tokens in slot contents UNCONDITIONALLY (before truncation and in verbose), then truncates;
verbose only lengthens the still-redacted preview. web_search/fetch_page are bridged elsewhere."""
from __future__ import annotations
import os, platform, re, shutil, subprocess
from collections import Counter

# Secret redaction for scheduler-facet slot contents — applied UNCONDITIONALLY (before truncation AND in verbose),
# so neither the model-settable verbose flag nor a short secret in the first chars can leak an artifact's secrets.
_SECRET_TOKEN_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{16,}|gho_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}"
    r"|xox[baprs]-[A-Za-z0-9-]{8,}|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,})")
_SECRET_KV_RE = re.compile(
    r"((?:api[_-]?key|secret|token|password|passwd|bearer|authorization|access[_-]?key)['\"]?\s*[:=]\s*['\"]?)"
    r"([^\s'\",}]{6,})", re.I)


def _redact(text: str) -> str:
    """Mask key-like tokens and secret-ish key=value assignments. Best-effort least-exposure, not a guarantee."""
    text = _SECRET_TOKEN_RE.sub("«redacted»", text)
    text = _SECRET_KV_RE.sub(lambda m: m.group(1) + "«redacted»", text)
    return text


# --------------------------------------------------------------------------- datetime
def datetime_handler(sched, inputs):
    import datetime as _dt
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    now_local = _dt.datetime.now().astimezone()
    return {
        "utc_iso": now_utc.isoformat(),
        "local_iso": now_local.isoformat(),
        "epoch": now_utc.timestamp(),
        "tz": str(now_local.tzinfo),
        "weekday": now_utc.strftime("%A"),
    }


# --------------------------------------------------------------------------- sysinfo: host facet
def _processor_brand():
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=2)
            if r.stdout.strip():
                return r.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _memory_info():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total": vm.total, "available": vm.available, "used": vm.used, "percent": vm.percent}
    except Exception:
        pass
    info = {}
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
            info["total"] = int(r.stdout.strip())
    except Exception:
        pass
    if "total" not in info:
        try:
            info["total"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            pass
    return info


def _all_partitions():
    parts = []
    try:
        import psutil
        for p in psutil.disk_partitions(all=False):
            try:
                u = shutil.disk_usage(p.mountpoint)
                parts.append({"mount": p.mountpoint, "fstype": p.fstype, "total": u.total, "free": u.free})
            except Exception:
                continue
        return parts
    except Exception:
        return [{"note": "psutil unavailable; install psutil for the all-partition view"}]


def _gpu_info():
    """Apple-Silicon GPU pressure via ioreg (NON-SUDO). 'Device Utilization %' is the GPU-busy percentage; the MAX
    across accelerator nodes is the binding GPU pressure. Sampled at a scheduler DECISION POINT (when this process is
    between generations) it reflects OTHER GPU consumers — the contention the CPU/mem/queue sensors miss. Returns
    None off-Darwin / if ioreg is unavailable (the cost model then contributes 0 GPU pressure, never raises)."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=2.0).stdout or ""
    except Exception:
        return None
    vals = [int(x) for x in re.findall(r'"Device Utilization %"=(\d+)', out)]
    return {"util_pct": max(vals)} if vals else None


def _host_info(expose_all=False):
    info = {
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": _processor_brand(),
        "cpu_count": os.cpu_count(),
        "memory": _memory_info(),
        "gpu": _gpu_info(),
    }
    try:
        info["load_avg"] = list(os.getloadavg())          # (1, 5, 15) min
    except (OSError, AttributeError):
        info["load_avg"] = None
    # free disk: ONLY the partition this process runs on (least-exposure default)
    try:
        du = shutil.disk_usage(os.getcwd())
        info["disk_running_partition"] = {"total": du.total, "used": du.used, "free": du.free}
    except Exception as e:
        info["disk_running_partition"] = {"error": f"{type(e).__name__}: {e}"}
    if expose_all:                                         # gated: explicit opt-in only
        info["disk_all_partitions"] = _all_partitions()
    return info


# --------------------------------------------------------------------------- sysinfo: scheduler facet
def _state_str(s):
    return getattr(s, "value", None) or getattr(s, "name", None) or (str(s) if s is not None else None)


_HELD_STATES = {"BLOCKED", "WAITING_FOR_PRIMITIVE", "WAITING_FOR_AGENT", "WAITING_FOR_TOOL", "REPAIRING"}


def _scheduler_info(sched, verbose=False):
    if sched is None:
        return {"note": "no scheduler handle"}
    out = {}
    try:
        tasks = getattr(sched, "tasks", {}) or {}
        procs = []
        for tid, t in tasks.items():
            procs.append({
                "id": tid, "kind": getattr(t, "kind", None), "owner": getattr(t, "owner", None),
                "state": _state_str(getattr(t, "status", None)),
                "deps": list(getattr(t, "dependencies", []) or []),
                "verification": _state_str(getattr(t, "verification", None)),
                "retries": getattr(t, "retries", None),
            })
        by_state = Counter(p["state"] for p in procs)
        held = [p for p in procs if p["state"] in _HELD_STATES]
        out["queue_depths"] = dict(by_state)
        out["ready_depth"] = by_state.get("READY", 0)
        out["running_depth"] = by_state.get("RUNNING", 0)
        out["held_depth"] = len(held)
        out["process_table"] = procs
        out["held_processes"] = held
        out["capabilities"] = sorted(getattr(sched, "capabilities", set()) or set())
    except Exception as e:
        out["scheduler_error"] = f"{type(e).__name__}: {e}"
    # memory slots — the ArtifactStore (content-addressed FDs)
    try:
        store = getattr(sched, "artifacts", None)
        a = getattr(store, "_a", {}) if store is not None else {}
        slots = []
        for ref, art in a.items():
            preview = _redact(repr(getattr(art, "value", None)))   # redact secrets UNCONDITIONALLY, before truncation
            slots.append({
                "ref": ref, "kind": getattr(art, "kind", None), "hash": getattr(art, "content_hash", None),
                "size": len(preview),
                "content": preview if verbose else (preview[:120] + ("…" if len(preview) > 120 else "")),
            })
        out["memory_slots"] = slots
        out["slot_count"] = len(slots)
    except Exception as e:
        out["slots_error"] = f"{type(e).__name__}: {e}"
    # trap budget (commit_trap.TrapBudget, lives at sched.budget)
    try:
        b = getattr(sched, "budget", None)
        if b is not None:
            out["trap_budget"] = {
                "max_traps": getattr(b, "max_traps", None),
                "max_same_op": getattr(b, "max_same_op", None),
                "disabled_ops": sorted(getattr(b, "disabled_ops", set()) or set()),
            }
    except Exception:
        pass
    return out


def sysinfo_handler(sched, inputs):
    inputs = inputs or {}
    facet = inputs.get("facet", "all")
    expose_all = bool(inputs.get("expose_all", False))
    verbose = bool(inputs.get("verbose", False))
    out = {}
    if facet in ("host", "all"):
        out["host"] = _host_info(expose_all)
    if facet in ("scheduler", "kernel", "all"):
        out["scheduler"] = _scheduler_info(sched, verbose)
    return out


__all__ = ["datetime_handler", "sysinfo_handler"]
