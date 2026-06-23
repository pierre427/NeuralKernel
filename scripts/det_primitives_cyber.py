#!/usr/bin/env python3
"""CYBER / security primitives — the deterministic building blocks for security & DFIR gaps:
hashing, payload decoding + magic-byte identification, single-byte XOR break, path-traversal /
zip-slip guard, hostname allowlisting, SSRF target detection, beacon (C2) interval detection,
impossible-travel velocity, and Windows/Sysmon event classification.

Each is a GENERAL algorithm (solves ANY instance of the shape — never a memorized table) and is
proven by hold_out_cyber.py against an INDEPENDENT ground truth on thousands of random/novel inputs.

Each is a TRUSTED, self-tested deterministic block the model composes around (the LLM-as-JIT links
these in and writes only the task-specific glue)."""
from __future__ import annotations
import hashlib
import base64
import binascii
import posixpath
import math
import ipaddress
from urllib.parse import urlparse
from collections import Counter


# ============================================================ SHA-256 hex
def sha256_hex(data) -> str:
    """SHA-256 of `data` (str -> utf-8 bytes, or raw bytes) as a lowercase hex digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be str or bytes")
    return hashlib.sha256(bytes(data)).hexdigest()


# ============================================================ decode + magic-byte file type
_MAGIC = [
    (b"MZ", "pe"),
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip"),          # empty archive
    (b"PK\x07\x08", "zip"),          # spanned
    (b"\x7fELF", "elf"),
    (b"%PDF", "pdf"),
    (b"GIF8", "gif"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\x89PNG", "png"),
]


def _try_decode(s: str):
    """Return decoded bytes if `s` is valid base64 OR valid hex, else None (prefer the one that fits)."""
    t = s.strip()
    if not t:
        return None
    # hex: even length, all hex digits
    ht = t
    if len(ht) % 2 == 0 and ht and all(c in "0123456789abcdefABCDEF" for c in ht):
        try:
            return bytes.fromhex(ht)
        except ValueError:
            pass
    # base64: strict validation (validate=True rejects non-alphabet chars)
    try:
        return base64.b64decode(t, validate=True)
    except (binascii.Error, ValueError):
        return None


def decode_and_magic(s: str) -> str:
    """Detect base64/hex encoding of `s`, decode, and return file type by magic bytes.
    Returns one of pe/zip/elf/pdf/gif/png/unknown, or 'invalid' if `s` does not decode."""
    raw = _try_decode(s)
    if raw is None:
        return "invalid"
    for sig, name in _MAGIC:
        if raw.startswith(sig):
            return name
    return "unknown"


# ============================================================ single-byte XOR break
def xor_break_single(data: bytes, crib: bytes) -> int:
    """Find the single-byte key k in 0..255 s.t. `crib` is a substring of bytes(b ^ k for b in data).
    Returns the key, or -1 if no key reveals the crib."""
    if not crib:
        return -1
    data = bytes(data)
    crib = bytes(crib)
    for k in range(256):
        dec = bytes(b ^ k for b in data)
        if crib in dec:
            return k
    return -1


# ============================================================ path-traversal / zip-slip guard
def normalize_path(base: str, path: str):
    """Join base + path and normalize '..'/'.'; return the normalized path if it stays within `base`,
    else None. Posix semantics (zip-slip / directory-traversal guard). Absolute `path` escapes base."""
    base_norm = posixpath.normpath("/" + base.strip("/"))   # canonical absolute-style base
    if path.startswith("/"):
        joined = posixpath.normpath(path)
    else:
        joined = posixpath.normpath(posixpath.join(base_norm, path))
    # within-base iff joined == base_norm or starts with base_norm + '/'
    if joined == base_norm or joined.startswith(base_norm.rstrip("/") + "/"):
        return joined
    return None


# ============================================================ hostname allowlist (exact)
def url_hostname_allowed(url: str, allowlist) -> bool:
    """True only if the EXACT parsed hostname of `url` is in `allowlist`. Rejects suffix tricks
    (app.example.com.evil.com) and substring tricks (xexample.com)."""
    host = urlparse(url).hostname
    if host is None:
        return False
    host = host.lower().rstrip(".")               # case-insensitive, drop FQDN trailing dot
    allowed = {h.lower().rstrip(".") for h in allowlist}
    return host in allowed


# ============================================================ SSRF target detection
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}


def is_ssrf_target(host: str) -> bool:
    """True if `host` is an IP in a private/loopback/link-local range (10/8, 172.16/12, 192.168/16,
    127/8, 169.254/16, ::1, fc00::/7) or the cloud metadata IP 169.254.169.254. Non-IP -> False."""
    h = host.strip()
    if h in _METADATA_IPS:
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local)


# ============================================================ beacon (C2) interval detection
def detect_beacon(timestamps, tol: int) -> int:
    """Given sorted epoch-second timestamps, return the dominant periodic inter-arrival interval if
    the gaps cluster within +/-tol of a common value, else 0."""
    ts = list(timestamps)
    if len(ts) < 3:
        return 0
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    if any(g <= 0 for g in gaps):
        return 0
    # The gaps cluster within +/-tol of a COMMON value C iff every gap lies in [C-tol, C+tol], i.e.
    # there exists a center C with max(gaps)-min(gaps) <= 2*tol. (Jitter of +/-tol around a true
    # period P can put two gaps as far as 2*tol apart, so a median-centered test is too tight.)
    lo, hi = min(gaps), max(gaps)
    if hi - lo <= 2 * tol:
        # dominant value: the clear mode if one exists, else the cluster center (midrange).
        mode_val, mode_cnt = Counter(gaps).most_common(1)[0]
        if mode_cnt > 1:
            return int(mode_val)
        return int(round((lo + hi) / 2))
    return 0


# ============================================================ impossible-travel velocity
def haversine_kmh(lat1, lon1, t1, lat2, lon2, t2) -> float:
    """Great-circle distance (km) between two points divided by hours elapsed (t in epoch seconds).
    Returns travel speed in km/h; 0.0 if t2 == t1."""
    if t2 == t1:
        return 0.0
    R = 6371.0088                       # mean Earth radius, km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    dist_km = 2 * R * math.asin(min(1.0, math.sqrt(a)))
    hours = abs(t2 - t1) / 3600.0
    return dist_km / hours


# ============================================================ Windows/Sysmon event classification
_WIN_EVENTS = {
    4624: "logon_success",
    4625: "logon_failure",
    4634: "logoff",
    4688: "process_create",
    4720: "account_created",
    4724: "password_reset",
    1102: "log_cleared",
    7045: "service_install",
    4698: "scheduled_task",
    10:   "process_access",
}


def classify_win_event(event_id: int) -> str:
    """Map a Windows Security / Sysmon event ID to a tactic label; unknown -> 'other'."""
    return _WIN_EVENTS.get(event_id, "other")


MANUAL_CYBER = """CYBER / security primitives (USE these — exact signatures):
  sha256_hex(data) -> str                                  # SHA-256 hex (str or bytes)
  decode_and_magic(s) -> str                               # base64/hex decode + magic byte file type
                                                           #   pe/zip/elf/pdf/gif/png/unknown/invalid
  xor_break_single(data: bytes, crib: bytes) -> int        # single-byte XOR key revealing crib, or -1
  normalize_path(base, path) -> str|None                   # zip-slip/traversal guard (posix); None if escapes
  url_hostname_allowed(url, allowlist) -> bool             # EXACT hostname membership (rejects suffix/substring)
  is_ssrf_target(host) -> bool                             # private/loopback/link-local IP or metadata IP
  detect_beacon(timestamps, tol) -> int                    # dominant C2 beacon interval, or 0
  haversine_kmh(lat1,lon1,t1,lat2,lon2,t2) -> float        # impossible-travel speed km/h (0 if t2==t1)
  classify_win_event(event_id) -> str                      # Windows/Sysmon event ID -> tactic label
"""


if __name__ == "__main__":
    # sha256
    assert sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_hex(b"abc") == sha256_hex("abc")
    # decode + magic
    assert decode_and_magic(base64.b64encode(b"MZ\x90\x00stub").decode()) == "pe"
    assert decode_and_magic((b"%PDF-1.7").hex()) == "pdf"
    assert decode_and_magic("\x89PNG\r\n\x1a\n".encode("latin-1").hex()) == "png"
    assert decode_and_magic("zzzz!!!!") == "invalid"
    # xor break
    pt = b"the quick brown fox jumps"
    assert xor_break_single(bytes(b ^ 0x5a for b in pt), b"quick") == 0x5a
    # path guard
    assert normalize_path("/srv/data", "a/b/c.txt") == "/srv/data/a/b/c.txt"
    assert normalize_path("/srv/data", "../../../etc/passwd") is None
    assert normalize_path("/srv/data", "a/../b") == "/srv/data/b"
    # hostname allowlist
    assert url_hostname_allowed("https://example.com/x", ["example.com"])
    assert not url_hostname_allowed("https://app.example.com.evil.com/x", ["app.example.com"])
    assert not url_hostname_allowed("https://xexample.com/x", ["example.com"])
    # ssrf
    assert is_ssrf_target("169.254.169.254") and is_ssrf_target("10.1.2.3")
    assert is_ssrf_target("127.0.0.1") and is_ssrf_target("::1")
    assert not is_ssrf_target("8.8.8.8") and not is_ssrf_target("not-an-ip")
    # beacon
    assert detect_beacon([0, 60, 120, 180, 240], 2) == 60
    assert detect_beacon([0, 5, 90, 91, 400], 2) == 0
    # haversine
    sp = haversine_kmh(0, 0, 0, 0, 1, 3600)   # ~111.32 km in 1h
    assert 110 < sp < 113, sp
    assert haversine_kmh(0, 0, 5, 0, 1, 5) == 0.0
    # win events
    assert classify_win_event(4625) == "logon_failure"
    assert classify_win_event(1102) == "log_cleared"
    assert classify_win_event(999999) == "other"
    print("det_primitives_cyber self-test OK — sha256_hex, decode_and_magic, xor_break_single, "
          "normalize_path, url_hostname_allowed, is_ssrf_target, detect_beacon, haversine_kmh, "
          "classify_win_event")
