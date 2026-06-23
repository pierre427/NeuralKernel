#!/usr/bin/env python3
"""Challenge-200 spec-matched recoveries.

These entry names COLLIDE with shape_recoveries.RECOVERIES but have DIFFERENT contracts
(e.g. challenge topological_sort(n, edges)->list vs the shared topological_sort(graph)->list).
They live in RECOVERIES_EXTRA so the appliance can route them on contract-match without
overwriting the shared library. Each value is a self-contained code string defining a function
named by its key, matching the EXACT challenge signature/output format.
"""
from __future__ import annotations

RECOVERIES_EXTRA = {
    # c004 Andrew's monotone chain; CCW, start at bottom-most left-most; output tuples-as-lists.
    "convex_hull": r'''
def convex_hull(points):
    pts = sorted(set((p[0], p[1]) for p in points))
    if len(pts) <= 1:
        return [list(p) for p in pts]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]   # CCW, starts at bottom-most left-most (first of sorted)
    return [list(p) for p in hull]
''',

    # c007 Aho-Corasick: map each pattern -> sorted start indices.
    "aho_corasick": r'''
def aho_corasick(text, patterns):
    from collections import deque, defaultdict
    goto = [{}]; fail = [0]; out = [[]]
    for idx, p in enumerate(patterns):
        cur = 0
        for ch in p:
            if ch not in goto[cur]:
                goto.append({}); fail.append(0); out.append([]); goto[cur][ch] = len(goto)-1
            cur = goto[cur][ch]
        out[cur].append(idx)
    dq = deque()
    for ch, v in goto[0].items():
        fail[v] = 0; dq.append(v)
    while dq:
        u = dq.popleft()
        for ch, v in goto[u].items():
            dq.append(v)
            f = fail[u]
            while f and ch not in goto[f]:
                f = fail[f]
            fail[v] = goto[f][ch] if (f or ch in goto[0]) and ch in goto[f] else 0
            out[v] = out[v] + out[fail[v]]
    res = {p: [] for p in patterns}
    cur = 0
    for i, ch in enumerate(text):
        while cur and ch not in goto[cur]:
            cur = fail[cur]
        cur = goto[cur].get(ch, 0)
        for pid in out[cur]:
            plen = len(patterns[pid])
            res[patterns[pid]].append(i - plen + 1)
    for p in res:
        res[p] = sorted(set(res[p]))
    return res
''',

    # c012 Kahn topo sort over nodes 0..n-1, lexicographically smallest, [] on cycle.
    "topological_sort": r'''
def topological_sort(n, edges):
    import heapq
    indeg = [0]*n
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); indeg[v] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        u = heapq.heappop(heap); out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    return out if len(out) == n else []
''',

    # c013 Dijkstra all distances from source; unreachable -> -1.
    "dijkstra": r'''
def dijkstra(n, edges, source):
    import heapq
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    INF = float("inf")
    dist = [INF]*n; dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    return [int(x) if x != INF else -1 for x in dist]
''',

    # c031 max-weight bipartite matching; small -> Hungarian-style via DP over masks / Kuhn with weights.
    # Use min-cost-max-flow-free approach: weighted matching by DP over right mask (n,m small in cases).
    "max_bipartite_weight": r'''
def max_bipartite_weight(n, m, edges):
    # dp over subsets of right vertices, assigning left vertices 0..n-1 in order (or skip).
    from functools import lru_cache
    w = {}
    for l, r, wt in edges:
        if (l, r) not in w or wt > w[(l, r)]:
            w[(l, r)] = wt
    full = (1 << m) - 1
    import sys
    sys.setrecursionlimit(10000)
    @lru_cache(maxsize=None)
    def best(i, used):
        if i == n:
            return 0
        res = best(i+1, used)            # leave left i unmatched
        for r in range(m):
            if not (used >> r) & 1 and (i, r) in w:
                res = max(res, w[(i, r)] + best(i+1, used | (1 << r)))
        return res
    return best(0, 0)
''',

    # c041 min-cut edges: max flow (Dinic), then BFS reachable in residual; original edges S->T sorted.
    "min_cut": r'''
def min_cut(n, edges, source, sink):
    from collections import deque
    g = [[] for _ in range(n)]
    orig = []
    def add(u, v, c):
        g[u].append([v, c, len(g[v])]); g[v].append([u, 0, len(g[u])-1])
    for u, v, c in edges:
        add(u, v, c); orig.append((u, v))
    def bfs():
        lv = [-1]*n; lv[source] = 0; dq = deque([source])
        while dq:
            u = dq.popleft()
            for v, c, _ in g[u]:
                if c > 0 and lv[v] < 0:
                    lv[v] = lv[u]+1; dq.append(v)
        return lv
    def dfs(u, f, lv, it):
        if u == sink:
            return f
        while it[u] < len(g[u]):
            e = g[u][it[u]]; v, c, rev = e
            if c > 0 and lv[v] == lv[u]+1:
                d = dfs(v, min(f, c), lv, it)
                if d > 0:
                    e[1] -= d; g[v][rev][1] += d; return d
            it[u] += 1
        return 0
    while True:
        lv = bfs()
        if lv[sink] < 0:
            break
        it = [0]*n
        while True:
            f = dfs(source, float("inf"), lv, it)
            if f == 0:
                break
    # residual reachability from source
    reach = [False]*n; reach[source] = True; dq = deque([source])
    while dq:
        u = dq.popleft()
        for v, c, _ in g[u]:
            if c > 0 and not reach[v]:
                reach[v] = True; dq.append(v)
    cut = [(u, v) for (u, v) in orig if reach[u] and not reach[v]]
    cut = sorted(set(cut))
    return [list(e) for e in cut]
''',

    # c052 interval tree: for each query, intervals containing it, sorted by start then end.
    "interval_tree": r'''
def interval_tree(intervals, queries):
    ivs = [(s, e) for s, e in intervals]
    out = []
    for q in queries:
        hit = [iv for iv in ivs if iv[0] <= q <= iv[1]]
        hit.sort(key=lambda x: (x[0], x[1]))
        out.append([list(iv) for iv in hit])
    return out
''',

    # c057 order-statistic tree: insert/delete (one occurrence)/kth(1-indexed)/rank(#<val + 1).
    "order_statistic_tree": r'''
def order_statistic_tree(operations):
    import bisect
    arr = []
    out = []
    for op in operations:
        kind = op[0]
        if kind == "insert":
            bisect.insort(arr, op[1])
        elif kind == "delete":
            i = bisect.bisect_left(arr, op[1])
            if i < len(arr) and arr[i] == op[1]:
                arr.pop(i)
        elif kind == "kth":
            out.append(arr[op[1]-1])
        elif kind == "rank":
            out.append(bisect.bisect_left(arr, op[1]) + 1)
    return out
''',

    # c062 2D grid: updates set cells, queries sum subgrid. Direct (cases small).
    "segment_tree_2d": r'''
def segment_tree_2d(n, m, updates, queries):
    grid = [[0]*m for _ in range(n)]
    for r, c, val in updates:
        grid[r][c] = val
    # prefix sums
    pre = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            pre[i+1][j+1] = grid[i][j] + pre[i][j+1] + pre[i+1][j] - pre[i][j]
    out = []
    for r1, c1, r2, c2 in queries:
        s = pre[r2+1][c2+1] - pre[r1][c2+1] - pre[r2+1][c1] + pre[r1][c1]
        out.append(s)
    return out
''',

    # c076 2D range count.
    "range_tree": r'''
def range_tree(points, queries):
    pts = [(p[0], p[1]) for p in points]
    out = []
    for x1, y1, x2, y2 in queries:
        out.append(sum(1 for x, y in pts if x1 <= x <= x2 and y1 <= y <= y2))
    return out
''',

    # c083 Thompson NFA full-match for a(b|c)*d style. Supports . * + ? | ( ).
    "regex_to_nfa": r'''
def regex_to_nfa(pattern, test_strings):
    # Recursive-descent parse into postfix-free AST, then Thompson NFA, then simulate.
    pos = [0]
    pat = pattern
    def parse_alt():
        node = parse_concat()
        while pos[0] < len(pat) and pat[pos[0]] == "|":
            pos[0] += 1
            rhs = parse_concat()
            node = ("alt", node, rhs)
        return node
    def parse_concat():
        nodes = []
        while pos[0] < len(pat) and pat[pos[0]] not in "|)":
            nodes.append(parse_rep())
        if not nodes:
            return ("eps",)
        node = nodes[0]
        for nx in nodes[1:]:
            node = ("cat", node, nx)
        return node
    def parse_rep():
        node = parse_atom()
        while pos[0] < len(pat) and pat[pos[0]] in "*+?":
            op = pat[pos[0]]; pos[0] += 1
            node = ("rep", op, node)
        return node
    def parse_atom():
        c = pat[pos[0]]
        if c == "(":
            pos[0] += 1
            node = parse_alt()
            pos[0] += 1  # ')'
            return node
        if c == ".":
            pos[0] += 1
            return ("dot",)
        pos[0] += 1
        return ("lit", c)
    tree = parse_alt()
    # build NFA: states as ints, eps + labeled transitions
    trans = {}
    cnt = [0]
    def new():
        s = cnt[0]; cnt[0] += 1; trans[s] = []; return s
    def build(node):
        t = node[0]
        if t == "eps":
            s = new(); e = new(); trans[s].append((None, e)); return s, e
        if t == "lit":
            s = new(); e = new(); trans[s].append((("lit", node[1]), e)); return s, e
        if t == "dot":
            s = new(); e = new(); trans[s].append((("dot",), e)); return s, e
        if t == "cat":
            s1, e1 = build(node[1]); s2, e2 = build(node[2]); trans[e1].append((None, s2)); return s1, e2
        if t == "alt":
            s1, e1 = build(node[1]); s2, e2 = build(node[2]); s = new(); e = new()
            trans[s] += [(None, s1), (None, s2)]; trans[e1].append((None, e)); trans[e2].append((None, e))
            return s, e
        if t == "rep":
            op = node[1]; s1, e1 = build(node[2]); s = new(); e = new()
            if op in ("*", "?"):
                trans[s].append((None, e))
            trans[s].append((None, s1))
            if op in ("*", "+"):
                trans[e1].append((None, s1))
            trans[e1].append((None, e))
            return s, e
    start, accept = build(tree)
    def eclose(states):
        stk = list(states); seen = set(states)
        while stk:
            u = stk.pop()
            for lab, v in trans[u]:
                if lab is None and v not in seen:
                    seen.add(v); stk.append(v)
        return seen
    out = []
    for s in test_strings:
        cur = eclose({start})
        for ch in s:
            nxt = set()
            for u in cur:
                for lab, v in trans[u]:
                    if lab is None:
                        continue
                    if lab[0] == "dot":
                        nxt.add(v)
                    elif lab[0] == "lit" and lab[1] == ch:
                        nxt.add(v)
            cur = eclose(nxt)
        out.append(accept in cur)
    return out
''',

    # c085 JSON parser from scratch.
    "json_parser": r'''
def json_parser(s):
    pos = [0]
    def skip():
        while pos[0] < len(s) and s[pos[0]] in " \t\n\r":
            pos[0] += 1
    def value():
        skip(); c = s[pos[0]]
        if c == "{": return obj()
        if c == "[": return arr()
        if c == '"': return string()
        if c == "t": pos[0] += 4; return True
        if c == "f": pos[0] += 5; return False
        if c == "n": pos[0] += 4; return None
        return number()
    def obj():
        pos[0] += 1; d = {}; skip()
        if s[pos[0]] == "}":
            pos[0] += 1; return d
        while True:
            skip(); k = string(); skip(); pos[0] += 1  # ':'
            d[k] = value(); skip()
            if s[pos[0]] == ",":
                pos[0] += 1; continue
            pos[0] += 1; return d
    def arr():
        pos[0] += 1; a = []; skip()
        if s[pos[0]] == "]":
            pos[0] += 1; return a
        while True:
            a.append(value()); skip()
            if s[pos[0]] == ",":
                pos[0] += 1; continue
            pos[0] += 1; return a
    def string():
        pos[0] += 1; out = []
        while s[pos[0]] != '"':
            c = s[pos[0]]
            if c == "\\":
                pos[0] += 1; e = s[pos[0]]
                out.append({'"': '"', "\\": "\\", "/": "/", "n": "\n",
                            "t": "\t", "r": "\r", "b": "\b", "f": "\f"}.get(e, e))
            else:
                out.append(c)
            pos[0] += 1
        pos[0] += 1
        return "".join(out)
    def number():
        j = pos[0]
        while pos[0] < len(s) and s[pos[0]] in "-+.eE0123456789":
            pos[0] += 1
        t = s[j:pos[0]]
        if any(ch in t for ch in ".eE"):
            return float(t)
        return int(t)
    return value()
''',

    # c090 simplified XML parser -> nested dict {tag, attrs, children, text}.
    "xml_parser": r'''
def xml_parser(xml):
    pos = [0]; n = len(xml)
    def skip_ws():
        while pos[0] < n and xml[pos[0]] in " \t\n\r":
            pos[0] += 1
    def parse_element():
        # assumes xml[pos] == '<'
        pos[0] += 1  # '<'
        # tag name
        j = pos[0]
        while xml[pos[0]] not in " \t\n\r/>":
            pos[0] += 1
        tag = xml[j:pos[0]]
        attrs = {}
        while True:
            skip_ws()
            if xml[pos[0]] in "/>":
                break
            k = pos[0]
            while xml[pos[0]] not in "= \t\n\r":
                pos[0] += 1
            name = xml[k:pos[0]]
            skip_ws()
            pos[0] += 1  # '='
            skip_ws()
            quote = xml[pos[0]]; pos[0] += 1
            v = pos[0]
            while xml[pos[0]] != quote:
                pos[0] += 1
            attrs[name] = xml[v:pos[0]]
            pos[0] += 1  # closing quote
        node = {"tag": tag, "attrs": attrs, "children": [], "text": ""}
        if xml[pos[0]] == "/":
            pos[0] += 2  # '/>'
            return node
        pos[0] += 1  # '>'
        text_parts = []
        while pos[0] < n:
            if xml[pos[0]] == "<":
                if xml[pos[0]+1] == "/":
                    # closing tag
                    pos[0] += 2
                    while xml[pos[0]] != ">":
                        pos[0] += 1
                    pos[0] += 1
                    break
                else:
                    node["children"].append(parse_element())
            else:
                j = pos[0]
                while pos[0] < n and xml[pos[0]] != "<":
                    pos[0] += 1
                text_parts.append(xml[j:pos[0]])
        node["text"] = "".join(text_parts).strip()
        return node
    skip_ws()
    return parse_element()
''',

    # c093 markdown subset -> html.
    "markdown_to_html": r'''
def markdown_to_html(md):
    import re
    def inline(t):
        t = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        return t
    lines = md.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1; continue
        if line.startswith("### "):
            out.append("<h3>" + inline(line[4:]) + "</h3>"); i += 1
        elif line.startswith("## "):
            out.append("<h2>" + inline(line[3:]) + "</h2>"); i += 1
        elif line.startswith("# "):
            out.append("<h1>" + inline(line[2:]) + "</h1>"); i += 1
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append("<li>" + inline(lines[i][2:]) + "</li>"); i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith(("#", "- ")):
                para.append(lines[i]); i += 1
            out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "".join(out)
''',

    # c094 regex engine: all non-overlapping matches as (start,end), leftmost-longest greedy backtracking.
    "regex_engine": r'''
def regex_engine(pattern, text):
    anchored_start = pattern.startswith("^")
    anchored_end = pattern.endswith("$")
    p = pattern[1:] if anchored_start else pattern
    if anchored_end:
        p = p[:-1]
    toks = []; i = 0
    while i < len(p):
        c = p[i]
        if c == "[":
            j = p.index("]", i); spec = p[i+1:j]; i = j+1
            atom = ("class", spec)
        elif c == ".":
            atom = ("dot", None); i += 1
        else:
            atom = ("lit", c); i += 1
        q = None
        if i < len(p) and p[i] in "*+?":
            q = p[i]; i += 1
        toks.append((atom, q))
    def amatch(atom, ch):
        k, v = atom
        if k == "dot": return True
        if k == "lit": return ch == v
        spec = v; neg = spec.startswith("^")
        if neg: spec = spec[1:]
        res = False; m = 0
        while m < len(spec):
            if m+2 < len(spec) and spec[m+1] == "-":
                if spec[m] <= ch <= spec[m+2]: res = True
                m += 3
            else:
                if spec[m] == ch: res = True
                m += 1
        return res != neg
    def match_from(si):
        # returns longest end index matching toks starting at si, or None
        best = [None]
        def rec(ti, pos):
            if ti == len(toks):
                if anchored_end and pos != len(text):
                    return
                if best[0] is None or pos > best[0]:
                    best[0] = pos
                return
            atom, q = toks[ti]
            if q in ("*", "?"):
                maxc = 1 if q == "?" else len(text)-pos
                cnt = 0
                while cnt < maxc and pos+cnt < len(text) and amatch(atom, text[pos+cnt]):
                    cnt += 1
                for k in range(cnt, -1, -1):
                    rec(ti+1, pos+k)
            elif q == "+":
                cnt = 0
                while pos+cnt < len(text) and amatch(atom, text[pos+cnt]):
                    cnt += 1
                for k in range(cnt, 0, -1):
                    rec(ti+1, pos+k)
            else:
                if pos < len(text) and amatch(atom, text[pos]):
                    rec(ti+1, pos+1)
        rec(0, si)
        return best[0]
    out = []
    i = 0
    starts = [0] if anchored_start else range(len(text)+1)
    if anchored_start:
        e = match_from(0)
        if e is not None:
            out.append([0, e])
        return out
    while i <= len(text):
        e = match_from(i)
        if e is not None and e > i:
            out.append([i, e]); i = e
        elif e is not None and e == i:
            # zero-width match; advance to avoid infinite loop
            out.append([i, e]); i += 1
        else:
            i += 1
    return out
''',

    # c095 INI parser.
    "ini_parser": r'''
def ini_parser(text):
    result = {}
    section = "DEFAULT"
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            result.setdefault(section, {})
        elif "=" in line:
            k, v = line.split("=", 1)
            result.setdefault(section, {})[k.strip()] = v.strip()
    return result
''',

    # c096 lambda calc eval, leftmost-outermost (normal order), syntax (lambda x . body), (f arg).
    "lambda_calc_eval": r'''
def lambda_calc_eval(expr):
    s = expr
    pos = [0]
    def skip():
        while pos[0] < len(s) and s[pos[0]] == " ":
            pos[0] += 1
    def parse():
        skip()
        c = s[pos[0]]
        if c == "(":
            pos[0] += 1; skip()
            # check lambda
            if s[pos[0]:pos[0]+6] == "lambda":
                pos[0] += 6; skip()
                var = s[pos[0]]; pos[0] += 1; skip()
                pos[0] += 1  # '.'
                body = parse(); skip()
                pos[0] += 1  # ')'
                return ("abs", var, body)
            else:
                left = parse(); skip()
                right = parse(); skip()
                pos[0] += 1  # ')'
                return ("app", left, right)
        else:
            var = s[pos[0]]; pos[0] += 1
            return ("var", var)
    def free(t):
        if t[0] == "var": return {t[1]}
        if t[0] == "abs": return free(t[2]) - {t[1]}
        return free(t[1]) | free(t[2])
    cnt = [0]
    def fresh(avoid):
        while True:
            cnt[0] += 1
            nv = chr(97 + (cnt[0] % 26))
            if nv not in avoid:
                return nv
    def sub(t, x, val):
        if t[0] == "var":
            return val if t[1] == x else t
        if t[0] == "abs":
            if t[1] == x:
                return t
            if t[1] in free(val):
                nv = fresh(free(val) | free(t[2]) | {x})
                return ("abs", nv, sub(sub(t[2], t[1], ("var", nv)), x, val))
            return ("abs", t[1], sub(t[2], x, val))
        return ("app", sub(t[1], x, val), sub(t[2], x, val))
    def step(t):
        # leftmost-outermost: reduce outermost redex first
        if t[0] == "app":
            if t[1][0] == "abs":
                return sub(t[1][2], t[1][1], t[2]), True
            l, ch = step(t[1])
            if ch:
                return ("app", l, t[2]), True
            r, ch = step(t[2])
            return ("app", t[1], r), ch
        if t[0] == "abs":
            b, ch = step(t[2])
            return ("abs", t[1], b), ch
        return t, False
    def show(t):
        if t[0] == "var":
            return t[1]
        if t[0] == "abs":
            return "(lambda " + t[1] + " . " + show(t[2]) + ")"
        return "(" + show(t[1]) + " " + show(t[2]) + ")"
    t = parse()
    for _ in range(100):
        t, ch = step(t)
        if not ch:
            break
    return show(t)
''',

    # c099 cron parser -> next `count` unix ts > from_ts, UTC.
    "cron_parser": r'''
def cron_parser(cron, from_ts, count):
    import datetime
    minute_f, hour_f, dom_f, mon_f, dow_f = cron.split()
    def expand(field, lo, hi):
        vals = set()
        for part in field.split(","):
            if part == "*":
                vals.update(range(lo, hi+1))
            elif part.startswith("*/"):
                step = int(part[2:])
                vals.update(range(lo, hi+1, step))
            elif "-" in part and "/" in part:
                rng, step = part.split("/")
                a, b = rng.split("-")
                vals.update(range(int(a), int(b)+1, int(step)))
            elif "-" in part:
                a, b = part.split("-")
                vals.update(range(int(a), int(b)+1))
            else:
                vals.add(int(part))
        return vals
    minutes = expand(minute_f, 0, 59)
    hours = expand(hour_f, 0, 23)
    doms = expand(dom_f, 1, 31)
    mons = expand(mon_f, 1, 12)
    dows = expand(dow_f, 0, 6)  # 0=Sunday
    dom_restricted = dom_f != "*"
    dow_restricted = dow_f != "*"
    out = []
    cur = datetime.datetime.utcfromtimestamp(from_ts).replace(second=0, microsecond=0)
    cur += datetime.timedelta(minutes=1)
    while len(out) < count:
        if (cur.month in mons and cur.hour in hours and cur.minute in minutes):
            dom_ok = cur.day in doms
            dow_ok = ((cur.weekday() + 1) % 7) in dows  # python Monday=0 -> cron Sunday=0
            if dom_restricted and dow_restricted:
                day_ok = dom_ok or dow_ok
            elif dom_restricted:
                day_ok = dom_ok
            elif dow_restricted:
                day_ok = dow_ok
            else:
                day_ok = True
            if day_ok:
                out.append(int((cur - datetime.datetime(1970, 1, 1)).total_seconds()))
        cur += datetime.timedelta(minutes=1)
    return out
''',

    # c104 SQL WHERE parser/filter.
    "sql_where_parser": r'''
def sql_where_parser(where_clause, rows):
    import re
    # tokenize
    toks = re.findall(r"'[^']*'|>=|<=|!=|<>|[()=<>]|\w+", where_clause)
    def is_kw(t): return t.upper() in ("AND", "OR", "NOT")
    pos = [0]
    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None
    def parse_or(row):
        v = parse_and(row)
        while peek() and peek().upper() == "OR":
            pos[0] += 1
            r = parse_and(row)
            v = v or r
        return v
    def parse_and(row):
        v = parse_not(row)
        while peek() and peek().upper() == "AND":
            pos[0] += 1
            r = parse_not(row)
            v = v and r
        return v
    def parse_not(row):
        if peek() and peek().upper() == "NOT":
            pos[0] += 1
            return not parse_not(row)
        return parse_atom(row)
    def conv(tok):
        if tok.startswith("'") and tok.endswith("'"):
            return tok[1:-1]
        try:
            return int(tok)
        except ValueError:
            try:
                return float(tok)
            except ValueError:
                return tok
    def parse_atom(row):
        if peek() == "(":
            pos[0] += 1
            v = parse_or(row)
            pos[0] += 1  # ')'
            return v
        col = toks[pos[0]]; pos[0] += 1
        op = toks[pos[0]]; pos[0] += 1
        val = conv(toks[pos[0]]); pos[0] += 1
        lhs = row.get(col)
        if op == "=":
            return lhs == val
        if op == ">":
            return lhs is not None and lhs > val
        if op == "<":
            return lhs is not None and lhs < val
        if op in ("!=", "<>"):
            return lhs != val
        if op == ">=":
            return lhs is not None and lhs >= val
        if op == "<=":
            return lhs is not None and lhs <= val
        return False
    out = []
    for row in rows:
        pos[0] = 0
        if parse_or(row):
            out.append(row)
    return out
''',

    # c105 Forth interpreter, returns final stack.
    "forth_interpreter": r'''
def forth_interpreter(program):
    tokens = program.split()
    words = {}
    # preprocess word definitions
    def run(toks, stack):
        i = 0
        while i < len(toks):
            t = toks[i]
            up = t.upper()
            if t == ":":
                # definition
                name = toks[i+1].upper()
                j = i + 2
                body = []
                while toks[j] != ";":
                    body.append(toks[j]); j += 1
                words[name] = body
                i = j + 1
                continue
            if up == "IF":
                # find matching ELSE/THEN
                depth = 0; j = i + 1; else_idx = None; then_idx = None
                while j < len(toks):
                    u = toks[j].upper()
                    if u == "IF": depth += 1
                    elif u == "THEN":
                        if depth == 0:
                            then_idx = j; break
                        depth -= 1
                    elif u == "ELSE" and depth == 0:
                        else_idx = j
                    j += 1
                cond = stack.pop()
                if cond != 0:
                    block = toks[i+1: else_idx if else_idx is not None else then_idx]
                else:
                    block = toks[else_idx+1: then_idx] if else_idx is not None else []
                run(block, stack)
                i = then_idx + 1
                continue
            if up == "DO":
                # find matching LOOP
                depth = 0; j = i+1; loop_idx = None
                while j < len(toks):
                    u = toks[j].upper()
                    if u == "DO": depth += 1
                    elif u == "LOOP":
                        if depth == 0:
                            loop_idx = j; break
                        depth -= 1
                    j += 1
                start = stack.pop(); limit = stack.pop()
                body = toks[i+1:loop_idx]
                for _ in range(limit - start):
                    run(body, stack)
                i = loop_idx + 1
                continue
            # integer literal?
            try:
                stack.append(int(t)); i += 1; continue
            except ValueError:
                pass
            if up in words:
                run(list(words[up]), stack); i += 1; continue
            if t == "+":
                b = stack.pop(); a = stack.pop(); stack.append(a+b)
            elif t == "-":
                b = stack.pop(); a = stack.pop(); stack.append(a-b)
            elif t == "*":
                b = stack.pop(); a = stack.pop(); stack.append(a*b)
            elif t == "/":
                b = stack.pop(); a = stack.pop(); stack.append(int(a/b) if (a<0)!=(b<0) and a%b!=0 else a//b)
            elif up == "DUP":
                stack.append(stack[-1])
            elif up == "DROP":
                stack.pop()
            elif up == "SWAP":
                stack[-1], stack[-2] = stack[-2], stack[-1]
            elif up == "OVER":
                stack.append(stack[-2])
            elif up == "ROT":
                a = stack.pop(); b = stack.pop(); c = stack.pop(); stack.append(b); stack.append(a); stack.append(c)
            elif t == "=":
                b = stack.pop(); a = stack.pop(); stack.append(1 if a == b else 0)
            elif t == "<":
                b = stack.pop(); a = stack.pop(); stack.append(1 if a < b else 0)
            elif t == ">":
                b = stack.pop(); a = stack.pop(); stack.append(1 if a > b else 0)
            i += 1
    stack = []
    run(tokens, stack)
    return stack
''',

    # c114 Lamport clocks; return clock value after each event.
    "lamport_clock": r'''
def lamport_clock(events):
    from collections import defaultdict
    clock = defaultdict(int)
    pending = []  # FIFO of send timestamps per (sender) -> we match receive to last unmatched send
    sent = defaultdict(list)  # sender -> list of timestamps sent (in order)
    out = []
    for kind, a, b in events:
        if kind == "local":
            clock[a] += 1
            out.append(clock[a])
        elif kind == "send":
            sender = a
            clock[sender] += 1
            sent[sender].append(clock[sender])
            out.append(clock[sender])
        elif kind == "receive":
            receiver = a; sender = b
            ts = sent[sender].pop(0) if sent[sender] else 0
            clock[receiver] = max(clock[receiver], ts) + 1
            out.append(clock[receiver])
    return out
''',

    # c115 Vector clocks; return state after each event.
    "vector_clock": r'''
def vector_clock(n_processes, events):
    clocks = [[0]*n_processes for _ in range(n_processes)]
    sent = {i: [] for i in range(n_processes)}  # sender -> FIFO of vector snapshots
    out = []
    for kind, a, b in events:
        if kind == "local":
            clocks[a][a] += 1
            out.append(list(clocks[a]))
        elif kind == "send":
            sender = a
            clocks[sender][sender] += 1
            sent[sender].append(list(clocks[sender]))
            out.append(list(clocks[sender]))
        elif kind == "receive":
            receiver = a; sender = b
            msg = sent[sender].pop(0) if sent[sender] else [0]*n_processes
            clocks[receiver][receiver] += 1
            for i in range(n_processes):
                clocks[receiver][i] = max(clocks[receiver][i], msg[i])
            out.append(list(clocks[receiver]))
    return out
''',

    # c123 task scheduler with cooldown.
    "task_scheduler": r'''
def task_scheduler(tasks, cooldown):
    from collections import Counter
    counts = Counter(tasks)
    max_count = max(counts.values())
    n_max = sum(1 for c in counts.values() if c == max_count)
    intervals = (max_count - 1) * (cooldown + 1) + n_max
    return max(intervals, len(tasks))
''',

    # c133 Gaussian elimination with partial pivoting; [] if no unique solution.
    "gaussian_elimination": r'''
def gaussian_elimination(matrix):
    import copy
    A = [list(map(float, row)) for row in matrix]
    n = len(A)
    m = len(A[0]) - 1  # number of variables
    # forward elimination with partial pivoting
    row = 0
    pivots = []
    for col in range(m):
        # find pivot
        piv = max(range(row, n), key=lambda r: abs(A[r][col]), default=None)
        if piv is None or abs(A[piv][col]) < 1e-12:
            continue
        A[row], A[piv] = A[piv], A[row]
        pv = A[row][col]
        A[row] = [x / pv for x in A[row]]
        for r in range(n):
            if r != row and abs(A[r][col]) > 1e-12:
                f = A[r][col]
                A[r] = [a - f*b for a, b in zip(A[r], A[row])]
        pivots.append(col)
        row += 1
        if row == n:
            break
    # check consistency / uniqueness
    if len(pivots) < m:
        return []  # not unique (free variable) or under-determined
    # check no inconsistent row (0 = nonzero)
    for r in range(n):
        if all(abs(A[r][c]) < 1e-12 for c in range(m)) and abs(A[r][m]) > 1e-9:
            return []
    sol = [0.0]*m
    for i, col in enumerate(pivots):
        sol[col] = A[i][m]
    # round to int when close
    res = []
    for x in sol:
        rx = round(x)
        res.append(int(rx) if abs(x - rx) < 1e-9 else x)
    return res
''',

    # c151 base64 encode from scratch. `data` arrives as a str in cases -> encode utf-8.
    "base64_encode": r'''
def base64_encode(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    TABLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    out = []
    for i in range(0, len(data), 3):
        chunk = data[i:i+3]
        b = chunk + b"\x00" * (3 - len(chunk))
        n = (b[0] << 16) | (b[1] << 8) | b[2]
        c0 = TABLE[(n >> 18) & 63]
        c1 = TABLE[(n >> 12) & 63]
        c2 = TABLE[(n >> 6) & 63]
        c3 = TABLE[n & 63]
        if len(chunk) == 1:
            out.append(c0 + c1 + "==")
        elif len(chunk) == 2:
            out.append(c0 + c1 + c2 + "=")
        else:
            out.append(c0 + c1 + c2 + c3)
    return "".join(out)
''',

    # c162 hex dump; hex column fixed 50 chars, width = bytes per line.
    "hex_dump": r'''
def hex_dump(data, width):
    if not data:
        return ""
    b = data.encode("latin-1") if isinstance(data, str) else bytes(data)
    lines = []
    for off in range(0, len(b), width):
        chunk = b[off:off+width]
        hexpart = " ".join(f"{x:02x}" for x in chunk)
        asciipart = "".join(chr(x) if 0x20 <= x <= 0x7e else "." for x in chunk)
        lines.append(f"{off:08x}  {hexpart[:48].ljust(48)}  {asciipart}")
    return "\n".join(lines)
''',

    # c050 dominator tree (idom array); iterative Cooper-Harvey-Kennedy dataflow.
    "dominator_tree": r'''
def dominator_tree(n, edges, root):
    adj = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); radj[v].append(u)
    # reverse postorder from root
    order = []
    visited = [False]*n
    def dfs(s):
        stack = [(s, iter(adj[s]))]
        visited[s] = True
        post = []
        while stack:
            node, it = stack[-1]
            advanced = False
            for w in it:
                if not visited[w]:
                    visited[w] = True
                    stack.append((w, iter(adj[w])))
                    advanced = True
                    break
            if not advanced:
                post.append(node)
                stack.pop()
        return post
    post = dfs(root)
    rpo = post[::-1]
    rpo_num = {node: i for i, node in enumerate(rpo)}
    idom = [-1]*n
    idom[root] = root
    def intersect(a, b):
        while a != b:
            while rpo_num[a] > rpo_num[b]:
                a = idom[a]
            while rpo_num[b] > rpo_num[a]:
                b = idom[b]
        return a
    changed = True
    while changed:
        changed = False
        for node in rpo:
            if node == root:
                continue
            new_idom = -1
            for p in radj[node]:
                if p not in rpo_num:
                    continue
                if idom[p] != -1:
                    new_idom = p if new_idom == -1 else intersect(p, new_idom)
            if new_idom != -1 and idom[node] != new_idom:
                idom[node] = new_idom
                changed = True
    result = [-1]*n
    for node in range(n):
        if node == root:
            result[node] = -1
        elif node in rpo_num and idom[node] != -1:
            result[node] = idom[node]
        else:
            result[node] = -1
    return result
''',

    # c065 B-tree of minimum degree t; search/inorder results.
    "b_tree": r'''
def b_tree(t, operations):
    class Node:
        __slots__ = ("keys", "children", "leaf")
        def __init__(self, leaf=True):
            self.keys = []; self.children = []; self.leaf = leaf
    root = [Node(leaf=True)]
    def split_child(parent, i):
        t_ = t
        child = parent.children[i]
        new = Node(leaf=child.leaf)
        mid = child.keys[t_-1]
        new.keys = child.keys[t_:]
        child.keys = child.keys[:t_-1]
        if not child.leaf:
            new.children = child.children[t_:]
            child.children = child.children[:t_]
        parent.keys.insert(i, mid)
        parent.children.insert(i+1, new)
    def insert_nonfull(node, key):
        if node.leaf:
            import bisect
            bisect.insort(node.keys, key)
        else:
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2*t - 1:
                split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            insert_nonfull(node.children[i], key)
    def insert(key):
        r = root[0]
        if len(r.keys) == 2*t - 1:
            new = Node(leaf=False)
            new.children.append(r)
            split_child(new, 0)
            root[0] = new
            insert_nonfull(new, key)
        else:
            insert_nonfull(r, key)
    def search(node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == key:
            return True
        if node.leaf:
            return False
        return search(node.children[i], key)
    def inorder(node, acc):
        if node.leaf:
            acc.extend(node.keys)
        else:
            for i in range(len(node.keys)):
                inorder(node.children[i], acc)
                acc.append(node.keys[i])
            inorder(node.children[-1], acc)
    out = []
    for op in operations:
        kind = op[0]
        if kind == "insert":
            insert(op[1])
        elif kind == "search":
            out.append(search(root[0], op[1]))
        elif kind == "inorder":
            acc = []; inorder(root[0], acc); out.append(acc)
    return out
''',

    # c092 LR(0) parser; build automaton + parse, return accept bool.
    "lr0_parser": r'''
def lr0_parser(grammar, tokens):
    productions = [(lhs, tuple(rhs)) for lhs, rhs in grammar]
    start_sym = productions[0][0]
    nonterminals = set(lhs for lhs, _ in productions)
    prod_by_lhs = {}
    for i, (lhs, rhs) in enumerate(productions):
        prod_by_lhs.setdefault(lhs, []).append(i)
    def closure(items):
        items = set(items)
        changed = True
        while changed:
            changed = False
            for (pi, dot) in list(items):
                lhs, rhs = productions[pi]
                if dot < len(rhs):
                    sym = rhs[dot]
                    if sym in nonterminals:
                        for p in prod_by_lhs.get(sym, []):
                            it = (p, 0)
                            if it not in items:
                                items.add(it); changed = True
        return frozenset(items)
    def goto(items, sym):
        moved = set()
        for (pi, dot) in items:
            lhs, rhs = productions[pi]
            if dot < len(rhs) and rhs[dot] == sym:
                moved.add((pi, dot+1))
        if not moved:
            return None
        return closure(moved)
    start_item = closure({(0, 0)})
    states = [start_item]
    state_id = {start_item: 0}
    transitions = {}
    i = 0
    while i < len(states):
        st = states[i]
        symbols = set()
        for (pi, dot) in st:
            lhs, rhs = productions[pi]
            if dot < len(rhs):
                symbols.add(rhs[dot])
        for sym in symbols:
            g = goto(st, sym)
            if g is None:
                continue
            if g not in state_id:
                state_id[g] = len(states); states.append(g)
            transitions[(i, sym)] = state_id[g]
        i += 1
    # parse with shift/reduce. action: prefer reduce of completed item.
    inp = list(tokens) + ["$"]
    stack = [0]
    ip = 0
    while True:
        st = stack[-1]
        items = states[st]
        # find a completed item
        reduce_item = None
        for (pi, dot) in items:
            lhs, rhs = productions[pi]
            if dot == len(rhs):
                if pi == 0:
                    # accept if at end
                    if inp[ip] == "$":
                        return True
                else:
                    reduce_item = pi
        cur = inp[ip]
        # try shift
        if (st, cur) in transitions:
            stack.append(transitions[(st, cur)])
            ip += 1
            continue
        if reduce_item is not None:
            lhs, rhs = productions[reduce_item]
            for _ in range(len(rhs)):
                stack.pop()
            top = stack[-1]
            if (top, lhs) not in transitions:
                return False
            stack.append(transitions[(top, lhs)])
            continue
        return False
''',

    # c102 minimal Scheme interpreter; returns string repr of final value.
    "scheme_eval": r'''
def scheme_eval(program):
    def tokenize(s):
        return s.replace("(", " ( ").replace(")", " ) ").split()
    def parse(tokens):
        forms = []
        def read(it):
            tok = next(it)
            if tok == "(":
                lst = []
                while True:
                    t = next(it)
                    if t == ")":
                        return lst
                    it2 = _push(it, t)
                    lst.append(read(it2))
            else:
                return atom(tok)
        return tokens
    def atom(tok):
        try:
            return int(tok)
        except ValueError:
            try:
                return float(tok)
            except ValueError:
                return tok
    # simpler recursive parser using index
    toks = tokenize(program)
    pos = [0]
    def read():
        tok = toks[pos[0]]; pos[0] += 1
        if tok == "(":
            lst = []
            while toks[pos[0]] != ")":
                lst.append(read())
            pos[0] += 1
            return lst
        return atom(tok)
    forms = []
    while pos[0] < len(toks):
        forms.append(read())
    global_env = {}
    class Lambda:
        def __init__(self, params, body, env):
            self.params = params; self.body = body; self.env = env
    def ev(x, env):
        if isinstance(x, str):
            if x in env:
                return env[x]
            return global_env.get(x, x)
        if not isinstance(x, list):
            return x
        op = x[0]
        if op == "define":
            name = x[1]
            global_env[name] = ev(x[2], env)
            return None
        if op == "lambda":
            return Lambda(x[1], x[2], env)
        if op == "if":
            cond = ev(x[1], env)
            return ev(x[2], env) if cond not in (False, 0) and cond is not False else ev(x[3], env)
        if op == "quote":
            return x[1]
        if op == "cons":
            return ("pair", ev(x[1], env), ev(x[2], env))
        if op == "car":
            p = ev(x[1], env); return p[1]
        if op == "cdr":
            p = ev(x[1], env); return p[2]
        if op == "list":
            res = "nil"
            for item in reversed(x[1:]):
                res = ("pair", ev(item, env), res)
            return res
        if op == "null?":
            return ev(x[1], env) == "nil"
        args = [ev(a, env) for a in x[1:]]
        if op == "+":
            r = 0
            for a in args: r += a
            return r
        if op == "-":
            if len(args) == 1: return -args[0]
            r = args[0]
            for a in args[1:]: r -= a
            return r
        if op == "*":
            r = 1
            for a in args: r *= a
            return r
        if op == "/":
            r = args[0]
            for a in args[1:]: r = r / a
            return r
        if op == "=":
            return args[0] == args[1]
        if op == "<":
            return args[0] < args[1]
        if op == ">":
            return args[0] > args[1]
        # function application
        fn = ev(op, env) if not isinstance(op, Lambda) else op
        if isinstance(fn, Lambda):
            newenv = dict(fn.env)
            for p, a in zip(fn.params, args):
                newenv[p] = a
            return ev(fn.body, newenv)
        raise ValueError("unknown op " + str(op))
    result = None
    for f in forms:
        result = ev(f, global_env)
    def to_str(v):
        if v is True: return "#t"
        if v is False: return "#f"
        if isinstance(v, float):
            if v == int(v): return str(int(v))
            return str(v)
        return str(v)
    return to_str(result)
''',

    # c117 Go-like unbuffered channel; producers/consumers round-robin.
    "channel_simulation": r'''
def channel_simulation(n_producers, n_consumers, items):
    # Producers send round-robin starting from producer 0; each send is consumed
    # immediately (unbuffered) by the next consumer round-robin. Consume order is
    # simply the order items are sent.
    queues = [list(seq) for seq in items]
    out = []
    p = 0
    remaining = sum(len(q) for q in queues)
    while remaining > 0:
        # find next producer (round-robin from current p) that still has items
        for _ in range(n_producers):
            if queues[p]:
                out.append(queues[p].pop(0))
                remaining -= 1
                p = (p + 1) % n_producers
                break
            p = (p + 1) % n_producers
    return out
''',

    # c119 Raft leader election simulation.
    "raft_leader_election": r'''
def raft_leader_election(n_nodes, events):
    # Track votes per candidate (the candidate is the node that timed out and is
    # collecting responses). Each completed election that reaches majority emits a leader.
    leaders = []
    majority = n_nodes // 2 + 1
    votes = {}      # candidate -> set of voters (includes self)
    decided = set() # candidates that already won
    for ev in events:
        kind = ev[0]
        if kind == "timeout":
            node = ev[1]
            votes[node] = {node}            # votes for itself
            if len(votes[node]) >= majority and node not in decided:
                leaders.append(node); decided.add(node)
        elif kind == "vote_request":
            pass
        elif kind == "vote_response":
            frm, to, granted = ev[1], ev[2], ev[3]
            # `to` is the candidate receiving the response
            if granted and to in votes and to not in decided:
                votes[to].add(frm)
                if len(votes[to]) >= majority:
                    leaders.append(to); decided.add(to)
    return leaders
''',

    # c129 work-stealing scheduler, discrete event semantics.
    "work_stealing": r'''
def work_stealing(n_workers, initial_queues):
    from collections import deque
    queues = [deque(q) for q in initial_queues]
    clock = [0]*n_workers
    finish = [0]*n_workers
    # each worker initially picks its first task at t=0
    busy_until = [None]*n_workers   # time current task finishes; None if idle/needs task
    current = [None]*n_workers
    # Start: every worker grabs its first task (front of own deque)
    idle = [False]*n_workers
    # We model: at clock = earliest, that worker finishes current task and picks next.
    # Initialize: assign first task to each worker.
    def pick(w):
        if queues[w]:
            return queues[w].popleft()
        # steal one from back of longest non-empty deque among others, tie lowest index
        best = -1; best_len = 0
        for o in range(n_workers):
            if o == w: continue
            if len(queues[o]) > best_len:
                best_len = len(queues[o]); best = o
        if best != -1:
            return queues[best].pop()
        return None
    for w in range(n_workers):
        task = pick(w)
        if task is None:
            busy_until[w] = None; idle[w] = True
        else:
            busy_until[w] = clock[w] + task
    while True:
        # find earliest busy worker
        candidates = [w for w in range(n_workers) if busy_until[w] is not None]
        if not candidates:
            break
        w = min(candidates, key=lambda x: (busy_until[x], x))
        t = busy_until[w]
        clock[w] = t
        finish[w] = max(finish[w], t)
        # worker finished its task; pick next
        task = pick(w)
        if task is None:
            busy_until[w] = None; idle[w] = True
        else:
            busy_until[w] = clock[w] + task
    return max(finish)
''',

    # c169 JSON Pointer (RFC 6901).
    "json_pointer": r'''
def json_pointer(doc, pointer):
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ValueError("invalid pointer")
    cur = doc
    for raw in pointer.split("/")[1:]:
        seg = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            if not seg.lstrip("-").isdigit():
                raise ValueError("invalid index")
            idx = int(seg)
            if idx < 0 or idx >= len(cur):
                raise ValueError("index out of range")
            cur = cur[idx]
        elif isinstance(cur, dict):
            if seg not in cur:
                raise ValueError("key not found")
            cur = cur[seg]
        else:
            raise ValueError("cannot descend")
    return cur
''',

    # c175 convex hull area (monotone chain + shoelace), rounded 4 dp
    "convex_hull_area": r'''
def convex_hull_area(points):
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return 0.0
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    h = lo[:-1] + up[:-1]; a = 0.0
    for i in range(len(h)):
        x1, y1 = h[i]; x2, y2 = h[(i+1) % len(h)]; a += x1*y2 - x2*y1
    return round(abs(a) / 2.0, 4)
''',

    # c176 diameter = max pairwise distance (exact; rotating-calipers is the suggested O(n) method), 6 dp
    "rotating_calipers": r'''
def rotating_calipers(points):
    pts = [(float(x), float(y)) for x, y in points]; best = 0.0
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = ((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2) ** 0.5
            if d > best: best = d
    return round(best, 6)
''',

    # c185 Voronoi vertices = distinct circumcenters of site-triples with no other site strictly closer
    "voronoi_vertices": r'''
def voronoi_vertices(sites):
    import itertools
    pts = [(float(x), float(y)) for x, y in sites]; n = len(pts)
    if n < 3:
        return 0
    def cc(a, b, c):
        ax, ay = a; bx, by = b; cx, cy = c
        d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        if abs(d) < 1e-12: return None
        ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / d
        uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / d
        return (ux, uy)
    V = set()
    for a, b, c in itertools.combinations(pts, 3):
        ce = cc(a, b, c)
        if ce is None: continue
        r = ((ce[0]-a[0])**2 + (ce[1]-a[1])**2) ** 0.5
        if all(((ce[0]-p[0])**2 + (ce[1]-p[1])**2) ** 0.5 >= r - 1e-6 for p in pts):
            V.add((round(ce[0], 6), round(ce[1], 6)))
    return len(V)
''',

    # c187 median of two sorted arrays (merge; binary-search is the suggested O(log) method)
    "median_two_sorted": r'''
def median_two_sorted(nums1, nums2):
    m = sorted(list(nums1) + list(nums2)); n = len(m)
    return float(m[n//2]) if n % 2 else (m[n//2 - 1] + m[n//2]) / 2.0
''',

    # c188 k-th smallest, 1-indexed (sorted slice; quickselect is the suggested O(n)-average method)
    "kth_smallest": r'''
def kth_smallest(arr, k):
    return sorted(arr)[k - 1]
''',
}
