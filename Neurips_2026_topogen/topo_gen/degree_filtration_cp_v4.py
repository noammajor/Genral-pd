"""
Degree-Based Filtration CP Solver — Fast Version
=================================================
Solves: given persistence diagrams (P0, P1) from a degree-based filtration,
find a graph H whose degree-based filtration produces the same PH.

Speed improvements over the original solver:

  OPT 1 — Skip unchanged stages
    If p0_births[k]=0, p0_deaths[k]=0, p1_births[k]=0: component structure
    at stage k is identical to k-1. Alias comp[k] = comp[k-1]; no new variables.

  OPT 2 — Skip connectivity at pure-birth stages
    Connectivity only needs enforcing at stages where merges or cycle births occur
    (p0_deaths[k]>0 or p1_births[k]>0). At pure-birth stages, all new vertices
    are isolated singletons — trivially connected. Old components carry their
    connectivity from the previous stage.

  OPT 3 — Iterate over used slots only (cc[k], not ns)
    By the symmetry-breaking constraint (C3), only the first cc[k] slots are
    occupied at stage k. Connectivity variables for slots cc[k]..ns-1 are
    wasted. We only generate parent/level variables for used slots.

  OPT 4 — Spanning-tree (parent+level) instead of bidirectional integer flow
    Replaces O(n_active^2) integer flow variables with O(n_active^2) boolean
    parent variables + O(n_active) integer level variables per slot/stage.
    Boolean variables are substantially cheaper for CP-SAT than integers.

Combined effect on Tree n=30 (hardest case):
  Old: ~44,000 vars, timeout at 60s → New: ~16,000 vars, solves in ~9s

Constraints:
  C1    Degree sequence
  C2    Partition (each active vertex in exactly one slot)
  C3    Symmetry breaking (slots filled in order)
  C4    Edge => same slot (linearised)
  C4b   Spanning-tree connectivity (same slot => connected path)
  C6    Ancestry forward (A[k][c,d]=1 => old vertices follow slot)
  C6c   Ancestry reflection (old vertex transition => A[k][c,d]=1)
  C7    A column-stochastic
  C8    Component count per stage
  C9    Cycle count per stage
  ALIVE Per-birth-time component survival (prevents premature merges)
"""

import time
from collections import defaultdict

import gudhi
import networkx as nx
import numpy as np
from ortools.sat.python import cp_model


# ─────────────────────────────────────────────────────────────────────────────
# PH computation (degree-based filtration) — via gudhi
# ─────────────────────────────────────────────────────────────────────────────

def degree_fil(G):
    """Degree-based filtration values for vertices and edges."""
    deg = {n: G.degree(n) for n in G.nodes()}
    vert_fil = [deg[n] for n in G.nodes()]
    edge_fil = [max(deg[u], deg[v]) for u, v in G.edges()]
    return [vert_fil, edge_fil]


def ph_from_nx(G):
    """
    Compute degree-filtration PH of G via gudhi.
    Returns raw gudhi list [(dim, (birth, death)), ...].
    death == inf for essential classes.
    Node labels are mapped to contiguous integers so gudhi accepts them
    (handles non-integer labels such as tuple nodes in grid graphs).
    Filtration values are cast to float as gudhi requires.
    """
    st = gudhi.SimplexTree()
    nodes = list(G.nodes())
    edges = list(G.edges())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    vert_fil, edge_fil = degree_fil(G)
    for v, val in zip(nodes, vert_fil):
        st.insert([node_to_int[v]], filtration=float(val))
    for e, val in zip(edges, edge_fil):
        u, v = e
        st.insert([node_to_int[u], node_to_int[v]], filtration=float(val))
    st.make_filtration_non_decreasing()
    return st.persistence(min_persistence=-1, persistence_dim_max=True)


def dgm_to_p0p1(dgm):
    """
    Convert gudhi persistence list to solver format.
      p0: list of (birth:int, death:int|None)  — None means essential (death=inf)
      p1: list of (birth:int, float('inf'))
    """
    p0, p1 = [], []
    for dim, (b, d) in dgm:
        b_int = int(round(b))
        if dim == 0:
            p0.append((b_int, None if d == float('inf') else int(round(d))))
        elif dim == 1:
            p1.append((b_int, float('inf')))
    return p0, p1


def degree_filtration_ph(G: nx.Graph):
    """Compute persistence diagrams (P0, P1) of the degree-based filtration of G."""
    if G.number_of_nodes() == 0:
        return [], []
    return dgm_to_p0p1(ph_from_nx(G))


def norm_ph(p0, p1):
    """Canonical sorted form for comparison."""
    sp0 = sorted((b, d if d is not None else float('inf')) for b, d in p0)
    return sp0, sorted(p1)


def ph_equal(p0a, p1a, p0b, p1b):
    return norm_ph(p0a, p1a) == norm_ph(p0b, p1b)


def fmt_p0(p0):
    return [(b, 'inf' if d is None else d)
            for b, d in sorted(p0, key=lambda x: (x[0], x[1] if x[1] is not None else 999))]


def ph_sort(ph):
    return sorted(ph, key=lambda x: (x[0], x[1] if x[1] is not None else float('inf')))


# ─────────────────────────────────────────────────────────────────────────────
# Boolean helpers
# ─────────────────────────────────────────────────────────────────────────────

def bool_and_var(model, a, b, name):
    """z = a AND b"""
    z = model.new_bool_var(name)
    model.add_bool_and([a, b]).only_enforce_if(z)
    model.add_bool_or([a.Not(), b.Not()]).only_enforce_if(z.Not())
    return z

def bool_or_vars(model, literals, name):
    """z = OR(literals)"""
    z = model.new_bool_var(name)
    model.add_bool_or(literals).only_enforce_if(z)
    for lit in literals: model.add_implication(z.Not(), lit.Not())
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

def solve(p0_target, p1_target, n_vertices=None, time_limit=60.0, verbose=True, cp_sat_workers=8):
    """
    Find graph H whose degree-based filtration PH matches (p0_target, p1_target).
    Returns nx.Graph or None if no solution found within time_limit.
    """

    # ── 1. Stage statistics ───────────────────────────────────────────────
    p0b  = defaultdict(int)   # births by stage
    p0d  = defaultdict(int)   # deaths by death-stage
    p0bd = defaultdict(int)   # deaths by (birth, death) pair
    p1b  = defaultdict(int)   # cycle births by stage

    for b, d in p0_target:
        p0b[b] += 1
        if d is not None: p0d[d] += 1; p0bd[(b, d)] += 1
    for b, _ in p1_target: p1b[b] += 1

    ms = max(max(p0b, default=0), max(p0d, default=0), max(p1b, default=0))

    # vertex i has degree vd[i] = its birth time in the degree filtration
    vd = []
    for k in range(ms + 1): vd.extend([k] * p0b[k])
    n = len(vd)
    if n_vertices is not None and n_vertices != n:
        print(f"[Warning] PH implies {n} vertices; using {n}.")

    na  = {k: sum(1 for d in vd if d <= k) for k in range(ms + 1)}
    cc  = {}; br = dr = 0
    for k in range(ms + 1): br += p0b[k]; dr += p0d[k]; cc[k] = br - dr
    cy  = {}; cr = 0
    for k in range(ms + 1): cr += p1b[k]; cy[k] = cr
    ns  = max(cc.values()) if cc else 1   # global slot count = max simultaneous components

    bt = sorted(p0b.keys())
    alive = {}
    for k in range(ms + 1):
        alive[k] = {}
        for b in bt:
            alive[k][b] = p0b[b] - sum(p0bd.get((b, d_), 0) for d_ in range(k + 1))

    # OPT 1: stages where nothing changes (alias to previous stage)
    unchanged = {k: (p0b[k] == 0 and p0d[k] == 0 and p1b[k] == 0)
                 for k in range(ms + 1)}
    unchanged[0] = False  # base case always active

    # OPT 2: stages needing connectivity enforcement (merges or cycle births)
    # Skip pure-birth stages: new vertices are isolated singletons, trivially connected.
    # Note: do NOT gate on cc[k]>1 — even when cc=1, C4b is needed to prove the
    # single slot is internally connected. C9 only fixes the total edge count,
    # not the connectivity within the slot.
    need_conn = {k: (p0d[k] > 0 or p1b[k] > 0) for k in range(ms + 1)}

    if verbose:
        print(f"n={n}, max_stage={ms}, comp_slots={ns}")
        print(f"vertex_degree : {vd}")
        print(f"cum_comp      : {dict(cc)}")
        print(f"cum_cyc       : {dict(cy)}")
        skipped = [k for k, u in unchanged.items() if u]
        noconn  = [k for k in range(ms+1) if not unchanged[k] and not need_conn[k]]
        print(f"slots per stage: {dict(cc)}")
        print(f"Skipped stages (unchanged): {skipped}")
        print(f"No-conn stages (births only): {noconn}")

    # ── 2. Build CP-SAT model ─────────────────────────────────────────────
    model = cp_model.CpModel()

    M = {(i, j): model.new_bool_var(f"M_{i}_{j}")
         for i in range(n) for j in range(i+1, n)}
    def ev(u, v): return M[min(u, v), max(u, v)]

    # comp[k][c][v]: vertex v in slot c at stage k (aliased for unchanged stages)
    comp = {}
    for k in range(ms + 1):
        if unchanged[k]: comp[k] = comp[k-1]; continue
        comp[k] = {c: {v: model.new_bool_var(f"cp_{k}_{c}_{v}")
                       for v in range(n) if vd[v] <= k}
                   for c in range(ns)}

    # comp_used[k][c]: slot c non-empty at stage k
    cu = {}
    for k in range(ms + 1):
        if unchanged[k]: cu[k] = cu[k-1]; continue
        av = [v for v in range(n) if vd[v] <= k]; cu[k] = {}
        for c in range(ns):
            u = model.new_bool_var(f"used_{k}_{c}"); cu[k][c] = u
            if av: model.add_max_equality(u, [comp[k][c][v] for v in av])
            else: model.add(u == 0)

    # A[k][(c,d)]: slot d at k-1 maps to slot c at k
    # Only create for valid pairs: c in range(cc[k]), d in range(cc[k-1]).
    # Entries outside this range are never referenced in C6/C6c/C7 and
    # would be dead variables eliminated during preprocessing.
    A = {}
    for k in range(1, ms + 1):
        if unchanged[k] or cc[k] == 0 or cc[k-1] == 0: continue
        A[k] = {(c, d): model.new_bool_var(f"A_{k}_{c}_{d}")
                for c in range(cc[k]) for d in range(cc[k-1])}

    # ── C1: Degree ───────────────────────────────────────────────────────
    for i in range(n):
        model.add(sum(ev(i, j) for j in range(n) if j != i) == vd[i])

    # ── C2: Partition ────────────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]: continue
        for v in range(n):
            if vd[v] <= k: model.add_exactly_one(comp[k][c][v] for c in range(ns))

    # ── C3: Symmetry breaking ────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]: continue
        for c in range(1, ns): model.add(cu[k][c] <= cu[k][c-1])

    # ── C8: Component count ───────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]: continue
        model.add(sum(cu[k][c] for c in range(ns)) == cc[k])

    # ── C4: Edge => same slot ─────────────────────────────────────────────
    for k in range(ms + 1):
        if unchanged[k]: continue
        for u in range(n):
            for v in range(u+1, n):
                if vd[u] <= k and vd[v] <= k:
                    e = ev(u, v)
                    for c in range(cc[k]):
                        model.add(e + comp[k][c][u] - comp[k][c][v] <= 1)
                        model.add(e + comp[k][c][v] - comp[k][c][u] <= 1)

    # ── C4b: Spanning-tree connectivity ───────────────────────────────────
    # OPT 2: only at stages with merges/cycle-births
    # OPT 3: only for used slots (first cc[k] slots by symmetry breaking)
    # OPT 4: parent+level vars (boolean parent, integer level)
    BIG = n
    for k in range(ms + 1):
        if unchanged[k] or not need_conn[k]: continue
        av = [v for v in range(n) if vd[v] <= k]
        if len(av) <= 1: continue
        n_used = cc[k]  # only first n_used slots occupied (C3 symmetry breaking)
        for c in range(n_used):
            # is_root[v]: v is min-index vertex in slot c at stage k
            is_root = {}
            for v in av:
                if v == 0:
                    is_root[v] = comp[k][c][v]
                else:
                    ir = model.new_bool_var(f"ir_{k}_{c}_{v}")
                    preds = [comp[k][c][u].Not() for u in av if u < v]
                    model.add_bool_and([comp[k][c][v]] + preds).only_enforce_if(ir)
                    model.add_bool_or([comp[k][c][v].Not()] +
                                      [comp[k][c][u] for u in av if u < v]).only_enforce_if(ir.Not())
                    is_root[v] = ir

            # Level variable: 0 for root, >= 1 for non-root in slot
            L = {v: model.new_int_var(0, n, f"L_{k}_{c}_{v}") for v in av}

            # Parent: par[i,j] = 1 iff i is parent of j in spanning tree of slot c
            par = {(i, j): model.new_bool_var(f"par_{k}_{c}_{i}_{j}")
                   for i in av for j in av if i != j}

            for i in av:
                for j in av:
                    if i == j: continue
                    # parent edge must be a real edge within the slot
                    model.add(par[i, j] <= ev(i, j))
                    model.add(par[i, j] <= comp[k][c][i])
                    model.add(par[i, j] <= comp[k][c][j])
                    # parent implies level ordering (exact: L[j] = L[i] + 1)
                    model.add(L[j] >= L[i] + 1 - BIG * (1 - par[i, j]))
                    model.add(L[j] <= L[i] + 1 + BIG * (1 - par[i, j]))

            for v in av:
                # Each non-root vertex in slot has exactly one parent
                in_par = sum(par[u, v] for u in av if u != v)
                model.add(in_par == comp[k][c][v] - is_root[v])
                # Root has level 0; vertex not in slot has level 0 (don't care)
                model.add(L[v] <= BIG * (1 - is_root[v]))
                model.add(L[v] <= BIG * comp[k][c][v])

    # ── C6: Ancestry forward (gated by comp_used) ────────────────────────
    # Only iterate over used slots: d in range(cc[k-1]), c in range(cc[k]).
    # Slots d >= cc[k-1] are empty at k-1 (cu[k-1][d]=0 by C3+C8), so
    # comp[k-1][d][v]=0 for all v and the constraint is trivially satisfied.
    # Slots c >= cc[k] are empty at k, but still participate — the constraint
    # comp[k-1][d][v] <= comp[k][c][v] + (1-a) + (1-ud) can still fire
    # usefully when a=1 forcing comp[k-1][d][v]=0 (slot d must be empty too).
    # However since C7 already forces A[k][c,d]=0 for c >= cc[k] (cu[k][c]=0),
    # those entries of A are fixed to 0, making (1-a)=1 and the constraint slack.
    # So we can safely restrict both loops to used slots.
    for k in range(1, ms + 1):
        if unchanged[k] or k not in A: continue
        for c in range(cc[k]):        # only used slots at k
            for d in range(cc[k-1]):  # only used slots at k-1
                a = A[k][c, d]; ud = cu[k-1][d]
                for v in range(n):
                    if vd[v] <= k-1:
                        model.add(comp[k-1][d][v] <= comp[k][c][v] + (1 - a) + (1 - ud))

    # ── C6c: A reflects actual vertex transitions ─────────────────────────
    # Same reasoning: restrict to used slots at both stages.
    for k in range(1, ms + 1):
        if unchanged[k] or k not in A: continue
        for v in range(n):
            if vd[v] <= k-1:
                for c in range(cc[k]):        # only used slots at k
                    for d in range(cc[k-1]):  # only used slots at k-1
                        model.add(comp[k-1][d][v] + comp[k][c][v] - 1 <= A[k][c, d])

    # ── C7: A column-stochastic ───────────────────────────────────────────
    for k in range(1, ms + 1):
        if unchanged[k] or k not in A: continue
        for d in range(cc[k-1]):   # only used slots at k-1
            model.add(sum(A[k][c, d] for c in range(cc[k])) == cu[k-1][d])

    # ── C9: Cycle count (paper: |E(Gk)| - |V(Gk)| + #comp = cum_cyc[k]) ─
    for k in range(ms + 1):
        if unchanged[k]: continue
        ae = [ev(u, v) for u in range(n) for v in range(u+1, n)
              if vd[u] <= k and vd[v] <= k]
        model.add(sum(ae) - na[k] + sum(cu[k][c] for c in range(ns)) == cy[k])

    # ── C_ALIVE: per-birth-time component survival counts ─────────────────
    # alive[k][b] = # components born at b still alive at k.
    # Encodes: slot c at k has "elder birth-time b" iff it is used, contains
    # a vertex born at b, and contains no vertex born earlier than b.
    # sum_c elder_is_b[k][c] = alive[k][b]
    # Prevents premature merges (the main source of P0 errors without this).
    # Loop over range(cc[k]) not range(ns): slots c >= cc[k] have cu[k][c]=0
    # so elder=0 and they contribute nothing to the sum. Creating their aux
    # variables (hasb, nolt, tmp, elder) is pure waste.
    for k in range(ms + 1):
        if unchanged[k]: continue
        av = [v for v in range(n) if vd[v] <= k]
        if not av: continue
        for b in bt:
            vob = [v for v in av if vd[v] == b]
            vlt = [v for v in av if vd[v] <  b]
            if not vob: continue
            elder_vars = []
            for c in range(cc[k]):   # cc[k] not ns: slots >= cc[k] are empty (elder=0)
                has_b = bool_or_vars(model, [comp[k][c][v] for v in vob],
                                     f"hasb_{k}_{c}_{b}")
                if vlt:
                    no_lt = model.new_bool_var(f"nolt_{k}_{c}_{b}")
                    model.add_bool_and([comp[k][c][v].Not() for v in vlt]).only_enforce_if(no_lt)
                    model.add_bool_or([comp[k][c][v] for v in vlt]).only_enforce_if(no_lt.Not())
                    tmp   = bool_and_var(model, has_b,    no_lt,   f"tmp_{k}_{c}_{b}")
                    elder = bool_and_var(model, cu[k][c], tmp,     f"elder_{k}_{c}_{b}")
                else:
                    elder = bool_and_var(model, cu[k][c], has_b,   f"elder_{k}_{c}_{b}")
                elder_vars.append(elder)
            model.add(sum(elder_vars) == alive[k][b])

    # ── 3. Solve ──────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers  = cp_sat_workers

    if verbose:
        print(f"\nModel: {len(model.proto.variables)} variables, "
              f"{len(model.proto.constraints)} constraints")
        print("Solving ...")

    t0     = time.time()
    status = solver.solve(model)
    elapsed = time.time() - t0

    if verbose:
        print(f"Status: {solver.status_name(status)}  ({elapsed:.2f}s)")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            adj[i, j] = adj[j, i] = solver.value(M[i, j])

    H = nx.from_numpy_array(adj)
    if verbose:
        print(f"Edges  : {sorted(H.edges())}")
        print(f"Degrees: {dict(H.degree())}")
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Full 95-case test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(time_limit=60.0, seed=123):
    """
    Run the full 95-case correctness and timing test suite.
    Covers: paths, cycles, stars, complete graphs, random trees, ER graphs
    (p=0.2/0.3/0.5), disconnected graphs, grids, barbells, lollipops,
    wheels, and named special graphs (Petersen, Ladder).

    For each source graph G, finds H such that PH(H) == PH(G) and checks:
      - P0 and P1 match exactly (correctness)
      - Whether H is isomorphic to G (new vs original solution)
    """
    import random, time as _time
    from collections import defaultdict as _dd

    random.seed(seed)
    results = []

    def _run(name, G, cat):
        p0, p1 = degree_filtration_ph(G)
        n, m = G.number_of_nodes(), G.number_of_edges()
        t0 = _time.time()
        H = solve(p0, p1, n_vertices=n, time_limit=time_limit, verbose=False)
        el = round(_time.time() - t0, 2)
        if H is None:
            results.append((name, cat, n, m, 'TIMEOUT', el, None)); return
        hp0, hp1 = degree_filtration_ph(H)
        ok  = ph_equal(p0, p1, hp0, hp1)
        iso = nx.is_isomorphic(G, H)
        results.append((name, cat, n, m, 'PASS' if ok else 'FAIL', el, iso))

    for n in [3,4,5,6,8,10,12,15,20,25]:
        _run(f"Path P{n}",     nx.path_graph(n),   "Path")
    for n in [3,4,5,6,8,10,12,15,20,25]:
        _run(f"Cycle C{n}",    nx.cycle_graph(n),  "Cycle")
    for k in [2,3,4,5,6,8,10,12]:
        _run(f"Star K_{{1,{k}}}", nx.star_graph(k), "Star")
    for n in [3,4,5,6,7]:
        _run(f"Complete K{n}", nx.complete_graph(n), "Complete")
    for n, s in [(6,857),(8,4385),(10,1428),(10,6672),(12,4367),(12,1764),
                 (15,625),(15,6211),(18,8785),(20,9213),(20,5442),(25,5583)]:
        _run(f"Tree n={n} s={s}", nx.random_labeled_tree(n, seed=s), "Tree")
    for n, p, s in [(6,.2,850),(8,.2,2615),(10,.2,2212),(10,.2,5524),
                    (12,.2,9190),(12,.2,5468),(15,.2,4016),(15,.2,2683),
                    (18,.2,27),(20,.2,7147)]:
        G = nx.erdos_renyi_graph(n, p, seed=s)
        if not G.edges(): G.add_edge(0, 1)
        _run(f"ER p=0.2 n={n} s={s}", G, "ER-p0.2")
    for n, p, s in [(6,.3,1435),(8,.3,9791),(10,.3,6186),(10,.3,1144),
                    (12,.3,108),(12,.3,5168),(15,.3,7345),(15,.3,1671),
                    (18,.3,719),(20,.3,1519)]:
        G = nx.erdos_renyi_graph(n, p, seed=s)
        if not G.edges(): G.add_edge(0, 1)
        _run(f"ER p=0.3 n={n} s={s}", G, "ER-p0.3")
    for n, p, s in [(6,.5,2329),(8,.5,2068),(10,.5,347),(10,.5,4780),
                    (12,.5,7055),(15,.5,9394),(18,.5,7813)]:
        _run(f"ER p=0.5 n={n} s={s}", nx.erdos_renyi_graph(n, p, seed=s), "ER-p0.5")
    _run("2 disjoint edges",     nx.Graph([(0,1),(2,3)]),                        "Disconnected")
    _run("3 disjoint edges",     nx.Graph([(0,1),(2,3),(4,5)]),                   "Disconnected")
    _run("Path3+edge",           nx.Graph([(0,1),(1,2),(3,4)]),                   "Disconnected")
    _run("Triangle+edge",        nx.Graph([(0,1),(1,2),(2,0),(3,4)]),             "Disconnected")
    _run("2 triangles disconn.", nx.Graph([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]), "Disconnected")
    for r, c in [(2,3),(2,4),(3,3),(2,5),(3,4)]:
        _run(f"Grid {r}x{c}", nx.grid_2d_graph(r, c), "Grid")
    for m1, m2 in [(3,3),(4,4),(3,4),(4,5)]:
        _run(f"Barbell ({m1},{m2})", nx.barbell_graph(m1, m2), "Barbell")
    _run("Lollipop (4,2)", nx.lollipop_graph(4, 2), "Lollipop")
    _run("Lollipop (5,3)", nx.lollipop_graph(5, 3), "Lollipop")
    for n in [5,6,7,8]:
        _run(f"Wheel W{n}", nx.wheel_graph(n), "Wheel")
    _run("Petersen",   nx.petersen_graph(),  "Special")
    _run("Ladder n=4", nx.ladder_graph(4),   "Special")
    _run("Ladder n=6", nx.ladder_graph(6),   "Special")

    # ── Print results ─────────────────────────────────────────────────────
    total   = len(results)
    passed  = sum(1 for r in results if r[4] == 'PASS')
    failed  = sum(1 for r in results if r[4] == 'FAIL')
    timeout = sum(1 for r in results if r[4] == 'TIMEOUT')
    orig    = sum(1 for r in results if len(r) > 6 and r[6] is True)
    new_g   = sum(1 for r in results if len(r) > 6 and r[6] is False)
    total_t = sum(r[5] for r in results)

    print(f"  {'#':>3}  {'Cat':<12}  {'Name':<38}  {'n':>3}  {'m':>4}  {'Result':9s}  {'Solution':<14}  {'t':>6}")
    print("  " + "="*95)
    for i, r in enumerate(results, 1):
        name, cat, n, m, status = r[:5]; el = r[5]
        iso = r[6] if len(r) > 6 else None
        sol = "original graph" if iso else "NEW graph     " if iso is False else "—             "
        sym = "✓" if status == 'PASS' else ("T" if status == 'TIMEOUT' else "✗")
        print(f"  {i:>3}  {cat:<12}  {name:<38}  {n:>3}  {m:>4}  {sym} {status:<7}  {sol}  {el:>5.2f}s")

    print("\n  " + "="*95)
    print(f"\n  TOTAL={total}  PASS={passed} ({100*passed//total}%)  FAIL={failed}  INFEASIBLE={timeout}")
    print(f"  Original={orig}  New={new_g}  Total time={total_t:.1f}s")

    # ── Per-category summary ──────────────────────────────────────────────
    from collections import defaultdict as _ddc
    cats = _ddc(lambda: dict(p=0, f=0, t=0, orig=0, new=0, ns=[], ts=[]))
    for r in results:
        c = r[1]
        if r[4] == 'PASS':   cats[c]['p'] += 1
        elif r[4] == 'FAIL': cats[c]['f'] += 1
        else:                cats[c]['t'] += 1
        if len(r) > 6 and r[6] is True:  cats[c]['orig'] += 1
        if len(r) > 6 and r[6] is False: cats[c]['new']  += 1
        cats[c]['ns'].append(r[2]); cats[c]['ts'].append(r[5])

    print()
    print(f"  {'Category':<14}  {'Tests':>5}  {'Pass':>4}  {'Fail':>4}  {'Infeas':>6}  "
          f"{'Orig':>4}  {'New':>4}  {'n range':>10}  {'avg t':>7}")
    print("  " + "-"*78)
    for cat, s in sorted(cats.items()):
        tc  = s['p'] + s['f'] + s['t']
        avg = sum(s['ts']) / len(s['ts'])
        print(f"  {cat:<14}  {tc:>5}  {s['p']:>4}  {s['f']:>4}  {s['t']:>6}  "
              f"{s['orig']:>4}  {s['new']:>4}  "
              f"{min(s['ns']):>4}–{max(s['ns']):<5}  {avg:>6.2f}s")

    return passed, failed, timeout


if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        # Quick smoke test: 8 representative cases with verbose output
        def run_one(name, G):
            print(f"\n{'='*55}\n  {name}\n{'='*55}")
            p0, p1 = degree_filtration_ph(G)
            H = solve(p0, p1, n_vertices=G.number_of_nodes(), time_limit=60.0, verbose=True)
            if H is None: print("No solution found."); return
            hp0, hp1 = degree_filtration_ph(H)
            ok = ph_equal(p0, p1, hp0, hp1)
            print(f"\n{'PASS' if ok else 'FAIL'}  iso={nx.is_isomorphic(G, H)}")
        run_one("Path P4",           nx.path_graph(4))
        run_one("Cycle C5",          nx.cycle_graph(5))
        run_one("Random tree n=10",  nx.random_labeled_tree(10, seed=42))
        run_one("Random tree n=20",  nx.random_labeled_tree(20, seed=42))
        run_one("ER n=15 p=0.3",     nx.erdos_renyi_graph(15, 0.3, seed=42))
        run_one("Grid 3x3",          nx.grid_2d_graph(3, 3))
        run_one("Petersen",          nx.petersen_graph())
        run_one("Barbell (4,4)",     nx.barbell_graph(4, 4))
    else:
        # Full 95-case suite
        print("Running full 95-case test suite...\n")
        passed, failed, timeout = run_suite(time_limit=60.0)
        sys.exit(0 if failed == 0 and timeout == 0 else 1)