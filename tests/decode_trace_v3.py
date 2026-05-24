#!/usr/bin/env python3
"""
Pipeline-aware analysis of MegaMoE trace data.

For each SM, produces a 38-row pipeline view:
  [Dispatch]
  [Wave 0 L1 TMA_A]  [Wave 0 L1 MMA]  [Wave 0 L1 Epilogue]
  [Wave 0 L2 TMA_A]  [Wave 0 L2 MMA]  [Wave 0 L2 Epilogue]   (NVLink writeback included)
  ... × 6 waves ...
  [Reduce]

Each interval is aggregated as [min(expert_starts), max(expert_ends)] across all
experts in that (sm, role, wave, phase) tuple.

Usage:
    python decode_trace_v3.py trace_v2_rank0.bin
    python decode_trace_v3.py trace_v2_rank0.bin --gantt-sm 0 --csv

Outputs:
    - Console: pipeline breakdown table
    - PNG:     Gantt chart of one SM with all 38 rows
    - CSV:     per-(sm, role, wave, phase) interval list
"""

import argparse
import struct
import sys
import numpy as np
import pandas as pd

# Match the C++ header layout
KEVENTS_PER_ROLE = 1024
SIZEOF_EVENT     = 8
WR_COUNT         = 7
KSMHEADER_BYTES  = 32
KBYTES_PER_ROLE  = KEVENTS_PER_ROLE * SIZEOF_EVENT
KBYTES_PER_SM    = KSMHEADER_BYTES + WR_COUNT * KBYTES_PER_ROLE

ROLE_NAMES = {0: "DISPATCH", 1: "TMA_A", 2: "TMA_B",
              3: "MMA", 4: "EPILOGUE", 5: "REDUCE",
              6: "SCHEDULER"}

EVENT_NAMES = {
    0: "KERNEL_BEGIN", 1: "KERNEL_END",
    10: "WAVE_L1_START", 11: "WAVE_L1_END",
    12: "WAVE_L2_START", 13: "WAVE_L2_END",
    14: "EXPERT_START", 15: "EXPERT_END",
    20: "AFTER_L1_ARRIVAL", 21: "AFTER_L2_ARRIVAL",
    22: "AFTER_EMPTY_BAR",  23: "AFTER_FULL_BAR",
    24: "AFTER_TMEM_FULL",  25: "AFTER_TMEM_EMPTY",
    30: "TMA_A_DONE",       32: "UMMA_ISSUED",
    33: "DISPATCH_START",   34: "DISPATCH_END",
    35: "REDUCE_START",     36: "REDUCE_END",
    40: "TILE_L1", 41: "TILE_L2",  
}

WAVE_L1_START = 10
WAVE_L1_END   = 11
WAVE_L2_START = 12
WAVE_L2_END   = 13


# ============================================================================
# Loading raw data
# ============================================================================

def load(path):
    """Read raw binary, return (events DataFrame, sm_headers dict, tiles DataFrame)."""
    with open(path, "rb") as f:
        num_sms = struct.unpack("<I", f.read(4))[0]
        raw = f.read(num_sms * KBYTES_PER_SM)

    rows = []
    tile_rows = []        # ← 新增
    sm_headers = {}
    for sm in range(num_sms):
        off = sm * KBYTES_PER_SM
        t_ns_start, clock_start, n_evts, t_ns_end, clock_end = \
            struct.unpack_from("<QIIQI", raw, off)
        if clock_start == 0:
            continue

        sm_headers[sm] = {
            "t_ns_start": t_ns_start, "clock_start": clock_start,
            "t_ns_end":   t_ns_end,   "clock_end":   clock_end,
        }

        off += KSMHEADER_BYTES
        for role in range(WR_COUNT):
            role_off = off + role * KBYTES_PER_ROLE
            role_name = ROLE_NAMES.get(role, f"role_{role}")
            prev_clock = clock_start
            wrap_offset = 0
            for slot in range(KEVENTS_PER_ROLE):
                ev_id, aux, clock_lo = struct.unpack_from(
                    "<HHI", raw, role_off + slot * SIZEOF_EVENT)
                if ev_id == 0 and clock_lo == 0:
                    break

                # === Special decoding for SCHEDULER role ===
                if role == 6:    # WR_SCHEDULER
                    if ev_id not in (40, 41):  # not a tile event
                        continue
                    m_block = (aux >> 8) & 0xff
                    n_block = aux & 0xff
                    wave_id = (clock_lo >> 24) & 0xff
                    expert  = (clock_lo >> 16) & 0xff
                    seq     = clock_lo & 0xffff
                    tile_rows.append({
                        "sm":      sm,
                        "phase":   "L1" if ev_id == 40 else "L2",
                        "wave":    wave_id,
                        "expert":  expert,
                        "m_block": m_block,
                        "n_block": n_block,
                        "seq":     seq,
                    })
                    continue

                # === Normal events (unchanged) ===
                if clock_lo < prev_clock:
                    wrap_offset += (1 << 32)
                prev_clock = clock_lo
                cycles = (clock_lo + wrap_offset) - clock_start
                rows.append({
                    "sm": sm,
                    "role": role_name,
                    "event_id": ev_id,
                    "event": EVENT_NAMES.get(ev_id, f"unk_{ev_id}"),
                    "wave": aux >> 8,
                    "expert": aux & 0xff,
                    "cycles_since_sm_start": cycles,
                    "t_ns_sm_start": t_ns_start,
                })
    return pd.DataFrame(rows), sm_headers, pd.DataFrame(tile_rows)


def calibrate_clock_rate_from_headers(sm_headers):
    rates = []
    for h in sm_headers.values():
        if h["t_ns_end"] <= h["t_ns_start"]:
            continue
        dc = h["clock_end"] - h["clock_start"]
        if h["clock_end"] < h["clock_start"]:
            dc += (1 << 32)
        dn_ns = h["t_ns_end"] - h["t_ns_start"]
        rates.append(dc / (dn_ns * 1e-9))
    if not rates:
        return 1.86e9
    rates = np.array(rates)
    print(f"Clock rate from device timing: {np.median(rates)/1e9:.3f} GHz "
          f"(min {rates.min()/1e9:.3f}, max {rates.max()/1e9:.3f}, "
          f"std {rates.std()/1e9:.4f} GHz across {len(rates)} SMs)")
    return float(np.median(rates))


def to_global_ns(df, clock_hz):
    df = df.copy()
    df["t_ns"] = df["t_ns_sm_start"] + (df["cycles_since_sm_start"] / clock_hz * 1e9).astype("int64")
    t0 = df["t_ns"].min()
    df["t_us"] = (df["t_ns"] - t0) / 1000.0
    return df.sort_values(["sm", "role", "t_us"]).reset_index(drop=True)


# ============================================================================
# Pipeline-level aggregation
# ============================================================================

def build_intervals(df, use_work_start=True):
    """
    Aggregate per-expert START/END events into per-(sm, role, wave, phase)
    intervals.

    Two modes:

    use_work_start=False (the "loose" view):
        interval start = min(WAVE_*_START) — when the warp ENTERS the
        for_each_block closure for this wave. The warp may still be waiting
        on a barrier for the first several microseconds, so this overstates
        the work range.

    use_work_start=True (the "tight" view, default):
        interval start = min(first relevant AFTER_* event) — when the warp
        has actually unblocked and started doing useful work.
            TMA_A    L1 → AFTER_L1_ARRIVAL  (21)
            TMA_A    L2 → AFTER_L2_ARRIVAL  (22)
            MMA      *  → AFTER_TMEM_EMPTY  (25)
            EPILOGUE *  → AFTER_TMEM_FULL   (24)

    Either way, interval end = max(WAVE_*_END) — the last expert finishes.
    DISPATCH and REDUCE use their START/END pairs directly (no wait events).
    """
    intervals = []

    # ---- Dispatch (one interval per SM) ----
    disp = df[df["role"] == "DISPATCH"]
    for sm, sm_df in disp.groupby("sm"):
        starts = sm_df[sm_df["event_id"] == 33]
        ends   = sm_df[sm_df["event_id"] == 34]
        if starts.empty or ends.empty:
            continue
        intervals.append({
            "sm": sm, "role": "DISPATCH", "wave": -1, "phase": "DISPATCH",
            "t_start_us": starts["t_us"].min(),
            "t_end_us":   ends["t_us"].max(),
        })

    # ---- Reduce (one interval per SM) ----
    red = df[df["role"] == "REDUCE"]
    for sm, sm_df in red.groupby("sm"):
        starts = sm_df[sm_df["event_id"] == 35]
        ends   = sm_df[sm_df["event_id"] == 36]
        if starts.empty or ends.empty:
            continue
        intervals.append({
            "sm": sm, "role": "REDUCE", "wave": -1, "phase": "REDUCE",
            "t_start_us": starts["t_us"].min(),
            "t_end_us":   ends["t_us"].max(),
        })

    # ---- TMA_A / MMA / EPILOGUE per wave/phase ----
    # Map (role, phase) → event_id that marks "real work start"
    AFTER_L1_ARRIVAL = 20
    AFTER_L2_ARRIVAL = 21
    AFTER_FULL_BAR   = 22   # MMA waits this after AFTER_TMEM_EMPTY — first SMEM stage ready
    AFTER_TMEM_FULL  = 24
    AFTER_TMEM_EMPTY = 25

    # Tight-mode "real work start" event per (role, phase):
    #   TMA_A: work starts after arrival spin (data in pool, TMA begins loading)
    #   MMA:   work starts after full_barrier for k_block=0 — this is when TMA has
    #          actually filled the first SMEM stage and MMA can issue its first UMMA.
    #          AFTER_TMEM_EMPTY fires before the k-block loop, so it's NOT a real
    #          work-start — the warp still waits for TMA inside the loop.
    #   EPILOGUE: work starts after tmem_full — result is in tmem, can be read.
    work_start_event = {
        ("TMA_A",   "L1"): AFTER_L1_ARRIVAL,
        ("TMA_A",   "L2"): AFTER_L2_ARRIVAL,
        ("MMA",     "L1"): AFTER_FULL_BAR,    # ← was AFTER_TMEM_EMPTY; now tracks true start
        ("MMA",     "L2"): AFTER_FULL_BAR,    # ← same
        ("EPILOGUE","L1"): AFTER_TMEM_FULL,
        ("EPILOGUE","L2"): AFTER_TMEM_FULL,
    }

    for role in ("TMA_A", "MMA", "EPILOGUE"):
        rd = df[df["role"] == role]
        if rd.empty:
            continue
        for (sm, wave), wave_df in rd.groupby(["sm", "wave"]):
            for phase, start_id, end_id in (
                    ("L1", WAVE_L1_START, WAVE_L1_END),
                    ("L2", WAVE_L2_START, WAVE_L2_END)):
                ends = wave_df[wave_df["event_id"] == end_id]
                if ends.empty:
                    continue
                end_t = ends["t_us"].max()

                if use_work_start:
                    # Use AFTER_* event as the "actually started working" marker
                    after_id = work_start_event[(role, phase)]
                    # The AFTER_* events fire per-k-block inside an expert.
                    # The first one in this wave/phase = first time this warp
                    # actually unblocked.
                    # Filter: AFTER events between the wave's first WAVE_*_START
                    # and last WAVE_*_END.
                    starts = wave_df[wave_df["event_id"] == start_id]
                    if starts.empty:
                        continue
                    wave_open  = starts["t_us"].min()
                    wave_close = end_t
                    afters = wave_df[(wave_df["event_id"] == after_id) &
                                     (wave_df["t_us"] >= wave_open) &
                                     (wave_df["t_us"] <= wave_close)]
                    if afters.empty:
                        # No AFTER_* event for this wave/phase — fall back to WAVE_*_START
                        start_t = wave_open
                    else:
                        start_t = afters["t_us"].min()
                else:
                    starts = wave_df[wave_df["event_id"] == start_id]
                    if starts.empty:
                        continue
                    start_t = starts["t_us"].min()

                intervals.append({
                    "sm": sm, "role": role, "wave": wave, "phase": phase,
                    "t_start_us": start_t,
                    "t_end_us":   end_t,
                })

    iv = pd.DataFrame(intervals)
    if not iv.empty:
        iv["duration_us"] = iv["t_end_us"] - iv["t_start_us"]
    return iv


# ============================================================================
# Reports
# ============================================================================

def summary(df, sm_headers):
    print(f"\n=== Event counts: {len(df):,} events across {df['sm'].nunique()} SMs ===")
    g = df.groupby(["role", "event"]).size().rename("count")
    print(g.to_string())

    spans = []
    for h in sm_headers.values():
        if h["t_ns_end"] > h["t_ns_start"]:
            spans.append((h["t_ns_end"] - h["t_ns_start"]) / 1000.0)
    if spans:
        print(f"\n=== Actual kernel span (from per-SM globaltimer) ===")
        print(f"  median = {np.median(spans):.1f} µs, max = {max(spans):.1f} µs")


def pipeline_breakdown(iv, focus_sm=0):
    """Print a per-wave pipeline table for one SM."""
    print(f"\n=== Pipeline breakdown — SM {focus_sm} ===")
    print(f"{'Phase':<14} {'Role':<10} {'start (µs)':>12} {'end (µs)':>12} {'dur (µs)':>10}")
    print("-" * 64)

    sm_iv = iv[iv["sm"] == focus_sm].sort_values("t_start_us")

    # Group by (wave, phase, role) and display
    if sm_iv.empty:
        print("  (no intervals on this SM)")
        return

    # Dispatch first
    d = sm_iv[sm_iv["role"] == "DISPATCH"]
    if not d.empty:
        r = d.iloc[0]
        print(f"{'Dispatch':<14} {'':<10} {r.t_start_us:>12.2f} {r.t_end_us:>12.2f} {r.duration_us:>10.2f}")

    # Then waves
    waves = sorted(sm_iv[sm_iv["wave"] >= 0]["wave"].unique())
    for w in waves:
        w_iv = sm_iv[sm_iv["wave"] == w]
        for phase in ("L1", "L2"):
            p_iv = w_iv[w_iv["phase"] == phase].sort_values("t_start_us")
            if p_iv.empty:
                continue
            print(f"--- Wave {w} {phase} ---")
            for _, r in p_iv.iterrows():
                print(f"  {'':<12} {r.role:<10} {r.t_start_us:>12.2f} {r.t_end_us:>12.2f} {r.duration_us:>10.2f}")

    # Reduce last
    rd = sm_iv[sm_iv["role"] == "REDUCE"]
    if not rd.empty:
        r = rd.iloc[0]
        print(f"{'Reduce':<14} {'':<10} {r.t_start_us:>12.2f} {r.t_end_us:>12.2f} {r.duration_us:>10.2f}")


def aggregate_across_sms(iv):
    """Show min/median/max duration of each phase across all SMs."""
    print(f"\n=== Phase durations aggregated across all SMs ===")
    iv2 = iv.copy()
    iv2["phase_label"] = iv2.apply(
        lambda r: r["phase"] if r["wave"] < 0 else f"W{r['wave']} {r['phase']}",
        axis=1)
    g = iv2.groupby(["phase_label", "role"])["duration_us"].agg(
        n="count",
        median=lambda x: x.median(),
        p95=lambda x: x.quantile(0.95),
        max="max",
    ).round(2)
    # Sort by phase_label naturally: Dispatch first, then waves in order, Reduce last
    print(g.to_string())


def gantt_chart(iv, focus_sm, out_path):
    """Render a Gantt-style timeline of one SM with all pipeline rows."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping Gantt)")
        return

    sm_iv = iv[iv["sm"] == focus_sm].copy()
    if sm_iv.empty:
        print(f"  (no intervals on SM {focus_sm})")
        return

    # Build the row ordering: Dispatch, then for each wave: L1 (TMA/MMA/EPI) then L2, then Reduce
    waves = sorted(sm_iv[sm_iv["wave"] >= 0]["wave"].unique())
    row_labels = ["Dispatch"]
    row_keys   = [("DISPATCH", -1, "DISPATCH")]   # (role, wave, phase)

    for w in waves:
        for phase in ("L1", "L2"):
            for role in ("TMA_A", "MMA", "EPILOGUE"):
                row_labels.append(f"W{w} {phase} {role}")
                row_keys.append((role, w, phase))

    row_labels.append("Reduce")
    row_keys.append(("REDUCE", -1, "REDUCE"))

    fig, ax = plt.subplots(figsize=(16, max(6, 0.25 * len(row_labels))))

    role_colors = {
        "DISPATCH": "#888888",
        "TMA_A":    "#4a90e2",
        "MMA":      "#e74c3c",
        "EPILOGUE": "#27ae60",
        "REDUCE":   "#f39c12",
    }

    for y, (role, w, phase) in enumerate(row_keys):
        matches = sm_iv[(sm_iv["role"] == role) &
                        (sm_iv["wave"] == w) &
                        (sm_iv["phase"] == phase)]
        for _, r in matches.iterrows():
            ax.barh(y, r["duration_us"], left=r["t_start_us"], height=0.7,
                    color=role_colors.get(role, "#aaa"),
                    edgecolor="black", linewidth=0.4, alpha=0.85)
            # Annotate duration
            if r["duration_us"] > 1.5:
                ax.text(r["t_start_us"] + r["duration_us"] / 2, y,
                        f"{r['duration_us']:.1f}", ha="center", va="center",
                        fontsize=6, color="white", weight="bold")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time (µs)")
    ax.set_title(f"MegaMoE pipeline — SM {focus_sm}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"  Wrote Gantt chart: {out_path}")


# ============================================================================
# Tile assignment analysis (Style 2: SM-as-rows view)
# ============================================================================

def tile_assignment_summary(tiles_df):
    """Print high-level stats about the task partitioning."""
    if tiles_df.empty:
        print("\n(no SCHEDULER tile events found — did you enable EV_TILE_*?)")
        return

    print(f"\n=== Tile assignment summary ===")
    n_sms = tiles_df["sm"].nunique()
    n_waves = tiles_df["wave"].nunique()
    n_l1 = (tiles_df["phase"] == "L1").sum()
    n_l2 = (tiles_df["phase"] == "L2").sum()
    print(f"  Total tiles:    L1={n_l1:,}  L2={n_l2:,}  (sum={n_l1+n_l2:,})")
    print(f"  Distinct waves: {n_waves}")
    print(f"  Active SMs:     {n_sms}")
    print(f"  Mean tiles per SM: L1={n_l1/n_sms:.1f}  L2={n_l2/n_sms:.1f}")

    # Per-(wave, phase) workload
    print(f"\n=== Per-(wave, phase) total tile count ===")
    g = tiles_df.groupby(["wave", "phase"]).size().unstack(fill_value=0)
    g["total"] = g.sum(axis=1)
    print(g.to_string())

    # Identify imbalance: per-(wave, phase) tile count per SM
    print(f"\n=== Per-(wave, phase) per-SM load distribution ===")
    print("(min/median/max tiles each SM got in that wave/phase)")
    per_sm = tiles_df.groupby(["wave", "phase", "sm"]).size().reset_index(name="n_tiles")
    g2 = per_sm.groupby(["wave", "phase"])["n_tiles"].agg(
        active_sms="count",
        min="min",
        median=lambda x: int(x.median()),
        max="max",
        mean=lambda x: round(x.mean(), 2),
    )
    print(g2.to_string())


def plot_tile_assignment_style2(tiles_df, out_path, num_sms_total=None):
    """
    Style 2: SM-as-rows heatmap with numeric annotations.
      Rows  = SMs (0 to num_sms-1)
      Cols  = (wave, phase) tuples in time order
      Cell  = number of tiles SM got in that (wave, phase)
              + background color (white = 0/empty, deeper blue = more tiles)
      Column headers show total tiles for that (wave, phase).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("  (matplotlib not available, skipping tile-assignment plot)")
        return
 
    if tiles_df.empty:
        print("  (no tile assignment data, skipping plot)")
        return
 
    if num_sms_total is None:
        num_sms_total = tiles_df["sm"].max() + 1
 
    waves = sorted(tiles_df["wave"].unique())
    columns = [(w, p) for w in waves for p in ("L1", "L2") if not tiles_df[
        (tiles_df["wave"] == w) & (tiles_df["phase"] == p)].empty]
    n_cols = len(columns)
 
    # Build the data matrix: rows = SM, cols = (wave, phase)
    counts = (tiles_df.groupby(["sm", "wave", "phase"])
                       .size().reset_index(name="n_tiles"))
 
    matrix = np.zeros((num_sms_total, n_cols), dtype=int)
    for _, r in counts.iterrows():
        col_idx = columns.index((r["wave"], r["phase"]))
        matrix[r["sm"], col_idx] = r["n_tiles"]
 
    column_totals = [tiles_df[(tiles_df["wave"] == w) & (tiles_df["phase"] == p)].shape[0]
                     for (w, p) in columns]
 
    # Cell size: keep numbers readable
    cell_w = 0.6
    cell_h = 0.20
    fig_w = max(8, cell_w * n_cols + 3)
    fig_h = max(8, cell_h * num_sms_total + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
 
    # Heatmap with white = 0, deeper blue = larger n_tiles
    max_val = matrix.max() if matrix.max() > 0 else 1
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_to_blue", ["#ffffff", "#cce0ff", "#3a78c2", "#0d3a73"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max_val)
 
    # Numeric annotation in each cell
    for sm_idx in range(num_sms_total):
        for col_idx in range(n_cols):
            n = matrix[sm_idx, col_idx]
            if n == 0:
                continue   # leave 0 cells blank to highlight "no work"
            # Text color: white on dark cells, black on light
            color = "white" if n > max_val * 0.55 else "black"
            ax.text(col_idx, sm_idx, str(n), ha="center", va="center",
                    fontsize=5, color=color)
 
    # Column header text: wave/phase label + total tile count
    for col_idx, (w, p) in enumerate(columns):
        ax.text(col_idx, -1.5, f"W{w}\n{p}\n{column_totals[col_idx]}",
                ha="center", va="bottom", fontsize=7, weight="bold")
 
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(["" for _ in columns])    # we drew our own headers above
    ax.set_yticks(range(num_sms_total))
    ax.set_yticklabels([str(i) for i in range(num_sms_total)], fontsize=5)
    ax.set_ylabel("SM ID")
    ax.set_xlabel("(wave, phase)   |   blank cell = SM did not get any tile in that phase")
 
    # Top padding so headers don't overlap data
    ax.set_ylim(num_sms_total - 0.5, -3.5)
 
    total_l1 = int((tiles_df["phase"] == "L1").sum())
    total_l2 = int((tiles_df["phase"] == "L2").sum())
    ax.set_title(
        f"Tile assignment by SM × (wave, phase)\n"
        f"Total L1 tiles = {total_l1:,}   Total L2 tiles = {total_l2:,}   "
        f"(grand total = {total_l1 + total_l2:,}, {num_sms_total} SMs)",
        fontsize=11)
 
    plt.colorbar(im, ax=ax, label="tiles per cell", shrink=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"  Wrote tile-assignment plot: {out_path}")


def export_tile_csv(tiles_df, out_path):
    """Dump the full tile assignment table for ad-hoc analysis."""
    tiles_df.sort_values(["sm", "wave", "phase", "seq"]).to_csv(out_path, index=False)
    print(f"  Wrote tile CSV: {out_path}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file")
    parser.add_argument("--gantt-sm", type=int, default=0)
    parser.add_argument("--csv", action="store_true",
                        help="Also dump intervals as CSV")
    parser.add_argument("--loose-intervals", action="store_true",
                        help="Use WAVE_*_START as interval start (the warp ENTERED "
                             "the closure but may still be waiting). Default is "
                             "tight intervals using AFTER_* events.")
    args = parser.parse_args()

    df, sm_headers, tiles_df = load(args.trace_file)
    if df.empty:
        print("No events found.")
        sys.exit(1)

    clock_hz = calibrate_clock_rate_from_headers(sm_headers)
    df = to_global_ns(df, clock_hz)
    print(f"Loaded {len(df):,} events from {args.trace_file}")

    summary(df, sm_headers)

    iv = build_intervals(df, use_work_start=not args.loose_intervals)
    if args.loose_intervals:
        print(f"\n(Built intervals in LOOSE mode: interval starts at WAVE_*_START — "
              f"includes wait-before-work time.)")
    else:
        print(f"\n(Built intervals in TIGHT mode: interval starts at first AFTER_* "
              f"event — actual work time only. Pass --loose-intervals to compare.)")
    if iv.empty:
        print("\nNo intervals could be built.")
        sys.exit(1)

    print(f"\n=== Built {len(iv):,} intervals "
          f"({iv['sm'].nunique()} SMs, "
          f"{iv['wave'].nunique()} distinct wave ids including -1) ===")

    pipeline_breakdown(iv, focus_sm=args.gantt_sm)
    aggregate_across_sms(iv)
    gantt_chart(iv, args.gantt_sm,
                args.trace_file.replace(".bin", f".pipeline.sm{args.gantt_sm}.png"))
    
    tile_assignment_summary(tiles_df)
    plot_tile_assignment_style2(
        tiles_df,
        args.trace_file.replace(".bin", ".tiles.png"),
        num_sms_total=max(sm_headers.keys()) + 1 if sm_headers else None)

    if args.csv:
        out_csv = args.trace_file.replace(".bin", ".intervals.csv")
        iv.to_csv(out_csv, index=False)
        print(f"\n  Wrote intervals CSV: {out_csv}")
        if not tiles_df.empty:
            export_tile_csv(tiles_df, args.trace_file.replace(".bin", ".tiles.csv"))