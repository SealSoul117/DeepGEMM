#!/usr/bin/env python3
"""
Analyze MegaMoE v2 trace data.

Usage:
    python analyze_trace_v2.py trace_v2_rank0.bin
    python analyze_trace_v2.py trace_v2_rank0.bin --clock-hz 1.86e9 --gantt-sm 0

Produces:
    1. Console: summary tables (event counts, wait durations, per-wave breakdown)
    2. PNG:     gantt chart of one SM's pipeline
    3. CSV:     full event list for further ad-hoc analysis
"""

import argparse
import struct
import sys
import numpy as np
import pandas as pd

# Match the C++ header layout
KEVENTS_PER_ROLE = 256
SIZEOF_EVENT     = 8
WR_COUNT         = 6
KSMHEADER_BYTES  = 32
KBYTES_PER_ROLE  = KEVENTS_PER_ROLE * SIZEOF_EVENT
KBYTES_PER_SM    = KSMHEADER_BYTES + WR_COUNT * KBYTES_PER_ROLE

ROLE_NAMES = {0: "DISPATCH", 1: "TMA_A", 2: "TMA_B",
              3: "MMA", 4: "EPILOGUE", 5: "COMBINE"}

EVENT_NAMES = {
    0: "KERNEL_BEGIN", 1: "KERNEL_END",
    2: "DISPATCH_START", 3: "DISPATCH_END",
    4: "COMBINE_START", 5: "COMBINE_END",
    10: "WAVE_L1_START", 11: "WAVE_L1_END",
    12: "WAVE_L2_START", 13: "WAVE_L2_END",
    14: "EXPERT_START", 15: "EXPERT_END",
    20: "AFTER_L1_ARRIVAL", 21: "AFTER_L2_ARRIVAL",
    22: "AFTER_EMPTY_BAR",  23: "AFTER_FULL_BAR",
    24: "AFTER_TMEM_FULL",  25: "AFTER_TMEM_EMPTY",
    30: "TMA_A_DONE",       32: "UMMA_ISSUED",
}

PHASE_START_IDS = {10, 12}   # WAVE_L1_START, WAVE_L2_START
PHASE_END_IDS   = {11, 13}   # WAVE_L1_END,   WAVE_L2_END
WAIT_EVENT_IDS  = {20, 21, 22, 23, 24, 25}  # all AFTER_* events


def load(path):
    """Read raw binary and return a DataFrame of all events."""
    with open(path, "rb") as f:
        num_sms = struct.unpack("I", f.read(4))[0]
        raw = f.read(num_sms * KBYTES_PER_SM)

    rows = []
    for sm in range(num_sms):
        off = sm * KBYTES_PER_SM
        t_ns_start, clock_start, _ = struct.unpack_from("QII", raw, off)
        if clock_start == 0:
            continue  # this SM never executed (more SMs available than launched)
        off += KSMHEADER_BYTES
        for role in range(WR_COUNT):
            role_off = off + role * KBYTES_PER_ROLE
            prev_clock = clock_start
            wrap_offset = 0
            for slot in range(KEVENTS_PER_ROLE):
                ev_id, aux, clock_lo = struct.unpack_from(
                    "HHI", raw, role_off + slot * SIZEOF_EVENT)
                if ev_id == 0 and clock_lo == 0:
                    break  # unwritten slot
                # u32 clock counter wraps every ~2 s at 2 GHz
                if clock_lo < prev_clock:
                    wrap_offset += (1 << 32)
                prev_clock = clock_lo
                cycles = (clock_lo + wrap_offset) - clock_start
                rows.append({
                    "sm": sm,
                    "role": ROLE_NAMES.get(role, f"role_{role}"),
                    "event_id": ev_id,
                    "event": EVENT_NAMES.get(ev_id, f"unk_{ev_id}"),
                    "aux": aux,
                    "wave": aux >> 8,
                    "expert": aux & 0xff,
                    "cycles_since_sm_start": cycles,
                    "t_ns_sm_start": t_ns_start,
                })
    return pd.DataFrame(rows)


def to_global_ns(df, clock_hz):
    """Translate per-SM cycles to a single ns-based timeline."""
    df = df.copy()
    df["t_ns"] = df["t_ns_sm_start"] + (df["cycles_since_sm_start"] / clock_hz * 1e9).astype("int64")
    t0 = df["t_ns"].min()
    df["t_us"] = (df["t_ns"] - t0) / 1000.0
    return df.sort_values(["sm", "role", "t_us"]).reset_index(drop=True)


# ============================================================================
# Analyses
# ============================================================================

def summary(df):
    print(f"\n=== Event counts: {len(df):,} events across {df['sm'].nunique()} SMs ===")
    g = df.groupby(["role", "event"]).size().rename("count")
    print(g.to_string())

    span_us = df["t_us"].max() - df["t_us"].min()
    print(f"\n=== Kernel time span: {span_us:.1f} µs ===")


def wait_breakdown(df):
    """For each AFTER_* event, the gap from the prior event on the same (sm, role)
    is the wait duration. Aggregate."""
    print("\n=== Wait-duration breakdown ===")
    print("(Gap from previous event to this AFTER_* event; this is how long the warp was stalled.)\n")

    d = df.sort_values(["sm", "role", "t_us"]).reset_index(drop=True)
    d["prev_t_us"] = d.groupby(["sm", "role"])["t_us"].shift(1)
    d["wait_us"] = d["t_us"] - d["prev_t_us"]

    waits = d[d["event_id"].isin(WAIT_EVENT_IDS)].dropna(subset=["wait_us"])
    if waits.empty:
        print("  (no AFTER_* events found)")
        return

    g = waits.groupby(["role", "event"])["wait_us"].agg(
        n="count",
        mean=lambda x: x.mean(),
        median=lambda x: x.median(),
        p95=lambda x: x.quantile(0.95),
        max="max",
    )
    g = g.round(2)
    print(g.to_string())

    # Flag the worst offender
    total_waits = waits.groupby(["role", "event"])["wait_us"].sum().sort_values(ascending=False)
    top = total_waits.head(3)
    print("\n--- Top 3 wait sources by total time ---")
    for (role, event), total in top.items():
        print(f"  {role:10s} {event:18s}  total = {total/1000:.1f} ms (across all SMs)")


def per_wave_breakdown(df):
    """For each wave, total time on MMA and EPILOGUE."""
    print("\n=== Per-wave breakdown (MMA role, SM 0) ===")
    d = df[(df["role"] == "MMA") & (df["sm"] == 0)].sort_values("t_us")
    if d.empty:
        print("  (no MMA events on SM 0)")
        return

    starts = d[d["event_id"].isin(PHASE_START_IDS)].reset_index(drop=True)
    ends   = d[d["event_id"].isin(PHASE_END_IDS)].reset_index(drop=True)
    n = min(len(starts), len(ends))

    rows = []
    for i in range(n):
        s, e = starts.iloc[i], ends.iloc[i]
        phase = "L1" if s["event_id"] == 10 else "L2"
        rows.append({
            "wave": s["wave"],
            "phase": phase,
            "expert": s["expert"],
            "duration_us": e["t_us"] - s["t_us"],
        })
    waves_df = pd.DataFrame(rows)
    if waves_df.empty:
        return
    g = waves_df.groupby(["wave", "phase"])["duration_us"].agg(["sum", "count"]).round(2)
    print(g.to_string())


def overlap_efficiency(df):
    """How well are dispatch / GEMM / combine overlapped?
    Compute the time during which each is active (defined as 'between START
    event and END event for that phase'), and the intersection."""
    print("\n=== Overlap efficiency ===")

    def phase_span(role, start_id, end_id):
        d = df[df["role"] == role]
        starts = d[d["event_id"] == start_id]
        ends   = d[d["event_id"] == end_id]
        if starts.empty or ends.empty:
            return None
        return (starts["t_us"].min(), ends["t_us"].max())

    dispatch_span = phase_span("DISPATCH", 2, 3)
    combine_span  = phase_span("COMBINE",  4, 5)

    # GEMM = aggregate of TMA_A / MMA / EPILOGUE phase intervals
    gemm_rows = df[df["role"].isin(["TMA_A", "MMA", "EPILOGUE"])
                   & df["event_id"].isin(PHASE_START_IDS | PHASE_END_IDS)]
    if not gemm_rows.empty:
        gemm_span = (gemm_rows["t_us"].min(), gemm_rows["t_us"].max())
    else:
        gemm_span = None

    def fmt(span):
        return f"[{span[0]:8.1f} → {span[1]:8.1f}] µs (len={span[1]-span[0]:.1f})" if span else "n/a"

    print(f"  dispatch span:  {fmt(dispatch_span)}")
    print(f"  gemm     span:  {fmt(gemm_span)}")
    print(f"  combine  span:  {fmt(combine_span)}")

    def overlap(a, b):
        if a is None or b is None:
            return None
        lo = max(a[0], b[0])
        hi = min(a[1], b[1])
        return max(0.0, hi - lo)

    if dispatch_span and gemm_span:
        o = overlap(dispatch_span, gemm_span)
        gemm_len = gemm_span[1] - gemm_span[0]
        print(f"  dispatch ∩ gemm = {o:.1f} µs  ({100*o/gemm_len:.1f}% of GEMM)")
    if gemm_span and combine_span:
        o = overlap(gemm_span, combine_span)
        gemm_len = gemm_span[1] - gemm_span[0]
        print(f"  gemm ∩ combine  = {o:.1f} µs  ({100*o/gemm_len:.1f}% of GEMM)")


def gantt_chart(df, sm, out_path):
    """Render a Gantt-style timeline of one SM."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping Gantt)")
        return

    d = df[df["sm"] == sm].copy()
    if d.empty:
        print(f"  (no events on SM {sm})")
        return

    roles = ["DISPATCH", "TMA_A", "TMA_B", "MMA", "EPILOGUE", "COMBINE"]
    role_y = {r: i for i, r in enumerate(roles)}
    colors = {
        "DISPATCH": "#888888", "TMA_A":   "#4a90e2", "TMA_B":   "#7b68ee",
        "MMA":      "#e74c3c", "EPILOGUE": "#27ae60", "COMBINE": "#f39c12",
    }

    fig, ax = plt.subplots(figsize=(16, 5))

    # For each role, pair up START / END events into intervals
    interval_start_ids = {2, 4, 10, 12, 14}    # DISPATCH/COMBINE/WAVE_L1/WAVE_L2/EXPERT START
    interval_end_ids   = {3, 5, 11, 13, 15}    # corresponding ENDs

    for role in roles:
        rd = d[d["role"] == role].sort_values("t_us").reset_index(drop=True)
        if rd.empty:
            continue
        starts = rd[rd["event_id"].isin(interval_start_ids)].reset_index(drop=True)
        ends   = rd[rd["event_id"].isin(interval_end_ids)].reset_index(drop=True)
        n = min(len(starts), len(ends))
        y = role_y[role]
        for i in range(n):
            s, e = starts.iloc[i], ends.iloc[i]
            ax.barh(y, e["t_us"] - s["t_us"], left=s["t_us"], height=0.6,
                    color=colors[role], edgecolor="black", linewidth=0.3, alpha=0.8)
            # Label
            if (e["t_us"] - s["t_us"]) > 2:  # only label intervals > 2 µs
                ax.text(s["t_us"], y + 0.25, f"w{s['wave']}e{s['expert']}",
                        fontsize=5, va="bottom")

    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(roles)
    ax.set_xlabel("time (µs)")
    ax.set_title(f"MegaMoE pipeline timeline — SM {sm}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"\n  Wrote Gantt chart: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file", help="Path to trace_v2_rank*.bin")
    parser.add_argument("--clock-hz", type=float, default=1.86e9,
                        help="SM clock rate in Hz (default 1.86e9 — check `nvidia-smi --query-gpu=clocks.sm`)")
    parser.add_argument("--gantt-sm", type=int, default=0,
                        help="Which SM to render in the Gantt chart (default 0)")
    parser.add_argument("--csv", action="store_true",
                        help="Also dump all events as CSV (one row per event)")
    args = parser.parse_args()

    df = load(args.trace_file)
    if df.empty:
        print("No events found. Did init/reset/launch happen in the right order?")
        sys.exit(1)

    df = to_global_ns(df, args.clock_hz)
    print(f"Loaded {len(df):,} events from {args.trace_file}")

    summary(df)
    overlap_efficiency(df)
    wait_breakdown(df)
    per_wave_breakdown(df)
    gantt_chart(df, args.gantt_sm, args.trace_file.replace(".bin", f".sm{args.gantt_sm}.png"))

    if args.csv:
        out_csv = args.trace_file.replace(".bin", ".csv")
        df.to_csv(out_csv, index=False)
        print(f"\n  Wrote full event CSV: {out_csv}")