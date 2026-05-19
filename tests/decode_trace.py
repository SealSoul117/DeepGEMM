#!/usr/bin/env python3
"""
Decode mega_moe trace_v2 binary dumps.

Per-SM cycle counters need:
  1. Conversion to ns via that SM's (t_ns_at_start, clock_at_start) baseline
  2. Wrap detection (clock_lo is u32; wraps ~every 2 sec at 2 GHz)
"""
import sys, struct
import numpy as np
import pandas as pd

# Must match the C++ header
KEVENTS_PER_ROLE = 256
SIZEOF_EVENT     = 8
WR_COUNT         = 6
KSMHEADER_BYTES  = 32
KBYTES_PER_ROLE  = KEVENTS_PER_ROLE * SIZEOF_EVENT
KBYTES_PER_SM    = KSMHEADER_BYTES + WR_COUNT * KBYTES_PER_ROLE

ROLE_NAMES = ["DISPATCH", "TMA_A", "TMA_B", "MMA", "EPILOGUE", "COMBINE"]
EVENT_NAMES = {
    0: "KERNEL_BEGIN", 1: "KERNEL_END",
    10: "WAVE_L1_START", 11: "WAVE_L1_END", 12: "WAVE_L2_START", 13: "WAVE_L2_END",
    14: "EXPERT_START", 15: "EXPERT_END",
    20: "AFTER_L1_ARRIVAL", 21: "AFTER_L2_ARRIVAL",
    22: "AFTER_EMPTY_BAR", 23: "AFTER_FULL_BAR",
    24: "AFTER_TMEM_FULL", 25: "AFTER_TMEM_EMPTY",
    30: "TMA_A_DONE", 32: "UMMA_ISSUED",
    33: "DISPATCH_START", 34: "DISPATCH_END",
    35: "COMBINE_START", 36: "COMBINE_END"
}


def decode(path):
    with open(path, "rb") as f:
        num_sms = struct.unpack("I", f.read(4))[0]
        raw = f.read(num_sms * KBYTES_PER_SM)
    rows = []
    for sm in range(num_sms):
        off = sm * KBYTES_PER_SM
        # Header
        t_ns_start, clock_start, n_evts = struct.unpack_from(
            "QII", raw, off)
        off += KSMHEADER_BYTES
        if clock_start == 0:
            continue  # this SM never ran (more SMs than active)
        for role in range(WR_COUNT):
            role_off = off + role * KBYTES_PER_ROLE
            # Read all 256 slots; valid ones have event_id != 0 OR clock_lo != 0.
            # We detect "end of writes" by the first slot that's all zeros.
            prev_clock = clock_start
            wrap_offset = 0
            for slot in range(KEVENTS_PER_ROLE):
                ev_id, aux, clock_lo = struct.unpack_from(
                    "HHI", raw, role_off + slot * SIZEOF_EVENT)
                if ev_id == 0 and clock_lo == 0:
                    break  # unwritten
                # Detect u32 wrap
                if clock_lo < prev_clock:
                    wrap_offset += (1 << 32)
                prev_clock = clock_lo
                # Cycles since this SM's start, then convert to ns assuming
                # a fixed clock rate (calibrated below).
                cycles_since_start = (clock_lo + wrap_offset) - clock_start
                rows.append({
                    "sm": sm,
                    "role": ROLE_NAMES[role],
                    "event": EVENT_NAMES.get(ev_id, f"unk_{ev_id}"),
                    "event_id": ev_id,
                    "aux": aux,
                    "wave_idx": aux >> 8,        # if you encoded it this way
                    "expert_idx": aux & 0xff,
                    "cycles_since_sm_start": cycles_since_start,
                    "t_ns_sm_start": t_ns_start,
                })
    return pd.DataFrame(rows)


def calibrate_clock_rate(df):
    """Estimate GPU clock rate from any two events on the same SM with known ns gap.
    
    Simplest approach: use KERNEL_BEGIN events from multiple SMs — they all
    happen at nearly the same moment in real time, but each SM's clock counter
    is independent. We use globaltimer (already in ns) as ground truth.
    
    Better approach if you have it: read --query-gpu=clocks.sm from nvidia-smi
    once, or just hard-code (e.g. H100 SM clock ≈ 1830 MHz).
    """
    # Without external info, fall back to a known clock rate.
    # SM100 (B200) base SM clock is around 1.86 GHz; boost can hit 2.2+.
    # For ns-level accuracy, you should measure this once on your hardware.
    return 1.86e9  # Hz


def to_global_ns(df, clock_hz):
    """Convert per-SM cycles to absolute ns, aligned across SMs."""
    df = df.copy()
    df["t_ns"] = df["t_ns_sm_start"] + (df["cycles_since_sm_start"] / clock_hz * 1e9).astype("int64")
    # Normalize: subtract the earliest event so t starts near 0.
    t0 = df["t_ns"].min()
    df["t_us"] = (df["t_ns"] - t0) / 1000.0
    return df.sort_values("t_us").reset_index(drop=True)


def summary(df):
    print(f"\n=== Trace summary: {len(df):,} events across {df['sm'].nunique()} SMs ===")
    print("\nEvents per (role, event):")
    print(df.groupby(["role", "event"]).size().to_string())


def interval_analysis(df, role, evt_pair):
    """Compute durations between consecutive event types on one role.
    
    Example: interval_analysis(df, "MMA", ("EXPERT_START", "EXPERT_END"))
    """
    d = df[df["role"] == role].sort_values(["sm", "t_us"])
    starts = d[d["event"] == evt_pair[0]].reset_index(drop=True)
    ends   = d[d["event"] == evt_pair[1]].reset_index(drop=True)
    if len(starts) != len(ends):
        print(f"WARNING: unmatched {evt_pair[0]}/{evt_pair[1]} on {role}: "
              f"{len(starts)} vs {len(ends)}")
        n = min(len(starts), len(ends))
        starts, ends = starts.head(n), ends.head(n)
    dur_us = ends["t_us"].values - starts["t_us"].values
    print(f"\n{role}: {evt_pair[0]} -> {evt_pair[1]}")
    print(f"  n        = {len(dur_us)}")
    print(f"  mean     = {dur_us.mean():.2f} us")
    print(f"  median   = {np.median(dur_us):.2f} us")
    print(f"  p95      = {np.percentile(dur_us, 95):.2f} us")
    print(f"  max      = {dur_us.max():.2f} us")


def wait_analysis(df):
    """For each AFTER_* event, compute time since the previous event on the
    same SM/role. That's the wait duration."""
    print("\n=== Wait durations (gap between prev event and this AFTER_* event) ===")
    df = df.sort_values(["sm", "role", "t_us"]).reset_index(drop=True)
    df["prev_t_us"] = df.groupby(["sm", "role"])["t_us"].shift(1)
    df["wait_us"] = df["t_us"] - df["prev_t_us"]
    waits = df[df["event"].str.startswith("AFTER_")]
    g = waits.groupby(["role", "event"])["wait_us"].describe(percentiles=[0.5, 0.95])
    print(g.to_string())


def timeline_csv(df, out_path):
    df.to_csv(out_path, index=False)
    print(f"\nWrote timeline CSV: {out_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "trace_v2.bin"
    df = decode(path)
    if df.empty:
        print("No events recorded.")
        sys.exit(0)
    clock_hz = calibrate_clock_rate(df)
    df = to_global_ns(df, clock_hz)
    summary(df)
    wait_analysis(df)
    # Example: MMA expert duration
    try:
        interval_analysis(df, "MMA", ("EXPERT_START", "EXPERT_END"))
    except Exception as e:
        print(f"(interval_analysis skipped: {e})")
    timeline_csv(df, "trace_v2_timeline.csv")