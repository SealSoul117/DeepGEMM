#pragma once
//
// Lightweight per-SM trace ringbuffer for MegaMoE.
//
// Design goals:
//   1. ZERO atomicAdd on the hot path (per-warp counter lives in a register)
//   2. ZERO globaltimer reads on the hot path (only at kernel entry, twice per SM)
//   3. Compact 8-byte events: just (event_id, clock_cycle_lo32)
//   4. Per-SM region with statically sized capacity — writes from different SMs
//      never touch the same cache line, so L2 pollution is bounded.
//
// Cost per record() call: ~8-12 ns (one clock() + one 8-byte store).
// At ~30 events/wave * 148 SMs = ~4400 events/launch, total HBM traffic ~35 KB.
//

#include <cstdint>

namespace deep_gemm::trace {

// 8 bytes per event. Two fields packed.
//   high 32 bits: clock_lo (cycle counter, low 32 bits — wraps ~every 2 sec at 2 GHz)
//   low  16 bits: event_id
//   bits 16..31: aux16  (e.g. wave_idx << 8 | expert_idx, both small)
//
// We deliberately drop sm_id, warp_role, and 64-bit timestamps. Those are
// reconstructed on the host from (a) the base header below and (b) the
// region in the buffer the event lives in.
struct Event {
    uint16_t event_id;
    uint16_t aux16;
    uint32_t clock_lo;
};
static_assert(sizeof(Event) == 8, "Event must be 8 bytes");

// Per-SM header — written ONCE at kernel entry by warp 0 of each SM.
// Used by the host to convert clock cycles to nanoseconds and align across SMs.
struct SmHeader {
    uint64_t t_ns_at_start;     // globaltimer reading at kernel entry on THIS sm
    uint32_t clock_at_start;    // clock() reading at kernel entry on THIS sm
    uint32_t n_events_written;  // total events written by all warps in this SM
                                // (sum of per-warp counters, filled at kernel exit)
    uint64_t t_ns_at_end;       // ← 新增
    uint32_t clock_at_end;
};
static_assert(sizeof(SmHeader) == 16, "SmHeader must be 16 bytes");

// Warp roles. Each role gets its own sub-region in the SM's slice.
enum WarpRole : uint16_t {
    WR_DISPATCH    = 0,
    WR_TMA_LOAD_A  = 1,
    WR_TMA_LOAD_B  = 2,
    WR_MMA         = 3,
    WR_EPILOGUE    = 4,
    WR_COMBINE     = 5,
    WR_COUNT       = 6,
};

enum EventId : uint16_t {
    // Lifecycle (point events)
    EV_KERNEL_BEGIN     = 0,
    EV_KERNEL_END       = 1,

    // Wave & phase boundaries (point events — emit at transition time)
    EV_WAVE_L1_START    = 10,
    EV_WAVE_L1_END      = 11,
    EV_WAVE_L2_START    = 12,
    EV_WAVE_L2_END      = 13,
    EV_EXPERT_START     = 14,   // per-expert granularity, aux16 = expert_idx
    EV_EXPERT_END       = 15,

    // Sync waits — emit one event AFTER the wait completes; the host computes
    // wait duration as (this event's cycle) − (previous event's cycle).
    // No need to bracket with two events.
    EV_AFTER_L1_ARRIVAL = 20,
    EV_AFTER_L2_ARRIVAL = 21,
    EV_AFTER_EMPTY_BAR  = 22,
    EV_AFTER_FULL_BAR   = 23,
    EV_AFTER_TMEM_FULL  = 24,
    EV_AFTER_TMEM_EMPTY = 25,

    // Per-block markers (use sparingly — these can flood the buffer)
    EV_TMA_A_DONE       = 30,
    EV_UMMA_ISSUED      = 32,

    EV_DISPATCH_START   = 33,
    EV_DISPATCH_END     = 34,

    EV_COMBINE_START    = 35,
    EV_COMBINE_END      = 36,
};

// Capacity per (SM, warp_role). Adjust based on expected events.
// At 256 events/role * 6 roles * 8 bytes = 12 KB per SM.
// On 148 SMs: ~1.8 MB total. Sized to stay in L2 footprint.
constexpr uint32_t kEventsPerRole = 256;
constexpr uint32_t kBytesPerRole  = kEventsPerRole * sizeof(Event);

// Buffer layout, per SM:
//   [SmHeader: 16 B] [pad to 32]
//   [WR_DISPATCH region:    kEventsPerRole * 8 B]
//   [WR_TMA_LOAD_A region:  kEventsPerRole * 8 B]
//   [WR_TMA_LOAD_B region:  ...]
//   [WR_MMA region:         ...]
//   [WR_EPILOGUE region:    ...]
//   [WR_COMBINE region:     ...]
constexpr uint32_t kSmHeaderBytes = 32;  // 16 B header + 16 B pad
constexpr uint32_t kBytesPerSm = kSmHeaderBytes + WR_COUNT * kBytesPerRole;

// Top-level buffer (allocated by host, one big global slab).
struct TraceBuf {
    uint8_t* data;        // size = num_sms * kBytesPerSm
    uint32_t num_sms;
};

#ifdef __CUDA_ARCH__

__device__ __forceinline__ uint32_t clock_lo() {
    uint32_t c;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
    return c;
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Get pointer to this SM's header. Call from any thread.
__device__ __forceinline__ SmHeader* sm_header(const TraceBuf& buf) {
    return reinterpret_cast<SmHeader*>(buf.data + blockIdx.x * kBytesPerSm);
}

// Get base pointer to this SM's region for `role`.
__device__ __forceinline__ Event* role_region(const TraceBuf& buf, uint32_t role) {
    return reinterpret_cast<Event*>(
        buf.data + blockIdx.x * kBytesPerSm + kSmHeaderBytes + role * kBytesPerRole);
}

__device__ __forceinline__ void trace_sm_finalize(const TraceBuf& buf) {
    if (buf.data == nullptr) return;
    if (threadIdx.x == 0) {
        SmHeader* h = sm_header(buf);
        h->t_ns_at_end  = globaltimer_ns();
        h->clock_at_end = clock_lo();
    }
}

// === The hot-path API ====================================================
//
// `local_idx` is a uint32_t kept in REGISTER, one per (warp-role, thread).
// Only ONE thread per warp (elected lane) actually writes. Other lanes still
// run clock() but discard the result — it's cheap and avoids divergence.
//
// Usage:
//
//   uint32_t mma_evt_idx = 0;   // declared once at start of warp role
//   ...
//   trace_event(trace_buf, WR_MMA, mma_evt_idx, EV_AFTER_TMEM_FULL, wave_idx);
//
__device__ __forceinline__ void trace_event(
    const TraceBuf& buf,
    uint32_t role,
    uint32_t& local_idx,        // register counter, incremented in-place
    uint16_t event_id,
    uint16_t aux16)
{
    if (buf.data == nullptr)
        return;
    // Read clock unconditionally (it's a single SASS instruction, ~1 cycle).
    // Branching on elect_one_sync first would cause divergent stall instead.
    uint32_t c = clock_lo();
    if (__builtin_expect(local_idx >= kEventsPerRole, 0)) return;
    // Only the elected lane writes. Other lanes skip the store.
    // Using a per-warp ballot is cheaper than `if (lane == 0)`.
    bool is_writer = (threadIdx.x & 31u) == 0;   // lane 0 of the warp
    if (is_writer) {
        Event* region = role_region(buf, role);
        region[local_idx] = Event{event_id, aux16, c};
    }
    local_idx++;  // ALL lanes increment (it's a register; cheap; keeps them in sync)
}

// Variant for bracketing: write the start cycle into a register, record
// "after" event later. Use when you genuinely need the start (rare; usually
// the previous event's cycle is enough).
__device__ __forceinline__ uint32_t trace_clock_now() {
    return clock_lo();
}

// Called by warp 0 of each SM at kernel entry. Records the (globaltimer, clock)
// pair so the host can do cycle→ns conversion per SM.
__device__ __forceinline__ void trace_sm_init(const TraceBuf& buf) {
    if (buf.data == nullptr)
        return;
    if (threadIdx.x == 0) {
        SmHeader* h = sm_header(buf);
        h->t_ns_at_start  = globaltimer_ns();
        h->clock_at_start = clock_lo();
        h->n_events_written = 0;  // filled later if you care
    }
}

#endif  // __CUDA_ARCH__

}  // namespace deep_gemm::trace
