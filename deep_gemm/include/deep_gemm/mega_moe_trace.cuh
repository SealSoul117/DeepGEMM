#pragma once
//
// Fine-grained timing for MegaMoE pipeline.
// All warps in all SMs append events into a single global ringbuffer.
// Host side replays them after kernel completion.
//
// Enable by defining DG_MEGA_MOE_TIMING before including any MegaMoE header.
//

#include <cstdint>
#include <cstdio>

namespace deep_gemm::trace {

// Per-event fixed-width record. Keep this small (32 B) so the buffer is cheap.
struct Event {
    uint64_t t_start;       // %globaltimer ns
    uint64_t t_end;         // %globaltimer ns (0 if it's a point event)
    uint32_t sm_id;         // blockIdx.x
    uint16_t warp_role;     // one of WarpRole below
    uint16_t event_id;      // one of EventId below
    uint32_t wave_idx;      // current wave index (or 0xFFFFFFFF if N/A)
    uint32_t aux;           // role-specific (e.g. expert_idx, stage_idx, k_block_idx)
};
static_assert(sizeof(Event) == 32, "Event must be 32 bytes");

// Warp roles. Use the same ids the kernel already implies.
enum WarpRole : uint16_t {
    WR_DISPATCH    = 0,
    WR_TMA_LOAD_A  = 1,
    WR_TMA_LOAD_B  = 2,
    WR_MMA         = 3,
    WR_EPILOGUE    = 4,
    WR_COMBINE     = 5,
};

// Event ids. Add more as needed; just keep total < 65k.
enum EventId : uint16_t {
    // Lifecycle
    EV_KERNEL_BEGIN          = 0,
    EV_KERNEL_END            = 1,

    // Wave & phase
    EV_WAVE_L1_PHASE         = 10,   // interval: one wave's L1 phase on this warp
    EV_WAVE_L2_PHASE         = 11,
    EV_EXPERT_L1             = 12,   // interval: one expert's L1 blocks on this warp
    EV_EXPERT_L2             = 13,

    // Sync primitives (these are the ones to watch carefully)
    EV_L1_ARRIVAL_WAIT       = 20,   // spin on workspace.get_l1_arrival_count_ptr
    EV_L2_ARRIVAL_WAIT       = 21,   // spin on workspace.get_l2_arrival_mask_ptr
    EV_EMPTY_BARRIER_WAIT    = 22,   // pipeline empty barrier
    EV_FULL_BARRIER_WAIT     = 23,   // pipeline full barrier
    EV_TMEM_FULL_WAIT        = 24,   // tmem accumulator full
    EV_TMEM_EMPTY_WAIT       = 25,   // tmem accumulator empty
    EV_DISPATCH_EPILOGUE_BAR = 26,   // dispatch↔epilogue cross-section barrier

    // Per-block TMA / MMA
    EV_TMA_LOAD_A_BLOCK      = 30,   // one (m_block, k_block) TMA copy for A+SFA
    EV_TMA_LOAD_B_BLOCK      = 31,
    EV_UMMA_ISSUE            = 32,   // one UMMA instruction sequence

    // Dispatch side
    EV_DISPATCH_TOKEN        = 40,   // one token pulled from a remote rank
    EV_DISPATCH_PHASE        = 41,   // entire dispatch phase on this warp

    // Combine side (L2 epilogue → NVLink writeback)
    EV_COMBINE_TOKEN         = 50,
    EV_COMBINE_PHASE         = 51,
};

constexpr uint32_t kMaxEvents = 1u << 20;   // 1M events ≈ 32 MiB ringbuffer

struct EventBuffer {
    Event*    events;
    uint32_t* counter;     // atomic append index
    uint32_t  capacity;
};

#ifdef __CUDA_ARCH__

// Read the global timer.
__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Append a single event. Returns false if buffer is full (silently drops).
// Only one thread per warp should call this (use cute::elect_one_sync or lane 0).
__device__ __forceinline__ bool record(
    const EventBuffer& buf,
    uint16_t warp_role, uint16_t event_id,
    uint64_t t_start, uint64_t t_end,
    uint32_t wave_idx, uint32_t aux)
{
    uint32_t idx = atomicAdd(buf.counter, 1u);
    if (idx >= buf.capacity) return false;
    Event& e = buf.events[idx];
    e.t_start   = t_start;
    e.t_end     = t_end;
    e.sm_id     = blockIdx.x;
    e.warp_role = warp_role;
    e.event_id  = event_id;
    e.wave_idx  = wave_idx;
    e.aux       = aux;
    return true;
}

#endif // __CUDA_ARCH__

} // namespace deep_gemm::trace

// === Scoped interval helper ====================================================
// Usage:
//   TRACE_INTERVAL(trace_buf, WR_TMA_LOAD_A, EV_TMA_LOAD_A_BLOCK, wave_idx, expert_idx) {
//       // code to be timed
//   }
// Expands to: take t0; run block; if elected lane, take t1 and record.
//
// Important: this only records from ONE thread per warp (lane 0 after elect_one).
// Don't put it inside divergent control flow within a warp.

#ifdef __CUDA_ARCH__
  #define TRACE_BEGIN(_t0)                                                     \
      uint64_t _t0 = ::deep_gemm::trace::now_ns()
  #define TRACE_END(_buf, _role, _ev, _t0, _wave, _aux)                        \
      do {                                                                     \
          if (cute::elect_one_sync()) {                                        \
              ::deep_gemm::trace::record(                                      \
                  (_buf), (_role), (_ev), (_t0),                               \
                  ::deep_gemm::trace::now_ns(), (_wave), (_aux));              \
          }                                                                    \
      } while (0)
#else
  #define TRACE_BEGIN(_t0)                              ((void)0)
  #define TRACE_END(_buf, _role, _ev, _t0, _w, _aux)    ((void)0)
#endif

#define TRACE_BEGIN(_t0)                            ((void)0)
#define TRACE_END(_buf, _role, _ev, _t0, _w, _aux)  ((void)0)
