#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

// ----------------------------------------------------------------------------
// `dispatch_impl_fast_path`: non-expand + intra-NVL72 fast-path dispatch
// kernel. Replaces the legacy two-kernel pipeline (dispatch_impl +
// dispatch_copy_epilogue) with one kernel that writes recv tensors directly
// into a `compact_recv_window` (Regions A..E, see CompactRecvBufferLayout).
//
// Differences vs dispatch_impl (legacy):
//   * All kNumThreads (= (kNumNotifyWarps + kNumDispatchWarps) * 32, typically
//     12 * 32 = 384) threads participate in BOTH phases (NOTIFY then
//     DISPATCH).  Warp-role partitioning is replaced with phase-based
//     partitioning.
//   * NOTIFY phase ends with a dst->src broadcast of psum_per_scaleup_rank
//     into Region G of the compact_recv_window, plus src-side wait until
//     every dst has populated its row.
//   * DISPATCH phase writes 5 separate TMA stores per token (hidden, sf,
//     topk_idx (local), topk_weights, src_metadata) to compact offsets, then
//     atomically increments a per-(src) arrival counter on the dst.
//   * The dispatch_copy_epilogue kernel is NOT launched in fast-path mode.
//
// Pre-conditions enforced at host-side dispatch() routing time:
//   * num_scaleout_ranks == 1 (intra-NVL72)
//   * do_expand == false
//   * cached_mode == false
//   * deterministic == false (no deterministic_rank_count_buffer dependency)
//
// Compatibility hooks: the kernel still writes the legacy outputs
// (`psum_num_recv_tokens_per_scaleup_rank`, `psum_num_recv_tokens_per_expert`,
//  `copied_topk_idx`) on the host-visible tensors handed in by the caller,
// so combine() and EPHandle bookkeeping remain unchanged.
// ----------------------------------------------------------------------------

namespace deep_ep::elastic {

template <bool kIsScaleupNVLink,
          bool kDoCPUSync,
          int kNumSMs,
          int kNumNotifyWarps, int kNumDispatchWarps,
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumThreads = (kNumNotifyWarps + kNumDispatchWarps) * 32,
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl_fast_path(
    // ----- inputs (same as legacy dispatch_impl) -----
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* buffer,
    void* workspace, void* mapped_host_workspace,
    const int rank_idx,
    // ----- fast-path additions -----
    void* compact_buffer,            // local LSA pointer to compact_recv_window base
    const ncclWindow_t compact_window
) {
    // -----------------------------------------------------------------------
    // Compile-time constants
    // -----------------------------------------------------------------------
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    constexpr int kNumNotifyThreads  = kNumNotifyWarps * 32;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");

    // -----------------------------------------------------------------------
    // Thread / warp identifiers
    // -----------------------------------------------------------------------
    const auto sm_idx     = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx   = ptx::get_warp_idx();
    const auto lane_idx   = ptx::get_lane_idx();

    // -----------------------------------------------------------------------
    // Workspace and Gin handle (same as legacy)
    // -----------------------------------------------------------------------
    const auto workspace_layout      = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);

    // Dynamic shared memory: NOTIFY counters region (first kNumSmemBytesForNotify)
    // followed by kNumDispatchWarps per-warp tma_buffer slots (same layout as
    // legacy dispatch_impl). Host launches with num_smem_bytes >= notify + warps*tma_buffer_bytes.
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];

    // Compact recv buffer layout helper (local copy; LSA-translated to dst at write time)
    constexpr int kSFBytesPerToken = kNumSFPacks * static_cast<int>(sizeof(sf_pack_t));
    const auto compact_layout = layout::CompactRecvBufferLayout(
        static_cast<int64_t>(kNumMaxTokensPerRank),
        static_cast<int64_t>(kNumRanks),
        static_cast<int64_t>(kNumHiddenBytes),
        static_cast<int64_t>(kSFBytesPerToken),
        static_cast<int64_t>(kNumTopk),
        compact_buffer);

    // Bring in QP / sharing mode (computed for the dispatch sub-warps; in fast
    // path we use the same QP layout as legacy so the underlying gin handle
    // selects sensible QPs even though the warp -> phase mapping is different).
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(
        sm_idx,
        /*dispatch_warp_idx_hint=*/(warp_idx >= kNumNotifyWarps ? warp_idx - kNumNotifyWarps : 0),
        warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);
    // Separate Gin handle for compact_recv_window: LSA pointer translation in
    // get_sym_ptr / put_value uses the window-specific lsa_base, so writes
    // targeted at compact_window must use a Gin built on compact_window.
    const auto gin_compact = handle::NCCLGin(nccl_dev_comm, compact_window, qp_idx, sharing_mode);

    // Cross-SM kickoff barrier (same as legacy).  No grid-sync prologue.
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles,
                      comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // -----------------------------------------------------------------------
    // PHASE 1: NOTIFY (only the first kNumNotifyWarps warps participate).
    // Identical to legacy NOTIFY, plus the dst->src G broadcast at the end.
    // -----------------------------------------------------------------------
    if (warp_idx < kNumNotifyWarps) {
        constexpr int kNotifyBarrierIndex = 1;
        constexpr int kNumAlignedElems    = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        int *rank_count   = rank_expert_count;
        int *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx_n = warp_idx * kNumSMs + sm_idx;
        for (int i = global_warp_idx_n; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            const auto dst_rank_idx_n = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx_n, lane_idx) and dst_rank_idx_n >= 0)
                atomicAdd_block(rank_count + dst_rank_idx_n, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Cross-SM reduction (encoded counters)
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        if (sm_idx == 0) {
            // Reduce all SM's count and prepare send buffer
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>(true, [=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        const auto encoded =
                            math::encode_decode_positive(static_cast<int>(status & 0xffffffffll));
                        rank_expert_count[i] = encoded;
                        if constexpr (not kIsScaleupNVLink)
                            workspace_layout.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }
                    if (is_last_check) {
                        printf("DeepEP[fast] notify reduction timeout rank=%d/%d thread=%d status=%d|%d\n",
                               rank_idx, kNumRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff));
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Cross-rank exchange: rank counts
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                const auto dst_rank_counter =
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;
                gin.put_value<team_t>(dst_rank_counter, static_cast<int64_t>(rank_count[i]), i,
                                      ncclGinOptFlagsAggregateRequests);
            }
            __syncwarp();

            // Cross-rank exchange: expert counts (intra-NVL72 path: per-element NVLink store)
            if constexpr (kIsScaleupNVLink) {
                for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads) {
                    const auto idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank);
                    gin.put_value<team_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), i / kNumExpertsPerRank);
                }
            } else {
                // Bulk copy via gin.put for RDMA path; fast-path is intra-NVL72 only,
                // so this branch is dead but kept for parity.
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                    const auto src_ptr = workspace_layout.get_scaleup_expert_count_ptr<true>() + kNumExpertsPerRank * i;
                    const auto dst_ptr = workspace_layout.get_scaleup_expert_count_ptr<false>() + kNumExpertsPerRank * rank_idx;
                    gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);
                }
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Wait for rank and expert counts from peers
            const auto start_clock = clock64();
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto count = static_cast<int>(
                        ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + i));
                    const auto decoded = math::encode_decode_positive(count);
                    if (math::is_decoded_positive_ready(decoded)) {
                        workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] = 0;
                        rank_expert_count[i] = decoded;
                        return true;
                    }
                    if (is_last_check)
                        printf("DeepEP[fast] notify peer-wait timeout rank=%d thread=%d count=%d\n",
                               rank_idx, i, decoded);
                    return false;
                }, start_clock);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Aggregate per-expert sums and update stats counter
            for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {
                int sum = 0;
                #pragma unroll
                for (int j = 0; j < kNumRanks; ++ j)
                    sum += expert_count[j * kNumExpertsPerRank + i];
                expert_count[i] = math::align(sum, kExpertAlignment);
                if (cumulative_local_expert_recv_stats != nullptr)
                    atomicAdd(cumulative_local_expert_recv_stats + i, sum);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Write host workspace if CPU sync is requested (lets host read N_recv)
            if constexpr (kDoCPUSync) {
                for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads) {
                    host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] =
                        math::encode_decode_positive(rank_expert_count[i]);
                }
                __syncwarp();
            }

            // Prefix sums (legacy outputs, fast path also produces these)
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);
                    if (idx < n + is_exclusive)
                        out[idx] = sum;
                    psum = ptx::exchange(sum, 31);
                }
            };
            if (warp_idx == 0) {
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);
            } else if (warp_idx == 1) {
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // ===== FAST-PATH-ONLY: dst -> src broadcast of psum_per_scaleup_rank =====
            // After psum is ready on this rank, push our entire `psum_per_scaleup_rank`
            // array to every peer's compact_window Region G row indexed by [rank_idx].
            // peer Region G layout: peer_psum_buf[dst_rank=peer][src_rank=who_wrote_it].
            // We write OUR psum (which is "this rank's view of how many tokens from each
            // src rank are arriving") into peer's row indexed by rank_idx.
            // After this, every peer has, for its own dst Region G row [rank_idx], the
            // psum that WE computed - i.e. peer-as-src learns "my (peer's) compact base
            // offset on dst=rank_idx".
            //
            // Equivalent recv-side: peer_psum_row_ptr(dst_rank=rank_idx) of this rank's
            // local Region G is written by each remote peer, providing this rank with
            // its compact base offset on each dst.
            //
            // Encoding: store (psum_val + 1) so 0 means "not arrived yet";
            // the src side spin-waits for non-zero then subtracts 1.
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                int* dst_psum_slot = compact_layout.peer_psum_ptr(/*dst_rank_idx=*/rank_idx,
                                                                  /*src_rank_idx=*/i);
                const int psum_val_exclusive = (i == 0) ? 0
                    : psum_num_recv_tokens_per_scaleup_rank[i - 1];
                gin_compact.put_value<team_t>(dst_psum_slot, psum_val_exclusive + 1, i,
                                              ncclGinOptFlagsAggregateRequests);
            }
            __threadfence_system();
            __syncwarp();
        }
    }

    // -----------------------------------------------------------------------
    // PHASE 1.5: spin-wait for peer_psum_buf to be populated by every dst.
    // The spin-wait reads "encoded + 1" values; non-zero means the remote
    // dst already wrote (psum_val + 1) into this slot.
    // -----------------------------------------------------------------------
    __syncthreads();

    // Load my row of peer_psum into shared memory once per block.
    __shared__ int s_my_compact_base[kNumRanks];
    if (thread_idx < kNumRanks) {
        int* slot = compact_layout.peer_psum_ptr(/*dst_rank_idx=*/thread_idx,
                                                  /*src_rank_idx=*/rank_idx);
        int v = 0;
        const auto t0 = clock64();
        while (v == 0) {
            v = ptx::ld_volatile<int>(slot);
            if (clock64() - t0 > kNumTimeoutCycles) {
                printf("DeepEP[fast] wait_psum TIMEOUT rank=%d sm=%d dst=%d\n",
                       rank_idx, sm_idx, thread_idx);
                ptx::trap();
            }
        }
        s_my_compact_base[thread_idx] = v - 1;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // PHASE 2: dispatch warps do staged TMA stores (same pipeline as legacy
    // dispatch_impl, but dst is compact_buffer regions A/B/C/D/E indexed by
    // compact_idx, not legacy per-rank packed slot).
    //
    // Per-warp tma_buffer (in dynamic smem, after the NOTIFY region) stages
    // hidden + sf + topk_idx + topk_weights + metadata. Then for each unique
    // dst_rank in the token's topk:
    //   - TMA store smem hidden -> peer compact A[compact_idx]
    //   - TMA store smem sf -> peer compact B[compact_idx]
    //   - master lane writes full topk_idx / topk_weights / src_metadata rows
    //     directly to peer regions (small, no smem staging needed)
    //   - master lane red_add_rel_sys to peer arrival_counter[my_rank]
    // -----------------------------------------------------------------------
    if (warp_idx >= kNumNotifyWarps) {
        const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;

        // Per-warp tma_buffer in dynamic smem after the notify region.
        const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kSFBytesPerToken, kNumTopk, true);
        const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumDispatchWarps, 1,
            math::advance_ptr<int8_t>(smem, kNumSmemBytesForNotify))
            .get_rank_buffer(dispatch_warp_idx).get_token_buffer(0);

        // Init mbarrier (legacy pattern)
        ptx::arrival_phase phase = 0;
        const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
        if (ptx::elect_one_sync())
            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        __syncwarp();

        const auto token_start  = dispatch_warp_idx * kNumSMs + sm_idx;
        const auto token_stride = kNumDispatchWarps * kNumSMs;

        for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
            const auto token_i64_idx = static_cast<int64_t>(token_idx);

            // Wait any previous TMA stores to drain
            ptx::tma_store_wait();
            __syncwarp();

            // Issue TMA load: x[token] -> smem hidden region
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(),
                                 math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue cp.async loads: sf[token] -> smem sf region
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(
                    sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();
                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }

            // Load topk and compute dst_rank per-lane
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for top-k");
            int my_expert_raw = -1;
            int my_dst_rank   = -1;
            if (lane_idx < kNumTopk) {
                my_expert_raw = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
                my_dst_rank   = my_expert_raw >= 0 ? my_expert_raw / kNumExpertsPerRank : -1;
            }
            const bool is_master_for_dst = ptx::deduplicate(my_dst_rank, lane_idx) and (my_dst_rank >= 0);
            const int  master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(is_master_for_dst));

            // Allocate src-local seq number for master lane of each unique dst
            int my_local_seq_to_dst = -1;
            if (is_master_for_dst) {
                my_local_seq_to_dst = atomicAdd(
                    workspace_layout.get_scaleup_atomic_sender_counter() + my_dst_rank, 1);
            }

            // Record dst_buffer_slot_idx (so combine() can reverse-route)
            if (lane_idx < kNumTopk) {
                const auto val = my_local_seq_to_dst >= 0
                    ? (rank_idx * kNumMaxTokensPerRank + my_local_seq_to_dst) : -1;
                dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = val;
            }

            // copied_topk_idx for EPHandle bookkeeping
            if (copied_topk_idx != nullptr and lane_idx < kNumTopk) {
                copied_topk_idx[token_idx * kNumTopk + lane_idx] =
                    static_cast<topk_idx_t>(my_expert_raw);
            }
            __syncwarp();

            // Wait TMA load completion (hidden + sf both arrived)
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();

            // For each unique dst_rank: TMA store smem -> peer compact buffer
            if (is_master_for_dst) {
                const int compact_idx = s_my_compact_base[my_dst_rank] + my_local_seq_to_dst;
                EP_DEVICE_ASSERT(compact_idx >= 0 and compact_idx < kNumMaxTokensPerRank * kNumRanks);

                // 1. Region A: hidden (TMA store smem -> peer)
                {
                    auto* local_dst = compact_layout.hidden_ptr(static_cast<int64_t>(compact_idx));
                    auto* sym_dst   = gin_compact.template get_sym_ptr<team_t>(local_dst, my_dst_rank);
                    if (sym_dst != nullptr)
                        ptx::tma_store_1d(sym_dst, tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
                }

                // 2. Region B: sf (TMA store smem -> peer, only if sf present)
                if constexpr (kNumSFPacks > 0) {
                    auto* local_dst = compact_layout.sf_ptr(static_cast<int64_t>(compact_idx));
                    auto* sym_dst   = gin_compact.template get_sym_ptr<team_t>(local_dst, my_dst_rank);
                    if (sym_dst != nullptr)
                        ptx::tma_store_1d(reinterpret_cast<void*>(sym_dst),
                                          tma_buffer.get_sf_ptr(), kSFBytesPerToken);
                }

                // 3. Region C: topk_idx — master writes the full kNumTopk row
                {
                    auto* dst_topk_local = static_cast<topk_idx_t*>(
                        compact_layout.topk_idx_ptr(static_cast<int64_t>(compact_idx)));
                    topk_idx_t* sym_dst = gin_compact.template get_sym_ptr<team_t, topk_idx_t>(
                        dst_topk_local, my_dst_rank);
                    if (sym_dst != nullptr) {
                        #pragma unroll
                        for (int k = 0; k < kNumTopk; ++ k) {
                            const int raw = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + k));
                            const bool in_dst = (raw >= 0) and (raw / kNumExpertsPerRank == my_dst_rank);
                            sym_dst[k] = in_dst
                                ? static_cast<topk_idx_t>(raw - my_dst_rank * kNumExpertsPerRank)
                                : static_cast<topk_idx_t>(-1);
                        }
                    }
                }

                // 4. Region D: topk_weights — master writes the full kNumTopk row
                if (topk_weights != nullptr) {
                    float* sym_dst = gin_compact.template get_sym_ptr<team_t>(
                        compact_layout.topk_weights_ptr(static_cast<int64_t>(compact_idx)),
                        my_dst_rank);
                    if (sym_dst != nullptr) {
                        #pragma unroll
                        for (int k = 0; k < kNumTopk; ++ k)
                            sym_dst[k] = __ldg(topk_weights + token_idx * kNumTopk + k);
                    }
                }

                // 5. Region E: src_metadata — master writes the full (2 + kNumTopk) row
                {
                    int* sym_dst = gin_compact.template get_sym_ptr<team_t>(
                        compact_layout.src_metadata_ptr(static_cast<int64_t>(compact_idx)),
                        my_dst_rank);
                    if (sym_dst != nullptr) {
                        sym_dst[0] = rank_idx * kNumMaxTokensPerRank + token_idx;
                        sym_dst[1] = rank_idx * kNumTopk + master_src_topk_idx;
                        #pragma unroll
                        for (int k = 0; k < kNumTopk; ++ k) {
                            const int raw = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + k));
                            const bool in_dst = (raw >= 0) and (raw / kNumExpertsPerRank == my_dst_rank);
                            sym_dst[2 + k] = in_dst ? (raw - my_dst_rank * kNumExpertsPerRank) : -1;
                        }
                    }
                }

                ptx::tma_store_commit();
            }
            __syncwarp();

            // Arrival counter on dst (one lane per unique dst)
            if (is_master_for_dst) {
                int* sym_counter = gin_compact.template get_sym_ptr<team_t>(
                    compact_layout.arrival_counter_ptr(rank_idx), my_dst_rank);
                if (sym_counter != nullptr)
                    ptx::red_add_rel_sys(sym_counter, 1);
            }
            __syncwarp();
        }

        // Drain remaining TMA stores before final barrier
        ptx::tma_store_wait();
        __syncwarp();
    }

    // -----------------------------------------------------------------------
    // PHASE 3: dst-side wait until arrival counters >= rank_count
    // -----------------------------------------------------------------------
    // After Phase 2 every src has finished writing; each dst now needs to wait
    // until its Region F arrival counters cumulatively account for all tokens
    // each src promised in the NOTIFY phase.
    if (sm_idx == 0 and warp_idx == 0) {
        // rank_count[i] was placed into the lower 32-bits of
        // workspace_layout.get_scaleup_rank_count_ptr<false>()[i] during NOTIFY.
        // We read it back and wait until our compact arrival counter has reached
        // that value for every src.
        for (int src = thread_idx; src < kNumRanks; src += 32) {
            const auto expected = static_cast<int>(ptx::ld_volatile<int64_t>(
                workspace_layout.get_scaleup_rank_count_ptr<false>() + src));
            int* counter_ptr = compact_layout.arrival_counter_ptr(src);
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto v = ptx::ld_volatile<int>(counter_ptr);
                if (v >= expected) return true;
                if (is_last_check) {
                    printf("DeepEP[fast] arrival wait timeout rank=%d src=%d got=%d expected=%d\n",
                           rank_idx, src, v, expected);
                }
                return false;
            });
            // Reset for next dispatch call.
            ptx::st_relaxed_sys(counter_ptr, 0);
        }
    }

    // Final cross-SM barrier to ensure all dst arrivals visible before kernel return.
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles,
                      comm::kDispatchTag1, true, true, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Reset workspace atomic sender counters (one per dst rank) so the next
    // dispatch() call starts from 0.  Same as legacy dispatch_impl end.
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (sm_idx == 0 and thread_idx < kNumRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;

    // Reset peer_psum_buf slots that THIS rank will spin-wait on next dispatch.
    // As src=my_rank, we read [d][my_rank] for d in [0, kNumRanks); clear them
    // here so the +1 sentinel is back to 0 for the next call.
    if (sm_idx == 0 and thread_idx < kNumRanks) {
        *compact_layout.peer_psum_ptr(/*dst=*/thread_idx, /*src=*/rank_idx) = 0;
    }
}

}  // namespace deep_ep::elastic
