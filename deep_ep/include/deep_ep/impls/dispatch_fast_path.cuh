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

    // Static smem (avoids the cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)
    // path entirely; that call was reporting CUDA_ERROR_INVALID_VALUE on sm_103
    // for this kernel for reasons not yet understood).  Sized via template
    // constants so each instantiation gets exactly the smem it needs.
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");
    __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[kNumSmemBytesForNotify > 0 ? kNumSmemBytesForNotify : 16];

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

    // Cross-SM kickoff barrier (same as legacy).  No grid-sync prologue.
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles,
                      comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // -----------------------------------------------------------------------
    // PHASE 1: NOTIFY (only the first kNumNotifyWarps warps participate).
    // This is intentionally identical to the legacy NOTIFY phase, since
    // letting all 12 warps participate gives <1us of headroom and complicates
    // smem layout.  The "12+12" design is fully captured by Phase 2 where all
    // warps stream the heavy hidden traffic.
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
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                int* dst_psum_slot = compact_layout.peer_psum_ptr(/*dst_rank_idx=*/rank_idx,
                                                                  /*src_rank_idx=*/i);
                // dtype_t deduced from sym_ptr (int*); value must be int too,
                // not int64_t, otherwise template deduction conflicts.
                const int psum_val = psum_num_recv_tokens_per_scaleup_rank[i];
                gin.put_value<team_t>(dst_psum_slot, psum_val, i,
                                      ncclGinOptFlagsAggregateRequests);
            }
            __syncwarp();
        }
    }

    // -----------------------------------------------------------------------
    // PHASE 1.5: src wait for peer_psum_buf to be populated by every dst.
    // After this barrier, every thread can safely read
    //   compact_layout.peer_psum_ptr(dst_rank, my_rank=rank_idx)
    // to get "my compact base offset on dst_rank".
    // -----------------------------------------------------------------------
    // Single global barrier across the whole grid (uses workspace tag 1).
    // ALL warps wait here.
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles,
                      comm::kDispatchTag0, true, true, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // All threads on all SMs now drop into PHASE 2 below.

    // -----------------------------------------------------------------------
    // PHASE 2: ALL warps do compact DISPATCH.
    // Each (warp, sm) pair iterates a stride of (kNumThreads/32) * kNumSMs
    // through `num_tokens`.  For each token's `kNumTopk` expert hits, find the
    // dst rank, look up dst's compact base offset (peer_psum row), add the
    // src-local sequence number stored in `dst_buffer_slot_idx`, then emit 5
    // TMA stores (hidden + sf + topk_idx + topk_weights + src_metadata).
    // After every token, atomically bump the dst's Region F arrival counter
    // for my src index.
    // -----------------------------------------------------------------------
    {
        // Per-dst compact base offset cache: 32 entries fit in registers.
        EP_STATIC_ASSERT(kNumRanks <= 64, "Compact base cache assumes kNumRanks <= 64");

        // Load my row of peer_psum into shared memory once for the whole block.
        // compact_layout.peer_psum_ptr(dst_rank, src_rank=rank_idx) on local rank gives
        // "in dst_rank's view, # tokens before me" which equals MY compact base on dst_rank.
        __shared__ int s_my_compact_base[kNumRanks];
        if (thread_idx < kNumRanks) {
            s_my_compact_base[thread_idx] = ptx::ld_volatile<int>(
                compact_layout.peer_psum_ptr(/*dst_rank_idx=*/thread_idx,
                                              /*src_rank_idx=*/rank_idx));
        }
        __syncthreads();

        // Token stride: each warp picks tokens at (warp_idx * kNumSMs + sm_idx + k * total_warps * kNumSMs).
        const auto total_warps = kNumThreads / 32;
        const auto warp_global = warp_idx * kNumSMs + sm_idx;
        const auto stride      = total_warps * kNumSMs;

        for (int token_idx = warp_global; token_idx < num_tokens; token_idx += stride) {
            // Each lane (lane_idx in [0, 32)) handles topk slot lane_idx (only first kNumTopk are valid).
            const int  topk_lane         = lane_idx < kNumTopk ? lane_idx : -1;
            const int  raw_expert        = topk_lane >= 0 ?
                static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + topk_lane)) : -1;
            const bool in_range          = raw_expert >= 0;
            const int  dst_rank          = in_range ? raw_expert / kNumExpertsPerRank : -1;
            const int  dst_expert_local  = in_range ? raw_expert - dst_rank * kNumExpertsPerRank : -1;

            // Deduplicate (token, dst_rank) so we only emit one TMA store per (token, unique dst).
            const bool is_master_for_dst = ptx::deduplicate(dst_rank, lane_idx) and in_range;

            // master_src_topk_idx = the lowest topk_lane among lanes that share this token's dst_rank,
            // used in src_metadata to point back at the originating topk slot.
            const int master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(is_master_for_dst));

            // Pull src-local sequence number from dst_buffer_slot_idx (assigned in prologue).
            // Note: dst_buffer_slot_idx[token_idx, topk_lane] is the slot number THIS src has
            // assigned for this (token, topk) under the LEGACY rank-slot layout, but its value
            // is exactly "this is my k-th token to dst_rank" which matches what we need for the
            // compact offset.
            const int my_local_seq_to_dst = (topk_lane >= 0 and in_range) ?
                __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + topk_lane) : -1;

            if (is_master_for_dst) {
                // Compact tensor index on dst rank.
                const int compact_idx = s_my_compact_base[dst_rank] + my_local_seq_to_dst;
                EP_DEVICE_ASSERT(compact_idx >= 0 and compact_idx < kNumMaxTokensPerRank * kNumRanks);

                // ----- 1. hidden (Region A) -----
                {
                    auto* local_dst = compact_layout.hidden_ptr(static_cast<int64_t>(compact_idx));
                    auto* sym_dst   = gin.template get_sym_ptr<team_t>(local_dst, dst_rank);
                    if (sym_dst != nullptr) {
                        const auto src_ptr = math::advance_ptr(x, static_cast<int64_t>(token_idx) * kNumHiddenBytes);
                        ptx::tma_store_1d(sym_dst, src_ptr, kNumHiddenBytes);
                    }
                }

                // ----- 2. sf (Region B), only when sf is present -----
                if constexpr (kNumSFPacks > 0) {
                    auto* local_dst = compact_layout.sf_ptr(static_cast<int64_t>(compact_idx));
                    auto* sym_dst   = gin.template get_sym_ptr<team_t>(local_dst, dst_rank);
                    if (sym_dst != nullptr) {
                        const auto src_ptr =
                            math::advance_ptr<sf_pack_t>(sf, static_cast<int64_t>(token_idx) * sf_token_stride);
                        // sf is interleaved by `sf_hidden_stride` between packs in legacy layout;
                        // for fast path we write the contiguous packed form.
                        constexpr int kSFBytes = kNumSFPacks * sizeof(sf_pack_t);
                        // Naive per-pack copy (small data, <<1 KB per token).
                        for (int k = 0; k < kNumSFPacks; ++ k)
                            sym_dst[k] = src_ptr[k * sf_hidden_stride];
                        (void)kSFBytes;
                    }
                }

                // ----- 3. topk_idx (Region C; topk_idx_t = int64_t by default) -----
                {
                    auto* dst_topk_local = static_cast<topk_idx_t*>(
                        compact_layout.topk_idx_ptr(static_cast<int64_t>(compact_idx)));
                    topk_idx_t* sym_dst = gin.template get_sym_ptr<team_t, topk_idx_t>(dst_topk_local, dst_rank);
                    if (sym_dst != nullptr and lane_idx < kNumTopk) {
                        const auto my_expert_raw = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                        const bool my_in_dst = (my_expert_raw >= 0)
                            and (my_expert_raw / kNumExpertsPerRank == dst_rank);
                        sym_dst[lane_idx] = my_in_dst
                            ? static_cast<topk_idx_t>(my_expert_raw - dst_rank * kNumExpertsPerRank)
                            : static_cast<topk_idx_t>(-1);
                    }
                }

                // ----- 4. topk_weights (Region D) -----
                if (topk_weights != nullptr) {
                    float* sym_dst = gin.template get_sym_ptr<team_t>(
                        compact_layout.topk_weights_ptr(static_cast<int64_t>(compact_idx)),
                        dst_rank);
                    if (sym_dst != nullptr and lane_idx < kNumTopk) {
                        sym_dst[lane_idx] = topk_weights[token_idx * kNumTopk + lane_idx];
                    }
                }

                // ----- 5. src_metadata (Region E): [src_rank_idx, src_token_rank_idx*kNumTopk+master_topk, expert_id_0..expert_id_{kNumTopk-1}] -----
                {
                    int* sym_dst = gin.template get_sym_ptr<team_t>(
                        compact_layout.src_metadata_ptr(static_cast<int64_t>(compact_idx)),
                        dst_rank);
                    if (sym_dst != nullptr) {
                        if (lane_idx == 0) {
                            sym_dst[0] = rank_idx * kNumMaxTokensPerRank + token_idx;
                            sym_dst[1] = rank_idx * kNumTopk + master_src_topk_idx;
                        }
                        if (lane_idx < kNumTopk) {
                            // Mirror legacy recv_src_metadata[i*(num_topk+2)+2+k] = dst_tensor_idx_in_local_expert
                            // For non-expand, this is -1 unless lane's expert is local to dst_rank, in which case
                            // it's the local expert index.  Legacy writes the local expert idx unconditionally
                            // (sentinel -1 for out-of-rank).
                            const int my_expert_raw = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
                            const bool my_in_dst = (my_expert_raw >= 0)
                                and (my_expert_raw / kNumExpertsPerRank == dst_rank);
                            sym_dst[2 + lane_idx] = my_in_dst ? (my_expert_raw - dst_rank * kNumExpertsPerRank) : -1;
                        }
                    }
                }

                // Make sure all TMA stores from this iteration are committed before signalling arrival.
                ptx::tma_store_commit();
            }
            __syncwarp();

            // ----- Arrival counter: one lane per (token, dst_rank) bumps it -----
            if (is_master_for_dst) {
                int* sym_counter = gin.template get_sym_ptr<team_t>(
                    compact_layout.arrival_counter_ptr(rank_idx),
                    dst_rank);
                if (sym_counter != nullptr) {
                    ptx::red_add_rel_sys(sym_counter, 1);
                }
            }
            __syncwarp();

            // Maintain copied_topk_idx for the EPHandle (host expects this).
            if (copied_topk_idx != nullptr and warp_idx == 0 and sm_idx == 0 and lane_idx < kNumTopk) {
                copied_topk_idx[token_idx * kNumTopk + lane_idx] =
                    __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
            }
        }
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
}

}  // namespace deep_ep::elastic
