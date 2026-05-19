#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/layout.cuh>

#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          bool kOverlapPushReduce,
          int kNumSMs, int kNumWarps,
          // TODO: merge these two variables into one (ensure the whole file does not contain "scaleup")
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleoutRanks,
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* recv_buffer,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    EP_STATIC_ASSERT(kNumExperts % (kNumScaleupRanks * kNumScaleoutRanks) == 0, "Invalid number of experts or ranks");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;   // NOTES: Here we prioritize distributing tasks to different SMs to ensure that the last wave is evenly concentrated on each SM.

    // Load buffers from scale-out or scale-up ranks
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto comm_buffer = layout::BufferLayout<false>(
        comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);
    auto* ready_flags = reinterpret_cast<int*>(comm_buffer.get_buffer_end_ptr());
    if constexpr (kOverlapPushReduce) {
        EP_STATIC_ASSERT(kNumScaleoutRanks == 1, "Combine overlap supports direct combine only");
        EP_STATIC_ASSERT(not kUseExpandedLayout, "Combine overlap does not support expanded layout");
        EP_STATIC_ASSERT(not kUseRankLayout, "Combine overlap currently requires top-k slot layout");
    }

    // Store buffers
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    // Bias layout
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);

    // Will block until the main combine kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();

    // Read from buffers and do reduction
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {
        // Preprocess all indices
        int stored_dst_rank_idx = -1, stored_dst_expert_idx = -1;
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        if (lane_idx < kNumTopk) {
            stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            stored_dst_rank_idx = stored_dst_expert_idx >= 0 ?
                stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout) : -1;
        }
        __syncwarp();

        // Sort valid top-k indices to front
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {
                // Activations are never reduced before
                return {false, 0};
            } else if constexpr (kNumScaleoutRanks != 1 and not kUseExpandedLayout and not kAllowMultipleReduction) {
                // Hybrid mode without expanded layout and multiple reduction. Should deduplicate on a per-rank basis
                return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / kNumExpertsPerRank : -1};
            } else {
                // Should deduplicate on a per-rank (for non-hybrid mode) or a per-scale-rank (for hybrid mode) basis
                return {true, stored_dst_rank_idx};
            }
        }();
        auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);
        int topk_slot_idx[kNumTokensInLayout];
        compute_topk_slots(
            topk_slot_idx, reduce_valid_mask,
            [=](const int& idx) {
                return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
            }
        );

        if constexpr (kOverlapPushReduce) {
            #pragma unroll
            for (int k = 0; k < kNumTokensInLayout; ++ k) {
                const int slot_idx = topk_slot_idx[k];
                if (slot_idx >= 0) {
                    const auto ready_ptr = ready_flags + slot_idx * kNumMaxTokensPerRank + token_idx;
                    comm::timeout_while<comm::kNumOneSecCycles>([=](const bool& is_last_check) {
                        const int ready = ptx::ld_acquire_sys<int>(ready_ptr);
                        if (ready == 1)
                            return true;
                        if (is_last_check and lane_idx == 0) {
                            printf("Combine overlap ready timeout: token=%d slot=%d ready=%d scaleout=%d scaleup=%d\n",
                                   token_idx, slot_idx, ready, scaleout_rank_idx, scaleup_rank_idx);
                        }
                        return false;
                    });
                }
            }
            __syncwarp();
        }

        // Iterate over per-hidden-chunk stage
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);
        constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
        combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(
            lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
            /* Get source base */ [=](const int& slot_idx) {
                return static_cast<combine_vec_t*>(
                    comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());
            },
            /* Wait buffer release */ [=]() {
                ptx::tma_store_wait();
                __syncwarp();
            },
            /* Bias 0 */ bias_0 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()),
            /* Bias 1 */ bias_1 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr())
        );
        ptx::tma_store_fence();
        __syncwarp();

        // Issue TMA copy
        if (ptx::elect_one_sync()) {
            ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),
                              tma_buffer.get_base_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();

        // Write top-k weights
        if (combined_topk_weights != nullptr) {
            const auto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));
            if (lane_idx < kNumTopk) {
                float value = 0;
                if (stored_dst_rank_idx >= 0) {
                    const auto dst_ptr = comm_buffer
                        .get_rank_buffer(kUseRankLayout ? stored_dst_rank_idx : master_lane_idx)
                        .get_token_buffer(token_idx).get_topk_weights_ptr() + lane_idx;
                    value = *dst_ptr;
                }
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;
            }
            __syncwarp();
        }
    }
}

}  // deep_ep::elastic
