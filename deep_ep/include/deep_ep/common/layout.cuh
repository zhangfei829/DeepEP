#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::layout {

struct WorkspaceLayout {
    void* workspace;

    int num_ranks;
    int num_scaleout_ranks, num_scaleup_ranks;
    int num_experts, num_experts_per_rank;

    // We want to fix the layout position for all settings,
    // so that one buffer can be reused for all cases
    static constexpr int kNumMaxRanks = 1024;
    static constexpr int kNumMaxExperts = 2048;
    static constexpr int kNumMaxExpertsPerRank = 256;
    static constexpr int kNumMaxInflightAGRS = 32;

    static constexpr int64_t kNumBarrierSignalBytes = 16;

    __forceinline__ __device__ __host__
    WorkspaceLayout(void* workspace,
                    const int& num_scaleout_ranks,
                    const int& num_scaleup_ranks,
                    const int& num_experts):
        workspace(workspace),
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),
        num_scaleout_ranks(num_scaleout_ranks),
        num_scaleup_ranks(num_scaleup_ranks),
        num_experts(num_experts) {
        num_experts_per_rank = num_experts / num_ranks;
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        EP_UNIFIED_ASSERT(num_experts <= kNumMaxExperts);
        EP_UNIFIED_ASSERT(num_experts_per_rank <= kNumMaxExpertsPerRank);
    }

    static int64_t get_num_bytes() {
        // Pure NVLink scaleup barrier signals
        int64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;

        // Notify reduction workspace
        num_bytes += (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t);

        // Scaleup notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 2;
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int64_t) * 2;

        // Scaleup atomic sender count
        num_bytes += kNumMaxRanks * sizeof(int);

        // Scaleout notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int) * 2;
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int) * 2;

        // Scaleout channel metadata (finish flag and tails)
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int64_t);

        // Channel aggregated into the scaleup domains
        // Also reused for channel scaleup tail
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int);

        // Rank send/recv count, for PP prev/next ranks
        num_bytes += 2 * 2 * sizeof(int64_t);

        // AGRS signals
        num_bytes += (kNumMaxInflightAGRS + 1) * kNumMaxRanks * sizeof(int);

        // Ensure LDG.256 work
        return math::align<int64_t>(num_bytes, 32);
    }

    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() const {
        return static_cast<unsigned long long*>(workspace);
    }

    __forceinline__ __device__ __host__ int* get_nvl_barrier_signal_ptr(const int& phase) const {
        return math::advance_ptr<int>(workspace, (2 + phase) * sizeof(int));
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        return math::advance_ptr<int64_t>(workspace, kNumBarrierSignalBytes);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int64_t>(get_notify_reduction_workspace_ptr(), (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + num_scaleup_ranks;
    }

    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {
        return math::advance_ptr<int>(
            get_scaleup_rank_expert_count_ptr<true>(), 2 * (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int>(get_scaleup_atomic_sender_counter(), kNumMaxRanks * sizeof(int));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>();
        return base_ptr + scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() + num_ranks;
        return base_ptr + scaleout_rank_idx * (num_scaleup_ranks * num_experts_per_rank) + expert_idx;
    }

    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr(
        const int& channel_idx, const int& scaleout_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_scaleout_rank_expert_count_ptr<true>(),
            (kNumMaxRanks + kNumMaxExperts) * sizeof(int) * 2);
        return base_ptr + (channel_idx * num_scaleout_ranks + scaleout_rank_idx);
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_scaleout_channel_signaled_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int64_t));
        return base_ptr + (channel_idx * num_scaleup_ranks + scaleup_rank_idx);
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_send_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_channel_scaleup_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int));
        return base_ptr + offset;
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_recv_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_pp_send_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + offset;
    }

    __forceinline__ __device__ __host__ int* get_agrs_recv_signal_ptr(const int& slot, const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_pp_recv_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + slot * kNumMaxRanks + rank_idx;
    }

    __forceinline__ __device__ __host__ int* get_agrs_session_signal_ptr(const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_agrs_recv_signal_ptr(0, 0), kNumMaxInflightAGRS * kNumMaxRanks * sizeof(int));
        return base_ptr + rank_idx;
    }
};

struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    // NOTES: the top-k index is always 32-bit
    bool with_metadata;
    int num_topk, num_metadata_bytes;
    void* base;

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        // Metadata includes: top-k indices, weight and source rank/token index
        with_metadata(with_metadata),
        num_topk(num_topk),
        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +
                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),
        base(base) {
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
    }

    template <bool kWithMBarrier, typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +
                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);
        return static_cast<dtype_t>(num_bytes);
    }

    __forceinline__ __device__ __host__ void* get_base_ptr() const {
        return base;
    }

    __forceinline__ __device__ __host__ void set_base_ptr(void* ptr) {
        base = ptr;
    }

    __forceinline__ __device__ __host__ void* get_hidden_ptr() const {
        return get_base_ptr();
    }

    __forceinline__ __device__ __host__ sf_pack_t* get_sf_ptr() const {
        return math::advance_ptr<sf_pack_t>(base, math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_metadata_ptr() const {
        return math::advance_ptr<int>(get_sf_ptr(), math::align(num_sf_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_topk_idx_ptr() const {
        return get_metadata_ptr();
    }

    __forceinline__ __device__ __host__ float* get_topk_weights_ptr() const {
        return math::advance_ptr<float>(get_metadata_ptr(), num_topk * sizeof(int));
    }

    __forceinline__ __device__ __host__ int* get_src_token_global_idx_ptr() const {
        return math::advance_ptr<int>(get_topk_weights_ptr(), num_topk * sizeof(float));
    }

    __forceinline__ __device__ __host__ int* get_linked_list_idx_ptr() const {
        return get_src_token_global_idx_ptr() + 1;
    }

    __forceinline__ __device__ ptx::mbarrier* get_mbarrier_ptr() const {
        return math::advance_ptr<ptx::mbarrier>(get_metadata_ptr(), math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes));
    }
};

template <bool kWithMBarrier>
struct BufferLayout {
    TokenLayout token_layout;
    int num_ranks;
    int num_max_tokens_per_rank;

    void* base;

    __forceinline__ __device__ __host__
    BufferLayout(const TokenLayout& token_layout,
                 const int& num_ranks,
                 const int& max_num_tokens_per_rank,
                 void* base = nullptr) :
        token_layout(token_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_token() const {
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    __forceinline__ __device__ __host__
    BufferLayout get_rank_buffer(const int& rank_idx) const {
        return BufferLayout(token_layout,
                            1, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    template <int kNumTokensPerChannel>
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx) const {
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % kNumTokensPerChannel == 0);
        return BufferLayout(token_layout,
                            // Do not use `num_max_tokens_per_rank / kNumTokensPerChannel` as the false stride
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * kNumTokensPerChannel * channel_idx);
    }

    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        return TokenLayout(token_layout.num_hidden_bytes, token_layout.num_sf_bytes, token_layout.num_topk, token_layout.with_metadata,
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx);
    }
};

// =============================================================================
// CompactRecvBufferLayout: Region-based, NO-COPY dispatch (fast path)
// =============================================================================
//
// Layout used by the fast-path dispatch (non-expand + intra-NVL72). Replaces
// the legacy "per-src-rank packed token slot" layout with five compact
// per-(N_recv) regions, written directly by the src-side dispatch warps:
//
//   Region A: hidden          [N_max * N_ranks, hidden]   <dtype>
//   Region B: sf              [N_max * N_ranks, num_sf_packs]   sf_pack_t
//   Region C: topk_idx        [N_max * N_ranks, num_topk]       int32   (local expert idx)
//   Region D: topk_weights    [N_max * N_ranks, num_topk]       float
//   Region E: src_metadata    [N_max * N_ranks, num_topk + 2]   int32
//   Region F: arrival counter [N_ranks]                          int32   (dst-side; how many tokens from src=i arrived)
//   Region G: peer_psum_buf   [N_ranks, N_ranks]                 int32   (src-side; per (dst, src_seen_in_dst) prefix sum, broadcast by dst)
//
// Each region is `ptx::kNumTMAAlignBytes`-aligned. `compact_window` is a
// single ncclWindow_t covering all six regions in this order.
//
// `recv_x`, `recv_sf`, `recv_topk_idx`, `recv_topk_weights`, `recv_src_metadata`
// in the Python API are `torch::from_blob(...)` views of Region A..E with the
// natural strides documented above (no extra copy).
struct CompactRecvBufferLayout {
    int64_t num_max_tokens_per_rank;
    int64_t num_ranks;
    int64_t hidden_bytes;        // sizeof(dtype) * hidden
    int64_t sf_bytes_per_token;  // sizeof(sf_pack_t) * num_sf_packs (0 for bf16)
    int64_t num_topk;
    void*   base;                // start of the registered ncclWindow buffer (host side)

    __forceinline__ __device__ __host__
    CompactRecvBufferLayout(const int64_t& num_max_tokens_per_rank,
                            const int64_t& num_ranks,
                            const int64_t& hidden_bytes,
                            const int64_t& sf_bytes_per_token,
                            const int64_t& num_topk,
                            void* base = nullptr) :
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ranks(num_ranks),
        hidden_bytes(hidden_bytes),
        sf_bytes_per_token(sf_bytes_per_token),
        num_topk(num_topk),
        base(base) {}

    __forceinline__ __device__ __host__
    int64_t get_max_tokens() const {
        return num_max_tokens_per_rank * num_ranks;
    }

    // math::align<T>(T,T) requires both args the same type; force int64_t.
    __forceinline__ __device__ __host__
    int64_t get_region_A_bytes() const {
        return math::align<int64_t>(get_max_tokens() * hidden_bytes,
                                    static_cast<int64_t>(ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__
    int64_t get_region_B_bytes() const {
        return math::align<int64_t>(get_max_tokens() * sf_bytes_per_token,
                                    static_cast<int64_t>(ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__
    int64_t get_region_C_bytes() const {
        // Region C stores recv_topk_idx[N_recv, num_topk] in topk_idx_t units
        // (default int64).  The size is computed at runtime since topk_idx_t
        // is a typedef and we can't use sizeof(topk_idx_t) in a generic helper.
        // Caller passes `topk_idx_size_bytes` via the dedicated overload below
        // if non-default sizes are needed; here we assume sizeof(int64_t).
        return math::align<int64_t>(get_max_tokens() * num_topk * static_cast<int64_t>(sizeof(int64_t)),
                                    static_cast<int64_t>(ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__
    int64_t get_region_D_bytes() const {
        return math::align<int64_t>(get_max_tokens() * num_topk * static_cast<int64_t>(sizeof(float)),
                                    static_cast<int64_t>(ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__
    int64_t get_region_E_bytes() const {
        return math::align<int64_t>(get_max_tokens() * (num_topk + 2) * static_cast<int64_t>(sizeof(int)),
                                    static_cast<int64_t>(ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__
    int64_t get_region_F_bytes() const {
        return math::align<int64_t>(num_ranks * static_cast<int64_t>(sizeof(int)),
                                    static_cast<int64_t>(ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__
    int64_t get_region_G_bytes() const {
        return math::align<int64_t>(num_ranks * num_ranks * static_cast<int64_t>(sizeof(int)),
                                    static_cast<int64_t>(ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__
    int64_t get_total_bytes() const {
        return get_region_A_bytes() + get_region_B_bytes() + get_region_C_bytes()
             + get_region_D_bytes() + get_region_E_bytes()
             + get_region_F_bytes() + get_region_G_bytes();
    }

    // Region offsets (within `base`).
    __forceinline__ __device__ __host__ int64_t offset_A() const { return 0; }
    __forceinline__ __device__ __host__ int64_t offset_B() const { return offset_A() + get_region_A_bytes(); }
    __forceinline__ __device__ __host__ int64_t offset_C() const { return offset_B() + get_region_B_bytes(); }
    __forceinline__ __device__ __host__ int64_t offset_D() const { return offset_C() + get_region_C_bytes(); }
    __forceinline__ __device__ __host__ int64_t offset_E() const { return offset_D() + get_region_D_bytes(); }
    __forceinline__ __device__ __host__ int64_t offset_F() const { return offset_E() + get_region_E_bytes(); }
    __forceinline__ __device__ __host__ int64_t offset_G() const { return offset_F() + get_region_F_bytes(); }

    // Per-token pointer helpers (compact_idx in [0, N_max * N_ranks)).
    __forceinline__ __device__ __host__
    void* hidden_ptr(const int64_t& compact_idx) const {
        return static_cast<int8_t*>(base) + offset_A() + compact_idx * hidden_bytes;
    }
    __forceinline__ __device__ __host__
    sf_pack_t* sf_ptr(const int64_t& compact_idx) const {
        return reinterpret_cast<sf_pack_t*>(
            static_cast<int8_t*>(base) + offset_B() + compact_idx * sf_bytes_per_token);
    }
    // Region C entries are sizeof(int64_t) wide (= sizeof(topk_idx_t) for the
    // default 64-bit config). Caller reinterprets the returned void* into
    // topk_idx_t* (int64_t*).
    __forceinline__ __device__ __host__
    void* topk_idx_ptr(const int64_t& compact_idx) const {
        return static_cast<int8_t*>(base) + offset_C() + compact_idx * num_topk * sizeof(int64_t);
    }
    __forceinline__ __device__ __host__
    float* topk_weights_ptr(const int64_t& compact_idx) const {
        return reinterpret_cast<float*>(
            static_cast<int8_t*>(base) + offset_D() + compact_idx * num_topk * sizeof(float));
    }
    __forceinline__ __device__ __host__
    int* src_metadata_ptr(const int64_t& compact_idx) const {
        return reinterpret_cast<int*>(
            static_cast<int8_t*>(base) + offset_E() + compact_idx * (num_topk + 2) * sizeof(int));
    }

    // Region F: per-(src_rank) arrival counter, on dst side.
    __forceinline__ __device__ __host__
    int* arrival_counter_ptr(const int& src_rank_idx) const {
        return reinterpret_cast<int*>(
            static_cast<int8_t*>(base) + offset_F() + src_rank_idx * sizeof(int));
    }

    // Region G: peer_psum_buf[dst_rank, src_rank] = "in dst_rank's view,
    // cumulative count of tokens from src ranks 0..src_rank-1 to dst_rank".
    // Used by src side to compute its compact_idx within dst's RECV buffer.
    __forceinline__ __device__ __host__
    int* peer_psum_ptr(const int& dst_rank_idx, const int& src_rank_idx) const {
        return reinterpret_cast<int*>(
            static_cast<int8_t*>(base) + offset_G()
            + (static_cast<int64_t>(dst_rank_idx) * num_ranks + src_rank_idx) * sizeof(int));
    }

    __forceinline__ __device__ __host__
    int* peer_psum_row_ptr(const int& dst_rank_idx) const {
        return peer_psum_ptr(dst_rank_idx, 0);
    }
};

}  // namespace deep_ep::elastic
