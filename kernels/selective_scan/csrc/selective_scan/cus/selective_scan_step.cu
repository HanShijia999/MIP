#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float softplus_f(float x) {
    // 稳一点的 softplus
    if (x > 20.f) return x;
    return log1pf(expf(x));
}

template<typename input_t>
__device__ __forceinline__ float to_float(input_t x) { return (float)x; }
template<>
__device__ __forceinline__ float to_float<at::Half>(at::Half x) { return (float)__half2float((__half)x); }

template<typename input_t>
__device__ __forceinline__ input_t from_float(float x);
template<>
__device__ __forceinline__ at::Half from_float<at::Half>(float x) { return (at::Half)__float2half_rn(x); }
template<>
__device__ __forceinline__ float from_float<float>(float x) { return x; }
template<>
__device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float x) { return (at::BFloat16)x; }

template<typename input_t>
__global__ void selective_scan_step_kernel(
    const input_t* __restrict__ u,        // [B, D]
    const input_t* __restrict__ delta,    // [B, D]
    const float* __restrict__ A,          // [D, dstate]
    const input_t* __restrict__ Bv,       // [B, G, dstate]
    const input_t* __restrict__ Cv,       // [B, G, dstate]
    const float* __restrict__ D,          // [D] or nullptr
    const float* __restrict__ delta_bias, // [D] or nullptr
    float* __restrict__ state,            // [B, D, dstate] in-place
    input_t* __restrict__ out,            // [B, D]
    int Bsz, int dim, int dstate, int ngroups,
    bool delta_softplus
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Bsz * dim;
    if (idx >= total) return;

    int b = idx / dim;
    int d = idx - b * dim;

    int dim_per_group = dim / ngroups;
    int g = d / dim_per_group;

    float uval = to_float(u[b * dim + d]);
    float dt   = to_float(delta[b * dim + d]);
    if (delta_bias) dt += delta_bias[d];
    if (delta_softplus) dt = softplus_f(dt);

    float Du = (D ? D[d] : 0.f) * uval;

    float y = Du;

    // 指向该 (b,d) 的 state 起点
    float* st = state + ( (b * dim + d) * dstate );

    // 指向该 (b,g) 的 B/C 起点
    const input_t* Bptr = Bv + ( (b * ngroups + g) * dstate );
    const input_t* Cptr = Cv + ( (b * ngroups + g) * dstate );

    // A 的起点
    const float* Aptr = A + d * dstate;

    float dt_u = dt * uval;

    #pragma unroll
    for (int s = 0; s < 256; ++s) {
        if (s >= dstate) break;
        float a = expf(dt * Aptr[s]);                 // a_t
        float bterm = to_float(Bptr[s]) * dt_u;       // b_t
        float x = a * st[s] + bterm;                  // state update
        st[s] = x;
        y += x * to_float(Cptr[s]);                   // output accumulate
    }

    out[b * dim + d] = from_float<input_t>(y);
}

template<typename input_t, typename weight_t>
void selective_scan_step_cuda(
    const at::Tensor &u, const at::Tensor &delta,
    const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
    const c10::optional<at::Tensor> &D_,
    const c10::optional<at::Tensor> &delta_bias_,
    at::Tensor &state,
    at::Tensor &out,
    bool delta_softplus,
    cudaStream_t stream
){
    int Bsz = u.size(0);
    int dim = u.size(1);
    int dstate = A.size(1);
    int ngroups = B.size(1);

    int threads = 256;
    int blocks = (Bsz * dim + threads - 1) / threads;

    selective_scan_step_kernel<input_t><<<blocks, threads, 0, stream>>>(
        (const input_t*)u.data_ptr(),
        (const input_t*)delta.data_ptr(),
        (const float*)A.data_ptr(),
        (const input_t*)B.data_ptr(),
        (const input_t*)C.data_ptr(),
        D_.has_value() ? (const float*)D_.value().data_ptr() : nullptr,
        delta_bias_.has_value() ? (const float*)delta_bias_.value().data_ptr() : nullptr,
        (float*)state.data_ptr(),
        (input_t*)out.data_ptr(),
        Bsz, dim, dstate, ngroups,
        delta_softplus
    );
}
// ===== explicit instantiations to satisfy linker =====
template void selective_scan_step_cuda<float, float>(
    const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    at::Tensor&, at::Tensor&,
    bool, cudaStream_t);

template void selective_scan_step_cuda<c10::Half, float>(
    const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    at::Tensor&, at::Tensor&,
    bool, cudaStream_t);

template void selective_scan_step_cuda<c10::BFloat16, float>(
    const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    at::Tensor&, at::Tensor&,
    bool, cudaStream_t);