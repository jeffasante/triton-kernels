import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    causal: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Block pointers for Q, K, V
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh \
             + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh \
             + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh \
             + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # Online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs)
    scale = (BLOCK_DMODEL ** -0.5)

    # Main loop over key/value blocks
    lo = 0
    hi = (start_m + 1) * BLOCK_M if causal else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.dot(q, tl.trans(k)) * scale

        if causal:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        l_i = alpha * l_i + tl.sum(p, 1)
        acc = alpha[:, None] * acc

        v = tl.load(v_ptrs + start_n * stride_vn)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_i_new

    acc = acc / l_i[:, None]

    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh \
               + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(tl.float16))

def triton_attention(q, k, v, causal=True):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, 64), B * H)
    fused_attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N,
        BLOCK_M=64, BLOCK_N=64, BLOCK_DMODEL=D,
        causal=causal,
    )
    return o
