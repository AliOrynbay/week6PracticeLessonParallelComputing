"""Task 1: Warp divergence microbenchmark (kernel-only timing)."""
import time
import numpy as np
from numba import cuda

N = 2 ** 20
ITERS = 1000
THREADS = 256
TRIALS = 10


@cuda.jit
def kernel_uniform(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        for _ in range(ITERS):
            v = v * 1.0001 + 0.0001
        y[idx] = v


@cuda.jit
def kernel_divergent(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        if idx % 2 == 0:  # adjacent threads diverge inside every warp
            for _ in range(ITERS):
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(ITERS):
                v = (v - 0.0001) / 1.0001
        y[idx] = v


@cuda.jit
def kernel_warp_aligned(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:  # whole warps take the same path
            for _ in range(ITERS):
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(ITERS):
                v = (v - 0.0001) / 1.0001
        y[idx] = v


def bench(kernel, d_y, blocks):
    kernel[blocks, THREADS](d_y, N)  # warm-up (includes JIT compile)
    cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[blocks, THREADS](d_y, N)
        cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def main():
    h_y = np.ones(N, dtype=np.float32)
    d_y = cuda.to_device(h_y)  # transfer happens once, outside all timing
    blocks = (N + THREADS - 1) // THREADS

    results = {}
    for name, k in [("Kernel A (Uniform)", kernel_uniform),
                    ("Kernel B (Full Divergence)", kernel_divergent),
                    ("Kernel C (Warp-Aligned)", kernel_warp_aligned)]:
        results[name] = bench(k, d_y, blocks)

    base = results["Kernel A (Uniform)"]
    print(f"GPU: {cuda.get_current_device().name}")
    print("| Kernel | Avg time (ms, 10 trials) | Slowdown vs A |")
    print("|---|---|---|")
    for name, t in results.items():
        print(f"| {name} | {t:.3f} | {t / base:.2f}x |")


if __name__ == "__main__":
    main()
