"""Task 2: 1D 3-point stencil with boundary clamping (halo replication)."""
import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        if idx == 0:
            left = d_in[0]
        else:
            left = d_in[idx - 1]
        if idx == N - 1:
            right = d_in[N - 1]
        else:
            right = d_in[idx + 1]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    N = h_in.shape[0]
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array(N, dtype=np.float32)
    blocks = (N + THREADS - 1) // THREADS
    stencil_1d[blocks, THREADS](d_in, d_out, N)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


if __name__ == "__main__":
    N = 100_007
    h_in = np.random.rand(N).astype(np.float32)
    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)
    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4)
    delta = float(np.max(np.abs(h_out_gpu - cpu_ref)))
    print(f"TASK 2 PASSED: MAX DELTA = {delta}")
