"""Task 3: Arbitrary-size vector scaling via a grid-stride loop."""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64  


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        d_arr, np.float32(factor), N)
    cuda.synchronize()
    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 2 ** 24  
    factor = 4.25
    h_arr = np.ones(N, dtype=np.float32)
    total_threads = THREADS_PER_BLOCK * BLOCKS_PER_GRID
    print(f"Launching {total_threads} threads for {N} elements")
    res = run_grid_stride(h_arr, factor)
    assert np.all(res == np.float32(factor)), "Mismatch in scaled values"
    print(f"All {N} elements match factor {factor}. TASK 3 PASSED")
