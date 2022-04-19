
/*
 * Benchmark for same address atomic operations
 * Does not report anything on its own, use Nsight Compute to gather statistics
 * In particular gpu__cycles_elapsed reports the total run time in cycles
 */

#include <cuda_runtime.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define check_error(x) (assert (x == 0))

__global__ void atomic_bmk(int *arr, int *addr, int limit, int iters, long *time) {
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x ;
  unsigned nthreads = gridDim.x * blockDim.x;
  int prev = 1;

  for(int i = tid; i < limit; i += nthreads) {
    for (int j = 0; j < iters; ++j) {
      prev = atomicAdd(arr+addr[tid], prev);
    }
  }
}

void generate_addrs(int *addr, int N, int offset) {
  for (int i = 0; i < N; ++i) {
    addr[i] = i*offset;
  }
}

int main(int argc, char *argv[]) {
  int blocks = atoi(argv[1]);
  int threads = atoi(argv[2]);
  int iters = atoi(argv[3]);

  int offset = 0;

  if (argc > 4) {
    offset = atoi(argv[4]);
  }

  int N = blocks*threads;

  int *addr = (int *)malloc(sizeof(int)*N);
  int *arr = (int *)malloc(sizeof(int)*N*offset+1);

  generate_addrs(addr, N, offset);

  int *d_addr = NULL;
  int *d_arr = NULL;

  check_error(cudaMalloc(&d_addr, sizeof(int)*N));
  check_error(cudaMalloc(&d_arr, sizeof(int)*N*offset+1));

  check_error(cudaMemcpy(d_arr, arr, sizeof(int)*N*offset+1, cudaMemcpyDefault));
  check_error(cudaMemcpy(d_addr, addr, sizeof(int)*N, cudaMemcpyDefault));

  atomic_bmk<<<blocks,threads>>>(d_arr, d_addr, N, iters, NULL);

  check_error(cudaDeviceSynchronize());

  atomic_bmk<<<blocks,threads>>>(d_arr, d_addr, N, iters, NULL);

  check_error(cudaDeviceSynchronize());

  check_error(cudaFree(d_addr));
  check_error(cudaFree(d_arr));
  free(addr);
  free(arr);

}

