
/*
 * Generates atomics to random addresses
 * Run using NSight Compute to gather statistics
 * In particular gpu__cycles_elapsed reports the total run time in cycles
 */

#include <cuda_runtime.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define check_error(x) (assert (x == 0))

__global__ void atomic_bmk(int *arr, long *addr, int limit, int iters, long *time) {
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x ;
  unsigned nthreads = gridDim.x * blockDim.x;
  int prev = 1;

  for(int i = tid; i < limit; i += nthreads) {
    for (int j = 0; j < iters; ++j) {
      prev = atomicAdd(arr+addr[tid], prev);
    }
  }
}

void generate_addrs(long *addr, int N, long max) {
  for (int i = 0; i < N; ++i) {
    long head = rand();
    long tail = rand();
    long total = (head<<32) | tail;
    addr[i] = total%max;
  }
}

int main(int argc, char *argv[]) {
  int blocks = atoi(argv[1]);
  int threads = atoi(argv[2]);
  int iters = atoi(argv[3]);

  int N = blocks*threads;
  long max = (long)1*1024*1024*1024*4;

  long *addr = (long *)malloc(sizeof(long)*N);
  int *arr = (int *)malloc(sizeof(int)*max);
  long *time = (long *)malloc(sizeof(long)*N*iters);

  generate_addrs(addr, N, max);
  long *d_addr = NULL;
  int *d_arr = NULL;
  long *d_time = NULL;

  check_error(cudaMalloc(&d_addr, sizeof(long)*N));
  check_error(cudaMalloc(&d_arr, sizeof(int)*max));
  check_error(cudaMalloc(&d_time, sizeof(long)*N*iters));

  check_error(cudaMemcpy(d_arr, arr, sizeof(int)*max, cudaMemcpyDefault));
  check_error(cudaMemcpy(d_addr, addr, sizeof(long)*N, cudaMemcpyDefault));

  atomic_bmk<<<blocks,threads>>>(d_arr, d_addr, N, iters, d_time);

  check_error(cudaDeviceSynchronize());

  check_error(cudaMemcpy(time, d_time, sizeof(long)*N*iters, cudaMemcpyDefault));

  double sum = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < iters; ++j) {
      sum += time[i*iters + j];
    }
  }
  sum /= (N*iters);
  printf("%lf\n", sum/N);

  check_error(cudaFree(d_addr));
  check_error(cudaFree(d_arr));
  free(addr);
  free(arr);
}

