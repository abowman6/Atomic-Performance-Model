
/*
 * Benchmark used to generate the service times
 * All atomics are to the same address
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
  long t1, t2;

  for(int i = tid; i < limit; i += nthreads) {
    for (int j = 0; j < iters; ++j) {
      t1 = clock64();
      prev = atomicAdd(arr+addr[tid], prev);
      t2 = clock64();
      time[tid*iters+j] = t2-t1;
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
  long *time = (long *)malloc(sizeof(long)*N*iters);
  int *arr = (int *)malloc(sizeof(int)*N*offset+1);

  generate_addrs(addr, N, offset);

  int *d_addr = NULL;
  int *d_arr = NULL;
  long *d_time = NULL;

  check_error(cudaMalloc(&d_addr, sizeof(int)*N));
  check_error(cudaMalloc(&d_time, sizeof(long)*N*iters));
  check_error(cudaMalloc(&d_arr, sizeof(int)*N*offset+1));

  check_error(cudaMemcpy(d_arr, arr, sizeof(int)*N*offset+1, cudaMemcpyDefault));
  check_error(cudaMemcpy(d_addr, addr, sizeof(int)*N, cudaMemcpyDefault));
  check_error(cudaMemcpy(d_time, time, sizeof(long)*N*iters, cudaMemcpyDefault));

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
  check_error(cudaFree(d_time));
  free(addr);
  free(arr);
  free(time);
}

