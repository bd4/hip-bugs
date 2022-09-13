/**
* Reproduce slow kernels after hipMemcpy involving managed memory as destination
* used in a computation kernel.
*
* Defines three arrays, to simulate a common pattern in GENE:
*
* (1) managed memory array allocated from Fortran (via C interfaces to hip)
* (2) initialized from host
* (3) pass to C computation kernel which does in-place operation
* (4) copy to second managed memory array so original is not overwritten
* (5) perform some computation using the copied to managed memory array
*     and some other data (could be in device mem or managed mem)
*/
#include <iostream>
#include <time.h>

#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }


__global__ void kernel_assign_1(const int size, double *lhs, double *rhs)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < size) {
    lhs[i] = rhs[i];
  }
}


int main (int argc, char *argv[])
{
  const int n = 10 * 1024 * 1024;
  const int nbytes = sizeof(double) * n;
  const int block_size = 256;
  const int nwarmup = 10;
  const double max_seconds = 2;
  double *d_a, *d_b, *d_c;
  struct timespec start, end;
  double elapsed, total;

  // data coming in is always managed
  CHECK(hipMallocManaged(&d_a, nbytes));

  // intermediate managed or device to compare
#ifdef MANAGED
  CHECK(hipMallocManaged(&d_b, nbytes));
#else
  CHECK(hipMalloc(&d_b, nbytes));
#endif

  // output array always in device
  CHECK(hipMalloc(&d_c, nbytes));

  total = 0.0;
  int niter = 0;
  dim3 nblocks(n / block_size);
  dim3 threads_per_block(block_size);
  while (total < max_seconds) {
    CHECK(hipMemcpy(d_b, d_a, nbytes, hipMemcpyDeviceToDevice));
    clock_gettime(CLOCK_MONOTONIC, &start);
    kernel_assign_1<<<nblocks, threads_per_block>>>(n, d_c, d_b);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (niter >= nwarmup) {
      elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
      total += elapsed;
    }
    niter++;
  }
  niter -= nwarmup;
  std::cout << float(total) / niter << "\t" << niter << std::endl;

  return 0;
}
