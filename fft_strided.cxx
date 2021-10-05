#include <complex>
#include <cstdio>
#include <iostream>

#ifdef CUDAHIPFFT
#include "cudahipfft.h"
#else
#include <hip/hip_runtime.h>
#include <hipfft.h>
#endif

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#define CHECK_FFT(cmd)                                                    \
  {                                                                       \
    hipfftResult_t error = cmd;                                           \
    if (error != HIPFFT_SUCCESS) {                                        \
      fprintf(stderr, "error: %d at %s:%d\n", error, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

template <typename E>
void print_1d(const char* name, int n, E* x);

template <>
void print_1d(const char* name, int n, float* x) {
  printf("%10s: [%6.3f", name, x[0]);
  for (int i = 1; i < n; i++) {
    printf(", %6.3f", x[i]);
  }
  printf("]\n");
}

template <>
void print_1d(const char* name, int n, std::complex<float>* x) {
  printf("%-12s: [(%6.3f, %6.3f)", name, x[0].real(), x[0].imag());
  for (int i = 1; i < n; i++) {
    printf(" (%6.3f, %6.3f)", x[i].real(), x[i].imag());
  }
  printf("]\n");
}

template <typename E>
inline void expect_near_array(const char* file, int line, int n,
                              const char* xname, E* x, const char* yname,
                              E* y) {
  constexpr double max_err = 1e-6;
  bool equal = true;
  int i;
  double err;

  for (i = 0; i < n; i++) {
    err = abs(x[i] - y[i]);
    if (err > max_err) {
      equal = false;
      break;
    }
  }

  if (!equal) {
    std::cerr << "Arrays not close (max " << max_err << ") at " << file << ":"
              << line << std::endl
              << " err " << err << " at [" << i << "]" << std::endl;
    print_1d<E>(xname, n, x);
    print_1d<E>(yname, n, y);
  }
}

#define EXPECT_NEAR_ARRAY(n, x, y) \
  expect_near_array(__FILE__, __LINE__, n, #x, x, #y, y)

template <typename E>
void fft_r2c_1d_strided() {
  constexpr int N = 4;
  constexpr int Nout = N / 2 + 1;
  constexpr int rstride = 2;
  constexpr int cstride = 3;
  constexpr int rdist = N * rstride;
  constexpr int cdist = Nout * cstride;
  constexpr int batch_size = 2;
  using T = std::complex<E>;

  int n[1] = {N};
  int ncomplex[1] = {Nout};

  E *h_A, *h_A2, *d_A, *d_A2;
  T *h_B, *h_B_expected, *d_B;

  size_t rbytes = rdist * batch_size * sizeof(E);
  size_t cbytes = cdist * batch_size * sizeof(T);

  CHECK(hipHostMalloc((void**)&h_A, rbytes));
  CHECK(hipMalloc((void**)&d_A, rbytes));

  CHECK(hipHostMalloc((void**)&h_A2, rbytes));
  CHECK(hipMalloc((void**)&d_A2, rbytes));

  CHECK(hipHostMalloc((void**)&h_B, cbytes));
  CHECK(hipHostMalloc((void**)&h_B_expected, cbytes));
  CHECK(hipMalloc((void**)&d_B, cbytes));

  CHECK(hipMemset(h_A, 0, rbytes));
  CHECK(hipMemset(h_B_expected, 0, cbytes));
  CHECK(hipMemset(d_B, 0, cbytes));
  CHECK(hipMemset(d_A2, 0, rbytes));

  // x = [2 3 -1 4];
  h_A[0 * rstride] = 2;
  h_A[1 * rstride] = 3;
  h_A[2 * rstride] = -1;
  h_A[3 * rstride] = 4;

  // y = [7 -21 11 1];
  h_A[0 * rstride + rdist] = 7;
  h_A[1 * rstride + rdist] = -21;
  h_A[2 * rstride + rdist] = 11;
  h_A[3 * rstride + rdist] = 1;

  print_1d("A", rdist * batch_size, h_A);

  h_B_expected[0 * cstride] = T(8, 0);
  h_B_expected[1 * cstride] = T(3, 1);
  h_B_expected[2 * cstride] = T(-6, 0);

  h_B_expected[0 * cstride + cdist] = T(-2, 0);
  h_B_expected[1 * cstride + cdist] = T(-4, 22);
  h_B_expected[2 * cstride + cdist] = T(38, 0);

  CHECK(hipMemcpy(d_A, h_A, rbytes, hipMemcpyHostToDevice));

  hipfftHandle plan_forward, plan_inverse;

  CHECK_FFT(hipfftCreate(&plan_forward));
  CHECK_FFT(hipfftCreate(&plan_inverse));

  CHECK_FFT(hipfftPlanMany(&plan_forward, 1, n, n, rstride, rdist, ncomplex, cstride,
                 cdist, HIPFFT_R2C, batch_size));
  CHECK_FFT(hipfftPlanMany(&plan_inverse, 1, n, ncomplex, cstride, cdist, n, rstride,
                 rdist, HIPFFT_C2R, batch_size));

  CHECK_FFT(hipfftExecR2C(plan_forward, (hipfftReal*)d_A, (hipfftComplex*)d_B));
  CHECK(hipMemcpy(h_B, d_B, cbytes, hipMemcpyDeviceToHost));

  // test roundtripping data
  CHECK_FFT(hipfftExecC2R(plan_inverse, (hipfftComplex*)d_B, (hipfftReal*)d_A2));
  CHECK(hipMemcpy(h_A2, d_A2, rbytes, hipMemcpyDeviceToHost));

  EXPECT_NEAR_ARRAY(cdist * batch_size, h_B_expected, h_B);

  for (int i = 0; i < rdist * batch_size; i++) {
    h_A2[i] = h_A2[i] / N;
  }
  EXPECT_NEAR_ARRAY(rdist * batch_size, h_A, h_A2);
}

int main(int argc, char** argv) { fft_r2c_1d_strided<float>(); }
