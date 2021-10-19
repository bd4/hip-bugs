#include <complex>
#include <iostream>
#include <fstream>
#include <numeric>
#include <time.h>

#define NRUNS 10

#ifdef CUDAHIPBLAS
#include "cudahipblas.h"
#else
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include "rocsolver.h"
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

#define CHECK_BLAS(cmd)                                                    \
  {                                                                       \
    rocblas_status status = cmd;                                           \
    if (status != rocblas_status_success) {                                        \
      fprintf(stderr, "error: %d at %s:%d\n", status, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

template <typename T>
inline void read_carray(std::ifstream& f, int n, std::complex<T>* Adata) {
    for (int i=0; i < n; i++) {
        //std::cout << i << " " << std::endl;
        f >> Adata[i];
    }
}

inline void read_iarray(std::ifstream& f, int n, int *data) {
    for (int i=0; i < n; i++) {
        f >> data[i];
    }
}

template <typename T, typename RT, typename F>
void test(F&& getrs) {
    rocblas_handle h;
    int n, nrhs, lda, ldb, batch_size;
    int Aptr_size, Bptr_size, Adata_size, Bdata_size, piv_size;
    std::complex<T> **h_Aptr, **d_Aptr, **h_Bptr, **d_Bptr;
    std::complex<T> *h_Adata, *d_Adata, *h_Bdata, *d_Bdata;
    int *h_piv, *d_piv;

    using CT = std::complex<T>;

#ifdef READ_INPUT
    std::ifstream f("zgetrs.txt", std::ifstream::in);

    f >> n;
    f >> nrhs;
    f >> lda;
    f >> ldb;
    f >> batch_size;

#else
    n = 140;
    nrhs = 1;
    lda = n;
    ldb = n;
    batch_size = 384;
#endif

    std::cout << "n    = " << n    << std::endl;
    std::cout << "nrhs = " << nrhs << std::endl;
    std::cout << "lda  = " << lda  << std::endl;
    std::cout << "ldb  = " << ldb  << std::endl;
    std::cout << "batch_size = " << batch_size << std::endl;

    Aptr_size = Bptr_size = sizeof(*h_Aptr) * batch_size;
    Adata_size = sizeof(*h_Adata) * n * n * batch_size;
    Bdata_size = sizeof(*h_Bdata) * n * nrhs * batch_size;
    piv_size = sizeof(*h_piv) * n * batch_size;

    std::cout << "Aptr size  = " << Aptr_size  << std::endl;
    std::cout << "Bptr size  = " << Bptr_size  << std::endl;
    std::cout << "Adata size = " << Adata_size << std::endl;
    std::cout << "Bdata size = " << Bdata_size << std::endl;
    std::cout << "piv_size   = " << piv_size   << std::endl;

    CHECK(hipHostMalloc((void**)&h_Aptr, Aptr_size));
    CHECK(hipHostMalloc((void**)&h_Bptr, Bptr_size));
    CHECK(hipMalloc((void**)&d_Aptr, Aptr_size));
    CHECK(hipMalloc((void**)&d_Bptr, Bptr_size));

    CHECK(hipHostMalloc((void**)&h_Adata, Adata_size));
    CHECK(hipHostMalloc((void**)&h_Bdata, Bdata_size));
    CHECK(hipMalloc((void**)&d_Adata, Adata_size));
    CHECK(hipMalloc((void**)&d_Bdata, Bdata_size));

    CHECK(hipHostMalloc((void**)&h_piv, piv_size));
    CHECK(hipMalloc((void**)&d_piv, piv_size));

    std::cout << "malloc done" << std::endl;

#ifdef READ_INPUT
    read_carray(f, n*n*batch_size, h_Adata);
    read_carray(f, n*nrhs*batch_size, h_Bdata);
    read_iarray(f, n*batch_size, h_piv);
    f.close();
#else
    CHECK(hipMemset(h_Adata, 0, Adata_size));
    CHECK(hipMemset(h_Bdata, 0, Bdata_size));
    CHECK(hipDeviceSynchronize());

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            h_Adata[b*n*n + i*n + i] = CT(1.0, 0.0);
            h_Bdata[b*n*nrhs + i] = CT(i, 0.0);
            h_piv[b*n + i] = i;
        }
    }
#endif

    for (int i = 0; i < batch_size; i++) {
        h_Aptr[i] = d_Adata + (n*n*i);
        h_Bptr[i] = d_Bdata + (n*nrhs*i);
    }

    std::cout << "read/init done" << std::endl;

    CHECK(hipMemcpy(d_Aptr, h_Aptr, Aptr_size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_Adata, h_Adata, Adata_size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_Bptr, h_Bptr, Bptr_size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_Bdata, h_Bdata, Bdata_size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_piv, h_piv, piv_size, hipMemcpyHostToDevice));

    std::cout << "memcpy done" << std::endl;

    CHECK_BLAS(rocblas_create_handle(&h));

    struct timespec start, end;
    double elapsed, total = 0.0;
    int *info, info_sum;

#ifdef CUDAHIPBLAS
    info = (int *)calloc(batch_size, sizeof(*info));
#endif

    for (int i=0; i<NRUNS; i++) {
        // std::cout << "run [" << i << "]: start" << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &start);
#ifdef CUDAHIPBLAS
        CHECK_BLAS(getrs(h, CUBLAS_OP_N, n, nrhs,
                       reinterpret_cast<RT**>(d_Aptr), lda,
                       d_piv,
                       reinterpret_cast<RT**>(d_Bptr), ldb,
                       info, batch_size));
#else
        CHECK_BLAS(getrs(h, rocblas_operation_none, n, nrhs,
                       reinterpret_cast<RT**>(d_Aptr), lda,
                       d_piv, n,
                       reinterpret_cast<RT**>(d_Bptr), ldb,
                       batch_size));
#endif
        CHECK(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
        if (i > 0)
            total += elapsed;
#ifdef CUDAHIPBLAS
        info_sum = std::accumulate(info, info+batch_size, 0);
        if (info_sum != 0)
            std::cout << "info sum: " << info_sum << std::endl;
#endif
        std::cout << "run [" << i << "]: " << elapsed << std::endl;
    }

    std::cout << "zgetrs done (avg " << total / (NRUNS-1) << ")" << std::endl;

    CHECK_BLAS(rocblas_destroy_handle(h));
    CHECK(hipDeviceSynchronize());

    std::cout << "destroy done" << std::endl;

    CHECK(hipHostFree((void*)h_Aptr));
    CHECK(hipFree((void*)d_Aptr));
    CHECK(hipHostFree(h_Adata));
    CHECK(hipFree(d_Adata));

    CHECK(hipHostFree(h_Bptr));
    CHECK(hipFree(d_Bptr));
    CHECK(hipHostFree(h_Bdata));
    CHECK(hipFree(d_Bdata));

    CHECK(hipHostFree(h_piv));
    CHECK(hipFree(d_piv));

    CHECK(hipDeviceSynchronize());

    std::cout << "free done" << std::endl;
}

int main(int argc, char **argv) {
#ifndef CUDAHIPBLAS
    rocblas_initialize();
#endif
    std::cout << "==== double ====" << std::endl;
    test<double, rocblas_double_complex>(&rocsolver_zgetrs_batched);
    std::cout << "==== float  ====" << std::endl;
    test<float, rocblas_float_complex>(&rocsolver_cgetrs_batched);
}
