#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define hipError_t cudaError_t
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipHostFree cudaFreeHost
#define hipHostMalloc cudaMallocHost
#define hipMemset cudaMemset
#define hipMemcpy cudaMemcpy
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetErrorString cudaGetErrorString

#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost

#define hipSuccess cudaSuccess

#define rocblas_status cublasStatus_t
#define rocblas_status_success CUBLAS_STATUS_SUCCESS

using rocblas_handle = cublasHandle_t;
using hipStream_t = cudaStream_t;

using rocblas_double_complex = cuDoubleComplex;
using rocblas_complex = cuComplex;

#define rocblas_create_handle cublasCreate
#define rocblas_destroy_handle cublasDestroy

#define rocblas_operation_none CUBLAS_OP_N

#define rocsolver_zgetrs_batched cublasZgetrsBatched
