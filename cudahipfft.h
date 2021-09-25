#include <cuda_runtime_api.h>
#include <cufft.h>

#define hipError_t cudaError_t
#define hipMalloc cudaMalloc
#define hipHostMalloc cudaMallocHost
#define hipMemset cudaMemset
#define hipMemcpy cudaMemcpy
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetErrorString cudaGetErrorString

#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost

#define hipSuccess cudaSuccess
#define HIPFFT_SUCCESS CUFFT_SUCCESS

#define HIPFFT_Z2Z CUFFT_Z2Z
#define HIPFFT_C2C CUFFT_C2C
#define HIPFFT_D2Z CUFFT_D2Z
#define HIPFFT_Z2D CUFFT_Z2D
#define HIPFFT_R2C CUFFT_R2C
#define HIPFFT_C2R CUFFT_C2R

#define HIPFFT_FORWARD CUFFT_FORWARD
#define HIPFFT_INVERSE CUFFT_BACKWARD
#define HIPFFT_BACKWARD CUFFT_BACKWARD

using hipfftHandle = cufftHandle;
using hipfftType = cufftType;
using hipStream_t = cudaStream_t;
using hipfftResult_t = cufftResult_t;

using hipfftDoubleComplex = cufftDoubleComplex;
using hipfftComplex = cufftComplex;
using hipfftDoubleReal = cufftDoubleReal;
using hipfftReal = cufftReal;

#define hipfftCreate cufftCreate
#define hipfftPlanMany cufftPlanMany
#define hipfftDestroy cufftDestroy

#define hipfftExecZ2Z cufftExecZ2Z
#define hipfftExecC2C cufftExecC2C
#define hipfftExecD2Z cufftExecD2Z
#define hipfftExecZ2D cufftExecZ2D
#define hipfftExecR2C cufftExecR2C
#define hipfftExecC2R cufftExecC2R
