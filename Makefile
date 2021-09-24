ROCM_PATH = /opt/rocm
CUDA_PATH = /usr/local/cuda
HIPFFT_CUDA_PATH = $(HOME)/soft/hipfft/cuda

fft_strided_hip: fft_strided.cxx
	HIP_PLATFORM="amd" hipcc -g -std=c++14 -L $(ROCM_PATH)/lib -lhipfft -I $(ROCM_PATH)/hipfft/include -o fft_strided_hip fft_strided.cxx

fft_strided_cuda: fft_strided.cxx
	HIP_PLATFORM="nvidia" hipcc -g -std=c++14 -L $(HIPFFT_CUDA_PATH)/lib -lhipfft -lcufft -I $(HIPFFT_CUDA_PATH)/include -I $(ROCM_PATH)/include -o fft_strided_cuda fft_strided.cxx

fft_strided_nvcc: fft_strided.cxx
	nvcc -g -std=c++14 -lcufft -DCUDAHIPFFT=1 -o fft_strided_nvcc fft_strided.cxx

.PHONY: clean
clean:
	rm -f fft_strided_cuda fft_strided_hip fft_strided_nvcc

.PHONY: all
all: fft_strided_hip fft_strided_cuda fft_strided_nvcc
