ROCM_PATH = /opt/rocm
CUDA_PATH = /usr/local/cuda
HIPFFT_CUDA_PATH = $(HOME)/soft/hipfft/cuda

.PHONY: all
all: fft_strided_hip fft_strided_nvcc batched_zgetrs_hip batched_zgetrs_nvcc

fft_strided_hip: fft_strided.cxx
	HIP_PLATFORM="amd" hipcc -g -std=c++14 -L $(ROCM_PATH)/lib -lhipfft -I $(ROCM_PATH)/hipfft/include -o fft_strided_hip fft_strided.cxx

fft_strided_cuda: fft_strided.cxx
	HIP_PLATFORM="nvidia" hipcc -g -std=c++14 -L $(HIPFFT_CUDA_PATH)/lib -lhipfft -lcufft -I $(HIPFFT_CUDA_PATH)/include -I $(ROCM_PATH)/include -o fft_strided_cuda fft_strided.cxx

fft_strided_nvcc: fft_strided.cxx
	nvcc -g -std=c++14 -lcufft -DCUDAHIPFFT=1 -o fft_strided_nvcc fft_strided.cxx

batched_zgetrs_hip: batched_zgetrs.cxx
	HIP_PLATFORM="amd" hipcc -g -std=c++14 -L $(ROCM_PATH)/lib -lrocblas -lrocsolver -I $(ROCM_PATH)/include -o $@ $<

batched_zgetrs_hip_read: batched_zgetrs.cxx
	HIP_PLATFORM="amd" hipcc -g -std=c++14 -D READ_INPUT -L $(ROCM_PATH)/lib -lrocblas -lrocsolver -I $(ROCM_PATH)/include -o $@ $<

batched_zgetrs_nvcc: batched_zgetrs.cxx
	nvcc -g -std=c++14 -lcublas -DCUDAHIPBLAS=1 -o $@ $<

.PHONY: clean
clean:
	rm -f fft_strided_{cuda,hip,nvcc} batched_zgetrs_{cuda,hip,nvcc}
