#include <torch/extension.h> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>

#ifdef DEBUG
#include "utils.cuh"
#include <iostream>
#endif

namespace {

// TODO: this can probabliy be parallelized using a gpu
template <typename scalar_t>
__device__ int64_t handlePadLen(scalar_t* str, int64_t strLen, int64_t padToken) {
    for (int i=0; i < strLen; i++)
	    if (str[i] == padToken) return i;
    return strLen;
}

template <typename scalar_t>
__device__ int32_t CmpChars(scalar_t a, scalar_t b) {
	if (a == int('?'))
		if (b != int('?'))
			return 0;
		else
			return 1;
	else
    	return (a == b ? 0 : 1);
}

template <typename scalar_t>
__global__ void distance_cuda_kernel(
    scalar_t* const __restrict__ src, 
    scalar_t* const __restrict__ trg, 
    float* __restrict__ result,
    int* __restrict__ workingM, 
	int64_t numBatch,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t padToken,
	int64_t insdelC,
	int64_t substC,
	int64_t f1N)
{
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
	if (batch >= numBatch) return;
    //const int batch = blockIdx.x;

    auto srcBatch = src + batch * srcLen;
	if (f1N)
    	srcBatch = src;
    auto trgBatch = trg + batch * trgLen;
    auto result_ = result + batch * 2;
	auto d = workingM + batch * (srcLen+1)*(trgLen+1);

    // handle padding
    srcLen = handlePadLen(srcBatch, srcLen, padToken);
    trgLen = handlePadLen(trgBatch, trgLen, padToken);

    // base case
    if (srcLen == 0) { result_[0] = (float)trgLen; result_[1] = 0.0; return; }
    if (trgLen == 0) { result_[0] = (float)srcLen; result_[1] = 0.0; return; }

    auto src_ = srcBatch, trg_ = trgBatch;
    auto srcLen_ = srcLen, trgLen_ = trgLen;
    //if (trgLen < srcLen) src_ = trgBatch, trg_ = srcBatch, srcLen_ = trgLen, trgLen_ = srcLen;

    int rows = srcLen_+1;
    int cols = trgLen_+1;

    // TODO: cudaMalloc is probably better, but first we need to fix the lengths
    //auto d = new int[rows*cols];

    d[0] = 0;
    for (int i = 0; i < rows; i++)
		d[i*cols] = i;
    for (int i = 0; i < cols; i++)
		d[i] = i;
    for (int i = 1; i < rows; i++) 
        for (int j = 1; j < cols; j++) 
            d[i*cols + j] = std::min(std::min(d[(i-1)*cols + j], d[i*cols + (j-1)]) + insdelC*(src_[i-1]!=int('*')), 
			    		 d[(i-1)*cols + (j-1)] + substC*CmpChars(src_[i-1], trg_[j-1])*(src_[i-1]!=int('*')));
    result_[0] = (float)d[srcLen_*cols + trgLen_];
	//result_[1] = result_[0] / std::max(srcLen_, trgLen_);
	result_[1] = 2.0*result_[0] / (srcLen_+trgLen_+result_[0]); // GED
}
}

torch::Tensor editdistance_cuda_kernel(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    torch::Tensor& result, 
    int64_t padToken,
	int64_t insdelC,
	int64_t substC)
{
    const auto numBatch = src.size(0);
    const auto srcLen = src.size(1);
    const auto trgLen = trg.size(1);

	//cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, 0);
	unsigned int threads = 256;
	unsigned int blocks = (numBatch + threads - 1) / threads;
	//unsigned int max_blocks = prop.maxGridSize[0];
	// ADJUST
	//if (blocks > max_blocks)
  	//	blocks = max_blocks;

/*
    const int threads = 1;
    const int blocks = numBatch;
*/

	int *workingM;
	cudaMalloc(&workingM, sizeof(int)*(srcLen+1)*(trgLen+1)*blocks*threads);

    // see https://github.com/pytorch/pytorch/issues/21819
    // to avoid random errors when executing on cuda:1 we need to set the device manually
    c10::cuda::set_device(static_cast<c10::DeviceIndex>(src.device().index()));

    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(),
        "editdistance_cuda",
        ([&] {
         distance_cuda_kernel<scalar_t><<<blocks, threads>>>(
            src.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            result.data_ptr<float>(),
			workingM,
			numBatch,
            srcLen, 
            trgLen, 
	    	padToken,
			insdelC,
			substC,
			0);
        }));

	cudaFree(workingM);
    return result;
}


torch::Tensor editdistance1N_cuda_kernel(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    torch::Tensor& result, 
    int64_t padToken,
	int64_t insdelC,
	int64_t substC)
{

    const auto numBatch = trg.size(0);
    const auto srcLen = src.size(1);
    const auto trgLen = trg.size(1);

	//cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, 0);
	unsigned int threads = 256;
	unsigned int blocks = (numBatch + threads - 1) / threads;
	//unsigned int max_blocks = prop.maxGridSize[0];
	// ADJUST
	//if (blocks > max_blocks)
  	//	blocks = max_blocks;

/*
    const int threads = 1;
    const int blocks = numBatch;
*/

	int *workingM;
	cudaMalloc(&workingM, sizeof(int)*(srcLen+1)*(trgLen+1)*blocks*threads);


    // see https://github.com/pytorch/pytorch/issues/21819
    // to avoid random errors when executing on cuda:1 we need to set the device manually
    c10::cuda::set_device(static_cast<c10::DeviceIndex>(src.device().index()));

    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(),
        "editdistance1N_cuda",
        ([&] {
         distance_cuda_kernel<scalar_t><<<blocks, threads>>>(
            src.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            result.data_ptr<float>(),
            workingM,
			numBatch,
            srcLen, 
            trgLen, 
	    	padToken,
			insdelC,
			substC,
			1);
        }));

	cudaFree(workingM);
    return result;
}
