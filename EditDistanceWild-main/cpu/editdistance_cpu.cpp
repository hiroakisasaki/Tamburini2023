#include <torch/extension.h> 

#ifdef DEBUG
#include <chrono>
#include <iostream>
#endif

namespace {

template <typename scalar_t>
int64_t handlePadLen(scalar_t* str, int64_t strLen, int64_t padToken) {
    for (int i=0; i < strLen; i++)
	    if (str[i] == padToken) return i;

    return strLen;
}


template <typename scalar_t>
int32_t CmpChars(scalar_t a, scalar_t b) {
    if (a == int('?'))
        if (b != int('?'))
            return 0;
        else
            return 1;
    else
        return (a == b ? 0 : 1);
}


// https://github.com/roy-ht/editdistance
template <typename scalar_t>
static void distance_single_batch_frame(
    scalar_t* const src, 
    scalar_t* const trg, 
    float* result,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t padToken,
	int64_t insdelC,
	int64_t substC) 
{
    // handle padding
    srcLen = handlePadLen(src, srcLen, padToken);
    trgLen = handlePadLen(trg, trgLen, padToken);

    // base case
    if (srcLen == 0) { result[0] = (float)trgLen; result[1] = 0.0; return; }
    if (trgLen == 0) { result[0] = (float)srcLen; result[1] = 0.0; return; }

    auto src_ = src, trg_ = trg;
    auto srcLen_ = srcLen, trgLen_ = trgLen;
    //if (trgLen < srcLen) src_ = trg, trg_ = src, srcLen_ = trgLen, trgLen_ = srcLen;

    std::vector<std::vector<int32_t>> d(srcLen_+1, std::vector<int32_t>(trgLen_+1));

    d[0][0] = 0;
    for (int i=1; i < srcLen_ + 1; i++) 
		d[i][0] = i;
    for (int i=1; i < trgLen_ + 1; i++) 
		d[0][i] = i;
    for (int i=1; i < srcLen_ + 1; i++) 
        for (int j=1; j < trgLen_ + 1; j++) 
            d[i][j] = std::min(std::min(d[(i-1)][j], d[i][j-1]) + insdelC*(src_[i-1]!=int('*')), 
								d[(i-1)][j-1] + substC*CmpChars(src_[i-1], trg_[j-1])*(src_[i-1]!=int('*')));
    result[0] = (float)(d[srcLen_][trgLen_]);
	//result[1] = result[0] / std::max(srcLen_, trgLen_);
	result[1] = 2.0*result[0] / (srcLen_+trgLen_+result[0]); // GED
}

template <typename scalar_t>
static void distance_frame(
    scalar_t* const src, 
    scalar_t* const trg, 
    float* result,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t numBatch, 
    int64_t padToken,
	int64_t insdelC,
	int64_t substC)
{
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
        for (const auto batch : c10::irange(start, end)) {
            distance_single_batch_frame<scalar_t>(
                src + batch * srcLen, 
                trg + batch * trgLen,
                result + 2*batch,
                srcLen, 
                trgLen,
				padToken,
				insdelC,
				substC
            );
        }
    });
}


template <typename scalar_t>
static void distance_frame1N(
    scalar_t* const src, 
    scalar_t* const trg, 
    float* result,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t numBatch, 
    int64_t padToken,
	int64_t insdelC, 
	int64_t substC)
{
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
        for (const auto batch : c10::irange(start, end)) {
            distance_single_batch_frame<scalar_t>(
                src, 
                trg + batch * trgLen,
                result + 2*batch,
                srcLen, 
                trgLen,
				padToken,
				insdelC,
				substC
            );
        }
    });
}

}

torch::Tensor editdistance_cpu(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken,
	int64_t insdelC, 
	int64_t substC)
{
    auto numBatch = src.size(0);
    auto srcLen = src.size(1);
    auto trgLen = trg.size(1);

    at::TensorOptions options(src.device());
    //options = options.dtype(at::ScalarType::Int);
    options = options.dtype(at::ScalarType::Float);
    auto result = at::empty({numBatch, 2}, options);

#ifdef DEBUG
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(),
        "editdistance_cpu",
        [&] {
            distance_frame<scalar_t>(
            src.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            result.data_ptr<float>(),
            srcLen, 
            trgLen,
            numBatch,
	    	padToken,
			insdelC,
			substC
          );
        }
    );

#ifdef DEBUG
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU timing = "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()
              << " us  " << std::flush;
#endif

    return result;
}

TORCH_LIBRARY_IMPL(editdistance, CPU, m) {
  m.impl("editdistance", editdistance_cpu);
}



torch::Tensor editdistance1N_cpu(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken,
	int64_t insdelC,
	int64_t substC)
{
    auto numBatch = trg.size(0);
    auto srcLen = src.size(1);
    auto trgLen = trg.size(1);

    at::TensorOptions options(src.device());
    //options = options.dtype(at::ScalarType::Int);
    options = options.dtype(at::ScalarType::Float);
    auto result = at::empty({numBatch, 2}, options);

#ifdef DEBUG
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(),
        "editdistance1N_cpu",
        [&] {
            distance_frame1N<scalar_t>(
            src.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            result.data_ptr<float>(),
            srcLen, 
            trgLen,
            numBatch,
	    	padToken,
			insdelC,
			substC
          );
        }
    );

#ifdef DEBUG
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU timing = "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()
              << " us  " << std::flush;
#endif

    return result;
}

TORCH_LIBRARY_IMPL(editdistance1N, CPU, m) {
  m.impl("editdistance1N", editdistance1N_cpu);
}
