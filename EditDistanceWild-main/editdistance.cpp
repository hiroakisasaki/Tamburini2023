#include "editdistance.h"

torch::Tensor editdistance(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken,
    int64_t insdelC,
    int64_t substC)
{
    int64_t srcDims = src.ndimension();
    int64_t trgDims = trg.ndimension();

    TORCH_CHECK(srcDims == 2 || srcDims == 1, 
                "editdistance: Expect 1D or 2D Tensor, got: ",
                src.sizes());

    TORCH_CHECK(trgDims == 2 || trgDims == 1, 
                "editdistance: Expect 1D or 2D Tensor, got: ",
                trg.sizes());

    TORCH_CHECK(srcDims == trgDims, 
                "editdistance: Expect src and trg to have the same number of dimensions");

    TORCH_CHECK(src.device() == trg.device(), 
	       "source and target tensor must be on the same device, got ",
	       "src on device ", src.device(),
	       " and trg on device ", trg.device());

    auto src_ = src;
    auto trg_ = trg; 
    if (srcDims == 1)
    {
        src_ = src_.reshape({1, src_.size(0)});
        trg_ = trg_.reshape({1, trg_.size(0)});
    }

    TORCH_CHECK(src_.size(0) == trg_.size(0), 
	        "editdistance: expected src and trg to have same batch size");

    // dispatch
    static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("editdistance::editdistance", "")
    .typed<decltype(editdistance)>();
    return op.call(src_, trg_, padToken, insdelC, substC);
}


torch::Tensor editdistance1N(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken,
    int64_t insdelC,
    int64_t substC)
{
    int64_t srcDims = src.ndimension();
    int64_t trgDims = trg.ndimension();

    TORCH_CHECK(srcDims == 2, 
                "editdistance: Expect 2D Tensor as src, got: ",
                src.sizes());

    TORCH_CHECK(trgDims == 2, 
                "editdistance: Expect 2D Tensor as trg, got: ",
                trg.sizes());

    TORCH_CHECK(srcDims == trgDims, 
                "editdistance: Expect src and trg to have the same number of dimensions");

    TORCH_CHECK(src.device() == trg.device(), 
	       "source and target tensor must be on the same device, got ",
	       "src on device ", src.device(),
	       " and trg on device ", trg.device());

    auto src_ = src;
    auto trg_ = trg; 

    TORCH_CHECK(src_.size(0) == 1, 
	        "editdistance1N: expected only one word in src");

    // dispatch
    static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("editdistance1N::editdistance1N", "")
    .typed<decltype(editdistance1N)>();
    return op.call(src_, trg_, padToken, insdelC, substC);
}


TORCH_LIBRARY(editdistance, m) {
  m.def("editdistance(Tensor self, Tensor other, int padToken, int insdelC, int substC) -> Tensor");
}

TORCH_LIBRARY(editdistance1N, m) {
  m.def("editdistance1N(Tensor self, Tensor other, int padToken, int insdelC, int substC) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("editdistance", &editdistance, "editdistance forward");
  m.def("editdistance1N", &editdistance1N, "editdistance1N forward");
}

