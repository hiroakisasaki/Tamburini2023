#pragma once
#include <torch/extension.h>

torch::Tensor editdistance(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken,
	int64_t insdelC,
	int64_t substC);

torch::Tensor editdistance1N(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken,
    int64_t insdelC,
    int64_t substC);

