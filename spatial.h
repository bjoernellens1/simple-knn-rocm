/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include <tuple>

torch::Tensor distCUDA2(const torch::Tensor& points);

// Cross-set k-NN. Returns (idx [Nq, k] int64 into reference, dist2 [Nq, k] float32).
// exclude_self skips the query's own index (use when query IS reference).
std::tuple<torch::Tensor, torch::Tensor> knnIndices(
    const torch::Tensor& query, const torch::Tensor& reference,
    int64_t k, bool exclude_self);