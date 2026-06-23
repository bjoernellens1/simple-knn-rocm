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

#include "spatial.h"
#include "simple_knn.h"
#include <limits>

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);

  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}

std::tuple<torch::Tensor, torch::Tensor>
knnIndices(const torch::Tensor& query, const torch::Tensor& reference, int64_t k, bool exclude_self)
{
  TORCH_CHECK(query.dim() == 2 && query.size(1) == 3, "query must be [Nq, 3]");
  TORCH_CHECK(reference.dim() == 2 && reference.size(1) == 3, "reference must be [Nr, 3]");
  TORCH_CHECK(k >= 1 && k <= 32, "k must be in [1, 32]");

  const int Q = query.size(0);
  const int R = reference.size(0);

  auto idx = torch::full({Q, k}, -1, query.options().dtype(torch::kInt64));
  auto dist2 = torch::full({Q, k}, std::numeric_limits<float>::max(),
                           query.options().dtype(torch::kFloat32));

  auto query_c = query.contiguous();
  auto reference_c = reference.contiguous();
  SimpleKNN::knnIndices(
      Q, (const float3*)query_c.data<float>(),
      R, (const float3*)reference_c.data<float>(),
      (int)k, exclude_self,
      idx.data<int64_t>(), dist2.data<float>());

  return std::make_tuple(idx, dist2);
}