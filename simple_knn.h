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

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED

#include <cstdint>

class SimpleKNN
{
public:
	static void knn(int P, float3* points, float* meanDists);

	// Cross-set k-NN: for each of the Q query points, find the k nearest of the R
	// reference points (Morton-grid accelerated, exact). Writes [Q, k] neighbour
	// indices (into reference) and squared distances, row-major. When exclude_self
	// is true (query IS reference), the point's own index is skipped.
	static void knnIndices(
		int Q, const float3* query,
		int R, const float3* reference,
		int k, bool exclude_self,
		int64_t* out_idx, float* out_dist2);
};

#endif