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

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#ifndef __HIP_PLATFORM_AMD__
#include "device_launch_parameters.h"  // CUDA-only IDE convenience header; absent on ROCm
#endif
#include "simple_knn.h"
#include <cfloat>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#ifndef __HIP_PLATFORM_AMD__
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
// cg is used only for the global thread index below.
#define SKNN_GLOBAL_THREAD_RANK() (cg::this_grid().thread_rank())
#else
// ROCm: cooperative_groups/reduce.h is absent and cg::reduce is unused here, so
// avoid cooperative_groups entirely — the only use is the 1D global thread rank.
#define SKNN_GLOBAL_THREAD_RANK() (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x)
#endif

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = SKNN_GLOBAL_THREAD_RANK();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	auto idx = SKNN_GLOBAL_THREAD_RANK();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = SKNN_GLOBAL_THREAD_RANK();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

// ---- Cross-set k-NN with returned indices (ROCm/HIP port addition) ----------
#ifndef SKNN_MAX_K
#define SKNN_MAX_K 32
#endif

// Insert (d, idx) into the per-thread top-k arrays kept sorted ascending by dist.
__device__ __forceinline__ void knnInsert(float* bd, int64_t* bi, int k, float d, int64_t idx)
{
	if (d >= bd[k - 1])
		return;
	int pos = k - 1;
	while (pos > 0 && bd[pos - 1] > d)
	{
		bd[pos] = bd[pos - 1];
		bi[pos] = bi[pos - 1];
		pos--;
	}
	bd[pos] = d;
	bi[pos] = idx;
}

// For each query point, find the k nearest reference points using the same
// Morton-box pruning as boxMeanDist. Reference is Morton-sorted (ref_sorted holds
// original reference indices); boxes are min/max over BOX_SIZE-sized blocks of it.
__global__ void knnIndicesKernel(
	uint32_t Q, const float3* query, const float3* reference,
	const uint32_t* ref_sorted, const MinMax* boxes, uint32_t num_boxes, uint32_t R,
	int k, int exclude_self, int64_t* out_idx, float* out_dist2)
{
	uint32_t qi = SKNN_GLOBAL_THREAD_RANK();
	if (qi >= Q)
		return;

	float3 p = query[qi];
	float bd[SKNN_MAX_K];
	int64_t bi[SKNN_MAX_K];
	for (int j = 0; j < k; j++) { bd[j] = FLT_MAX; bi[j] = -1; }

	for (uint32_t b = 0; b < num_boxes; b++)
	{
		if (distBoxPoint(boxes[b], p) > bd[k - 1])
			continue;
		uint32_t start = b * BOX_SIZE;
		uint32_t end = min(R, (b + 1) * BOX_SIZE);
		for (uint32_t i = start; i < end; i++)
		{
			uint32_t ridx = ref_sorted[i];
			if (exclude_self && ridx == qi)
				continue;
			float3 r = reference[ridx];
			float3 dd = { p.x - r.x, p.y - r.y, p.z - r.z };
			float d = dd.x * dd.x + dd.y * dd.y + dd.z * dd.z;
			knnInsert(bd, bi, k, d, (int64_t)ridx);
		}
	}

	for (int j = 0; j < k; j++)
	{
		out_idx[(size_t)qi * k + j] = bi[j];
		out_dist2[(size_t)qi * k + j] = bd[j];
	}
}

void SimpleKNN::knnIndices(
	int Q, const float3* query,
	int R, const float3* reference,
	int k, bool exclude_self,
	int64_t* out_idx, float* out_dist2)
{
	if (Q <= 0 || R <= 0)
		return;  // outputs are pre-filled with -1 / FLT_MAX by the caller

	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;
	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, reference, result, R, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, reference, result, R, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, reference, result, R, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(R), morton_sorted(R), indices(R), indices_sorted(R);
	coord2Morton<<<(R + 255) / 256, 256>>> (R, reference, minn, maxx, morton.data().get());
	thrust::sequence(indices.begin(), indices.end());
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), R);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), R);

	uint32_t num_boxes = (R + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax<<<num_boxes, BOX_SIZE>>> (R, (float3*)reference, indices_sorted.data().get(), boxes.data().get());
	knnIndicesKernel<<<(Q + 255) / 256, 256>>> (
		Q, query, reference, indices_sorted.data().get(), boxes.data().get(), num_boxes, R,
		k, exclude_self ? 1 : 0, out_idx, out_dist2);

	cudaFree(result);
}

void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton<<<(P + 255) / 256, 256 >>> (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax<<<num_boxes, BOX_SIZE >>> (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist<<<num_boxes, BOX_SIZE >>> (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}