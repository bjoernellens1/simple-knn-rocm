# simple-knn-rocm

A **ROCm / HIP port** of Inria GRAPHDECO's [`simple-knn`](https://gitlab.inria.fr/bkerbl/simple-knn)
— the `distCUDA2` spatial-KNN extension used by 3D Gaussian Splatting to initialise
Gaussian scales (exact O(N·log N) Morton-grid mean squared distance to the 3 nearest
neighbours).

The upstream extension is CUDA-only. This fork makes it build and run on AMD GPUs
(RDNA3 / RDNA3.5 / RDNA4) via PyTorch's HIP toolchain, so the pure-PyTorch O(N²)
`cdist` fallbacks people use on ROCm can be dropped. On a Radeon AI PRO R9700
(gfx1201) the HIP kernel is **~31× faster at 50k points and ~291× faster at 300k**
than the O(N²) torch fallback (6.8 s → 23 ms at 300k).

> Upstream is distributed under the Inria GRAPHDECO research/evaluation license
> (see [`LICENSE.md`](LICENSE.md)); those terms are preserved here unchanged. This
> repository only ports the build to ROCm — no algorithmic changes.

## What changed for ROCm

All edits are guarded with `#ifndef __HIP_PLATFORM_AMD__`, so the sources still build
unchanged on CUDA:

1. **`device_launch_parameters.h`** — a CUDA-only IDE convenience header with no HIP
   equivalent; guarded out (`threadIdx`/`blockIdx` are built-ins on both platforms).
2. **`cooperative_groups`** — ROCm ships no `cooperative_groups/reduce.h`, and
   `cg::reduce` was never used here. The single `cg::this_grid().thread_rank()` is
   replaced on HIP by the plain 1-D global thread index, dropping the dependency.
3. **Kernel-launch chevrons** — upstream uses Windows-style spaced `<< <` / `>> >`,
   which hipify's regex doesn't match; normalised to `<<<` / `>>>` so they transpile.

`cub` → `hipCUB` and `thrust` → rocThrust are handled automatically by PyTorch's
`BuildExtension` hipify. The block reduction uses `__shared__` + `__syncthreads()`
with no warp-size assumption, so it is correct on wave32 (RDNA) and wave64 (CDNA).

## Install

Requires a ROCm PyTorch environment (`torch.version.hip` set) and the ROCm dev
headers (`hipcub`, `rocthrust`, `rocprim` — present in `rocm/pytorch` images).

```bash
# Build for your GPU arch(es); semicolon-separate to make one fat wheel.
PYTORCH_ROCM_ARCH="gfx1100;gfx1151;gfx1201" \
CUDA_HOME=/opt/rocm ROCM_HOME=/opt/rocm \
  pip install --no-build-isolation .
```

Or grab a prebuilt wheel from the [Actions artifacts](../../actions) /
[Releases](../../releases).

## Usage

```python
import torch
from simple_knn._C import distCUDA2

pts = torch.rand(100_000, 3, device="cuda")      # [N, 3] float32 on GPU
mean_sq_dist = distCUDA2(pts)                     # [N] mean squared dist to 3-NN
scales = torch.sqrt(mean_sq_dist)                 # typical 3DGS scale init
```

## Building locally without a system ROCm install

Use a ROCm PyTorch container (no GPU needed to *compile*):

```bash
docker run --rm -v "$PWD":/src -w /src \
  -e PYTORCH_ROCM_ARCH="gfx1100;gfx1151;gfx1201" \
  -e CUDA_HOME=/opt/rocm -e ROCM_HOME=/opt/rocm \
  rocm/pytorch:latest \
  bash -lc "pip wheel . --no-build-isolation -w dist"
```

## Credits

- Original `simple-knn`: Inria GRAPHDECO (Bernhard Kerbl et al.), part of the
  [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
  project.
- ROCm/HIP port: maintained for the
  [Splatograph](https://github.com/bjoernellens1/splatograph) ROCm 3DGS stack.
