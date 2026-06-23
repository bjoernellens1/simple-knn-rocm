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
#include "spatial.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("distCUDA2", &distCUDA2);
  m.def("knnIndices", &knnIndices,
        "Cross-set k-NN -> (idx [Nq,k] int64, dist2 [Nq,k] float32)",
        pybind11::arg("query"), pybind11::arg("reference"),
        pybind11::arg("k"), pybind11::arg("exclude_self") = false);
}
