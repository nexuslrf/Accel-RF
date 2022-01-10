#pragma once

#include <torch/extension.h>
#include <utility>

std::tuple<at::Tensor, at::Tensor, at::Tensor> aabb_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor points, 
               const float voxelsize, const int n_max);