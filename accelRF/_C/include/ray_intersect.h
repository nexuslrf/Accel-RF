#pragma once

#include <torch/extension.h>
#include <utility>

std::tuple<at::Tensor, at::Tensor, at::Tensor> aabb_intersect_old(at::Tensor rays_o, at::Tensor rays_d, at::Tensor points, 
               const float voxelsize, const int n_max);
std::tuple<at::Tensor, at::Tensor, at::Tensor> aabb_intersect(at::Tensor rays_o, at::Tensor rays_d, at::Tensor points, 
               const float voxelsize, const int n_max);