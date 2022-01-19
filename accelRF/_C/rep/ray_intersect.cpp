#include "ray_intersect.h"
#include "utils.h"
#include <utility>
// #include <torch/script.h>

void aabb_intersect_old_kernel_wrapper(
  int b, int n, int m, float voxelsize, int n_max,
  const float *rays_o, const float *rays_d, const float *points,
  int *idx, float *min_depth, float *max_depth);

void aabb_intersect_kernel_wrapper(
  int n_blk, int rays_per_blk, int n_pts, int n_rays, int max_hit, float radius,
  at::DeviceIndex device_idx, const float *rays_o, const float *rays_d, const float *points,
  int *idx, float *min_depth, float *max_depth);

std::tuple< at::Tensor, at::Tensor, at::Tensor > aabb_intersect_old(at::Tensor rays_o, at::Tensor rays_d, at::Tensor points, 
               const float voxelsize, const int n_max){
  CHECK_CONTIGUOUS(rays_o);
  CHECK_CONTIGUOUS(rays_d);
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(rays_o);
  CHECK_IS_FLOAT(rays_d);
  CHECK_IS_FLOAT(points);
  CHECK_CUDA(rays_o);
  CHECK_CUDA(rays_d);
  CHECK_CUDA(points);
  at::Tensor idx =
      torch::zeros({rays_o.size(0), rays_o.size(1), n_max},
                    at::device(rays_o.device()).dtype(at::ScalarType::Int));
  at::Tensor min_depth =
      torch::zeros({rays_o.size(0), rays_o.size(1), n_max},
                    at::device(rays_o.device()).dtype(at::ScalarType::Float));
  at::Tensor max_depth =
      torch::zeros({rays_o.size(0), rays_o.size(1), n_max},
                    at::device(rays_o.device()).dtype(at::ScalarType::Float));
  aabb_intersect_old_kernel_wrapper(rays_o.size(0), points.size(1), rays_o.size(1),
                                      voxelsize, n_max,
                                      rays_o.data_ptr <float>(), rays_d.data_ptr <float>(), points.data_ptr <float>(),
                                      idx.data_ptr <int>(), min_depth.data_ptr <float>(), max_depth.data_ptr <float>());
  return std::make_tuple(idx, min_depth, max_depth);
}

std::tuple< at::Tensor, at::Tensor, at::Tensor > aabb_intersect(at::Tensor rays_o, at::Tensor rays_d, at::Tensor points, 
               const float voxelsize, const int max_hit){
  CHECK_CONTIGUOUS(rays_o);
  CHECK_CONTIGUOUS(rays_d);
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(rays_o);
  CHECK_IS_FLOAT(rays_d);
  CHECK_IS_FLOAT(points);
  CHECK_CUDA(rays_o);
  CHECK_CUDA(rays_d);
  CHECK_CUDA(points);
  /*
  Args:
    rays: (N_rays, 3)
    points: (..., 3)
  Return:
    inds: *1d* index of points.
    max_depth, min_depth.
    
  Note: this function has wiser implementations, if points is a regular dense grid (4D Tensor).
   TODO@nexuslrf: decide whether to further improve it or not based on later use cases.
  */

  // assert(points.dim()==4 && "Points should be a 4-D tensor.");
  // int s_x = points.size(0), s_y = points.size(1), s_z = points.size(2);
  // int max_hit = s_x + s_y + s_z, n_pts = s_x * s_y * s_z;

  int n_pts = points.numel() / 3;
  int n_rays = rays_o.size(0);
  int rays_per_blk = 256;
  int n_blocks = (n_rays - 1) / rays_per_blk + 1;
  float half_voxel = voxelsize * 0.5; 
  at::Tensor idx =
      torch::zeros({n_rays, max_hit}, at::device(rays_o.device()).dtype(at::ScalarType::Int));
  at::Tensor min_depth =
      torch::zeros({n_rays, max_hit}, at::device(rays_o.device()).dtype(at::ScalarType::Float));
  at::Tensor max_depth =
      torch::zeros({n_rays, max_hit}, at::device(rays_o.device()).dtype(at::ScalarType::Float));
  at::DeviceIndex device_idx = rays_o.device().index();
  aabb_intersect_kernel_wrapper(n_blocks, rays_per_blk, n_pts, n_rays, max_hit, half_voxel, device_idx,
                                  rays_o.data_ptr <float>(), rays_d.data_ptr <float>(), points.data_ptr <float>(),
                                  idx.data_ptr <int>(), min_depth.data_ptr <float>(), max_depth.data_ptr <float>());
  return std::make_tuple(idx, min_depth, max_depth);
}
