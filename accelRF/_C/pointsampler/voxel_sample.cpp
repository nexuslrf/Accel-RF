#include "point_sample.h"
#include "utils.h"
#include <utility> 


void voxel_uniform_sample_kernel_wrapper(
  int b, int rays_per_blk, int max_hits, int max_steps, float step_size,
  const int *pts_idx, const float *min_depth, const float *max_depth, const float *uniform_noise,
  int *sampled_idx, float *sampled_depth, float *sampled_dists);

void voxel_cdf_sample_kernel_wrapper(
  int b, int rays_per_blk, int n_rays, int max_hits, int max_steps, float fixed_step_size,
  const int *pts_idx, const float *min_depth, const float *max_depth,
  const float *uniform_noise, const float *probs, const float *steps,
  int *sampled_idx, float *sampled_depth, float *sampled_dists);

                  
std::tuple< at::Tensor, at::Tensor, at::Tensor> voxel_uniform_sample(
  at::Tensor pts_idx, at::Tensor min_depth, at::Tensor max_depth, at::Tensor uniform_noise,
  const float step_size, const int max_steps){

  CHECK_CONTIGUOUS(pts_idx);
  CHECK_CONTIGUOUS(min_depth);
  CHECK_CONTIGUOUS(max_depth);
  CHECK_CONTIGUOUS(uniform_noise);
  CHECK_IS_FLOAT(min_depth);
  CHECK_IS_FLOAT(max_depth);
  CHECK_IS_FLOAT(uniform_noise);
  CHECK_IS_INT(pts_idx);
  CHECK_CUDA(pts_idx);
  CHECK_CUDA(min_depth);
  CHECK_CUDA(max_depth);
  CHECK_CUDA(uniform_noise);

  at::Tensor sampled_idx =
      -torch::ones({pts_idx.size(0), pts_idx.size(1), max_steps},
                    at::device(pts_idx.device()).dtype(at::ScalarType::Int));
  at::Tensor sampled_depth =
      torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                    at::device(min_depth.device()).dtype(at::ScalarType::Float));
  at::Tensor sampled_dists =
      torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                    at::device(min_depth.device()).dtype(at::ScalarType::Float));
  voxel_uniform_sample_kernel_wrapper(min_depth.size(0), min_depth.size(1), min_depth.size(2), sampled_depth.size(2),
                                      step_size,
                                      pts_idx.data_ptr <int>(), min_depth.data_ptr <float>(), max_depth.data_ptr <float>(),
                                      uniform_noise.data_ptr <float>(), sampled_idx.data_ptr <int>(), 
                                      sampled_depth.data_ptr <float>(), sampled_dists.data_ptr <float>());
  return std::make_tuple(sampled_idx, sampled_depth, sampled_dists);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> voxel_cdf_sample(
    at::Tensor pts_idx, at::Tensor min_depth, at::Tensor max_depth, at::Tensor uniform_noise,
    at::Tensor probs, at::Tensor steps, float fixed_step_size) {
  
  CHECK_CONTIGUOUS(pts_idx);
  CHECK_CONTIGUOUS(min_depth);
  CHECK_CONTIGUOUS(max_depth);
  CHECK_CONTIGUOUS(probs);
  CHECK_CONTIGUOUS(steps);
  CHECK_CONTIGUOUS(uniform_noise);
  CHECK_IS_FLOAT(min_depth);
  CHECK_IS_FLOAT(max_depth);
  CHECK_IS_FLOAT(uniform_noise);
  CHECK_IS_FLOAT(probs);
  CHECK_IS_FLOAT(steps);
  CHECK_IS_INT(pts_idx);
  CHECK_CUDA(pts_idx);
  CHECK_CUDA(min_depth);
  CHECK_CUDA(max_depth);
  CHECK_CUDA(uniform_noise);
  CHECK_CUDA(probs);
  CHECK_CUDA(steps);

  int max_steps = uniform_noise.size(-1), max_hits = min_depth.size(-1);
  int n_rays = pts_idx.size(0);
  int rays_per_blk = 128;
  int n_blocks = (n_rays - 1) / rays_per_blk + 1;
  at::Tensor sampled_idx =
      -torch::ones({n_rays, max_steps}, at::device(pts_idx.device()).dtype(at::ScalarType::Int));
  at::Tensor sampled_depth =
      torch::zeros({n_rays, max_steps}, at::device(min_depth.device()).dtype(at::ScalarType::Float));
  at::Tensor sampled_dists =
      torch::zeros({n_rays, max_steps}, at::device(min_depth.device()).dtype(at::ScalarType::Float));
  voxel_cdf_sample_kernel_wrapper(n_blocks, rays_per_blk, n_rays, max_hits, max_steps, fixed_step_size,
                                      pts_idx.data_ptr <int>(), min_depth.data_ptr <float>(), max_depth.data_ptr <float>(),
                                      uniform_noise.data_ptr <float>(), probs.data_ptr <float>(), steps.data_ptr <float>(),
                                      sampled_idx.data_ptr <int>(), sampled_depth.data_ptr <float>(), sampled_dists.data_ptr <float>());
  return std::make_tuple(sampled_idx, sampled_depth, sampled_dists);
}