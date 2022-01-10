#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

__device__ float2 RayAABBIntersection(
  const float3 &ori,
  const float3 &dir,
  const float3 &center,
  float half_voxel) {

  float f_low = 0;
  float f_high = 100000.;
  float f_dim_low, f_dim_high, temp, inv_rays_d, start, aabb;

  for (int d = 0; d < 3; ++d) {  
    switch (d) {
      case 0:
        inv_rays_d = __fdividef(1.0f, dir.x); start = ori.x; aabb = center.x; break;
      case 1:
        inv_rays_d = __fdividef(1.0f, dir.y); start = ori.y; aabb = center.y; break;
      case 2:
        inv_rays_d = __fdividef(1.0f, dir.z); start = ori.z; aabb = center.z; break;
    }
  
    f_dim_low  = (aabb - half_voxel - start) * inv_rays_d; // pay attention to half_voxel --> center -> corner.
    f_dim_high = (aabb + half_voxel - start) * inv_rays_d;
  
    // Make sure low is less than high
    if (f_dim_high < f_dim_low) {
      temp = f_dim_low;
      f_dim_low = f_dim_high;
      f_dim_high = temp;
    }

    // If this dimension's high is less than the low we got then we definitely missed.
    if (f_dim_high < f_low) {
      return make_float2(-1.0f, -1.0f);
    }
  
    // Likewise if the low is less than the high.
    if (f_dim_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
      
    // Add the clip from this dimension to the previous results 
    f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
    f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
    
    if (f_low >= f_high) { // The corner case: if f_low == f_high, also treated as miss.
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
}

__global__ void aabb_intersect_old_kernel(
            int b, int n, int m, float voxelsize,
            int n_max,
            const float *__restrict__ rays_o,
            const float *__restrict__ rays_d,
            const float *__restrict__ points,
            int *__restrict__ idx,
            float *__restrict__ min_depth,
            float *__restrict__ max_depth) {
  
  int batch_index = blockIdx.x;
  rays_o += batch_index * m * 3;
  rays_d += batch_index * m * 3;
  idx += batch_index * m * n_max;
  min_depth += batch_index * m * n_max;
  max_depth += batch_index * m * n_max;
    
  int index = threadIdx.x;
  int stride = blockDim.x;
  float half_voxel = voxelsize * 0.5; 

  for (int j = index; j < m; j += stride) {
    for (int l = 0; l < n_max; ++l) {
      idx[j * n_max + l] = -1;
    }

    for (int k = 0, cnt = 0; k < n && cnt < n_max; ++k) {
      float2 depths = RayAABBIntersection(
        make_float3(rays_o[j * 3 + 0], rays_o[j * 3 + 1], rays_o[j * 3 + 2]),
        make_float3(rays_d[j * 3 + 0], rays_d[j * 3 + 1], rays_d[j * 3 + 2]),
        make_float3(points[k * 3 + 0], points[k * 3 + 1], points[k * 3 + 2]),
        half_voxel);

      if (depths.x > -1.0f){
        idx[j * n_max + cnt] = k;
        min_depth[j * n_max + cnt] = depths.x;
        max_depth[j * n_max + cnt] = depths.y;
        ++cnt;
      }
    }
  }
}

void aabb_intersect_old_kernel_wrapper(
  int b, int n, int m, float voxelsize, int n_max,
  const float *rays_o, const float *rays_d, const float *points,
  int *idx, float *min_depth, float *max_depth) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  aabb_intersect_old_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, voxelsize, n_max, rays_o, rays_d, points, idx, min_depth, max_depth);
  
  CUDA_CHECK_ERRORS();
}

__global__ void aabb_intersect_kernel(
            int rays_per_blk, int n_pts, int n_rays,
            int max_hit, float radius,
            const float *__restrict__ rays_o,
            const float *__restrict__ rays_d,
            const float *__restrict__ points,
            int *__restrict__ idx,
            float *__restrict__ min_depth,
            float *__restrict__ max_depth) {
  
  int blk_index = blockIdx.x;
  int offset = blk_index * rays_per_blk;
  int index = threadIdx.x + offset;
  int stride = blockDim.x;

  for (int j = index; j < offset + rays_per_blk && j < n_rays; j += stride) {
    for (int l = 0; l < max_hit; ++l) {
      idx[j * max_hit + l] = -1;
    }

    for (int k = 0, cnt = 0; k < n_pts && cnt < max_hit; ++k) {
      float2 depths = RayAABBIntersection(
        make_float3(rays_o[j * 3 + 0], rays_o[j * 3 + 1], rays_o[j * 3 + 2]),
        make_float3(rays_d[j * 3 + 0], rays_d[j * 3 + 1], rays_d[j * 3 + 2]),
        make_float3(points[k * 3 + 0], points[k * 3 + 1], points[k * 3 + 2]),
        radius);

      if (depths.x > -1.0f){
        idx[j * max_hit + cnt] = k;
        min_depth[j * max_hit + cnt] = depths.x;
        max_depth[j * max_hit + cnt] = depths.y;
        ++cnt;
      }
    }
  }
}

void aabb_intersect_kernel_wrapper(
  int n_blk, int rays_per_blk, int n_pts, int n_rays, int max_hit, float radius,
  const float *rays_o, const float *rays_d, const float *points,
  int *idx, float *min_depth, float *max_depth) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  aabb_intersect_kernel<<<n_blk, opt_n_threads(rays_per_blk), 0, stream>>>(
    rays_per_blk, n_pts, n_rays, max_hit, radius, 
    rays_o, rays_d, points, idx, min_depth, max_depth);
  
  CUDA_CHECK_ERRORS();
}
