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
  float f_dim_low, f_dim_high, temp, inv_ray_dir, start, aabb;

  for (int d = 0; d < 3; ++d) {  
    switch (d) {
      case 0:
        inv_ray_dir = __fdividef(1.0f, dir.x); start = ori.x; aabb = center.x; break;
      case 1:
        inv_ray_dir = __fdividef(1.0f, dir.y); start = ori.y; aabb = center.y; break;
      case 2:
        inv_ray_dir = __fdividef(1.0f, dir.z); start = ori.z; aabb = center.z; break;
    }
  
    f_dim_low  = (aabb - half_voxel - start) * inv_ray_dir; // pay attention to half_voxel --> center -> corner.
    f_dim_high = (aabb + half_voxel - start) * inv_ray_dir;
  
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
    
    if (f_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
}

__global__ void aabb_intersect_point_kernel(
            int b, int n, int m, float voxelsize,
            int n_max,
            const float *__restrict__ ray_start,
            const float *__restrict__ ray_dir,
            const float *__restrict__ points,
            int *__restrict__ idx,
            float *__restrict__ min_depth,
            float *__restrict__ max_depth) {
  
  int batch_index = blockIdx.x;
  points += batch_index * n * 3; // points may not need expand..
  ray_start += batch_index * m * 3;
  ray_dir += batch_index * m * 3;
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
        make_float3(ray_start[j * 3 + 0], ray_start[j * 3 + 1], ray_start[j * 3 + 2]),
        make_float3(ray_dir[j * 3 + 0], ray_dir[j * 3 + 1], ray_dir[j * 3 + 2]),
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

void aabb_intersect_point_kernel_wrapper(
  int b, int n, int m, float voxelsize, int n_max,
  const float *ray_start, const float *ray_dir, const float *points,
  int *idx, float *min_depth, float *max_depth) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  aabb_intersect_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, voxelsize, n_max, ray_start, ray_dir, points, idx, min_depth, max_depth);
  
  CUDA_CHECK_ERRORS();
}

