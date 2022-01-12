#include "point_sample.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_uniform_sample", &voxel_uniform_sample);
  m.def("voxel_cdf_sample", &voxel_cdf_sample);
}