#include "point_sample.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("uniform_ray_sampling", &uniform_ray_sampling);
  m.def("inverse_cdf_sampling", &inverse_cdf_sampling);
}