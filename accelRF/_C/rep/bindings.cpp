#include "ray_intersect.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aabb_intersect", &aabb_intersect);
  m.def("aabb_intersect_old", &aabb_intersect_old);
}