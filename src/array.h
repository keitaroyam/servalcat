#pragma once
#include <nanobind/ndarray.h>
namespace nb = nanobind;

template<class T, size_t N=1>
using np_array = nb::ndarray<T, nb::ndim<N>, nb::device::cpu>;

// gemmi python/array.h
template<typename T>
auto make_numpy_array(std::initializer_list<size_t> size,
                      std::initializer_list<int64_t> strides={}) {
  size_t total_size = 1;
  for (size_t i : size)
    total_size *= i;
  T* c_array = new T[total_size];
  nb::capsule owner(c_array, [](void* p) noexcept { delete [] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T>(c_array, size, owner, strides);
}
