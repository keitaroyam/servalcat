// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <pybind11/stl.h>
#include <pybind11/numpy.h>    // for vectorize

namespace py = pybind11;
void add_refine(py::module& m); // refine.cpp

PYBIND11_MODULE(ext, mg) {
  mg.doc() = "Servalcat extension";

  add_refine(mg);
}
