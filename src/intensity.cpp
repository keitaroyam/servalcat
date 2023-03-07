// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <pybind11/numpy.h>
namespace py = pybind11;

double f1_exp2_value(double x, double z, double t0) {
  double ex = std::exp(-x);
  double exx = x - ex;
  double ex1 = std::exp(2.0 * exx);
  return (ex1 * ex1 - 2.0 * t0 * ex1) / 2.0 - 4 * z * exx - std::log(1.0 + ex);
}
double f1_orig2_value(double x, double z, double t0) {
  double x2 = x * x;
  return 0.5 * (x2 * x2 - 2 * t0 * x2) - (4 * z - 1) * std::log(x);
}

template<bool J1>
void add_integ_j(py::module& m, const char* name) {
  m.def(name, [](py::array_t<double> delta, py::array_t<double> root,
		 py::array_t<double> to1, py::array_t<double> f1val,
		 double z, int N, double ewmax) {
    const int size = (int) delta.shape(0);
    assert(size == root.shape(0));
    assert(size == to1.shape(0));
    assert(size == f1val.shape(0));
    auto d = delta.unchecked<1>();
    auto r = root.unchecked<1>();
    auto t = to1.unchecked<1>();
    auto f1 = f1val.unchecked<1>();
    auto result = py::array_t<double>(size);
    py::buffer_info buf = result.request();
    double *ptr = static_cast<double *>(buf.ptr);
    auto fun = J1 ? f1_exp2_value : f1_orig2_value;
    for (int j = 0; j < size; ++j) {
      double s = 1; // for i = 0
      for (int sign : {-1, 1}) {
        for (int i = 1; i < N; ++i) {
          const double xx = sign * d(j) * i + r(j);
          if (!J1 && xx <= 0) continue; //break?
          const double ff = fun(xx, z, t(j)) - f1(j);
          s += std::exp(-ff);
          if (ff > ewmax) break;
        }
      }
      ptr[j] = s;
    }
    return result;
  }, py::arg("delta"), py::arg("root"), py::arg("to1"), py::arg("f1val"), py::arg("z"),
     py::arg("N")=200, py::arg("ewmax")=20.);
}

template<bool J1>
void add_integ_j_ratio(py::module& m, const char* name) {
  m.def(name, [](py::array_t<double> delta, py::array_t<double> root,
		 py::array_t<double> to1, py::array_t<double> f1val,
		 double z, double deltaz, int N, double ewmax) {
    const int size = (int) delta.shape(0);
    assert(size == root.shape(0));
    assert(size == to1.shape(0));
    assert(size == f1val.shape(0));
    auto d = delta.unchecked<1>();
    auto r = root.unchecked<1>();
    auto t = to1.unchecked<1>();
    auto f1 = f1val.unchecked<1>();
    auto result = py::array_t<double>(size);
    py::buffer_info buf = result.request();
    double *ptr = static_cast<double *>(buf.ptr);
    auto fun1 = J1 ? f1_exp2_value : f1_orig2_value;
    
    for (int j = 0; j < size; ++j) {
      const double tmp = J1 ? r(j) - std::exp(-r(j)) : std::log(r(j));
      double sd = 1, sn = std::exp(4 * deltaz * tmp);
      for (int sign : {-1, 1}) {
        for (int i = 1; i < N; ++i) {
          const double xx = sign * d(j) * i + r(j);
	  if (!J1 && xx <= 0) continue; //break?
	  double ff = fun1(xx, z, t(j)) - f1(j);
          const double tmp = std::exp(-ff);
          const double g = std::exp(4 * deltaz * (J1 ? xx - std::exp(-xx) : std::log(xx)));
          sd += tmp;
          sn += tmp * g;
          if (ff > ewmax) break;
        }
      }
      ptr[j] = sn / sd;
    }
    return result;
  }, py::arg("delta"), py::arg("root"), py::arg("to1"), py::arg("f1val"), py::arg("z"), py::arg("deltaz"),
     py::arg("N")=200, py::arg("ewmax")=20.);
}

void add_intensity(py::module& m) {
  add_integ_j<true>(m, "integ_J_1");
  add_integ_j<false>(m, "integ_J_2");
  add_integ_j_ratio<true>(m, "integ_J_ratio_1");
  add_integ_j_ratio<false>(m, "integ_J_ratio_2");

}
