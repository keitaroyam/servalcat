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
double integ_j_fw(double delta, double root, double to1,
                  double f1val, double z, int N, double ewmax) {
  auto fun = J1 ? f1_exp2_value : f1_orig2_value;
  double s = 1; // for i = 0
  for (int sign : {-1, 1}) {
    for (int i = 1; i < N; ++i) {
      const double xx = sign * delta * i + root;
      if (!J1 && xx <= 0) continue; //break?
      const double ff = fun(xx, z, to1) - f1val;
      s += std::exp(-ff);
      if (ff > ewmax) break;
    }
  }
  return s;
}

template<bool J1>
double integ_j_ratio_fw(double delta, double root, double to1,
                        double f1val, double z, double deltaz, int N, double ewmax) {
  auto fun1 = J1 ? f1_exp2_value : f1_orig2_value;
  const double tmp = J1 ? root - std::exp(-root) : std::log(root);
  double sd = 1, sn = std::exp(4 * deltaz * tmp);
  for (int sign : {-1, 1}) {
    for (int i = 1; i < N; ++i) {
      const double xx = sign * delta * i + root;
      if (!J1 && xx <= 0) continue; //break?
      double ff = fun1(xx, z, to1) - f1val;
      const double tmp = std::exp(-ff);
      const double g = std::exp(4 * deltaz * (J1 ? xx - std::exp(-xx) : std::log(xx)));
      sd += tmp;
      sn += tmp * g;
      if (ff > ewmax) break;
    }
  }
  return sn / sd;
}

void add_intensity(py::module& m) {
  m.def("integ_J_1_fw", py::vectorize(integ_j_fw<true>),
        py::arg("delta"), py::arg("root"), py::arg("to1"), py::arg("f1val"), py::arg("z"),
        py::arg("N")=200, py::arg("ewmax")=20.);
  m.def("integ_J_2_fw", py::vectorize(integ_j_fw<false>),
        py::arg("delta"), py::arg("root"), py::arg("to1"), py::arg("f1val"), py::arg("z"),
        py::arg("N")=200, py::arg("ewmax")=20.);
  m.def("integ_J_ratio_1_fw", py::vectorize(integ_j_ratio_fw<true>),
        py::arg("delta"), py::arg("root"), py::arg("to1"), py::arg("f1val"), py::arg("z"), py::arg("deltaz"),
        py::arg("N")=200, py::arg("ewmax")=20.);
  m.def("integ_J_ratio_2_fw", py::vectorize(integ_j_ratio_fw<false>),
        py::arg("delta"), py::arg("root"), py::arg("to1"), py::arg("f1val"), py::arg("z"), py::arg("deltaz"),
        py::arg("N")=200, py::arg("ewmax")=20.);
}
