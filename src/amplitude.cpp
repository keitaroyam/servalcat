// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <gemmi/bessel.hpp>
#include <gemmi/math.hpp>
#include "math.hpp"
namespace py = pybind11;
using namespace servalcat;

// ML amplitude target; -log(|Fo|; Fc) without constants
// https://doi.org/10.1107/S0907444911001314 eqn (4)
// S: must include epsilon.
// Fc = |sum D*Fc|
double ll_amp(double Fo, double sigFo, double k_ani, double S, double Fc, int c) {
  assert(c == 1 || c == 2); // c=1: acentric, 2: centric
  if (std::isnan(Fo) || S <= 0) return NAN;
  const double Fo_iso = Fo / k_ani;
  const double Sigma = (3 - c) * sq(sigFo / k_ani) + S;
  const double log_ic0 = log_i0_or_cosh(Fo_iso * Fc / Sigma, c);
  const double tmp = (c == 1) ? std::log(2.) + std::log(Fo_iso) : std::log(2 / gemmi::pi()) / c; // don't need this
  return std::log(Sigma) / c + (sq(Fo_iso) + sq(Fc)) / (Sigma * c) - log_ic0 - tmp;
}

py::array_t<double>
ll_amp_der1_params_py(py::array_t<double> Fo, py::array_t<double> sigFo, py::array_t<double> k_ani,
                      double S, py::array_t<std::complex<double>> Fcs, std::vector<double> Ds,
                      py::array_t<int> c, py::array_t<int> eps) {
  if (Ds.size() != (size_t)Fcs.shape(1)) throw std::runtime_error("Fc and D shape mismatch");
  const size_t n_models = Fcs.shape(1);
  const size_t n_ref = Fcs.shape(0);
  const size_t n_cols = n_models + 1; //for_DS ? n_models + 1 : 1;
  auto Fo_ = Fo.unchecked<1>();
  auto sigFo_ = sigFo.unchecked<1>();
  auto k_ani_ = k_ani.unchecked<1>();
  auto Fcs_ = Fcs.unchecked<2>();
  auto c_ = c.unchecked<1>();
  auto eps_ = eps.unchecked<1>();

  // der1 wrt D1, D2, .., S //, or k_ani
  auto ret = py::array_t<double>({n_ref, n_cols});
  double* ptr = (double*) ret.request().ptr;
  auto sum_Fc = [&](int i) {
                  std::complex<double> s = Fcs_(i, 0) * Ds[0];
                  for (size_t j = 1; j < n_models; ++j)
                    s += Fcs_(i, j) * Ds[j];
                  return s;
                };
  for (size_t i = 0; i < n_ref; ++i) {
    if (S <= 0 || std::isnan(Fo_(i))) {
      for (size_t j = 0; j < n_cols; ++j)
        ptr[i * n_cols + j] = NAN;
      continue;
    }
    const double Fo_iso = Fo_(i) / k_ani_(i);
    const double Sigma = (3 - c_(i)) * sq(sigFo_(i) / k_ani_(i)) + S * eps_(i);
    const std::complex<double> Fc_total_conj = std::conj(sum_Fc(i));
    const double Fc_abs = std::abs(Fc_total_conj);
    if (1){ //for_DS) {
      const double m = fom(Fo_iso * Fc_abs / Sigma, c_(i));
      for (size_t j = 0; j < n_models; ++j) {
        const double r_fcj_fc = (Fcs_(i, j) * Fc_total_conj).real();
        // wrt Dj
        ptr[i*n_cols + j] = 2 * r_fcj_fc / (Sigma * c_(i)) * (1. - m * Fo_iso / Fc_abs);
      }
      // wrt S
      const double tmp = (sq(Fo_iso) + sq(Fc_abs)) / c_(i) - m * (3 - c_(i)) * Fo_iso * Fc_abs;
      ptr[i*n_cols + n_models] = eps_(i) * (1. / (c_(i) * Sigma) - tmp / sq(Sigma));
    }
    else {
      // k_aniso * d/dk_aniso -log p(Io; Fc)
      // note k_aniso is multiplied to the derivative
      // not implemented
    }
  }
  return ret;
}

void add_amplitude(py::module& m) {
  m.def("ll_amp", py::vectorize(ll_amp),
        py::arg("Fo"), py::arg("sigFo"), py::arg("k_ani"), py::arg("S"), py::arg("Fc"), py::arg("c"));
  m.def("ll_amp_der1_DS", &ll_amp_der1_params_py);
  //m.def("ll_int_der1_ani", &ll_int_der1_params_py<false>);
}
