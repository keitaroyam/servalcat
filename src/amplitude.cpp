// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>
#include <gemmi/bessel.hpp>
#include <gemmi/math.hpp>
#include "math.hpp"
#include "array.h"
namespace nb = nanobind;
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

auto
ll_amp_der1_params_py(np_array<double> Fo, np_array<double> sigFo, np_array<double> k_ani,
                      double S, np_array<std::complex<double>, 2> Fcs, std::vector<double> Ds,
                      np_array<int> c, np_array<int> eps, np_array<double> w) {
  auto Fo_ = Fo.view();
  auto sigFo_ = sigFo.view();
  auto k_ani_ = k_ani.view();
  auto Fcs_ = Fcs.view();
  auto c_ = c.view();
  auto eps_ = eps.view();
  auto w_ = w.view();
  if (Ds.size() != Fcs_.shape(1)) throw std::runtime_error("Fc and D shape mismatch");
  const size_t n_models = Fcs_.shape(1);
  const size_t n_ref = Fcs_.shape(0);
  const size_t n_cols = n_models + 1; //for_DS ? n_models + 1 : 1;

  // der1 wrt D1, D2, .., S //, or k_ani
  auto ret = make_numpy_array<double>({n_ref, n_cols});
  double* ptr = ret.data();
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
        ptr[i*n_cols + j] = w_(i) * 2 * r_fcj_fc / (Sigma * c_(i)) * (1. - m * Fo_iso / Fc_abs);
      }
      // wrt S
      const double tmp = (sq(Fo_iso) + sq(Fc_abs)) / c_(i) - m * (3 - c_(i)) * Fo_iso * Fc_abs;
      ptr[i*n_cols + n_models] = w_(i) * eps_(i) * (1. / (c_(i) * Sigma) - tmp / sq(Sigma));
    }
    else {
      // k_aniso * d/dk_aniso -log p(Io; Fc)
      // note k_aniso is multiplied to the derivative
      // not implemented
    }
  }
  return ret;
}

void add_amplitude(nb::module_& m) {
  m.def("ll_amp", [](np_array<double> Fo, np_array<double> sigFo, np_array<double> k_ani,
                     np_array<double> S, np_array<double> Fc, np_array<int> c, np_array<double> w) {
    auto Fo_ = Fo.view();
    auto sigFo_ = sigFo.view();
    auto k_ani_ = k_ani.view();
    auto S_ = S.view();
    auto Fc_ = Fc.view();
    auto c_ = c.view();
    auto w_ = w.view();
    size_t len = Fo_.shape(0);
    if (len != sigFo_.shape(0) || len != k_ani_.shape(0) || len != S_.shape(0) ||
        len != Fc_.shape(0) || len != c_.shape(0) || len != w_.shape(0))
      throw std::runtime_error("ll_amp: shape mismatch");
    auto ret = make_numpy_array<double>({len});
    double* retp = ret.data();
    for (size_t i = 0; i < len; ++i)
      retp[i] = w_(i) * ll_amp(Fo_(i), sigFo_(i), k_ani_(i), S_(i), Fc_(i), c_(i));
    return ret;
  },
        nb::arg("Fo"), nb::arg("sigFo"), nb::arg("k_ani"), nb::arg("S"), nb::arg("Fc"), nb::arg("c"), nb::arg("w"));
  m.def("ll_amp_der1_DS", &ll_amp_der1_params_py);
  //m.def("ll_int_der1_ani", &ll_int_der1_params_py<false>);
}
