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

// for MLI
double f1_orig2(double x, double z, double to, double tf, double sig1, int c) {
  // z = 2k+1
  const double x2 = x * x;
  const double ret = 0.5 * x2 * x2 - to * x2 - z * std::log(x);
  if (tf == 0.) return ret;
  const double X = x * tf / sig1;
  const double log_ic0 = c == 1 ? gemmi::log_bessel_i0(2*X) : gemmi::log_cosh(X);
  return ret - log_ic0;
}
double f1_orig2_der1(double x, double z, double to, double tf, double sig1, int c) {
  const double ret = 2 * x*x*x - 2 * to * x - z / x;
  if (tf == 0.) return ret;
  const double X = x * tf / sig1;
  const double m = fom(X, c);
  return ret - m * (3-c) * tf / sig1;
}
double f1_orig2_der2(double x, double z, double to, double tf, double sig1, int c) {
  const double x2 = x * x;
  const double ret = 6 * x2 - 2 * to + z / x2;
  if (tf == 0.) return ret;
  const double X = x * tf / sig1;
  const double m = fom(X, c);
  const double m_der = fom_der(m, X, c);
  const double ret2 = ret - (3-c) * tf*tf * m_der / (sig1*sig1);
  // if (ret2 <= 0)
  //   printf("f1_orig2_der2 %f x %f z= %f; to= %f; tf= %f; sig1= %f; c= %d\n", ret2, x, z, to, tf, sig1, c);
  return ret2;
}
double f1_exp2(double y, double z, double to, double tf, double sig1, int c) {
  // z = 2k+2
  const double e_y = std::exp(-y);
  const double exp2 = std::exp(2 * (y - e_y));
  const double ret = 0.5 * exp2*exp2 - to * exp2 - z * (y - e_y) - std::log(1 + e_y);
  if (tf == 0.) return ret;
  const double X = std::exp(y - e_y) * tf / sig1;
  const double log_ic0 = c == 1 ? gemmi::log_bessel_i0(2*X) : gemmi::log_cosh(X);
  return ret - log_ic0;
}
double f1_exp2_der1(double y, double z, double to, double tf, double sig1, int c) {
  const double e_y = std::exp(-y);
  const double exp2 = std::exp(2 * (y - e_y));
  const double ret = (1 + e_y) * (2 * exp2*exp2 - 2 * to * exp2 - z) + e_y / (1 + e_y);
  if (tf == 0.) return ret;
  const double X = std::exp(y - e_y) * tf / sig1;
  const double m = fom(X, c);
  return ret - (1 + e_y) * ((3-c) * tf / sig1 * m * std::exp(y - e_y));
}
double f1_exp2_der2(double y, double z, double to, double tf, double sig1, int c) {
  const double e_y = std::exp(-y);
  const double exp2 = std::exp(2 * (y - e_y));
  const double ret = -e_y * (2 * exp2*exp2 - 2 * to * exp2 - z) + sq(1 + e_y) * (8 * sq(exp2) - 4 * to * exp2) - e_y / sq(1 + e_y);
  if (tf == 0.) return ret;
  const double X = std::exp(y - e_y) * tf / sig1;
  const double m = fom(X, c);
  const double m_der = (tf == 0 || m > 0.9999) ? 0 : fom_der(m, X, c);
  const double ret2 = ret + e_y * (3-c) * tf / sig1 * m * std::exp(y - e_y) - sq(1 + e_y) * ((3-c) * tf / sig1 * m * std::exp(y - e_y) + (3-c) * sq(tf / sig1) * m_der * exp2);
  // if (ret2 <= 0)
  //   printf("f1_exp2_der2 %f y %f z= %f; to= %f; tf= %f; sig1= %f; c= %d\n", ret2, y, z, to, tf, sig1, c);
  return ret2;
}
double find_root(double k, double to, double tf, double sig1, int c, double det, bool use_exp2) {
  if (tf == 0.) {
    if (use_exp2)
      return solve_y_minus_exp_minus_y(0.5 * std::log(0.5 * x_plus_sqrt_xsq_plus_y(to, 4 * k + 4)), 1e-4); // XXX may not be exact
    return det;
  }
  const double z = use_exp2 ? 2 * k + 2 : 2 * k + 1;
  auto fder1 = use_exp2 ? f1_exp2_der1 : f1_orig2_der1;
  auto fder2 = use_exp2 ? f1_exp2_der2 : f1_orig2_der2;
  double x0 = det, x1 = NAN;
  if (use_exp2) {
    const double X1 = tf * 0.5 / sig1 * std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 4 * k + 3.5));
    const double m1 = fom(X1, c);
    x0 = solve_y_minus_exp_minus_y(0.5 * std::log(0.5 * x_plus_sqrt_xsq_plus_y(to, 4 * k + 4 * (3-c) * X1 * m1 + 3.5)), 1e-4);
    if (std::isnan(x0))
      printf("ERROR: x0= %e, use_exp2= %d, X1=%e, m1=%e in_log=%e\n", x0, use_exp2, X1, m1,
             0.5 * x_plus_sqrt_xsq_plus_y(to, 4 * k + 4 * (3-c) * X1 * m1 + 3.5));
    const double A = std::max(to, std::max((3-c) * tf / 4 / sig1, k + 1));
    x1 = solve_y_minus_exp_minus_y(std::log(0.5 * (std::sqrt(A) + std::sqrt(A + 4 * std::sqrt(A)))), 1e-4);
  } else {
    const double A = std::max(to, std::max((3-c) * tf / 4 / sig1, k + 0.5));
    x1 = 0.5 * (std::sqrt(A) + std::sqrt(A + 4 * std::sqrt(A)));
  }
  if (std::isinf(x0) || std::isnan(x0))
    printf("ERROR: x0= %e, use_exp2= %d, z=%e, to=%e, tf=%e, sig1=%e, c=%d, det=%e\n", x0, use_exp2, z, to, tf, sig1, c, det);
  auto func = [&](double x) { return fder1(x, z, to, tf, sig1, c); };
  auto fprime = [&](double x) { return fder2(x, z, to, tf, sig1, c); };
  //printf("debug %d x0= %f x1= %f f'_x0= %f f'_x1= %f\n", use_exp2, x0, x1, func(x0), func(x1));
  if (std::abs(func(x0)) < 1e-4) // need to check. 1e-2 is small enough?
    return x0;
  auto ret = newton(func, fprime, x0);
  if (ret.success && std::abs(func(ret.x)) < 1e-4)
    return ret.x;
  //   printf("newton_success iter %d k=%f; to=%f; tf=%f; sig1=%f; c=%d; det=%f; use_exp2=%d\n", ret.iter, k, to, tf,sig1, c, det, use_exp2);

  double a = x0, b = std::isnan(x1) ? x0 : x1;
  double fa = func(a), fb = func(b);
  if (fa * fb > 0) { // should not happen, just for safety
    printf("DEBUG: bad_interval x0= %e, x1= %e, use_exp2= %d, z=%e, to=%e, tf=%e, sig1=%e, c=%d, det=%e\n",
           x0, x1, use_exp2, z, to, tf, sig1, c, det);
    double inc = fa < 0 ? 0.1 : -0.1;
    for (int i = 0; i < 10000; ++i, b+=inc) { // to prevent infinite loop
      fb = func(b);
      if (fa * fb < 0) break;
    }
    if (fa * fb >= 0) throw std::runtime_error("interval not found");
  }
  ret = bisect(func, a, b, 10000, 1e-4);
  if (!ret.success) {
    printf("DEBUG: bisect_fail x0= %e, x1= %e, use_exp2= %d, z=%e, to=%e, tf=%e, sig1=%e, c=%d, det=%e\n",
           x0, x1, use_exp2, z, to, tf, sig1, c, det);
    return NAN;
  }
  // printf("bisect_success %f iter %d k=%f; to=%f; tf=%f; sig1=%f; c=%d; det=%f; use_exp2=%d\n", func(ret.x), ret.iter, k, to, tf,sig1, c, det, use_exp2);
  return ret.x;
}

// factor of (sig/k_ani^2)^(k+1) is needed, which should be done outside.
double integ_j(double k, double to, double tf, double sig1, int c, bool return_log,
               double exp2_threshold=3., double h=0.5, int N=200, double ewmax=20.) {
  if (std::isnan(to) || sig1 <= 0) return NAN; // perhaps sig1<0 should not return nan.. they considered 0 in sum
  const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k + 1)));
  const bool use_exp2 = det < exp2_threshold;
  const double z = use_exp2 ? 2 * k + 2 : 2 * k + 1;
  auto f = use_exp2 ? f1_exp2 : f1_orig2;
  auto fder2 = use_exp2 ? f1_exp2_der2 : f1_orig2_der2;
  const double root = find_root(k, to, tf, sig1, c, det, use_exp2);
  const double f1val = f(root, z, to, tf, sig1, c);
  const double f2der = fder2(root, z, to, tf, sig1, c);
  if (f2der <= 0)
    printf("integ_j bad f2der %f exp2=%d root=%f z=%f; to=%f; tf=%f; sig1=%f; c=%d;\n", f2der, use_exp2, root, z, to, tf, sig1, c);
  if (f2der * f1val * f1val > 1e10) { // Laplace approximation threshould needs to be revisited
    if (use_exp2) {
      const double lap = -f1val - 0.5 * std::log(f2der) + 0.5 * std::log(gemmi::pi() * 0.5);
      if (std::isinf(lap))
        printf("log integ_j (Laplace) inf f1val %f f2der %f exp2=%d root=%f z=%f; to=%f; tf=%f; sig1=%f; c=%d;\n", f1val, f2der, use_exp2, root, z, to, tf, sig1, c);
      return return_log ? lap : std::exp(lap);
    }
  }
  const double delta = h * std::sqrt(2 / f2der);
  double s = 1; // for i = 0
  for (int sign : {-1, 1}) {
    for (int i = 1; i < N; ++i) {
      const double xx = sign * delta * i + root;
      if (!use_exp2 && xx <= 0) continue; //break?
      const double ff = f(xx, z, to, tf, sig1, c) - f1val;
      s += std::exp(-ff);
      if (ff > ewmax) break;
    }
  }
  const double laplace_correct = s * h;
  const double expon = -f1val + 0.5 * (std::log(2.) - std::log(f2der));
  if (std::isinf(expon + std::log(laplace_correct)))
    printf("log integ_j inf f1val %f laplace_correct %f f2der %f exp2=%d root=%f z=%f; to=%f; tf=%f; sig1=%f; c=%d;\n", f1val, laplace_correct, f2der, use_exp2, root, z, to, tf, sig1, c);
  return return_log ? expon + std::log(laplace_correct) : std::exp(expon) * laplace_correct;
}

double integ_j_ratio(double k_num, double k_den, bool l, double to, double tf, double sig1, int c,
                     double exp2_threshold=3., double h=0.5, int N=200, double ewmax=20.) {
  // factor of sig^{k_num - k_den} is needed, which should be done outside.
  if (std::isnan(to) || sig1 <= 0) return NAN;
  const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
  const bool use_exp2 = det < exp2_threshold;
  const double z = use_exp2 ? 2 * k_den + 2 : 2 * k_den + 1;
  const double deltaz = 2 * (k_num - k_den);
  auto f = use_exp2 ? f1_exp2 : f1_orig2;
  auto fder2 = use_exp2 ? f1_exp2_der2 : f1_orig2_der2;
  const double root = find_root(k_den, to, tf, sig1, c, det, use_exp2);
  const double f1val = f(root, z, to, tf, sig1, c);
  const double f2der = fder2(root, z, to, tf, sig1, c);
  if (f2der <= 0)
    printf("integ_j_ratio bad f2der %f exp2=%d root=%f z=%f; to=%f; tf=%f; sig1=%f; c=%d;\n", f2der, use_exp2, root, z, to, tf, sig1, c);
  const double delta = h * std::sqrt(2 / f2der);
  auto calc_fom = [&](double xx) { return fom((use_exp2 ? std::exp(xx - std::exp(-xx)) : xx) * tf / sig1, c); };
  const double tmp = use_exp2 ? root - std::exp(-root) : std::log(root);
  double sd = 1, sn = std::exp(deltaz * tmp);
  if (l) sn *= calc_fom(root);
  for (int sign : {-1, 1}) {
    for (int i = 1; i < N; ++i) {
      const double xx = sign * delta * i + root;
      if (!use_exp2 && xx <= 0) continue; //break?
      const double ff = f(xx, z, to, tf, sig1, c) - f1val;
      const double tmp = std::exp(-ff);
      double g = std::exp(deltaz * (use_exp2 ? xx - std::exp(-xx) : std::log(xx)));
      if (l) g *= calc_fom(xx);
      sd += tmp;
      sn += tmp * g;
      if (ff > ewmax) break;
    }
  }
  return sn / sd;
}

// d/dDj -log(Io; Fc)
// note Re(Fcj Fc*) needs to be multiplied
double ll_int_der1_D(double S, double Fc, int c, double eps, double j_ratio_2) {
  const double invepsS = 1. / (S * eps);
  if (c == 1) // acentrics
    return (2 - (3-c) / Fc * j_ratio_2) * invepsS;
  else
    return (1. - (3-c) * j_ratio_2 / Fc) * invepsS;
}
// d/dS -log(Io; Fc)
double ll_int_der1_S(double S, double Fc, int c, double eps, double j_ratio_1, double j_ratio_2) {
  const double invepsS2 = 1. / (S * S * eps);
  if (c == 1) // acentrics
    return 1. / S - (sq(Fc) + j_ratio_1 / c - (3-c) * Fc * j_ratio_2) * invepsS2;
  else
    return 0.5 / S - 0.5 * sq(Fc) * invepsS2 - (j_ratio_1 / c - (3-c) * Fc * j_ratio_2) * invepsS2;
}

struct IntensityIntegrator {
  double h = 0.5;
  int N = 200;
  double ewmax = 20.;
  double exp2_threshold = 3.;

  // ML intensity target; -log(Io; Fc) without constants
  double ll_int(double Io, double sigIo, double k_ani, double S, double Fc, int c) const {
    if (std::isnan(Io) || S <= 0) return NAN;
    const double k = c == 1 ? 0 : -0.5;
    const double to = Io / sigIo - sigIo / c / sq(k_ani) / S;
    const double Ic = sq(Fc);
    const double tf = k_ani * Fc / std::sqrt(sigIo);
    const double sig1 = sq(k_ani) * S / sigIo;
    if (sig1 < 0)
      printf("ERROR: negative sig1= %f k_ani= %f S= %f sigIo= %f\n", sig1, k_ani, S, sigIo);
    const double logj = integ_j(k, to, tf, sig1, c, true, exp2_threshold, h, N, ewmax);
    const double ret = (3-c) * std::log(k_ani) + (std::log(S) + Ic / S) / c - logj;
    if (std::isinf(ret))
      printf("-LL Inf S=%f; logj=%f; to=%f; tf=%f; sig1=%f; k=%f; c=%d\n", S, logj, to, tf, sig1, k, c);
    return ret;
  }

  template<bool for_DS>
  nb::ndarray<nb::numpy, double>
  ll_int_der1_params_py(np_array<double> Io, np_array<double> sigIo, np_array<double> k_ani,
                        double S, np_array<std::complex<double>, 2> Fcs, std::vector<double> Ds,
                        np_array<int> c, np_array<int> eps, np_array<double> w) const {
    auto Io_ = Io.view();
    auto sigIo_ = sigIo.view();
    auto k_ani_ = k_ani.view();
    auto Fcs_ = Fcs.view();
    auto c_ = c.view();
    auto eps_ = eps.view();
    auto w_ = w.view();
    const size_t n_models = Fcs_.shape(1);
    const size_t n_ref = Fcs_.shape(0);
    const size_t n_cols = for_DS ? n_models + 1 : 1;
    if (Ds.size() != n_models) throw std::runtime_error("Fc and D shape mismatch");

    // der1 wrt D1, D2, .., S, or k_ani
    auto ret = make_numpy_array<double>({n_ref, n_cols});
    double* ptr = ret.data();
    auto sum_Fc = [&](int i) {
      std::complex<double> s = Fcs_(i, 0) * Ds[0];
      for (size_t j = 1; j < n_models; ++j)
        s += Fcs_(i, j) * Ds[j];
      return s;
    };
    for (size_t i = 0; i < n_ref; ++i) {
      if (S <= 0 || std::isnan(Io_(i))) {
        for (size_t j = 0; j < n_cols; ++j)
          ptr[i * n_cols + j] = NAN;
        continue;
      }
      const std::complex<double> Fc_total_conj = std::conj(sum_Fc(i));
      const double Fc_abs = std::abs(Fc_total_conj);
      const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S / eps_(i);
      const double sqrt_sigIo = std::sqrt(sigIo_(i));
      const double tf = k_ani_(i) * Fc_abs / sqrt_sigIo;
      const double sig1 = sq(k_ani_(i)) * S * eps_(i) / sigIo_(i);
      if (sig1 < 0)
        printf("ERROR2: negative sig1= %f k_ani= %f S= %f eps= %d sigIo= %f\n", sig1, k_ani_(i), S, eps_(i), sigIo_(i));
      const double k_num_1 = c_(i) == 1 ? 1. : 0.5;
      const double k_num_2 = c_(i) == 1 ? 0.5 : 0.;
      double j_ratio_1 = integ_j_ratio(k_num_1, k_num_1 - 1, false, to, tf, sig1, c_(i), exp2_threshold, h, N, ewmax);
      const double j_ratio_2 = integ_j_ratio(k_num_2, k_num_2 - 0.5, true, to, tf, sig1, c_(i),
                                             exp2_threshold, h, N, ewmax) * sqrt_sigIo / k_ani_(i);
      if (for_DS) {
        j_ratio_1 *= sigIo_(i) / sq(k_ani_(i));
        const double tmp = ll_int_der1_D(S, Fc_abs, c_(i), eps_(i), j_ratio_2);
        for (size_t j = 0; j < n_models; ++j) {
          const double r_fcj_fc = (Fcs_(i, j) * Fc_total_conj).real();
          ptr[i*n_cols + j] = w_(i) * tmp * r_fcj_fc;
        }
        ptr[i*n_cols + n_models] = w_(i) * ll_int_der1_S(S, Fc_abs, c_(i), eps_(i), j_ratio_1, j_ratio_2);
      }
      else {
        // k_aniso * d/dk_aniso -log p(Io; Fc)
        // note k_aniso is multiplied to the derivative
        const double k_num_3 = c_(i) == 1 ? 2 : 1.5;
        const double j_ratio_3 = integ_j_ratio(k_num_3, k_num_3 - 2, false, to, tf, sig1, c_(i), exp2_threshold, h, N, ewmax); // * sq(sigIo_(i)) / sq(sq(k_ani_(i)));
        ptr[i] = w_(i) * (2 * j_ratio_3 - 2 * Io_(i) / sigIo_(i) * j_ratio_1);
      }
    }
    return ret;
  }

  // an attempt to fast update of Sigma, but it does look good.
  double find_ll_int_S_from_current_estimates_py(np_array<double> Io, np_array<double> sigIo, np_array<double> k_ani,
                                                 double S, np_array<std::complex<double>, 2> Fcs, std::vector<double> Ds,
                                                 np_array<int> c, np_array<int> eps, np_array<double> w) const {
    auto Io_ = Io.view();
    auto sigIo_ = sigIo.view();
    auto k_ani_ = k_ani.view();
    auto Fcs_ = Fcs.view();
    auto c_ = c.view();
    auto eps_ = eps.view();
    auto w_ = w.view();
    const size_t n_models = Fcs_.shape(1);
    const size_t n_ref = Fcs_.shape(0);
    if (Ds.size() != (size_t)Fcs.shape(1)) throw std::runtime_error("Fc and D shape mismatch");

    auto sum_Fc = [&](int i) {
      std::complex<double> s = Fcs_(i, 0) * Ds[0];
      for (size_t j = 1; j < n_models; ++j)
        s += Fcs_(i, j) * Ds[j];
      return s;
    };
    double count = 0;
    double ret = 0;
    for (size_t i = 0; i < n_ref; ++i) {
      if (S <= 0 || std::isnan(Io_(i)) || std::isnan(w_(i)))
        continue;
      const std::complex<double> Fc_total_conj = std::conj(sum_Fc(i));
      const double Fc_abs = std::abs(Fc_total_conj);
      const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S / eps_(i);
      const double sqrt_sigIo = std::sqrt(sigIo_(i));
      const double tf = k_ani_(i) * Fc_abs / sqrt_sigIo;
      const double sig1 = sq(k_ani_(i)) * S * eps_(i) / sigIo_(i);
      if (sig1 < 0)
        printf("ERROR2: negative sig1= %f k_ani= %f S= %f eps= %d sigIo= %f\n", sig1, k_ani_(i), S, eps_(i), sigIo_(i));
      const double k_num_1 = c_(i) == 1 ? 1. : 0.5;
      const double k_num_2 = c_(i) == 1 ? 0.5 : 0.;
      const double j_ratio_1 = integ_j_ratio(k_num_1, k_num_1 - 1, false, to, tf, sig1, c_(i),
                                             exp2_threshold, h, N, ewmax) * sigIo_(i) / sq(k_ani_(i));
      const double j_ratio_2 = integ_j_ratio(k_num_2, k_num_2 - 0.5, true, to, tf, sig1, c_(i),
                                             exp2_threshold, h, N, ewmax) * sqrt_sigIo / k_ani_(i);
      const double tmp = ll_int_der1_D(S, Fc_abs, c_(i), eps_(i), j_ratio_2);
      if (c_(i) == 1) // acentrics
        ret += w_(i) * (sq(Fc_abs) + j_ratio_1 / c_(i) - (3-c_(i)) * Fc_abs * j_ratio_2) / eps_(i);
      else
        ret += w_(i) * (sq(Fc_abs) + 2 * (j_ratio_1 / c_(i) - (3-c_(i)) * Fc_abs * j_ratio_2)) / eps_(i);
      count += w_(i);
    }
    if (count == 0) return NAN;
    return ret / count;
  }

// For French-Wilson
  template<bool for_S>
  nb::ndarray<nb::numpy, double>
  ll_int_fw_der1_params_py(np_array<double> Io, np_array<double> sigIo, np_array<double> k_ani,
                           double S, np_array<int> c, np_array<int> eps, np_array<double> w) const {
    auto Io_ = Io.view();
    auto sigIo_ = sigIo.view();
    auto k_ani_ = k_ani.view();
    auto c_ = c.view();
    auto eps_ = eps.view();
    auto w_ = w.view();
    size_t n_ref = Io.shape(0);

    // der1 wrt S or k_ani
    auto ret = make_numpy_array<double>({n_ref});
    double* ptr = ret.data();
    for (size_t i = 0; i < n_ref; ++i) {
      if (std::isnan(Io_(i))) {
        ptr[i] = NAN;
        continue;
      }
      const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S / eps_(i);
      const double k_num = c_(i) == 1 ? 1. : 0.5;
      double j_ratio_1 = integ_j_ratio(k_num, k_num - 1, false, to, 0., 1., c_(i), exp2_threshold, h, N, ewmax);
      if (for_S) {
        j_ratio_1 *= sigIo_(i) / sq(k_ani_(i));
        ptr[i] = w_(i) * ll_int_der1_S(S, 0., c_(i), eps_(i), j_ratio_1, 0.);
      }
      else {
        // note k_aniso is multiplied to the derivative
        const double k_num_3 = c_(i) == 1 ? 2 : 1.5;
        const double j_ratio_3 = integ_j_ratio(k_num_3, k_num_3 - 2, false, to, 0., 1., c_(i), exp2_threshold, h, N, ewmax);
        ptr[i] = w_(i) * (2 * j_ratio_3 - 2 * Io_(i) / sigIo_(i) * j_ratio_1);
      }
    }
    return ret;
  }
};

void add_intensity(nb::module_& m) {
  m.def("integ_J", [](double k, np_array<double> to, np_array<double> tf, np_array<double> sig1, int c, bool return_log,
                      double exp2_threshold, double h, int N, double ewmax) {
    auto to_ = to.view();
    auto tf_ = tf.view();
    auto sig1_ = sig1.view();
    size_t len = to_.shape(0);
    if (len != tf_.shape(0) || len != sig1_.shape(0))
      throw std::runtime_error("integ_j: shape mismatch");
    auto ret = make_numpy_array<double>({len});
    double* retp = ret.data();
    for (size_t i = 0; i < len; ++i)
      retp[i] = integ_j(k, to_(i), tf_(i), sig1_(i), c, return_log, exp2_threshold, h, N, ewmax);
    return ret;
  },
    nb::arg("k"), nb::arg("to"), nb::arg("tf"), nb::arg("sig1"), nb::arg("c"), nb::arg("return_log"),
    nb::arg("exp2_threshold")=10, nb::arg("h")=0.5, nb::arg("N")=200, nb::arg("ewmax")=20.);
  m.def("integ_J_ratio", [](np_array<double> k_num, np_array<double> k_den, bool l, np_array<double> to, np_array<double> tf,
                            np_array<double> sig1, np_array<int> c, double exp2_threshold, double h, int N, double ewmax) {
    auto k_num_ = k_num.view();
    auto k_den_ = k_den.view();
    auto to_ = to.view();
    auto tf_ = tf.view();
    auto sig1_ = sig1.view();
    auto c_ = c.view();
    size_t len = to_.shape(0);
    if (len != k_num_.shape(0) || len != k_den_.shape(0) || len != tf_.shape(0) ||
        len != sig1_.shape(0) || len != c_.shape(0))
      throw std::runtime_error("integ_j_ratio: shape mismatch");
    auto ret = make_numpy_array<double>({len});
    double* retp = ret.data();
    for (size_t i = 0; i < len; ++i)
      retp[i] = integ_j_ratio(k_num_(i), k_den_(i), l, to_(i), tf_(i), sig1_(i), c_(i), exp2_threshold, h, N, ewmax);
    return ret;
  },
    nb::arg("k_num"), nb::arg("k_den"), nb::arg("l"), nb::arg("to"), nb::arg("tf"),
    nb::arg("sig1"), nb::arg("c"),
    nb::arg("exp2_threshold")=10, nb::arg("h")=0.5, nb::arg("N")=200, nb::arg("ewmax")=20.);
  nb::class_<IntensityIntegrator>(m, "IntensityIntegrator")
    .def(nb::init<>())
    .def_rw("h", &IntensityIntegrator::h)
    .def_rw("N", &IntensityIntegrator::N)
    .def_rw("ewmax", &IntensityIntegrator::ewmax)
    .def_rw("exp2_threshold", &IntensityIntegrator::exp2_threshold)
    .def("ll_int", [](const IntensityIntegrator& self, np_array<double> Io, np_array<double> sigIo, np_array<double> k_ani,
                      np_array<double> S, np_array<double> Fc, np_array<int> c, np_array<double> w) {
      auto Io_ = Io.view();
      auto sigIo_ = sigIo.view();
      auto k_ani_ = k_ani.view();
      auto S_ = S.view();
      auto Fc_ = Fc.view();
      auto c_ = c.view();
      auto w_ = w.view();
      size_t len = Io_.shape(0);
      if (len != sigIo_.shape(0) || len != k_ani_.shape(0) || len != S_.shape(0) || len != Fc_.shape(0) || len != c_.shape(0)
          || len != w_.shape(0))
        throw std::runtime_error("ll_int: shape mismatch");
      auto ret = make_numpy_array<double>({len});
      double* retp = ret.data();
      for (size_t i = 0; i < len; ++i)
        retp[i] = w_(i) * self.ll_int(Io_(i), sigIo_(i), k_ani_(i), S_(i), Fc_(i), c_(i));
      return ret;
    },
         nb::arg("Io"), nb::arg("sigIo"), nb::arg("k_ani"), nb::arg("S"), nb::arg("Fc"), nb::arg("c"), nb::arg("w"))
    .def("ll_int_der1_DS", &IntensityIntegrator::ll_int_der1_params_py<true>)
    .def("ll_int_der1_ani", &IntensityIntegrator::ll_int_der1_params_py<false>)
    .def("find_ll_int_S_from_current_estimates", &IntensityIntegrator::find_ll_int_S_from_current_estimates_py)
    .def("ll_int_fw_der1_S", &IntensityIntegrator::ll_int_fw_der1_params_py<true>)
    .def("ll_int_fw_der1_ani", &IntensityIntegrator::ll_int_fw_der1_params_py<false>)
    ;
  m.def("lambertw", &lambertw::lambertw);
  m.def("find_root", &find_root);
  m.def("f1_orig2", &f1_orig2);
  m.def("f1_orig2_der1", &f1_orig2_der1);
  m.def("f1_orig2_der2", &f1_orig2_der2);
  m.def("f1_exp2", &f1_exp2);
  m.def("f1_exp2_der1", &f1_exp2_der1);
  m.def("f1_exp2_der2", &f1_exp2_der2);
}
