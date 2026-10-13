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
double f1_orig2(double x, double z, double to, double tf, double sig1, int c, bool l=false) {
  // z = 2k+1
  const double x2 = x * x;
  double ret = 0.5 * x2 * x2 - to * x2 - z * std::log(x);
  if (tf == 0.) return ret;
  const double X = x * tf / sig1;
  const double log_ic0 = c == 1 ? gemmi::log_bessel_i0(2*X) : gemmi::log_cosh(X);
  ret -= log_ic0;
  if (!l) return ret;
  return ret - std::log(fom_safe(X, c));
}
double f1_orig2_der1(double x, double z, double to, double tf, double sig1, int c, bool l=false) {
  const double ret = 2 * x*x*x - 2 * to * x - z / x;
  if (tf == 0.) return ret;
  const double X_der = tf / sig1;
  const double X = x * X_der;
  if (l) {
    double m_der;
    const double m = fom_safe(X, c, &m_der);
    return ret - ((3-c) * m + m_der / m) * X_der;
  } else {
    const double m = fom_safe(X, c);
    return ret - (3-c) * m * X_der;
  }
}
double f1_orig2_der2(double x, double z, double to, double tf, double sig1, int c, bool l=false) {
  const double x2 = x * x;
  const double ret = 6 * x2 - 2 * to + z / x2;
  if (tf == 0.) return ret;
  const double X_der = tf / sig1;
  const double X = x * X_der;
  double m_der;
  if (l) {
    double m_der2;
    const double m = fom_safe(X, c, &m_der, &m_der2);
    return ret - X_der * X_der * ((3 - c) * m_der + m_der2 / m - sq(m_der / m));
  }
  fom_safe(X, c, &m_der);
  return ret - (3-c) * X_der * X_der * m_der;
}
double f1_exp2(double y, double z, double to, double tf, double sig1, int c, bool l=false) {
  // z = 2k+2
  const double e_y = std::exp(-y);
  const double exp2 = std::exp(2 * (y - e_y));
  double ret = 0.5 * exp2*exp2 - to * exp2 - z * (y - e_y) - std::log(1 + e_y);
  if (tf == 0.) return ret;
  const double X = std::exp(y - e_y) * tf / sig1;
  const double log_ic0 = c == 1 ? gemmi::log_bessel_i0(2*X) : gemmi::log_cosh(X);
  ret -= log_ic0;
  if (!l) return ret;
  return ret - std::log(fom_safe(X, c));
}
double f1_exp2_der1(double y, double z, double to, double tf, double sig1, int c, bool l=false) {
  const double e_y = std::exp(-y);
  const double exp1 = std::exp(y - e_y);
  const double exp2 = exp1 * exp1;
  const double ret = (1 + e_y) * (2 * exp2*exp2 - 2 * to * exp2 - z) + e_y / (1 + e_y);
  if (tf == 0.) return ret;
  const double X_der = tf / sig1;
  const double X = exp1 * X_der;
  if (l) {
    double m_der;
    const double m = fom_safe(X, c, &m_der);
    return ret - (1 + e_y) * X * ((3-c) * m + m_der / m);
  } else {
    const double m = fom_safe(X, c);
    return ret - (1 + e_y) * X * (3-c) * m;
  }
}
double f1_exp2_der2(double y, double z, double to, double tf, double sig1, int c, bool l=false) {
  const double e_y = std::exp(-y);
  const double exp1 = std::exp(y - e_y);
  const double exp2 = exp1 * exp1;
  double ret = -e_y * (2 * exp2*exp2 - 2 * to * exp2 - z) + sq(1 + e_y) * (8 * sq(exp2) - 4 * to * exp2) - e_y / sq(1 + e_y);
  if (tf == 0.) return ret;
  const double X_der = tf / sig1;
  const double X = exp1 * X_der;
  double m_der;
  if (l) {
    double m_der2;
    const double m = fom_safe(X, c, &m_der, &m_der2);
    ret += e_y * (3-c) * X * m - sq(1 + e_y) * ((3-c) * X * m + (3-c) * sq(X) * m_der);
    return ret - (m_der2 / m - sq(m_der / m)) * sq((1 + e_y) * X) - m_der / m * X * (-e_y + sq(1 + e_y));
  } else {
    const double m = fom_safe(X, c, &m_der);
    return ret + e_y * (3-c) * X * m - sq(1 + e_y) * ((3-c) * X * m + (3-c) * sq(X) * m_der);
  }
}
double find_root(double k, double to, double tf, double sig1, int c, double det, bool l, bool use_exp2) {
  if (tf == 0.) {
    if (use_exp2)
      return solve_y_minus_exp_minus_y(0.5 * std::log(0.5 * x_plus_sqrt_xsq_plus_y(to, 4 * k + 4)), 1e-4); // XXX may not be exact
    return det;
  }

  const auto func = [&](double x) {return use_exp2
      ? f1_exp2_der1(x, 2 * k + 2, to, tf, sig1, c, l)
      : f1_orig2_der1(x, 2 * k + 1, to, tf, sig1, c, l);
  };
  double x0, x1;
  constexpr double xmin = 1e-30; // for orig2
  if (!use_exp2) { // orig2
    x0 = std::max(xmin, det);
    const double A = std::max(to, std::max((3-c) * tf / 4.0 / sig1, k + 0.5));
    const double sqrt_A = std::sqrt(A);
    x1 = 0.5 * (sqrt_A + std::sqrt(A + 4.0 * sqrt_A));
  } else {
    const double X1 = tf * 0.5 / sig1 * std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 4.0 * k + 3.5));
    const double m1 = fom_safe(X1, c);
    const double log_arg = 0.5 * x_plus_sqrt_xsq_plus_y(to, 4.0 * k + 4.0 * (3 - c) * X1 * m1 + 3.5);
    x0 = solve_y_minus_exp_minus_y(0.5 * std::log(log_arg), 1e-4);
    if (std::isnan(x0))
      printf("ERROR: x0= %e, use_exp2, X1=%e, m1=%e in_log=%e\n", x0, X1, m1, log_arg);
    const double A = std::max(to, std::max((3-c) * tf / 4.0 / sig1, k + 1.0));
    const double sqrt_A = std::sqrt(A);
    x1 = solve_y_minus_exp_minus_y(std::log(0.5 * (sqrt_A + std::sqrt(A + 4.0 * sqrt_A))), 1e-4);
  }

  double f0 = func(x0), f1 = func(x1);
  if (f0 == 0.0) return x0;
  if (f1 == 0.0) return x1;
  RootfindResult ret;
  if ((f0 > 0.0) != (f1 > 0.0))
    ret = brent(func, std::min(x0, x1), std::max(x0, x1));

  // printf("initial brent. ok=%d iter=%d", ret.success, ret.iter);
  // if (ret.success) printf("\n");
  // else printf(" why= %s det= %e f0= %e f1= %e \n", ret.reason.c_str(), det, f0, f1);

  if (ret.success)
    return ret.x;

  double x_curr = (f0 > 0.0) ? std::min(x0, x1) : std::max(x0, x1);
  double f_curr = (x_curr == x0) ? f0 : f1;
  if (!use_exp2) {
    // Geometric search in log-space around initial estimate
    const double factor = (f_curr > 0.0) ? (1.0 / 1.5) : 1.5;

    // Step left towards xmin, or right until sign flip / floating-point overflow
    while (factor < 1.0 ? (x_curr > xmin) : std::isfinite(x_curr)) {
      const double x_next = (factor < 1.0) ? std::max(xmin, x_curr * factor) : x_curr * factor;
      const double f_next = func(x_next);

      if ((f_curr > 0.0) != (f_next > 0.0)) {
        ret = brent(func, std::min(x_curr, x_next), std::max(x_curr, x_next));
        // printf("second brent. ok=%d iter=%d\n", ret.success, ret.iter);
        return ret.x;
      }

      if (x_next == x_curr) break; // Reached xmin lower bound cap
      x_curr = x_next;
      f_curr = f_next;
    }
  } else { // exp2
    // Both initial points are on the same side of the root.
    // Select the outer boundary and step outward directionally.
    const double dir = (f_curr > 0.0) ? -1.0 : 1.0;
    double step = 1.0;

    for (int i = 0; i < 20; ++i) {
      const double x_next = x_curr + dir * step;
      const double f_next = func(x_next);

      if ((f_curr > 0.0) != (f_next > 0.0)) {
        ret = brent(func, std::min(x_curr, x_next), std::max(x_curr, x_next));
        // printf("second brent. ok=%d iter=%d\n", ret.success, ret.iter);
        return ret.x;
      }

      x_curr = x_next;
      f_curr = f_next;
      step *= 1.5;
    }
  }
  return (std::abs(f0) < std::abs(f_curr)) ? x0 : x_curr;
}

// factor of (sig/k_ani^2)^(k+1) is needed, which should be done outside.
double integ_j(double k, double to, double tf, double sig1, int c, bool l, bool return_log,
               double exp2_threshold=2., double h=0.5, int N=200, double ewmax=20.) {
  if (std::isnan(to) || sig1 <= 0) return NAN; // perhaps sig1<0 should not return nan.. they considered 0 in sum
  const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k + 1)));
  const bool use_exp2 = det < exp2_threshold;
  const double z = use_exp2 ? 2 * k + 2 : 2 * k + 1;
  auto f = use_exp2 ? f1_exp2 : f1_orig2;
  auto fder2 = use_exp2 ? f1_exp2_der2 : f1_orig2_der2;
  const double root = find_root(k, to, tf, sig1, c, det, l, use_exp2);
  const double f1val = f(root, z, to, tf, sig1, c, l);
  const double f2der = fder2(root, z, to, tf, sig1, c, l);
  if (f2der <= 0)
    printf("integ_j bad f2der %f exp2=%d root=%f z=%f; to=%f; tf=%f; sig1=%f; c=%d;\n", f2der, use_exp2, root, z, to, tf, sig1, c);
  // Laplace approximation, <0.2% rel err
  if (!use_exp2 && f2der * f1val * f1val > 1e7) {
    const double lap = -f1val - 0.5 * std::log(f2der) + 0.5 * std::log(gemmi::pi() * 2.0);
    if (std::isinf(lap))
      printf("log integ_j (Laplace) inf f1val %f f2der %f exp2=%d root=%f z=%f; to=%f; tf=%f; sig1=%f; c=%d;\n", f1val, f2der, use_exp2, root, z, to, tf, sig1, c);
    return return_log ? lap : std::exp(lap);
  }
  const double delta = h * std::sqrt(2 / f2der);
  double s = 1; // for i = 0
  for (int sign : {-1, 1}) {
    for (int i = 1; i < N; ++i) {
      const double xx = sign * delta * i + root;
      if (!use_exp2 && xx <= 0) continue; //break?
      const double ff = f(xx, z, to, tf, sig1, c, l) - f1val;
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

double integ_j_ratio(double k_num, double k_den, bool l, double to, double tf, double sig1, int c, double root_den,
                     bool use_exp2, double h=0.5, int N=200, double ewmax=20.) {
  // factor of sig^{k_num - k_den} is needed, which should be done outside.
  if (std::isnan(to) || sig1 <= 0) return NAN;
  const double z = use_exp2 ? 2 * k_den + 2 : 2 * k_den + 1;
  const double deltaz = 2 * (k_num - k_den);
  auto f = use_exp2 ? f1_exp2 : f1_orig2;
  auto fder2 = use_exp2 ? f1_exp2_der2 : f1_orig2_der2;
  const double f1val = f(root_den, z, to, tf, sig1, c, false);
  const double f2der = fder2(root_den, z, to, tf, sig1, c, false);
  if (f2der <= 0)
    printf("integ_j_ratio bad f2der %f exp2=%d root=%f z=%f; to=%f; tf=%f; sig1=%f; c=%d;\n", f2der, use_exp2, root_den, z, to, tf, sig1, c);

  auto calc_fom = [&](double xx) { return fom_safe((use_exp2 ? std::exp(xx - std::exp(-xx)) : xx) * tf / sig1, c); };
  const double tmp = use_exp2 ? root_den - std::exp(-root_den) : std::log(root_den);
  double sd = 1, sn = std::exp(deltaz * tmp);
  if (l) sn *= calc_fom(root_den);

  // Laplace approximation, assuming the same root for numerator: \int g(x)exp(-f(x))dx / \int exp(-f(x))dx ~ g(x_0).
  // thresholds adjusted empirically so that <0.2% rel error for orig2 and <0.08% for exp2 would be achieved
  if (use_exp2 ? (f2der * f1val * f1val > 1e9) : (f2der * f1val * f1val > 1e6))
    return sn;

  const double delta = h * std::sqrt(2 / f2der);
  for (int sign : {-1, 1}) {
    for (int i = 1; i < N; ++i) {
      const double xx = sign * delta * i + root_den;
      if (!use_exp2 && xx <= 0) continue; //break?
      const double ff = f(xx, z, to, tf, sig1, c, false) - f1val;
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
  return (3 - c) * (1. - j_ratio_2 / Fc) / (S * eps);
}
// d/dS -log(Io; Fc)
double ll_int_der1_S(double S, double Fc, int c, double eps, double j_ratio_1, double j_ratio_2) {
  const double invepsS2 = 1. / (S * S * eps);
  return 1. / (c * S) - ((sq(Fc) + j_ratio_1) / c - (3-c) * Fc * j_ratio_2) * invepsS2;
}

struct IntensityIntegrator {
  double h = 0.5;
  int N = 200;
  double ewmax = 20.;
  double exp2_threshold = 2.;

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
    const double logj = integ_j(k, to, tf, sig1, c, false, true, exp2_threshold, h, N, ewmax);
    const double ret = (3-c) * std::log(k_ani) + (std::log(S) + Ic / S) / c - logj;
    if (std::isinf(ret))
      printf("-LL Inf S=%f; logj=%f; to=%f; tf=%f; sig1=%f; k=%f; c=%d\n", S, logj, to, tf, sig1, k, c);
    return ret;
  }

  template<bool for_DS>
  nb::ndarray<nb::numpy, double>
  ll_int_der1_params_py(np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                        double S, np_array<const std::complex<double>, 2> Fcs, std::vector<double> Ds,
                        np_array<const int> c, np_array<const int> eps, np_array<const double> w) const {
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
    if (Ds.size() != n_models) throw std::runtime_error("ll_int_der1_params: Fc and D shape mismatch");
    if (n_ref != Io_.shape(0) || n_ref != sigIo_.shape(0) || n_ref != k_ani_.shape(0) || n_ref != c_.shape(0) || n_ref != eps_.shape(0)
        || n_ref != w_.shape(0))
      throw std::runtime_error("ll_int_der_fc: shape mismatch");

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
      const double k_den = c_(i) == 1 ? 0. : -0.5;
      const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
      const bool use_exp2 = det < exp2_threshold;
      const double root_den = find_root(k_den, to, tf, sig1, c_(i), det, false, use_exp2);
      double j_ratio_1 = integ_j_ratio(k_den + 1.0, k_den, false, to, tf, sig1, c_(i), root_den, use_exp2, h, N, ewmax);
      const double j_ratio_2 = integ_j_ratio(k_den + 0.5, k_den, true, to, tf, sig1, c_(i), root_den, use_exp2, h, N, ewmax) * sqrt_sigIo / k_ani_(i);
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
        const double j_ratio_3 = integ_j_ratio(k_den + 2.0, k_den, false, to, tf, sig1, c_(i), root_den, use_exp2, h, N, ewmax); // * sq(sigIo_(i)) / sq(sq(k_ani_(i)));
        ptr[i] = w_(i) * (2 * j_ratio_3 - 2 * Io_(i) / sigIo_(i) * j_ratio_1);
      }
    }
    return ret;
  }

  // an attempt to fast update of Sigma, but it does look good.
  double find_ll_int_S_from_current_estimates_py(np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                                                 double S, np_array<const std::complex<double>, 2> Fcs, std::vector<double> Ds,
                                                 np_array<const int> c, np_array<const int> eps, np_array<const double> w) const {
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
    if (n_ref != Io_.shape(0) || n_ref != sigIo_.shape(0) || n_ref != k_ani_.shape(0) || n_ref != c_.shape(0) || n_ref != eps_.shape(0)
        || n_ref != w_.shape(0))
      throw std::runtime_error("find_ll_int_S_from_current_estimates: shape mismatch");

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
      const double k_den = c_(i) == 1 ? 0. : -0.5;
      const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
      const bool use_exp2 = det < exp2_threshold;
      const double root_den = find_root(k_den, to, tf, sig1, c_(i), det, false, use_exp2);
      const double j_ratio_1 = integ_j_ratio(k_den + 1.0, k_den, false, to, tf, sig1, c_(i),
                                             root_den, use_exp2, h, N, ewmax) * sigIo_(i) / sq(k_ani_(i));
      const double j_ratio_2 = integ_j_ratio(k_den + 0.5, k_den, true, to, tf, sig1, c_(i),
                                             root_den, use_exp2, h, N, ewmax) * sqrt_sigIo / k_ani_(i);
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

  nb::ndarray<nb::numpy, double>
  ll_int_der_fc_py(np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                   np_array<const double> S, np_array<const double> Fc_abs,
                   np_array<const int> c, np_array<const int> eps) const {
    auto Io_ = Io.view();
    auto sigIo_ = sigIo.view();
    auto k_ani_ = k_ani.view();
    auto S_ = S.view();
    auto Fc_abs_ = Fc_abs.view();
    auto c_ = c.view();
    auto eps_ = eps.view();
    const size_t len = Io_.shape(0);
    if (len != sigIo_.shape(0) || len != k_ani_.shape(0) || len != S_.shape(0) || len != Fc_abs_.shape(0) || len != c_.shape(0)
        || len != eps_.shape(0))
      throw std::runtime_error("ll_int_der_fc: shape mismatch");

    auto ret = make_numpy_array<double>({len});
    double* ptr = ret.data();
    for (size_t i = 0; i < len; ++i) {
      if (std::isnan(Io_(i))) {
        ptr[i] = NAN;
        continue;
      }
      const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S_(i) / eps_(i);
      const double sqrt_sigIo = std::sqrt(sigIo_(i));
      const double tf = k_ani_(i) * Fc_abs_(i) / sqrt_sigIo;
      const double sig1 = sq(k_ani_(i)) * S_(i) * eps_(i) / sigIo_(i);
      if (sig1 < 0)
        printf("ERROR2: negative sig1= %f k_ani= %f S= %f eps= %d sigIo= %f\n", sig1, k_ani_(i), S_(i), eps_(i), sigIo_(i));
      const double k_den = c_(i) == 1 ? 0. : -0.5;
      const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
      const bool use_exp2 = det < exp2_threshold;
      const double root_den = find_root(k_den, to, tf, sig1, c_(i), det, false, use_exp2);
      auto f = use_exp2 ? f1_exp2_der1 : f1_orig2_der1;
      const double j_ratio = integ_j_ratio(k_den + 0.5, k_den, true, to, tf, sig1, c_(i), root_den, use_exp2, h, N, ewmax) * sqrt_sigIo / k_ani_(i);
      ptr[i] = (3 - c_(i)) * (Fc_abs_(i) - j_ratio) / eps_(i) / S_(i);
    }
    return ret;
  }

// For French-Wilson
  template<bool for_S>
  nb::ndarray<nb::numpy, double>
  ll_int_fw_der1_params_py(np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                           double S, np_array<const int> c, np_array<const int> eps, np_array<const double> w) const {
    auto Io_ = Io.view();
    auto sigIo_ = sigIo.view();
    auto k_ani_ = k_ani.view();
    auto c_ = c.view();
    auto eps_ = eps.view();
    auto w_ = w.view();
    size_t n_ref = Io.shape(0);
    if (n_ref != sigIo_.shape(0) || n_ref != k_ani_.shape(0) || n_ref != c_.shape(0) || n_ref != eps_.shape(0)
        || n_ref != w_.shape(0))
      throw std::runtime_error("ll_int_fw_der1_params: shape mismatch");

    // der1 wrt S or k_ani
    auto ret = make_numpy_array<double>({n_ref});
    double* ptr = ret.data();
    for (size_t i = 0; i < n_ref; ++i) {
      if (std::isnan(Io_(i))) {
        ptr[i] = NAN;
        continue;
      }
      const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S / eps_(i);
      const double k_den = c_(i) == 1 ? 0. : -0.5;
      const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
      const bool use_exp2 = det < exp2_threshold;
      const double root_den = find_root(k_den, to, 0., 1., c_(i), det, false, use_exp2);
      double j_ratio_1 = integ_j_ratio(k_den + 1, k_den, false, to, 0., 1., c_(i), root_den, use_exp2, h, N, ewmax);
      if (for_S) {
        j_ratio_1 *= sigIo_(i) / sq(k_ani_(i));
        ptr[i] = w_(i) * ll_int_der1_S(S, 0., c_(i), eps_(i), j_ratio_1, 0.);
      }
      else {
        // note k_aniso is multiplied to the derivative
        const double j_ratio_3 = integ_j_ratio(k_den + 2.0, k_den, false, to, 0., 1., c_(i), root_den, use_exp2, h, N, ewmax);
        ptr[i] = w_(i) * (2 * j_ratio_3 - 2 * Io_(i) / sigIo_(i) * j_ratio_1);
      }
    }
    return ret;
  }

  nb::ndarray<nb::numpy, double>
  ll_refine_D_S_py(np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                   np_array<const double> S, np_array<const std::complex<double>, 2> Fc, np_array<const double, 2> D,
                   np_array<const int> c, np_array<const int> eps, np_array<const double> w, np_array<const int> bin,
                   int max_cyc, bool single_D) const {
    const bool use_exp = true; //par == 1;
    auto Io_ = Io.view();
    auto sigIo_ = sigIo.view();
    auto k_ani_ = k_ani.view();
    auto S_ = S.view();
    auto Fc_ = Fc.view();
    auto D_ = D.view();
    auto c_ = c.view();
    auto eps_ = eps.view();
    auto w_ = w.view();
    auto bin_ = bin.view();
    const size_t n_models = Fc_.shape(1);
    const size_t n_ref = Fc_.shape(0);
    const size_t n_bins = *std::max_element(bin.data(), bin.data() + bin.size()) + 1;
    if (D_.shape(1) != n_models) throw std::runtime_error("Fc and D shape mismatch");
    if (D_.shape(0) != n_bins) throw std::runtime_error("D and n_bins mismatch");
    if (S_.shape(0) != n_bins) throw std::runtime_error("S and n_bins mismatch");
    if (n_ref != Io_.shape(0) || n_ref != sigIo_.shape(0) || n_ref != k_ani_.shape(0) || n_ref != c_.shape(0) ||
        n_ref != eps_.shape(0) || n_ref != bin_.shape(0) || n_ref != w_.shape(0))
      throw std::runtime_error("ll_refine_D_S: shape mismatch");

    auto ret = make_numpy_array<double>({n_bins, n_models + 1}); // [D0, D1, .., S]
    auto DS = ret.view<double, nb::ndim<2>>();
    for (int i_bin = 0; i_bin < n_bins; ++i_bin) {
      for (int j = 0; j < n_models; ++j)
        DS(i_bin, j) = D_(i_bin, j);
      DS(i_bin, n_models) = S_(i_bin);
    }
    auto sum_Fc = [&](int i, int i_bin) {
      std::complex<double> s = Fc_(i, 0) * DS(i_bin, 0);
      for (size_t j = 1; j < n_models; ++j)
        s += Fc_(i, j) * DS(i_bin, j);
      return s;
    };
    auto calc_f_ders = [&](int i_bin, bool ll_only) {
      double ll = 0;
      const int n_par = n_models + 1;
      Eigen::VectorXd der1 = Eigen::VectorXd::Zero(n_par);
      Eigen::MatrixXd der2 = Eigen::MatrixXd::Zero(n_par, n_par);
      for (int i = 0; i < n_ref; ++i)
        if (bin_(i) == i_bin && !std::isnan(Io_(i)) && !std::isnan(sigIo_(i)) && w_(i) != 0) {
          const double S = DS(i_bin, n_models);
          const std::complex<double> Fc_total_conj = std::conj(sum_Fc(i, i_bin));
          const double Fc_abs = std::abs(Fc_total_conj);
          const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S / eps_(i);
          const double sqrt_sigIo = std::sqrt(sigIo_(i));
          const double tf = k_ani_(i) * Fc_abs / sqrt_sigIo;
          const double sig1 = sq(k_ani_(i)) * S * eps_(i) / sigIo_(i);
          if (sig1 < 0) {
            printf("ERROR2: negative sig1= %f k_ani= %f S= %f eps= %d sigIo= %f\n", sig1, k_ani_(i), S, eps_(i), sigIo_(i));
            continue;
          }
          const double logj = integ_j(c_(i) == 1 ? 0 : -0.5, to, tf, sig1, c_(i), false, true, exp2_threshold, h, N, ewmax);
          ll += w_(i) * ((3-c_(i)) * std::log(k_ani_(i)) + (std::log(S) + sq(Fc_abs) / S) / c_(i) - logj);
          if (!ll_only) {
            const double k_den = c_(i) == 1 ? 0. : -0.5;
            const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
            const bool use_exp2 = det < exp2_threshold;
            const double root_den = find_root(k_den, to, tf, sig1, c_(i), det, false, use_exp2);
            const double j_ratio_1 = integ_j_ratio(k_den + 1.0, k_den, false, to, tf, sig1, c_(i),
                                                   root_den, use_exp2, h, N, ewmax) * sigIo_(i) / sq(k_ani_(i));
            const double j_ratio_2 = integ_j_ratio(k_den + 0.5, k_den, true, to, tf, sig1, c_(i),
                                                   root_den, use_exp2, h, N, ewmax) * sqrt_sigIo / k_ani_(i);
            Eigen::VectorXd g_i = Eigen::VectorXd::Zero(n_par);
            const double tmp = ll_int_der1_D(S, Fc_abs, c_(i), eps_(i), j_ratio_2);
            for (size_t j = 0; j < n_models; ++j)
              g_i(j) = tmp * (Fc_(i, j) * Fc_total_conj).real();
            g_i(n_models) = ll_int_der1_S(S, Fc_abs, c_(i), eps_(i), j_ratio_1, j_ratio_2);
            der1 += w_(i) * g_i;
            der2 += w_(i) * (g_i * g_i.transpose());
          }
        }
      return std::make_pair(ll, std::make_pair(der1, der2));
    };
    auto get_par = [&](int i_bin, bool use_exp) {
      Eigen::VectorXd pval(n_models + 1);
      for (size_t j = 0; j < n_models; ++j)
        pval(j) = DS(i_bin, j);
      pval(pval.size()-1) = DS(i_bin, n_models);
      return use_exp ? pval.array().log() : pval;
    };
    auto set_par = [&](int i_bin, const auto &val, bool use_exp) {
      for (size_t j = 0; j < n_models; ++j)
        DS(i_bin, j) = std::max(1e-4, use_exp ? std::exp(val(j)) : val(j));
      DS(i_bin, n_models) = std::max(1e-1, use_exp ? std::exp(val(val.size()-1)) : val(val.size()-1));
    };
    for (int i_bin = 0; i_bin < n_bins; ++i_bin) {
      for (int cyc = 0; cyc < max_cyc; ++cyc) {
        bool no_shift = false;
        auto f0_ders = calc_f_ders(i_bin, false);
        const Eigen::VectorXd pval = get_par(i_bin, use_exp);
        Eigen::VectorXd shift(pval.size());
        if (use_exp) {
          f0_ders.second.first = f0_ders.second.first.array() * pval.array().exp();
          for (int i = 0; i < pval.size(); ++i)
            for (int j = 0; j < pval.size(); ++j)
              f0_ders.second.second(i,j) *= std::exp(pval(i) + pval(j));
        }
        // Ensure strictly positive definite to avoid singular inversion
        for (int i = 0; i < f0_ders.second.second.rows(); ++i)
          f0_ders.second.second(i,i) += 1e-6;
        if (single_D) {
          Eigen::Vector2d red_g;
          red_g(0) = f0_ders.second.first.head(n_models).sum();
          red_g(1) = f0_ders.second.first(n_models);
          Eigen::Matrix2d red_H;
          red_H(0, 0) = f0_ders.second.second.topLeftCorner(n_models, n_models).sum();
          red_H(0, 1) = f0_ders.second.second.col(n_models).head(n_models).sum();
          red_H(1, 0) = red_H(0, 1);
          red_H(1, 1) = f0_ders.second.second(n_models, n_models);
          const Eigen::VectorXd red_shift = -SymMatEig(red_H).inv() * (red_g);
          shift.head(n_models).setConstant(red_shift(0));
          shift(n_models) = red_shift(1);
        } else
          shift = -SymMatEig(f0_ders.second.second).inv() * (f0_ders.second.first);
        if (use_exp)
          shift = shift.cwiseMin(5).cwiseMax(-5); // cap shift
        for (int i = 0; i < 6; ++i) {
          // printf("bin %d cyc %d ", i_bin, cyc);
          // printf("D ");
          // for (size_t j = 0; j < n_models; ++j)
          //   printf("%f%+f ", DS(i_bin, j), shift(j));
          // printf("S %f%+f ", DS(i_bin, n_models), shift(pval.size()-1));
          set_par(i_bin, pval + shift, use_exp);
          const double f1 = calc_f_ders(i_bin, true).first;
          // printf("ll %f\n", f1 - f0_ders.first);
          if (f1 < f0_ders.first)
            break;
          shift /= 2;
          if (shift.array().abs().maxCoeff() < (use_exp ? 1e-2 : 1e-4)) { // arbitrary
            no_shift = true;
            break;
          }
        }
        if (no_shift)
          break;
      }
    }
    return ret;
  }
};

void add_intensity(nb::module_& m) {
  m.def("integ_J", [](double k, np_array<const double> to, np_array<const double> tf, np_array<const double> sig1, int c, bool l, bool return_log,
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
      retp[i] = integ_j(k, to_(i), tf_(i), sig1_(i), c, l, return_log, exp2_threshold, h, N, ewmax);
    return ret;
  },
        nb::arg("k"), nb::arg("to"), nb::arg("tf"), nb::arg("sig1"), nb::arg("c"), nb::arg("l"), nb::arg("return_log"),
        nb::arg("exp2_threshold")=10, nb::arg("h")=0.5, nb::arg("N")=200, nb::arg("ewmax")=20.);
  m.def("integ_J_ratio", [](np_array<const double> k_num, np_array<const double> k_den, bool l, np_array<const double> to, np_array<const double> tf,
                            np_array<const double> sig1, np_array<const int> c, np_array<const double> root_den, np_array<const bool> use_exp2,
                            double h, int N, double ewmax) {
    auto k_num_ = k_num.view();
    auto k_den_ = k_den.view();
    auto to_ = to.view();
    auto tf_ = tf.view();
    auto sig1_ = sig1.view();
    auto c_ = c.view();
    auto root_ = root_den.view();
    auto use_exp2_ = use_exp2.view();
    size_t len = to_.shape(0);
    if (len != k_num_.shape(0) || len != k_den_.shape(0) || len != tf_.shape(0) ||
        len != sig1_.shape(0) || len != c_.shape(0) || len != root_.shape(0) || len != use_exp2_.shape(0))
      throw std::runtime_error("integ_j_ratio: shape mismatch");
    auto ret = make_numpy_array<double>({len});
    double* retp = ret.data();
    for (size_t i = 0; i < len; ++i)
      retp[i] = integ_j_ratio(k_num_(i), k_den_(i), l, to_(i), tf_(i), sig1_(i), c_(i), root_(i), use_exp2_(i), h, N, ewmax);
    return ret;
  },
        nb::arg("k_num"), nb::arg("k_den"), nb::arg("l"), nb::arg("to"), nb::arg("tf"),
        nb::arg("sig1"), nb::arg("c"), nb::arg("root_den"), nb::arg("use_exp2"),
        nb::arg("h")=0.5, nb::arg("N")=200, nb::arg("ewmax")=20.);
  nb::class_<IntensityIntegrator>(m, "IntensityIntegrator")
    .def(nb::init<>())
    .def_rw("h", &IntensityIntegrator::h)
    .def_rw("N", &IntensityIntegrator::N)
    .def_rw("ewmax", &IntensityIntegrator::ewmax)
    .def_rw("exp2_threshold", &IntensityIntegrator::exp2_threshold)
    .def("ll_int", [](const IntensityIntegrator& self, np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                      np_array<const double> S, np_array<const double> Fc, np_array<const int> c, np_array<const double> w) {
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
    .def("expected_F", [](const IntensityIntegrator& self, np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                          np_array<const double> S, np_array<const double> Fc, np_array<const int> c) {
      auto Io_ = Io.view();
      auto sigIo_ = sigIo.view();
      auto k_ani_ = k_ani.view();
      auto S_ = S.view(); // including epsilon
      auto Fc_ = Fc.view();
      auto c_ = c.view();
      size_t len = Io_.shape(0);
      if (len != sigIo_.shape(0) || len != k_ani_.shape(0) || len != S_.shape(0) || len != Fc_.shape(0) || len != c_.shape(0))
        throw std::runtime_error("expected_F: shape mismatch");
      auto ret_f = make_numpy_array<double>({len}); // <F>
      auto ret_m = make_numpy_array<double>({len}); // FOM proxy
      double *retp1 = ret_f.data(), *retp2 = ret_m.data();
      for (size_t i = 0; i < len; ++i) {
        if (std::isnan(Io_(i))) {
          retp1[i] = retp2[i] = NAN;
          continue;
        }
        const double k_den = c_(i) == 1 ? 0. : -0.5;
        const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S_(i);
        const double sqrt_sigIo = std::sqrt(sigIo_(i));
        const double tf = k_ani_(i) * Fc_(i) / sqrt_sigIo;
        const double sig1 = sq(k_ani_(i)) * S_(i) / sigIo_(i);
        const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
        const bool use_exp2 = det < self.exp2_threshold;
        const double root_den = find_root(k_den, to, tf, sig1, c_(i), det, false, use_exp2);

        const double r_1 = integ_j_ratio(k_den + 0.5, k_den, true, to, tf, sig1, c_(i), root_den, use_exp2, self.h, self.N, self.ewmax);
        const double r_2 = integ_j_ratio(k_den + 0.5, k_den, false, to, tf, sig1, c_(i), root_den, use_exp2, self.h, self.N, self.ewmax);
        retp1[i] = r_1 * std::sqrt(sigIo_(i)) / k_ani_(i);
        retp2[i] = r_1 / r_2; // should we calculate J(k_den+0.5, 1) / J(k_den+0.5, 0) directly?
      }
      return nb::make_tuple(ret_f, ret_m);
    },
         nb::arg("Io"), nb::arg("sigIo"), nb::arg("k_ani"), nb::arg("S"), nb::arg("Fc"), nb::arg("c"))
    .def("fw", [](const IntensityIntegrator& self, np_array<const double> Io, np_array<const double> sigIo, np_array<const double> k_ani,
                  np_array<const double> S0, np_array<const int> c) {
      auto Io_ = Io.view();
      auto sigIo_ = sigIo.view();
      auto k_ani_ = k_ani.view();
      auto S0_ = S0.view(); // should include epsilon
      auto c_ = c.view();
      size_t len = Io_.shape(0);
      if (len != sigIo_.shape(0) || len != k_ani_.shape(0) || len != S0_.shape(0) || len != c_.shape(0))
        throw std::runtime_error("fw: shape mismatch");
      auto ret_f = make_numpy_array<double>({len}); // <|F|>
      auto ret_f2 = make_numpy_array<double>({len}); // <|F|^2>
      double *retp1 = ret_f.data(), *retp2 = ret_f2.data();
      for (size_t i = 0; i < len; ++i) {
        if (std::isnan(Io_(i))) {
          retp1[i] = retp2[i] = NAN;
          continue;
        }
        const double k_den = c_(i) == 1 ? 0. : -0.5;
        const double to = Io_(i) / sigIo_(i) - sigIo_(i) / c_(i) / sq(k_ani_(i)) / S0_(i);
        const double sqrt_sigIo = std::sqrt(sigIo_(i));
        const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
        const bool use_exp2 = det < self.exp2_threshold;
        // tf == 0 and sig1 doesn't matter
        const double root_den = find_root(k_den, to, 0.0, 1.0, c_(i), det, false, use_exp2);

        retp1[i] = integ_j_ratio(k_den + 0.5, k_den, false, to, 0.0, 1.0, c_(i), root_den, use_exp2, self.h, self.N, self.ewmax);
        retp1[i] *= std::sqrt(sigIo_(i));
        retp2[i] = integ_j_ratio(k_den + 1.0, k_den, false, to, 0.0, 1.0, c_(i), root_den, use_exp2, self.h, self.N, self.ewmax);
        retp2[i] *= sigIo_(i);
      }
      return nb::make_tuple(ret_f, ret_f2);
    })
    .def("ll_int_der1_DS", &IntensityIntegrator::ll_int_der1_params_py<true>)
    .def("ll_int_der1_ani", &IntensityIntegrator::ll_int_der1_params_py<false>)
    .def("find_ll_int_S_from_current_estimates", &IntensityIntegrator::find_ll_int_S_from_current_estimates_py)
    .def("ll_int_der_fc", &IntensityIntegrator::ll_int_der_fc_py)
    .def("ll_int_fw_der1_S", &IntensityIntegrator::ll_int_fw_der1_params_py<true>)
    .def("ll_int_fw_der1_ani", &IntensityIntegrator::ll_int_fw_der1_params_py<false>)
    .def("ll_refine_D_S", &IntensityIntegrator::ll_refine_D_S_py)
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
