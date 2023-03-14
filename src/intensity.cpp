// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <gemmi/bessel.hpp>
#include "math.hpp"
namespace py = pybind11;
using namespace servalcat;

// for MLI
double f1_orig2(double x, double z, double to, double tf, double sig1, int c) {
  // z = 2k+1
  const double x2 = x * x;
  const double X = x * tf / sig1;
  const double log_ic0 = c == 1 ? gemmi::log_bessel_i0(2*X) : log_cosh(X);
  return 0.5 * x2 * x2 - to * x2 - z * std::log(x) - log_ic0;
}
double f1_orig2_der1(double x, double z, double to, double tf, double sig1, int c) {
  const double X = x * tf / sig1;
  const double m = fom(X, c);
  return 2 * x*x*x - 2 * to * x - z / x - m * (3-c) * tf / sig1;
}
double f1_orig2_der2(double x, double z, double to, double tf, double sig1, int c) {
  const double x2 = x * x;
  const double X = x * tf / sig1;
  const double m = fom(X, c);
  const double m_der = fom_der(m, X, c);
  return 6 * x2 - 2 * to + z / x2 - (3-c)*(3-c) * tf*tf * m_der / (sig1*sig1);
}
double f1_exp2(double y, double z, double to, double tf, double sig1, int c) {
  // z = 2k+2
  const double e_y = std::exp(-y);
  const double exp2 = std::exp(2 * (y - e_y));
  const double X = std::exp(y - e_y) * tf / sig1;
  const double log_ic0 = c == 1 ? gemmi::log_bessel_i0(2*X) : log_cosh(X);
  return 0.5 * exp2*exp2 - to * exp2 - z * (y - e_y) - log_ic0 - std::log(1 + e_y);
}
double f1_exp2_der1(double y, double z, double to, double tf, double sig1, int c) {
  const double e_y = std::exp(-y);
  const double exp2 = std::exp(2 * (y - e_y));
  const double X = std::exp(y - e_y) * tf / sig1;
  const double m = fom(X, c);
  return (1 + e_y) * (2 * exp2*exp2 - 2 * to * exp2 - z - (3-c) * tf / sig1 * m * std::exp(y - e_y)) + e_y / (1 + e_y);
}
double f1_exp2_der2(double y, double z, double to, double tf, double sig1, int c) {
  const double e_y = std::exp(-y);
  const double exp2 = std::exp(2 * (y - e_y));
  const double X = std::exp(y - e_y) * tf / sig1;
  const double m = fom(X, c);
  const double m_der = fom_der(m, X, c);
  double ret = -e_y * (2 * exp2*exp2 - 2 * to * exp2 - z - (3-c) * tf / sig1 * m * std::exp(y - e_y));
  const double tmp = (8 * sq(exp2) - 4 * to * exp2 - (3-c) * tf / sig1 * m * std::exp(y - e_y) - sq(3-c) * sq(tf / sig1) * m_der * exp2);
  ret += sq(1 + e_y) * tmp - e_y / sq(1 + e_y);
  return ret;
}
double find_initial_root_exp2(double z, double to, double tf, double sig1, int c) {
  //double a = 0;
  // if (to < 10)
  //   // small to case
  //   a = std::log(std::sqrt(0.5 * (to + std::sqrt(sq(to) + 2 * z + 2 * (3-c)))));
  // else {
  //   // assuming case 3: m = 1
  //   const double x1 = std::max(to, 0.) * tf / sig1;
  //   a = std::log(std::sqrt(0.5 * (to + std::sqrt(sq(to) + 2 * z + 2 * (3-c) * x1))));
  // }
  const double a = std::log(std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * z)));
  return a >= 0 ? a : -std::log(-a);
}

template<typename Func, typename Fprime>
double find_root(Func&& func, Fprime&& fprime, double x0) {
  // const double root = newton_or_secant([&](double x) { return fder1(x, z, to, tf, sig1, c); },
  //                                   [&](double x) { return fder2(x, z, to, tf, sig1, c); },
  //                                   x0);
  double root;
  try {
    root = newton(func, fprime, x0);
  } catch (const std::runtime_error& err) {
    // root = newton([&](double x) { return fder1(x, z, to, tf, sig1, c); },
    //               [&](double x) {
    //                 const double e = 1e-1;
    //                 return (fder1(x+e, z, to, tf, sig1, c) - fder1(x, z, to, tf, sig1, c)) / e;
    //               }, x0);
    double a = x0, b = x0, fa = func(x0), fb = func(x0);
    double inc = fa < 0 ? 0.1 : -0.1;
    for (int i = 0; i < 10000; ++i, b+=inc) { // to prevent infinite loop
      fb = func(b);
      if (fa * fb < 0) break;
    }
    if (fa * fb >= 0) throw std::runtime_error("interval not found");
    root = bisect(func, a, b, 100, 1e-2);
  }
  return root;
}

double integ_j(double k, double to, double tf, double sig1, int c, bool return_log,
               double exp2_threshold=10., double h=0.5, int N=200, double ewmax=20.) {
  if (std::isnan(to)) return NAN;
  const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k + 1)));
  const bool use_exp2 = det < exp2_threshold;
  const double z = use_exp2 ? 2 * k + 2 : 2 * k + 1;
  auto f = use_exp2 ? f1_exp2 : f1_orig2;
  auto fder1 = use_exp2 ? f1_exp2_der1 : f1_orig2_der1;
  auto fder2 = use_exp2 ? f1_exp2_der2 : f1_orig2_der2;
  const double x0 = use_exp2 ? find_initial_root_exp2(z, to, tf, sig1, c) : det;
  if (std::isinf(x0))
    printf("ERROR: x0= %f, use_exp2= %d, z=%f, to=%f, tf=%f, sig1=%f, c=%d, det=%f\n", x0, use_exp2, z, to, tf, sig1, c, det);
  const double root = find_root([&](double x) { return fder1(x, z, to, tf, sig1, c); },
                                [&](double x) { return fder2(x, z, to, tf, sig1, c); },
                                x0);
  const double f1val = f(root, z, to, tf, sig1, c);
  const double f2der = fder2(root, z, to, tf, sig1, c);
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
  return return_log ? expon + std::log(laplace_correct) : std::exp(expon) * laplace_correct;
}

double integ_j_ratio(double k_num, double k_den, bool l, double to, double tf, double sig1, int c,
                     double exp2_threshold=10., double h=0.5, int N=200, double ewmax=20.) {
  // factor of sig^{k_num - k_den} is needed, which should be done outside.
  if (std::isnan(to)) return NAN;
  const double det = std::sqrt(0.5 * x_plus_sqrt_xsq_plus_y(to, 2 * (2 * k_den + 1)));
  const bool use_exp2 = det < exp2_threshold;
  const double z = use_exp2 ? 2 * k_den + 2 : 2 * k_den + 1;
  const double deltaz = 2 * (k_num - k_den);
  auto f = use_exp2 ? f1_exp2 : f1_orig2;
  auto fder1 = use_exp2 ? f1_exp2_der1 : f1_orig2_der1;
  auto fder2 = use_exp2 ? f1_exp2_der2 : f1_orig2_der2;
  const double x0 = use_exp2 ? find_initial_root_exp2(z, to, tf, sig1, c) : det;
  if (std::isinf(x0))
    printf("ERROR: x0= %f, use_exp2= %d, z=%f, to=%f, tf=%f, sig1=%f, c=%d, det=%f\n", x0, use_exp2, z, to, tf, sig1, c, det);
  const double root = find_root([&](double x) { return fder1(x, z, to, tf, sig1, c); },
                                [&](double x) { return fder2(x, z, to, tf, sig1, c); },
                                x0);
  const double f1val = f(root, z, to, tf, sig1, c);
  const double f2der = fder2(root, z, to, tf, sig1, c);
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

// ML intensity target; -log(Io; Fc) without constants
double ll_int(double Io, double sigIo, double k_ani, double S, double Fc, int c) {
  if (std::isnan(Io)) return NAN;
  const double k = c == 1 ? 0 : -0.5;
  const double to = Io / sigIo - sigIo / c / sq(k_ani) / S;
  const double Ic = sq(Fc);
  const double tf = k_ani * Fc / std::sqrt(sigIo);
  const double sig1 = sq(k_ani) * S / sigIo;
  const double logj = integ_j(k, to, tf, sig1, c, true);
  if (c == 1) // acentrics
    return 2 * std::log(k_ani) + std::log(S) + Ic / S - logj;
  else
    return std::log(k_ani) + 0.5 * std::log(S) + 0.5 * Ic / S - logj;
}

// d/dx -log(Io; Fc) for x = D, S, k_aniso
// for Dj, Re(Fcj Fc*) needs to be multiplied
std::tuple<double,double,double>
ll_int_der1_params(double Io, double sigIo, double k_ani, double S, double Fc, int c, double eps) {
  if (std::isnan(Io)) return std::make_tuple(NAN, NAN, NAN);
  const double to = Io / sigIo - sigIo / c / sq(k_ani) / S / eps;
  const double Ic = sq(Fc);
  const double sqrt_sigIo = std::sqrt(sigIo);
  const double tf = k_ani * Fc / sqrt_sigIo;
  const double sig1 = sq(k_ani) * S / sigIo;
  const double j_ratio_1 = integ_j_ratio(c==1 ? 1 : 0.5, c==1 ? 0 : -0.5, false, to, tf, sig1, c) * sigIo;
  const double j_ratio_2 = integ_j_ratio(c==1 ? 0.5 : 0, c==1 ? 0 : -0.5, true, to, tf, sig1, c) * sqrt_sigIo;
  const double invepsS = 1. / (S * eps);
  const double invepsS2 = invepsS / S;
  if (c == 1) // acentrics
    return std::make_tuple((2 - (3-c) / k_ani / Fc * j_ratio_2) * invepsS,
                           1. / S - (Ic + j_ratio_1 / sq(k_ani) / c - (3-c) * Fc * j_ratio_2 / k_ani) * invepsS2,
                           2 / k_ani - (2 / c / k_ani * j_ratio_1 - (3-c) * Fc * j_ratio_2) / sq(k_ani) * invepsS);
  else
    return std::make_tuple((1. - (3-c) * j_ratio_2 / k_ani / Fc) * invepsS,
                           0.5 / S - 0.5 * Ic * invepsS2 - (j_ratio_1 / c / k_ani - (3-c) * Fc * j_ratio_2) / k_ani * invepsS2,
                           1 / k_ani - (2 * j_ratio_1 / c / k_ani - (3-c) * Fc * j_ratio_2) * invepsS / sq(k_ani));
}

py::array_t<double>
ll_int_der1_params_py(py::array_t<double> Io, py::array_t<double> sigIo, py::array_t<double> k_ani,
                      double S, py::array_t<std::complex<double>> Fcs, std::vector<double> Ds,
                      py::array_t<int> c, py::array_t<int> eps) {
  if (Ds.size() != (size_t)Fcs.shape(1)) throw std::runtime_error("Fc and D shape mismatch");
  size_t n_models = Fcs.shape(1);
  size_t n_ref = Fcs.shape(0);
  size_t n_cols = n_models + 2;
  auto Io_ = Io.unchecked<1>();
  auto sigIo_ = sigIo.unchecked<1>();
  auto k_ani_ = k_ani.unchecked<1>();
  //auto S_ = S.unchecked<1>(); // should take just one?
  auto Fcs_ = Fcs.unchecked<2>();
  //auto Ds_ = Ds.unchecked<2>();
  auto c_ = c.unchecked<1>();
  auto eps_ = eps.unchecked<1>();

  // der1 wrt D1, D2, .., S, k_ani
  auto ret = py::array_t<double>({n_ref, n_cols});
  double* ptr = (double*) ret.request().ptr;
  auto sum_Fc = [&](int i) {
                  std::complex<double> s = Fcs_(i, 0) * Ds[0];
                  for (size_t j = 1; j < n_models; ++j)
                    s += Fcs_(i, j) * Ds[j];
                  return s;
                };
  for (size_t i = 0; i < n_ref; ++i) {
    const std::complex<double> Fc_total_conj = std::conj(sum_Fc(i));
    const auto v = ll_int_der1_params(Io_(i), sigIo_(i), k_ani_(i), S, std::abs(Fc_total_conj), c_(i), eps_(i));
    for (size_t j = 0; j < n_models; ++j) {
      const double r_fcj_fc = (Fcs_(i, j) * Fc_total_conj).real();
      ptr[i*n_cols + j] = std::get<0>(v) * r_fcj_fc;
    }
    ptr[i*n_cols + n_models] = std::get<1>(v);
    ptr[i*n_cols + n_models + 1] = std::get<2>(v);
  }
  return ret;
}

// For French-Wilson

double f1_exp2_fw(double x, double z, double t0) {
  double ex = std::exp(-x);
  double exx = x - ex;
  double ex1 = std::exp(2.0 * exx);
  return (ex1 * ex1 - 2.0 * t0 * ex1) / 2.0 - 4 * z * exx - std::log(1.0 + ex);
}
double f1_orig2_fw(double x, double z, double t0) {
  double x2 = x * x;
  return 0.5 * (x2 * x2 - 2 * t0 * x2) - (4 * z - 1) * std::log(x);
}

template<bool J1>
double integ_j_fw(double delta, double root, double to1,
                  double f1val, double z, int N, double ewmax) {
  auto fun = J1 ? f1_exp2_fw : f1_orig2_fw;
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
  auto fun1 = J1 ? f1_exp2_fw : f1_orig2_fw;
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
  m.def("integ_J", py::vectorize(integ_j),
        py::arg("k"), py::arg("to"), py::arg("tf"), py::arg("sig1"), py::arg("c"), py::arg("return_log"),
        py::arg("exp2_threshold")=10, py::arg("h")=0.5, py::arg("N")=200, py::arg("ewmax")=20.);
  m.def("integ_J_ratio", py::vectorize(integ_j_ratio),
        py::arg("k_num"), py::arg("k_den"), py::arg("l"), py::arg("to"), py::arg("tf"),
        py::arg("sig1"), py::arg("c"),
        py::arg("exp2_threshold")=10, py::arg("h")=0.5, py::arg("N")=200, py::arg("ewmax")=20.);
  m.def("ll_int", py::vectorize(ll_int),
        py::arg("Io"), py::arg("sigIo"), py::arg("k_ani"), py::arg("S"), py::arg("Fc"), py::arg("c"));
  m.def("ll_int_der1_params", &ll_int_der1_params_py);
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
