// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_MATH_HPP_
#define SERVALCAT_MATH_HPP_

#include "lambertw.hpp"
#include <gemmi/bessel.hpp>    // for log_bessel_i0, bessel_i1_over_i0
#include <gemmi/math.hpp>    //   for log_cosh
#include <Eigen/Dense>

namespace servalcat {

constexpr double sq(double x) {return x * x;}

inline double log_i0_or_cosh(double X, int c) {
  return c == 1 ? gemmi::log_bessel_i0(2*X) : gemmi::log_cosh(X);
}

inline double fom(double X, int c) {
  return c == 1 ? gemmi::bessel_i1_over_i0(2*X) : std::tanh(X);
}

inline double fom_der(double m, double X, int c) {
  // c=1: d/dX I1(2X)/I0(2X)
  // c=2: d/dX tanh(X)
  return c == 1 ? 2 - m / X - 2 * sq(m) : 1 - sq(m);
}

inline double fom_der2(double m, double X, int c) {
  // c=1: d^2/dX^2 I1(2X)/I0(2X)
  // c=2: d^2/dX^2 tanh(X)
  return c == 1 ? m / sq(X) - (4 * m + 1 / X) * (2 - m / X - 2 * sq(m)) : 2 * m * (sq(m) - 1);
}

inline double x_plus_sqrt_xsq_plus_y(double x, double y) {
  // avoid precision loss
  const double tmp = std::sqrt(sq(x) + y);
  return x < 0 ? y / (tmp - x) : x + tmp;
}

// solve y - exp(-y) = x for y.
// solution is y = W(exp(-x)) + x
inline double solve_y_minus_exp_minus_y(double x, double prec) {
  if (x > 20) return x;
  return lambertw::lambertw(std::exp(-x), prec) + x;
}

struct RootfindResult {
  bool success = false;
  std::string reason;
  int iter = 0;
  double x;
};

template<typename Func, typename Fprime>
RootfindResult newton(Func&& func, Fprime&& fprime, double x0,
                      int maxiter=50, double tol=1.48e-8) {
  RootfindResult ret;
  ret.x = x0;
  for (int itr = 0; itr < maxiter; ++itr) {
    ++ret.iter;
    double fval = func(x0);
    if (fval == 0) {
      ret.success = true;
      return ret;
    }
    double fder = fprime(x0);
    if (fder <= 0) {
      ret.reason = "fder <= 0";
      return ret;
    }
    ret.x = x0 - fval / fder;
    if (std::abs(ret.x - x0) < tol) {
      ret.success = true;
      return ret;
    }
    x0 = ret.x;
  }
  ret.reason = "maxiter reached";
  return ret;
}

template<typename Func>
RootfindResult secant(Func&& func, double x0,
                      int maxiter=50, double tol=1.48e-8) {
  const double eps = 1e-1;
  RootfindResult ret;
  double p = x0, p0 = x0;
  double p1 = x0 * (1 + eps);
  p1 += p1 >= 0 ? eps : -eps;
  double q0 = func(p0);
  double q1 = func(p1);
  if (std::abs(q1) < std::abs(q0)) {
    std::swap(p0, p1);
    std::swap(q0, q1);
  }
  for (int itr = 0; itr < maxiter; ++itr) {
    ++ret.iter;
    if (q1 == q0) {
      ret.x = 0.5 * (p1 + p0);
      if (p1 != p0)
        return ret;
      ret.success = true;
      return ret;
    } else {
      if (std::abs(q1) > std::abs(q0))
        p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1);
      else
        p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
    }
    ret.x = p;
    if (std::abs(p - p1) < tol) {
      ret.success = true;
      return ret;
    }
    p0 = p1;
    q0 = q1;
    p1 = p;
    q1 = func(p1);
  }
  ret.reason = "maxiter reached";
  return ret;
}

template<typename Func, typename Fprime>
RootfindResult newton_or_secant(Func&& func, Fprime&& fprime, double x0,
                                int maxiter=50, double tol=1.48e-8) {
  auto res = newton(func, fprime, x0, maxiter, tol);
  if (res.success)
    return res;
  return secant(func, x0, maxiter, tol);
}


template<typename Func>
RootfindResult bisect(Func&& func, double a, double b,
                      int maxiter=100, double tol=1.48e-8) {
  RootfindResult ret;
  if (a > b)
    std::swap(a, b);
  if (func(a) * func(b) >= 0) {
    ret.reason = "fa * fb >= 0";
    return ret;
  }
  for (int itr = 0; itr < maxiter; ++itr) {
    ++ret.iter;
    ret.x = 0.5 * (a + b);
    if (func(ret.x) == 0 || 0.5 * (b - a) < tol) {
      ret.success = true;
      return ret;
    }
    if (func(ret.x) * func(a) >= 0)
      a = ret.x;
    else
      b = ret.x;
  }
  ret.reason = "maxiter reached";
  return ret;
}

inline double procrust_dist(Eigen::MatrixXd x, Eigen::MatrixXd y) {
  if (x.rows() != y.rows() || x.cols() != y.cols() || x.cols() != 3)
    throw std::runtime_error("procrust_dist: dimension mismatch");
  if (!x.size()) return NAN;
  const Eigen::Vector3d xmean = x.colwise().mean(), ymean = y.colwise().mean();
  x.rowwise() -= xmean.transpose();
  y.rowwise() -= ymean.transpose();
  const Eigen::Matrix3d xty = x.transpose() * y;
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(xty);
  double dist = -2.0 * svd.singularValues().sum() + x.squaredNorm() + y.squaredNorm();
  dist = std::sqrt(std::max(0., dist) / x.rows());
  return dist;
}

struct SymMatEig {
  SymMatEig(const Eigen::MatrixXd &m) : es(m) {}
  double det() const {
    return es.eigenvalues().prod();
  }
  Eigen::MatrixXd inv(double e=1e-8) const {
    Eigen::VectorXd eig_inv = es.eigenvalues();
    for (int i = 0; i < eig_inv.size(); ++i)
      eig_inv(i) = std::abs(eig_inv(i)) < e ? 1 : (1. / eig_inv(i));
    return es.eigenvectors() * eig_inv.asDiagonal() * es.eigenvectors().adjoint();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
};

} // namespace servalcat
#endif
