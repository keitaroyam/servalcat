// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_MATH_HPP_
#define SERVALCAT_MATH_HPP_

#include "lambertw.hpp"
#include <gemmi/bessel.hpp>    // for log_bessel_i0, bessel_i1_over_i0
#include <gemmi/math.hpp>    //   for log_cosh
#include <Eigen/Dense>

namespace servalcat {

template<typename T>
constexpr T sq(T x) {return x * x;}

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

inline double fom_safe(double X, int c, double *der1=nullptr, double *der2=nullptr) {
  double m;
  if (c == 1) // acentrics
    m = X > 50. ? (1.0 - 1.0 / (4.0 * X) - 1.0 / (32.0 * X * X)) : gemmi::bessel_i1_over_i0(2 * X);
  else // c == 2, centrics
    m = X > 50. ? 1.0 : std::tanh(X);

  if (!der1 && !der2) return m;

  double m_der = 0.0;
  if (c == 1) {
    if (X > 50.0) {
      const double invX2 = 1.0 / (X * X);
      m_der = invX2 * (0.25 + 0.0625 / X);
    } else if (X < 1e-4)
      m_der = 1.0 - 1.5 * X * X;
    else
      m_der = 2.0 * (1.0 - m / (2.0 * X) - m * m);
  } else
    m_der = X > 50. ? 0.0 : (1. - m * m);

  if (der1) *der1 = m_der;

  if (der2) {
    if (c == 1) {
      if (X > 50.0) {
        const double invX = 1.0 / X;
        const double invX3 = invX * invX * invX;
        *der2 = -0.5 * invX3 - 0.1875 * invX3 * invX - 0.09375 * invX3 * invX * invX;
      } else if (X < 1e-3) {
        const double X2 = X * X;
        *der2 = -3.0 * X + (20.0 / 3.0) * X * X2 - 8.75 * X * X2 * X2;
      } else
        *der2 = m_der * (-4.0 * m - 1.0 / X) + m / (X * X);
    } else
      *der2 = X > 50.0 ? 0.0 : (-2.0 * m * m_der);
  }

  return m;
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

template <typename Func>
double brent(Func&& func, double a, double b, int maxiter=100, double atol=1e-15) {
  double fa = func(a);
  double fb = func(b);

  if (fa == 0.0) return a;
  if (fb == 0.0) return b;

  // Endpoints must bracket a root
  if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)) {
    throw std::invalid_argument("Endpoints a and b must have opposite signs in brent().");
  }

  if (std::abs(fa) < std::abs(fb)) {
    std::swap(a, b);
    std::swap(fa, fb);
  }

  double c = a;
  double fc = fa;
  double d = b - a;
  double e = d;

  const double eps = std::numeric_limits<double>::epsilon();

  for (int iter = 0; iter < maxiter; ++iter) {
    if (fb == 0.0) return b;

    // Reset step if sign changed
    if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }

    if (std::abs(fc) < std::abs(fb)) {
      a = b; fa = fb;
      b = c; fb = fc;
      c = a; fc = fa;
    }

    double m = 0.5 * (c - b);
    double tol1 = 2.0 * eps * std::abs(b) + atol;

    if (std::abs(m) <= tol1 || m == 0.0) {
      return b;
    }

    // Try inverse quadratic interpolation or secant method
    if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
      double p, q;
      if (a != c) {
        // Inverse quadratic interpolation
        double r1 = fa / fc;
        double r2 = fb / fc;
        double r3 = fb / fa;
        p = r3 * (2.0 * m * r1 * (r1 - r2) - (b - a) * (r2 - 1.0));
        q = (r1 - 1.0) * (r2 - 1.0) * (r3 - 1.0);
      } else {
        // Secant method
        double s = fb / fa;
        p = 2.0 * m * s;
        q = 1.0 - s;
      }

      if (p > 0.0) {
        q = -q;
      } else {
        p = -p;
      }

      // Accept interpolation step or fall back to bisection
      if ((2.0 * p < 3.0 * m * q - std::abs(tol1 * q)) && (p < std::abs(0.5 * e * q))) {
        e = d;
        d = p / q;
      } else {
        d = m;
        e = m;
      }
    } else {
      // Bisection step
      d = m;
      e = m;
    }

    a = b;
    fa = fb;

    if (std::abs(d) > tol1) {
      b += d;
    } else {
      b += (m > 0.0) ? tol1 : -tol1;
    }

    fb = func(b);
  }

  return b;
}

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
  SymMatEig(const Eigen::MatrixXd &m) : es(m), n_nonzero_rows(m.rows()) {
    // check zero-valued rows which should give zero eigenvalues
    for (int i = 0; i < m.rows(); ++i)
      if ((m.row(i).array() == 0).all())
        --n_nonzero_rows;
  }
  double det(bool exclude_zero=false) const {
    if (exclude_zero) {
      auto v = es.eigenvalues();
      // eigenvalue is sorted, but not with its magnitude
      std::sort(v.data(), v.data() + v.size(),
                [](double a, double b) { return std::abs(a) < std::abs(b); });
      v = v.tail(n_nonzero_rows);
      // filtering no longer needed, but just in case
      return (v.array().abs() < 1e-13).select(1, v).prod();
    }
    return es.eigenvalues().prod();
  }
  Eigen::MatrixXd inv(double e=1e-8) const {
    Eigen::VectorXd eig_inv = es.eigenvalues();
    for (int i = 0; i < eig_inv.size(); ++i)
      eig_inv(i) = std::abs(eig_inv(i)) < e ? 1 : (1. / eig_inv(i));
    return es.eigenvectors() * eig_inv.asDiagonal() * es.eigenvectors().adjoint();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
  int n_nonzero_rows;
};

} // namespace servalcat
#endif
