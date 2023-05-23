// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_MATH_HPP_
#define SERVALCAT_MATH_HPP_

#include "lambertw.hpp"

namespace servalcat {

double sq(double x) {return x * x;}

double log_cosh(double x) {
  // from gemmi/python/gemmi.cpp
  x = std::abs(x);
  return x + std::log1p(std::exp(-2 * x)) - std::log(2);
}

double fom(double X, int c) {
  return c == 1 ? gemmi::bessel_i1_over_i0(2*X) : std::tanh(X);
}

double fom_der(double m, double X, int c) {
  return c == 1 ? 1 - 0.5 * m / X - m * m : 1 - m * m;
}

double x_plus_sqrt_xsq_plus_y(double x, double y) {
  // avoid precision loss
  const double tmp = std::sqrt(sq(x) + y);
  return x < 0 ? y / (tmp - x) : x + tmp;
}

// solve y - exp(-y) = x for y.
// solution is y = W(exp(-x)) + x
double solve_y_minus_exp_minus_y(double x, double prec) {
  if (x > 20) return x;
  return lambertw::lambertw(std::exp(-x), prec) + x;
}

template<typename Func, typename Fprime>
double newton(Func&& func, Fprime&& fprime, double x0,
	      int maxiter=50, double tol=1.48e-8) {
  double x = x0;
  for (int itr = 0; itr < maxiter; ++itr) {
    double fval = func(x0);
    if (fval == 0)
      return x0;
    double fder = fprime(x0);
    if (fder == 0)
      throw std::runtime_error("newton did not converge (der=0)");
    x = x0 - fval / fder;
    if (std::abs(x - x0) < tol)
      return x;
    x0 = x;
  }
  throw std::runtime_error("newton did not converge");
}

template<typename Func>
double secant(Func&& func, double x0,
	      int maxiter=50, double tol=1.48e-8) {
  const double eps = 1e-1;
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
    if (q1 == q0) {
      if (p1 != p0)
	throw std::runtime_error("secant did not converge: x= " + std::to_string(q1));
      return 0.5 * (p1 + p0);
    } else {
      if (std::abs(q1) > std::abs(q0))
	p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1);
      else
	p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
    }
    if (std::abs(p - p1) < tol)
      return p;
    p0 = p1;
    q0 = q1;
    p1 = p;
    q1 = func(p1);
  }
  throw std::runtime_error("secant did not converge: x= " + std::to_string(q1));
}

template<typename Func, typename Fprime>
double newton_or_secant(Func&& func, Fprime&& fprime, double x0,
			int maxiter=50, double tol=1.48e-8) {
  try {
    return newton(func, fprime, x0, maxiter, tol);
  } catch (const std::runtime_error& e) {
    return secant(func, x0, maxiter, tol);
  }  
}


template<typename Func>
double bisect(Func&& func, double a, double b,
 	      int maxiter=100, double tol=1.48e-8) {
  if (a > b)
    std::swap(a, b);
  if (func(a) * func(b) >= 0)
    throw std::runtime_error("fa * fb >= 0");

  for (int itr = 0; itr < maxiter; ++itr) {
    double c = 0.5 * (a + b);
    if (func(c) == 0 || 0.5 * (b - a) < tol)
      return c;
    if (func(c) * func(a) >= 0)
      a = c;
    else
      b = c;
  }
  throw std::runtime_error("bisect did not converge: c= " + std::to_string(0.5*(a+b)));
}

} // namespace servalcat
#endif
