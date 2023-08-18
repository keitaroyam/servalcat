#ifndef ROBERTDJ_LAMBERTW_HPP_
#define ROBERTDJ_LAMBERTW_HPP_
#include <cmath>
// This code was taken from https://github.com/robertdj/LambertW.jl

// The LambertW.jl package is licensed under the MIT "Expat" License:
//     Copyright (c) 2014: Robert Dahl Jacobsen.
//     Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//     The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

namespace lambertw {
inline double lambertw_approx(double x) {
  const double e = std::exp(1.);
  if (x <= 1) {
    const double sqrt2 = std::sqrt(2.);
    const double sqeta = std::sqrt(2 + 2 * e * x);
    const double N2 = 3 * sqrt2 + 6 - (((2237. + 1457 * sqrt2) * e - 4108 * sqrt2 - 5764) * sqeta) / ((215. + 199 * sqrt2) * e - 430 * sqrt2 - 796);
    const double N1 = (1. - 1. / sqrt2) * (N2 + sqrt2);
    return -1 + sqeta / (1. + N1 * sqeta / (N2 + sqeta));
  } else
    return std::log( 6 * x / (5 * std::log( 12. / 5. * (x / std::log(1. + 12 * x / 5.)))));
}

// may not work if x < exp(-36)
inline double lambertw(double x, double prec) {
  if (x < -std::exp(-1.))
    return NAN;

  // First approximation
  double W = lambertw_approx(x);

  // Compute residual using logarithms to avoid numerical overflow
  // When x == 0, r = NaN, but here the approximation is exact and
  // the while loop below is unnecessary
  double r = std::abs(W - std::log(std::abs(x)) + std::log(std::abs(W)));

  // Apply Fritsch’s method to increase precision
  for (int i = 0; i < 5; ++i) {
    if (r <= prec) break;
    const double z = std::log(x / W) - W;
    const double q = 2 * (1. + W) * (1 + W + 2. / 3. * z);
    const double epsilon = z * (q - z) / ((1 + W) * (q - 2 * z));
    W *= 1 + epsilon;
    r = std::abs(W - std::log(std::abs(x)) + std::log(std::abs(W)));
  }
  return W;
}

}
#endif
