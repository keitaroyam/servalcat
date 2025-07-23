// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <cmath> // for NAN
#include <gemmi/symmetry.hpp>
#include <gemmi/unitcell.hpp>
#include <complex>
#include <vector>
#include <map>
#include <numeric>  // for iota
#include <iostream>
#include <fstream>
#include "math.hpp"
#include "array.h"
namespace nb = nanobind;
using namespace servalcat;

struct TwinData {
  std::vector<gemmi::Op::Miller> asu;
  std::vector<int> centric;
  std::vector<double> epsilon;
  std::vector<double> alphas;
  std::vector<double> f_true_max;
  std::vector<double> s2_array;
  std::vector<std::complex<double>> f_calc_;
  std::vector<double> mlparams; // (Sigma, D0, D1, ..), ...
  std::ofstream ofs_debug;
  std::complex<double> &f_calc(size_t idx, size_t i_model) {
    return f_calc_[idx * n_models + i_model];
  }
  const std::complex<double> &f_calc(size_t idx, size_t i_model) const {
    return const_cast<TwinData*>(this)->f_calc(idx, i_model);
  }
  double &ml_scale(size_t ibin, size_t i_model) {
    return mlparams.at(ibin * (n_models + 1) + i_model + 1);
  }
  const double &ml_scale(size_t ibin, size_t i_model) const {
    return const_cast<TwinData*>(this)->ml_scale(ibin, i_model);
  }
  double &ml_sigma(size_t ibin) {
    return mlparams[ibin * (n_models + 1)];
  }
  const double &ml_sigma(size_t ibin) const {
    return const_cast<TwinData*>(this)->ml_sigma(ibin);
  }
  std::complex<double> sum_fcalc(size_t idx, bool with_D) const {
    std::complex<double> ret = 0;
    for (int i = 0; i < n_models; ++i)
      ret += f_calc(idx, i) * (with_D ? ml_scale(bin[idx], i) : 1);
    return ret;
  }
  std::vector<int> bin;
  std::vector<gemmi::Op> ops;
  size_t n_models = 0; // number of models (including solvent)
  //gemmi::UnitCell cell;
  //const gemmi::SpaceGroup *sg;

  // References
  // this may be slow. should we use 1d array and access via function?
  std::vector<std::vector<size_t>> rb2o; // [i_block][i_obs] -> index of iobs
  std::vector<std::vector<size_t>> rb2a; // [i_block][i] -> index of asu (for P(F; Fc))
  std::vector<std::vector<std::vector<size_t>>> rbo2a; // [i_block][i_obs][i] -> index of rb2a
  std::vector<std::vector<std::vector<size_t>>> rbo2c; // [i_block][i_obs][i] -> index of alpha
  std::vector<int> rbin; // [i_block] -> bin

  void clear() {
    *this = TwinData(); // ok?
    // asu.clear();
    // centric.clear();
    // epsilon.clear();
    // alphas.clear();
    // rb2o.clear();
    // rb2a.clear();
    // rbo2a.clear();
    // rbo2c.clear();
    // n_models = 0;
  }

  double d_min(const gemmi::UnitCell &cell) const {
    double s2max = 0;
    for (const auto &h : asu)
      s2max = std::max(s2max, cell.calculate_1_d2(h));
    return 1. / std::sqrt(s2max);
  }

  int idx_of_asu(const gemmi::Op::Miller &h) const {
    auto it = std::lower_bound(asu.begin(), asu.end(), h);
    if (it != asu.end() && *it == h)
      return std::distance(asu.begin(), it);
    //throw std::runtime_error("hkl not found in asu");
    return -1; // may happen due to non/pseudo-merohedral and systematic absence
  }
  size_t n_obs() const {
    size_t ret = 0;
    for (const auto x : rb2o)
      ret += x.size();
    return ret;
  }
  size_t n_ops() const { // include identity
    return ops.size() + 1;
  }

  void setup_f_calc(size_t n) {
    n_models = n;
    f_calc_.assign(n_models * asu.size(), 0);
    if (!bin.empty()) {
      const int bin_max = *std::max_element(bin.begin(), bin.end());
      mlparams.assign((n_models + 1) * (bin_max + 1), 0);
    }
  }

  void setup(const std::vector<gemmi::Op::Miller> &hkls,
             const std::vector<int> &bins,
             const gemmi::SpaceGroup &sg,
             const gemmi::UnitCell &cell,
             const std::vector<gemmi::Op> &operators) {
    clear();
    ops = operators;
    const gemmi::GroupOps gops = sg.operations();
    const gemmi::ReciprocalAsu rasu(&sg);
    auto apply_and_asu = [&rasu, &gops](const gemmi::Op &op, const gemmi::Op::Miller &h) {
      return rasu.to_asu(op.apply_to_hkl(h), gops).first;
    };
    alphas.assign(operators.size() + 1, 0.);
    std::map<gemmi::Op::Miller, int> bin_map;
    // Set asu
    for (int i = 0; i < hkls.size(); ++i) {
      const auto h = hkls[i];
      // assuming hkl is in ASU - but may not be complete?
      if (!gops.is_systematically_absent(h)) {
        asu.push_back(h);
        bin_map.emplace(h, bins[i]);
      }
      for (const auto &op : operators) {
        const auto hr = apply_and_asu(op, h);
        if (!gops.is_systematically_absent(hr)) {
          asu.push_back(hr);
          bin_map.emplace(hr, bins[i]); // this isn't always correct, if pseudo-merohedral
        }
      }
    }
    std::sort(asu.begin(), asu.end());
    asu.erase(std::unique(asu.begin(), asu.end()), asu.end());
    epsilon.reserve(asu.size());
    s2_array.reserve(asu.size());
    centric.reserve(asu.size());
    bin.reserve(asu.size());
    f_true_max.assign(asu.size(), NAN);
    for (const auto &h : asu) {
      epsilon.push_back(gops.epsilon_factor_without_centering(h));
      centric.push_back(gops.is_reflection_centric(h));
      s2_array.push_back(cell.calculate_1_d2(h));
      bin.push_back(bin_map[h]);
    }
    // Permutation sort
    std::vector<int> perm(hkls.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](int lhs, int rhs) {return hkls[lhs] < hkls[rhs];});
    // Loop over hkls
    std::vector<bool> done(hkls.size());
    const auto append_if_good = [](std::vector<size_t> &vec, int i) {
      if (i < 0) return false;
      if (std::find(vec.begin(), vec.end(), i) != vec.end()) // if already exists
        return false;
      vec.push_back(i);
      return true;
    };
    for (int i = 0; i < perm.size(); ++i) {
      const auto &h = hkls[perm[i]];
      if (done[perm[i]]) continue;
      rbin.push_back(bins[perm[i]]); // first encounter
      rbo2a.emplace_back();
      rbo2c.emplace_back();
      rb2o.emplace_back();
      rb2a.emplace_back();
      append_if_good(rb2a.back(), idx_of_asu(h));
      // loop over same hkls (would not happen if unique set was given)
      for (int j = i; j < perm.size() && hkls[perm[j]] == h; ++j) {
        rb2o.back().push_back(perm[j]);
        done[perm[j]] = true;
      }
      // A loop instead of recursion for safety.
      // Even in merohedral cases, if some operators are omitted,
      // single loop is not sufficient to generate all related hkls
      // TODO: do we need the same treatment when setting up asu?
      std::vector<gemmi::Op::Miller> tmp_stack = {h};
      for (int i_cyc = 0; ; ++i_cyc) {
        if (i_cyc > 99)
          throw std::runtime_error("twin: maximum iteration reached. twin operator invalid.");
        if (tmp_stack.empty())
          break;
        const auto hh = tmp_stack.back();
        tmp_stack.pop_back();
        // loop over twin related
        for (const auto &op : operators) {
          const auto hr = apply_and_asu(op, hh);
          for (auto it = std::lower_bound(perm.begin(), perm.end(), hr,
                                          [&](int lhs, const gemmi::Op::Miller &rhs) {return hkls[lhs] < rhs;});
               it != perm.end() && hkls[*it] == hr; ++it) {
            size_t j = std::distance(perm.begin(), it);
            if (done[perm[j]]) continue;
            rb2o.back().push_back(perm[j]);
            done[perm[j]] = true;
          }
          if (append_if_good(rb2a.back(), idx_of_asu(hr)))
            tmp_stack.push_back(hr);
        }
      }
      std::sort(rb2a.back().begin(), rb2a.back().end());
      rb2a.back().erase(std::unique(rb2a.back().begin(), rb2a.back().end()), rb2a.back().end());
      const auto idx_of_rb2a = [&](size_t h) {
        auto it = std::lower_bound(rb2a.back().begin(), rb2a.back().end(), h);
        if (it != rb2a.back().end() && *it == h)
          return std::distance(rb2a.back().begin(), it);
        throw std::runtime_error("hkl not found in rb2a"); // should not happen
      };
      for (auto j : rb2o.back()) {
        const auto &h2 = hkls[j];
        rbo2a.back().emplace_back();
        rbo2c.back().emplace_back();
        const int ia  = idx_of_asu(h2);
        if (ia >= 0) {
          rbo2a.back().back().push_back(idx_of_rb2a(ia));
          rbo2c.back().back().push_back(0);
        }
        for (int k = 0; k < operators.size(); ++k) {
          const auto h2r = apply_and_asu(operators[k], h2);
          const int iar  = idx_of_asu(h2r);
          if (iar >= 0) {
            rbo2a.back().back().push_back(idx_of_rb2a(iar));
            rbo2c.back().back().push_back(k + 1);
          }
        }
      }
    }
  }

  // calculation of f(x), which is part of -LL = -log \int exp(-f(x)) dx + (1+c)^-1 log Sigma
  double calc_f(int ib, const double *iobs, const double *sigo, const Eigen::VectorXd &f_true) const {
    double ret = 0;
    for (int io = 0; io < rb2o[ib].size(); ++io) {
      const int obs_idx = rb2o[ib][io];
      if (std::isnan(iobs[obs_idx]))
        continue;
      double i_true_twin = 0;
      for (int ic = 0; ic < rbo2a[ib][io].size(); ++ic)
        i_true_twin += alphas[rbo2c[ib][io][ic]] * gemmi::sq(f_true(rbo2a[ib][io][ic]));
      ret += gemmi::sq((iobs[obs_idx] - i_true_twin) / sigo[obs_idx]) * 0.5;
    }
    for (int ia = 0; ia < rb2a[ib].size(); ++ia) {
      const int a_idx = rb2a[ib][ia];
      const int c = centric[a_idx] + 1;
      const double den = epsilon[a_idx] * ml_sigma(bin[a_idx]);
      const std::complex<double> DFc = sum_fcalc(a_idx, true);
      ret += (gemmi::sq(f_true(ia)) + std::norm(DFc)) / den / c;
      const double X = std::abs(DFc) * f_true(ia) / den;
      ret -= log_i0_or_cosh(X, c);
      if (c == 1) // acentric
        ret -= std::log(f_true(ia));
    }
    if (0 && std::isnan(ret)) {
      std::cout << "f_nan [";
      for (int ia = 0; ia < rb2a[ib].size(); ++ia) {
        const int a_idx = rb2a[ib][ia];
        std::cout << f_true(ia) << " "
                  << asu[a_idx][0] << " " << asu[a_idx][1] << " " << asu[a_idx][2]
                  << ", ";
      }
      std::cout << "]" << std::endl;
    }
    return ret;
  }

  // first and second derivative matrix of f(x)
  std::pair<Eigen::VectorXd, Eigen::MatrixXd>
  calc_f_der(int ib, const double *iobs, const double *sigo, const Eigen::VectorXd &ft,
             bool unstable_mode=false) const {
    const size_t n_a = rb2a[ib].size();
    Eigen::VectorXd der1 = Eigen::VectorXd::Zero(n_a);
    Eigen::MatrixXd der2 = Eigen::MatrixXd::Zero(n_a, n_a);
    for (int io = 0; io < rb2o[ib].size(); ++io) {
      const int obs_idx = rb2o[ib][io];
      if (std::isnan(iobs[obs_idx]))
        continue;
      const double inv_varobs = 1. / gemmi::sq(sigo[obs_idx]);
      double i_true_twin = 0;
      for (int ic = 0; ic < rbo2a[ib][io].size(); ++ic)
        i_true_twin += alphas[rbo2c[ib][io][ic]] * gemmi::sq(ft(rbo2a[ib][io][ic]));
      for (int ic = 0; ic < rbo2a[ib][io].size(); ++ic) {
        const double tmp = 2 * alphas[rbo2c[ib][io][ic]] * ft(rbo2a[ib][io][ic]);
        der1(rbo2a[ib][io][ic]) -= (iobs[obs_idx] - i_true_twin) * inv_varobs * tmp;
      }
    }
    for (int i = 0; i < n_a; ++i)
      for (int j = i; j < n_a; ++j) {
        for (int io = 0; io < rb2o[ib].size(); ++io) {
          const int obs_idx = rb2o[ib][io];
          if (std::isnan(iobs[obs_idx]))
            continue;
          double i_true_twin = 0;
          for (int ic = 0; ic < rbo2a[ib][io].size(); ++ic)
            i_true_twin += alphas[rbo2c[ib][io][ic]] * gemmi::sq(ft(rbo2a[ib][io][ic]));
          double tmp1 = 0, tmp2 = 0, tmp3 = 0;
          const double inv_varobs = 1. / gemmi::sq(sigo[obs_idx]);
          for (int ic = 0; ic < rbo2a[ib][io].size(); ++ic) {
            const double a_f = 2 * alphas[rbo2c[ib][io][ic]] * ft(rbo2a[ib][io][ic]);
            if (rbo2a[ib][io][ic] == i)
              tmp1 += a_f;
            if (rbo2a[ib][io][ic] == j)
              tmp2 += a_f;
            if (i == j && rbo2a[ib][io][ic] == i)
              tmp3 += 2 * (iobs[obs_idx] - i_true_twin) * alphas[rbo2c[ib][io][ic]];
          }
          // der2(i, j) += (tmp1 * tmp2 - tmp3) * inv_varobs;
          der2(i, j) += (tmp1 * tmp2) * inv_varobs; // should be more stable?
        }
        if (i != j)
          der2(j, i) = der2(i, j);
      }
    for (int ia = 0; ia < n_a; ++ia) {
      const int a_idx = rb2a[ib][ia];
      const int c = centric[a_idx] + 1;
      const double inv_den = 1. / (epsilon[a_idx] * ml_sigma(bin[a_idx]));
      const std::complex<double> DFc = sum_fcalc(a_idx, true);
      // printf("ia = %d (%d %d %d) c = %d eps= %f S= %f inv_den = %f\n",
      //     ia, asu[a_idx][0], asu[a_idx][1], asu[a_idx][2],
      //     c, epsilon[a_idx], S, inv_den);
      der1(ia) += 2 * ft(ia) * inv_den / c;
      der2(ia, ia) += 2 * inv_den / (1. + c);
      const double X = std::abs(DFc) * ft(ia) * inv_den;
      const double m = fom(X, c);
      const double f_inv_den = std::abs(DFc) * inv_den;
      der1(ia) -= m * f_inv_den * (3 - c);
      if (unstable_mode)
        der2(ia, ia) -= fom_der(m, X, c) * gemmi::sq(f_inv_den) * (3 - c); // omit for stability
      if (c == 1) { // acentric
        der1(ia) -= 1. / ft(ia);
        der2(ia, ia) += 1. / gemmi::sq(ft(ia));
      }
    }
    return std::make_pair(der1, der2);
  }

  // Note that f_calc refers to asu, while iobs/sigo refer to observation list
  void est_f_true(int ib, const double *iobs, const double *sigo, int max_cycle=10) {
    if (ib < 0 || ib > rb2o.size())
      throw std::out_of_range("twin_ll: bad ib");
    for (int i = 0; i < rb2a[ib].size(); ++i)
      f_true_max[rb2a[ib][i]] = NAN;
    // skip if no observation at all
    bool has_obs = false;
    for (int io = 0; io < rb2o[ib].size(); ++io)
      if (!std::isnan(iobs[rb2o[ib][io]]))
        has_obs = true;
    if (!has_obs)
      return;

    // Initial estimate
    std::vector<double> f_est(rb2a[ib].size());
    for (int io = 0; io < rb2o[ib].size(); ++io) {
      const int obs_idx = rb2o[ib][io];
      if (std::isnan(iobs[obs_idx]))
        continue;
      const double i_obs = std::max(0.001 * sigo[obs_idx], iobs[obs_idx]);
      double i_calc_twin = 0;
      for (int ic = 0; ic < rbo2a[ib][io].size(); ++ic)
        if (alphas[rbo2c[ib][io][ic]] < 0)
          throw std::runtime_error("negative alpha");
        else
          i_calc_twin += alphas[rbo2c[ib][io][ic]] * std::norm(sum_fcalc(rb2a[ib][rbo2a[ib][io][ic]], false));
      // printf("debug: i_obs %f i_calc_twin %f\n", i_obs, i_calc_twin);
      for (int ic = 0; ic < rbo2a[ib][io].size(); ++ic) {
        f_est[rbo2a[ib][io][ic]] += alphas[rbo2c[ib][io][ic]] * std::sqrt(i_obs / i_calc_twin);
        //std::cout << "debug2: f_est[" << rbo2a[ib][io][ic] << "] += " <<  alphas[rbo2c[ib][io][ic]] * std::sqrt(i_obs / i_calc_twin) * f_calc[rb2a[ib][rbo2a[ib][io][ic]]] << "\n";
      }
      //std::cout << "debug3: f_est= " << f_est[0] << "\n";
    }
    Eigen::VectorXd f_true(rb2a[ib].size());
    for (int ia = 0; ia < rb2a[ib].size(); ++ia) {
      f_true(ia) = std::max(1e-6, f_est[ia] * std::abs(sum_fcalc(rb2a[ib][ia], false)));
      if (std::isnan(f_true(ia)))
        throw std::runtime_error("initial f_true is nan");
    }
    Eigen::IOFormat FmtV(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");
    if (ofs_debug.is_open()) {
      Eigen::VectorXd v(rb2o[ib].size()), v2(rb2a[ib].size());
      ofs_debug << "{";
      ofs_debug << "\"ib\":" << ib;
      for (int io = 0; io < rb2o[ib].size(); ++io) v(io) = iobs[rb2o[ib][io]];
      ofs_debug << ",\"iobs\":" << v.format(FmtV);
      for (int io = 0; io < rb2o[ib].size(); ++io) v(io) = sigo[rb2o[ib][io]];
      ofs_debug << ",\"sigma\":" << v.format(FmtV);
      ofs_debug << ",\"idx\":[";
      for (int io = 0; io < rb2o[ib].size(); ++io) {
        Eigen::VectorXd vv(rbo2a[ib][io].size());
        for (int ic = 0; ic < rbo2c[ib][io].size(); ++ic) vv(ic) = rbo2c[ib][io][ic];
        ofs_debug << vv.format(FmtV);
        ofs_debug << ((io == rb2o[ib].size() - 1) ? "]" : ",");
      }
      ofs_debug << ",\"alpha\":" << Eigen::VectorXd::Map(alphas.data(), alphas.size()).format(FmtV);
      for (int ia = 0; ia < rb2a[ib].size(); ++ia) v2(ia) = std::abs(sum_fcalc(rb2a[ib][ia], true));
      ofs_debug << ",\"dfc\":" << v2.format(FmtV);
      for (int ia = 0; ia < rb2a[ib].size(); ++ia) v2(ia) = epsilon[rb2a[ib][ia]] * ml_sigma(bin[rb2a[ib][ia]]);
      ofs_debug << ",\"eps_s\":" << v2.format(FmtV);
      for (int ia = 0; ia < rb2a[ib].size(); ++ia) v2(ia) = centric[rb2a[ib][ia]];
      ofs_debug << ",\"c\":" << v2.format(FmtV);
      ofs_debug << ",\"ini\":" << f_true.format(FmtV);
      double tmpf = calc_f(ib, iobs, sigo, f_true);
      ofs_debug << ",\"f_ini\":" << tmpf;
    }

    Eigen::VectorXd f_true_old = f_true;
    Eigen::IOFormat Fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");
    const double tol = 1.e-5; // enough?
    double det = 1, f0 = 0;
    std::pair<Eigen::VectorXd,Eigen::MatrixXd> ders;
    int i_cyc = 0;
    for (; i_cyc < max_cycle; ++i_cyc) {
      f0 = calc_f(ib, iobs, sigo, f_true);
      // printf("f = %.e\n", f0);
      // printf("f_der = [");
      ders = calc_f_der(ib, iobs, sigo, f_true);
      // std::cout << ders.first;
      // printf("]\n");

      const double e = 1e-2;
      // // test der1
      // printf("f_num = [");
      // for (int ia = 0; ia < rb2a[ib].size(); ++ia) {
      //        for (int ia2 = 0; ia2 < rb2a[ib].size(); ++ia2)
      //          f_true(ia2) = f_calc[rb2a[ib][ia2]] + (ia2 == ia ? e : 0);
      //        const double f1 = f(f_true);
      //        printf("%.6e, ", (f1-f0)/e);
      // }
      // printf("\n");
      if (0) {
        std::cout << "f_der2 = \n" << ders.second << "\n";
        std::cout << "f_der2_num = \n";
        for (int ia = 0; ia < rb2a[ib].size(); ++ia) {
          Eigen::VectorXd f_true2 = f_true;
          for (int ia2 = 0; ia2 < rb2a[ib].size(); ++ia2)
            //f_true2(ia2) += (ia2 == ia ? e : 0);
            if (ia2 == ia)
              f_true2(ia2) += e;
          const auto f1_der = calc_f_der(ib, iobs, sigo, f_true2);
          // std::cout << "debug: ft = " << f_true2.format(Fmt) << "\n"
          //      << "f1_der = " << f1_der.first.format(Fmt) << "\n";
          const auto nder2 = (f1_der.first(ia) - ders.first(ia)) / e;
          printf("num= %.6e ratio= %.5e\n", nder2, nder2/ders.second(ia,ia));
        }
        printf("\n");
      }
      // std::cout << "f_der2_num_ver2 = \n";
      // for (int ia = 0; ia < rb2a[ib].size(); ++ia) {
      //        Eigen::VectorXd f_true2 = f_true;
      //        for (int ia2 = 0; ia2 < rb2a[ib].size(); ++ia2)
      //          f_true2(ia2) += (ia2 == ia ? 2*e : 0);
      //        const double f1_2h = f(f_true2);
      //        f_true2 = f_true;
      //        for (int ia2 = 0; ia2 < rb2a[ib].size(); ++ia2)
      //          f_true2(ia2) += (ia2 == ia ? e : 0);
      //        const double f1_h = f(f_true2);
      //        printf("%.6e, ", (f1_2h - 2 * f1_h + f0)/e/e);
      // }
      // printf("\n");
      SymMatEig eig(ders.second);
      det = eig.det();
      auto a_inv = eig.inv(1e-4);
      f_true_old = f_true;
      Eigen::VectorXd shift = a_inv * ders.first;
      // std::cout << ib << " " << f_true.format(Fmt)
      //           << " first " << ders.first.format(Fmt)
      //           << " second " << ders.second.format(Fmt)
      //                << " shift " << shift.format(Fmt);

      if (shift.array().isNaN().any()) {
        std::cout << "nanshift " << ib << " " << f_true.format(Fmt)
                  << " first " << ders.first.format(Fmt)
                  << " second " << ders.second.format(Fmt)
                  << " a_inv " << a_inv.format(Fmt)
                  << " shift " << shift.format(Fmt) << std::endl;
        throw std::runtime_error("nan shift");
      }
      const double g2p = ders.first.dot(shift);
      const double tol_conv = 1e-6;

      // Line search
      double lambda = 1, lambda_old = 1;
      double f1 = 0, f2 = 0;
      while (((f_true_old - shift * lambda).array() < 0).any()) {
        if (lambda < 0.1)
          break;
        lambda *= 0.75;
      }
      int i_ls = 0;
      for (; i_ls < 20; ++i_ls) {
        f_true = (f_true_old - lambda * shift).cwiseMax(f_true_old / 2).cwiseMax(1e-6);
        f1 = calc_f(ib, iobs, sigo, f_true);
        if (f1 <= f0 - 1e-4 * lambda * g2p)
          break;
        double tmp = 0.5;
        if (i_ls > 0) {
          double l12 = lambda - lambda_old;
          double r1 = f1 - f0 + lambda * g2p;
          double r2 = f2 - f0 + lambda_old * g2p;
          double a = (r1 / gemmi::sq(lambda) - r2 / gemmi::sq(lambda_old)) / l12;
          double b = (-lambda_old * r1 / gemmi::sq(lambda) + lambda * r2 / gemmi::sq(lambda_old)) / l12;
          //printf("debug l12 r1 r2 a b %f %f %f %f %f\n", l12, r1, r2, a, b);
          if (a == 0)
            tmp = g2p / b * 0.5;
          else
            tmp = (-b + std::sqrt(std::max(0., gemmi::sq(b) + 3 * a * g2p))) / 3. / a;
          if (std::isnan(tmp))
            std::cout << "lambda_tmp_nan " << l12 << " " << r1 << " " << r2 << " " << a << " " << b << " " << g2p << std::endl;;
        }
        tmp = std::min(tmp, 0.9 * lambda);
        lambda_old = lambda;
        lambda = std::max(tmp, 0.1 * lambda);
        f2 = f1;
      }
      //std::cout << "i_cyc " << i_cyc << " i_ls " << i_ls
      //<< " first " << ders.first.format(Fmt) << "\n";
      // std::cout << " lambda " << lambda << "\n";
      if (std::isnan(lambda))
        throw std::runtime_error("nan lambda");

      if (g2p * lambda / f_true.size() < tol_conv)
        break;

      // //if (ders.first.norm() < tol || shift.norm() < tol) {
      //        std::cout << "\n";
      //        double det = 1;
      //        for (int i = 0; i < es.eigenvalues().size(); ++i)
      //          det *= es.eigenvalues()(i);
      //        return f0 + 0.5 * std::log(det); // Laplace approximation. omitted (2pi)**N/2
      // }

      //f_true = (f_true - lambda * shift).cwiseMax(1e-6);
      // XXX f_true should not be negative

      // std::cout << "now = \n" << f_true << "\n";
      // auto ders2 = f_der(f_true);
      // std::cout << "f_der1 = \n" << ders2.first << "\n";
      // std::cout << "f_der2 = \n" << ders2.second << "\n";
    }
    // should we keep derivative?
    // std::cout << "debug: " << ib <<  " "
    //           << f0 << " " << det << " " << std::log(det) << " der "
    //           << ders.first.format(Fmt) << std::endl;// << f_true.format(Fmt) <<
    //return f0 + 0.5 * std::log(det); // Laplace approximation. omitted (2pi)**N/2

    if (ofs_debug.is_open()) {
      ofs_debug << ",\"final\":" << f_true.format(FmtV);
      ofs_debug << ",\"cyc\":" << i_cyc;
      ders = calc_f_der(ib, iobs, sigo, f_true);
      double tmpf = calc_f(ib, iobs, sigo, f_true);
      ofs_debug << ",\"f\": " << tmpf;
      ofs_debug << ",\"f_der\":" << ders.first.format(FmtV);
      auto eig = SymMatEig(ders.second);
      ofs_debug << ",\"f_sec_eig\":" << eig.es.eigenvalues().format(FmtV);
      ofs_debug << ",\"f_sec_det\":" << eig.det();
      ofs_debug << "},";
    }

    for (int i = 0; i < rb2a[ib].size(); ++i)
      f_true_max[rb2a[ib][i]] = std::max(f_true(i), 1e-6);

    //throw std::runtime_error("did not converge. ib = " + std::to_string(ib));
  }
};

void add_twin(nb::module_& m) {
  using T = double;
  nb::class_<TwinData>(m, "TwinData")
    .def(nb::init<>())
    .def_ro("rb2o", &TwinData::rb2o)
    .def_ro("rb2a", &TwinData::rb2a)
    .def_ro("rbo2a", &TwinData::rbo2a)
    .def_ro("rbo2c", &TwinData::rbo2c)
    .def_ro("rbin", &TwinData::rbin)
    .def_prop_ro("asu", [](TwinData &self) {
      int64_t stride = static_cast<int64_t>(sizeof(gemmi::Op::Miller) / sizeof(int));
      return nb::ndarray<nb::numpy, int, nb::shape<-1,3>>(&self.asu.front()[0],
                                                          {self.asu.size(), 3}, // shape
                                                          nb::handle(), {stride, 1});
    }, nb::rv_policy::reference_internal)
    .def_ro("centric", &TwinData::centric)
    .def_ro("epsilon", &TwinData::epsilon)
    .def_ro("bin", &TwinData::bin)
    .def_ro("s2_array", &TwinData::s2_array)
    .def_ro("f_true_max", &TwinData::f_true_max)
    .def_prop_ro("f_calc", [](TwinData &self) {
      const size_t size = sizeof(std::complex<T>);
      return nb::ndarray<nb::numpy, std::complex<T>>(self.f_calc_.data(),
                                                     {self.asu.size(), self.n_models},
                                                     nb::handle(),
                                                     {(int64_t)self.n_models, 1});
    }, nb::rv_policy::reference_internal)
    .def_ro("mlparams", &TwinData::mlparams) // debug raw access
    // Sigma
    .def_prop_ro("ml_sigma", [](TwinData &self) {
      const size_t size = sizeof(T);
      const size_t bin_max = self.mlparams.size() / (self.n_models + 1);
      return nb::ndarray<nb::numpy, T>(self.mlparams.data(),
                                       {bin_max},
                                       nb::handle(),
                                       {(int64_t)self.n_models+1});
    }, nb::rv_policy::reference_internal)
    // D
    .def_prop_ro("ml_scale", [](TwinData &self) {
      const size_t size = sizeof(T);
      const size_t bin_max = self.mlparams.size() / (self.n_models + 1);
      return nb::ndarray<nb::numpy, T>(self.mlparams.data() + 1,
                                       {bin_max, self.n_models},
                                       nb::handle(),
                                       {(int64_t)self.n_models+1, 1});
    }, nb::rv_policy::reference_internal)
    .def("ml_scale_array", [](const TwinData &self) {
      auto ret =  make_numpy_array<double>({self.asu.size(), self.n_models});
      double* ptr = ret.data();
      for (size_t ia = 0; ia < self.asu.size(); ++ia) {
        const int ibin = self.bin[ia];
        for (int j = 0; j < self.n_models; ++j)
          ptr[ia * self.n_models + j] = self.ml_scale(ibin, j);
      }
      return ret;
    })
    .def("ml_sigma_array", [](const TwinData &self) {
      auto ret = make_numpy_array<double>({self.asu.size()});
      double* ptr = ret.data();
      for (size_t ia = 0; ia < self.asu.size(); ++ia)
        ptr[ia] = self.ml_sigma(self.bin[ia]);
      return ret;
    })
    .def_ro("ops", &TwinData::ops)
    .def_rw("alphas", &TwinData::alphas)
    .def("idx_of_asu", [](const TwinData &self, np_array<int, 2> hkl, bool inv){
      auto h = hkl.view();
      if (h.shape(1) < 3)
        throw std::domain_error("error: the size of the second dimension < 3");
      const size_t ret_size = inv ? self.asu.size() : h.shape(0);
      auto ret = make_numpy_array<int>({ret_size});
      int* ptr = ret.data();
      for (size_t i = 0; i < ret_size; ++i)
        ptr[i] = -1;
      for (size_t i = 0; i < h.shape(0); ++i) {
        int j = self.idx_of_asu({h(i, 0), h(i, 1), h(i, 2)});
        // if (j >= h.shape(0))
        //   throw std::runtime_error("bad idx_of_asu " +
        //                            std::to_string(h(i,0))+" "+
        //                            std::to_string(h(i,1))+" "+
        //                            std::to_string(h(i,2)));
        if (inv)
          ptr[j] = i;
        else
          ptr[i] = j;
      }
      return ret;
    }, nb::arg("hkl"), nb::arg("inv")=false)
    .def("setup_f_calc", &TwinData::setup_f_calc)
    .def("d_min", &TwinData::d_min)
    .def("setup", [](TwinData &self, np_array<int, 2> hkl, const std::vector<int> &bin,
                     const gemmi::SpaceGroup &sg, const gemmi::UnitCell &cell,
                     const std::vector<gemmi::Op> &operators) {
      auto h = hkl.view();
      if (h.shape(1) < 3)
        throw std::domain_error("error: the size of the second dimension < 3");
      std::vector<gemmi::Op::Miller> hkls;
      hkls.reserve(h.shape(0));
      for (size_t i = 0; i < h.shape(0); ++i)
        hkls.push_back({h(i, 0), h(i, 1), h(i, 2)});
      self.setup(hkls, bin, sg, cell, operators);
    })
    .def("pairs", [](const TwinData &self, int i_op, int i_bin) {
      // i_op = 0 is the first twin operator (not identity)
      if (i_op < 0 || i_op >= self.alphas.size())
        throw std::runtime_error("bad i_op");
      std::vector<std::array<size_t, 2>> idxes;
      idxes.reserve(self.rb2o.size());
      for (int ib = 0; ib < self.rb2o.size(); ++ib) {
        if (i_bin >= 0 && self.rbin[ib] != i_bin)
          continue;
        for (int io = 0; io < self.rb2o[ib].size(); ++io) {
          const int tofind = [&]()->int{
            for (int ic = 0; ic < self.rbo2c[ib][io].size(); ++ic)
              if (self.rbo2c[ib][io][ic] == i_op + 1)
                return self.rbo2a[ib][io][ic];
            return -1;
          }();
          if (tofind < 0)
            continue;
          // look for obs data indices corresponding to the twin operator
          for (int io2 = io+1; io2 < self.rb2o[ib].size(); ++io2)
            if (!self.rbo2a[ib][io2].empty() && self.rbo2c[ib][io2][0] == 0 &&
                self.rbo2a[ib][io2][0] == tofind)
              idxes.push_back({self.rb2o[ib][io], self.rb2o[ib][io2]});
        }
      }
      return idxes;
    }, nb::arg("i_op"), nb::arg("i_bin")=-1)
    // note: this may lead to higher cc, as each pair may include the same indices
    .def("twin_related_obs", [](const TwinData &self, int i_bin) {
      const size_t n_ops = self.n_ops(); // include identity
      const size_t n_obs = [&]() {
        if (i_bin < 0)
          return self.n_obs();
        size_t n = 0;
        for (int ib = 0; ib < self.rb2o.size(); ++ib)
          if (self.rbin[ib] == i_bin)
            n += self.rb2o[ib].size();
        return n;
      }();
      auto ret = make_numpy_array<int>({n_obs, n_ops});
      int *ptr = ret.data();
      for (int i = 0; i < ret.size(); ++i)
        ptr[i] = -1;
      for (int ib = 0; ib < self.rb2o.size(); ++ib)
        if (i_bin < 0 || self.rbin[ib] == i_bin)
          for (int io = 0; io < self.rb2o[ib].size(); ++io) {
            for (int ic = 0; ic < self.rbo2c[ib][io].size(); ++ic) {
              // look for obs data indices corresponding to twin operators
              // XXX not properly tested
              for (int io2 = 0; io2 < self.rb2o[ib].size(); ++io2)
                if (!self.rbo2a[ib][io2].empty() && self.rbo2a[ib][io2][0] == self.rbo2a[ib][io][ic]) {
                  ptr[self.rbo2c[ib][io][ic]] = self.rb2o[ib][io2];
                  break;
                }
            }
            ptr += n_ops;
          }
      return ret;
    })
    .def("obs_related_asu", [](const TwinData &self) {
      const size_t n_ops = self.n_ops(); // include identity
      auto ret = make_numpy_array<int>({self.n_obs(), n_ops});
      int *ptr = ret.data();
      for (int i = 0; i < ret.size(); ++i)
        ptr[i] = -1;
      for (int ib = 0; ib < self.rb2o.size(); ++ib)
        for (int io = 0; io < self.rb2o[ib].size(); ++io) {
          int *ptr2 = ptr + self.rb2o[ib][io] * n_ops;
          for (int ic = 0; ic < self.rbo2a[ib][io].size(); ++ic)
            // FIXME? It may be safer to put -1 for all if self.rbo2a[ib][io].size() != n_ops
            ptr2[ic] = self.rb2a[ib][self.rbo2a[ib][io][ic]];
        }
      return ret;
    })
    .def("twin_related", [](const TwinData &self,
                            const gemmi::SpaceGroup &sg) {
      const size_t n_asu = self.asu.size();
      //if (data.shape(0) != n_asu)
      //  throw std::runtime_error("data and asu shapes mismatch");
      const gemmi::GroupOps gops = sg.operations();
      const gemmi::ReciprocalAsu rasu(&sg);
      auto apply_and_asu = [&rasu, &gops](const gemmi::Op &op, const gemmi::Op::Miller &h) {
        return rasu.to_asu(op.apply_to_hkl(h), gops).first;
      };
      const size_t n_ops = self.n_ops();
      auto ret = make_numpy_array<int>({n_asu, n_ops});
      int* ptr = ret.data();
      //auto data_ = data.unchecked<1>();
      for (int i = 0; i < n_asu; ++i) {
        const auto h = self.asu[i];
        ptr[i * n_ops] = i; //data_(i);
        for (int j = 1; j < n_ops; ++j) {
          const auto hr = apply_and_asu(self.ops[j-1], h);
          ptr[i * n_ops + j] = self.idx_of_asu(hr);
        }
      }
      return ret;
    })
    .def("est_f_true", [](TwinData &self, np_array<double> Io, np_array<double> sigIo, int max_cycle) {
      auto Io_ = Io.view();
      auto sigIo_ = sigIo.view();
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib)
        self.est_f_true(ib, Io_.data(), sigIo_.data(), max_cycle);
    }, nb::arg("Io"), nb::arg("sigIo"), nb::arg("max_cycle")=10)
    .def("ll_rice", [](const TwinData &self) { // pseudo -LL
      double ret = 0;
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib) {
        const int b = self.rbin[ib];
        // if (bin >= 0 && b != bin)
        //   continue;
        // calculate Rice distribution using Ftrue as Fobs
        for (int i = 0; i < self.rb2a[ib].size(); ++i) {
          const int ia = self.rb2a[ib][i];
          if (std::isnan(self.f_true_max[ia]))
            continue;
          const int c = self.centric[ia] + 1;
          const double eps = self.epsilon[ia];
          const std::complex<double> DFc = self.sum_fcalc(ia, true);
          const double S = self.ml_sigma(self.bin[ia]);
          const double log_ic0 = log_i0_or_cosh(self.f_true_max[ia] * std::abs(DFc) / (S * eps), c);
          const double log_fmax_a = c == 1 ? std::log(self.f_true_max[ia]) : 0;
          ret +=  std::log(S) / c - log_fmax_a + (sq(self.f_true_max[ia]) + std::norm(DFc)) / (eps * S * c) - log_ic0;
        }
      }
      return ret;
    })
    .def("ll", [](const TwinData &self, np_array<double> Io, np_array<double> sigIo) { // Laplace approximation
      auto Io_ = Io.view();
      auto sigIo_ = sigIo.view();
      double ret = 0;
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib) {
        const int b = self.rbin[ib];
        Eigen::VectorXd f_true(self.rb2a[ib].size());
        for (size_t i = 0; i < self.rb2a[ib].size(); ++i)
          f_true(i) = self.f_true_max[self.rb2a[ib][i]];
        if (f_true.array().isNaN().all())
          continue;
        const auto ders = self.calc_f_der(ib, Io_.data(), sigIo_.data(), f_true);
        const double f = self.calc_f(ib, Io_.data(), sigIo_.data(), f_true);
        Eigen::IOFormat Fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");
        if (std::isnan(f)) {
          std::cout << "nan f ib " << ib << " ft " << f_true.format(Fmt) << "\n";
          continue;
        }
        const double h_inv_der = SymMatEig(ders.second).det(); // could be negative?
        if (h_inv_der <= 0) { // how can we handle this..
          std::cout << "h_inv_der " << h_inv_der << " f " << f
                    << " f_der " << ders.first.format(Fmt)
                    << " f_max " << f_true.format(Fmt) << "\n";
          continue;
        }
        ret += f + 0.5 * std::log(h_inv_der);
        for (int i = 0; i < self.rb2a[ib].size(); ++i) {
          const int ia = self.rb2a[ib][i];
          const int c = self.centric[ia];
          const double S = self.ml_sigma(self.bin[ia]); // omitting eps
          ret += std::log(S) / (c + 1);
        }
      }
      return ret;
    })
    .def("ll_der_D_S", [](const TwinData &self) {
      const size_t n_cols = self.n_models + 1;
      auto ret = make_numpy_array<double>({self.asu.size(), n_cols});
      double* ptr = ret.data();
      for (int i = 0; i < ret.size(); ++i)
        ptr[i] = NAN;
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib) {
        const int b = self.rbin[ib];
        for (int i = 0; i < self.rb2a[ib].size(); ++i) {
          const int ia = self.rb2a[ib][i];
          if (std::isnan(self.f_true_max[ia]))
            continue;
          const int c = self.centric[ia] + 1;
          const double eps = self.epsilon[ia];
          const std::complex<double> DFc = self.sum_fcalc(ia, true);
          const std::complex<double> DFc_conj = std::conj(DFc);
          const double S = self.ml_sigma(self.bin[ia]);
          const double m = fom(self.f_true_max[ia] * std::abs(DFc) / (eps * S), c);
          for (size_t j = 0; j < self.n_models; ++j) {
            const double r_fcj_fc = (self.f_calc(ia, j) * DFc_conj).real();
            // wrt Dj
            ptr[ia*n_cols + j] = 2 * r_fcj_fc / (eps * S * c) * (1. - m * self.f_true_max[ia] / std::abs(DFc));
          }
          // wrt S
          const double tmp = (sq(self.f_true_max[ia]) + std::norm(DFc)) / c - m * (3 - c) * self.f_true_max[ia] * std::abs(DFc);
          ptr[(ia+1)*n_cols - 1] = 1. / (c * S) - tmp / (eps * sq(S));
        }
      }
      return ret;
    })
    .def("ll_der_alpha", [](const TwinData &self, np_array<double> Io, np_array<double> sigIo, bool accurate) {
      auto Io_ = Io.view();
      auto sigIo_ = sigIo.view();
      auto ret = make_numpy_array<double>({self.n_ops()});
      double* ptr = ret.data();
      for (int i = 0; i < ret.size(); ++i)
        ptr[i] = 0.;
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib) {
        // not needed for inaccurate ver
        Eigen::VectorXd f_true(self.rb2a[ib].size());
        for (size_t i = 0; i < self.rb2a[ib].size(); ++i)
          f_true(i) = self.f_true_max[self.rb2a[ib][i]];
        auto ders = self.calc_f_der(ib, Io_.data(), sigIo_.data(), f_true);
        auto eig_f = SymMatEig(ders.second);
        auto f_inv = eig_f.inv();
        // __
        for (int io = 0; io < self.rb2o[ib].size(); ++io) {
          const int obs_idx = self.rb2o[ib][io];
          const double i_obs = Io_(obs_idx);
          if (std::isnan(i_obs)) continue;
          const double sig2inv =  1. / sq(sigIo_(obs_idx));
          for (int ic = 0; ic < self.rbo2a[ib][io].size(); ++ic) {
            // |Fi|^2 part
            const int i = self.rbo2a[ib][io][ic];
            const int ia = self.rb2a[ib][i];
            const int alpha_idx = self.rbo2c[ib][io][ic];
            if (std::isnan(self.f_true_max[ia])) continue;
            const double fac1 = -sig2inv * i_obs;
            if (accurate) {
              const double g = sq(self.f_true_max[ia]);
              const double g2 = g * g;
              const double g_der_sq = 4 * g;
              const double g_der2 = 2;
              const double denom = g2 + (-g_der2 * g + g_der_sq) * f_inv(i,i);
              if (denom <= 0) {
                printf("negative denom %f fmax %f Hii %f\n", denom, self.f_true_max[ia], f_inv(i,i));
                std::cout << ders.second << "\n" << f_inv << "\n";
              }
              ptr[alpha_idx] += fac1 * g2 / std::sqrt(denom) * std::exp(0.5 * g_der_sq * f_inv(i,i) / denom);
            } else
              ptr[alpha_idx] += fac1 * sq(self.f_true_max[ia]);
            // |Fi|^2 |Fj|^2 part
            for (int ic2 = 0; ic2 < self.rbo2a[ib][io].size(); ++ic2) {
              const int j = self.rbo2a[ib][io][ic2];
              const int ia2 = self.rb2a[ib][j];
              const int alpha_idx2 = self.rbo2c[ib][io][ic2];
              if (std::isnan(self.f_true_max[ia2])) continue;
              const double fac2 = sig2inv * self.alphas[alpha_idx2];
              if (accurate) {
                const double g = sq(self.f_true_max[ia] * self.f_true_max[ia2]);
                if (i == j) {
                  const double g2 = g * g;
                  const double g_der_sq = 16 * sq(sq(self.f_true_max[ia]) * self.f_true_max[ia]);
                  const double g_der2 = 12 * sq(self.f_true_max[ia]);
                  const double denom = g2 + (-g_der2 * g + g_der_sq) * f_inv(i,i);
                  if (denom <= 0) {
                    printf("negative denom %f fmax %f Hii %f\n", denom, self.f_true_max[ia], f_inv(i,i));
                    std::cout << ders.second << "\n" << f_inv << "\n";
                  }
                  ptr[alpha_idx] += fac2 * g2 / std::sqrt(denom) * std::exp(0.5 * g_der_sq * f_inv(i,i) / denom);
                } else { // cannot use Sherman-Morrison update
                  Eigen::VectorXd g_der = Eigen::VectorXd::Zero(self.rb2a[ib].size());
                  g_der(i) = 2 * self.f_true_max[ia] * sq(self.f_true_max[ia2]);
                  g_der(j) = 2 * sq(self.f_true_max[ia]) * self.f_true_max[ia2];
                  Eigen::MatrixXd g_der2 = Eigen::MatrixXd::Zero(g_der.size(), g_der.size());
                  g_der2(i,i) = 2 * sq(self.f_true_max[ia2]);
                  g_der2(j,j) = 2 * sq(self.f_true_max[ia]);
                  g_der2(i,j) = g_der2(j,i) = 4 * self.f_true_max[ia] * self.f_true_max[ia2];
                  const auto H_h = ders.second - g_der2 / g + g_der * g_der.transpose() / sq(g);
                  const auto eig_h = SymMatEig(H_h);
                  const double det_f = eig_f.det(), det_h = eig_h.det();
                  const double tmp = g_der.transpose() * eig_h.inv() * g_der;
                  if (det_f <= 0) {
                    printf("negative det_f %f\n", det_f);
                    std::cout << ders.second << "\n";
                  }
                  if (det_h <= 0) {
                    printf("negative det_h %f\n", det_h);
                    std::cout << "f_true " << f_true << "\n";
                    std::cout << H_h << "\n" << g_der2 << "\n";
                  }
                  ptr[alpha_idx] += fac2 * g * std::sqrt(det_f / det_h) * std::exp(0.5 * tmp / sq(g));
                }
              } else
                ptr[alpha_idx] += fac2 * sq(self.f_true_max[ia] * self.f_true_max[ia2]);
            }
          }
        }
      }
      return ret;
    })
    // F_true must be estimated beforehand
    .def("calc_fom", [](const TwinData &self) {
      auto ret = make_numpy_array<double>({self.asu.size()});
      double* ptr = ret.data();
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib) {
        const int b = self.rbin[ib];
        for (int i = 0; i < self.rb2a[ib].size(); ++i) {
          const int ia = self.rb2a[ib][i];
          const int c = self.centric[ia] + 1;
          const double eps = self.epsilon[ia];
          const std::complex<double> DFc = self.sum_fcalc(ia, true);
          const double S = self.ml_sigma(self.bin[ia]);
          ptr[ia] = fom(self.f_true_max[ia] * std::abs(DFc) / (S * eps), c);
        }
      }
      return ret;
    })
    // for FWT and DELFWT: <m|F|>
    .def("expected_F", [](TwinData &self, np_array<double> Io, np_array<double> sigIo, bool accurate) {
      auto ret = make_numpy_array<double>({self.asu.size()});
      double* ptr = ret.data();
      auto Io_ = Io.view();
      auto sigIo_ = sigIo.view();
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib) {
        const size_t b = self.rbin[ib];
        Eigen::VectorXd f_true(self.rb2a[ib].size());
        for (size_t i = 0; i < self.rb2a[ib].size(); ++i)
          f_true(i) = self.f_true_max[self.rb2a[ib][i]];
        auto ders = self.calc_f_der(ib, Io_.data(), sigIo_.data(), f_true);
        auto f_inv = SymMatEig(ders.second).inv();
        for (size_t i = 0; i < self.rb2a[ib].size(); ++i) {
          const size_t ia = self.rb2a[ib][i];
          const double Ft = self.f_true_max[ia];
          const int c = self.centric[ia] + 1;
          const double eps = self.epsilon[ia];
          const std::complex<double> DFc = self.sum_fcalc(ia, true);
          const double S = self.ml_sigma(self.bin[ia]);
          const double r = std::abs(DFc) / (S * eps); // X without |F|
          const double m = fom(Ft * r, c);
          const double m_der = fom_der(m, Ft * r, c);
          const double g = m * Ft;
          if (accurate) {
            const double g2 = g * g;
            const double g_der_sq = sq(m_der * Ft * r + m);
            const double g_der2 = fom_der2(m, Ft * r, c) * Ft * sq(r) + 2 * m_der * r;
            const double denom = g2 + (-g_der2 * g + g_der_sq) * f_inv(i,i);
            ptr[ia] = g2 / std::sqrt(denom) * std::exp(0.5 * g_der_sq * f_inv(i,i) / denom);
          } else {
            ptr[ia] = g;
          }
        }
      }
      return ret;
    }, nb::arg("Io"), nb::arg("sigIo"), nb::arg("accurate")=true)
    .def("ll_der_fc0", [](const TwinData &self) {
      auto ret1 = make_numpy_array<std::complex<double>>({self.asu.size()});
      auto ret2 = make_numpy_array<double>({self.asu.size()});
      std::complex<double>* ptr1 = ret1.data();
      double* ptr2 = ret2.data();
      for (int i = 0; i < ret1.size(); ++i) {
        ptr1[i] = NAN;
        ptr2[i] = NAN;
      }
      for (int ia = 0; ia < self.asu.size(); ++ia) {
        const int b = self.bin[ia];
        const std::complex<double> DFc = self.sum_fcalc(ia, true);
        const double fmax = self.f_true_max[ia];
        if (!std::isnan(fmax)) {
          const int c = self.centric[ia] + 1;
          const double eps = self.epsilon[ia];
          const double S = self.ml_sigma(b);
          const double X_der = fmax / (S * eps);
          const double X = std::abs(DFc) * X_der;
          const double m = fom(X, c);
          const std::complex<double> exp_ip = std::exp(std::arg(DFc) * std::complex<double>(0, 1));
          ptr1[ia] = (3 - c) * (std::abs(DFc) - m * fmax) / (eps * S) * self.ml_scale(b, 0) * exp_ip; // assuming 0 is atomic structure
          ptr2[ia] = ((3 - c) / (eps * S) - fom_der(m, X, c) * sq((3 - c) * X_der)) * sq(self.ml_scale(b, 0));
        }
      }
      return nb::make_tuple(ret1, ret2);
    })
    // helper function for least-square scaling
    // n_models should include mask, but last f_falc is ignored
    // 0: F_calc = sqrt(sum(alpha * |Fc,0 + Fc,1 * k_sol * exp(-B_sol s^2/4) |^2))
    // 1: dF/dk_sol = 1/F_calc * Re((Fc,0+...) * (Fc,1 * exp(...)).conj)
    // 2: dF/dB_sol = 1/F_calc * Re((Fc,0+...) * (Fc,1 * k_sol * exp(...) * (-s^2/4)).conj)
    .def("scaling_fc_and_mask_grad", [](const TwinData &self,
                                        np_array<std::complex<double>> f_mask,
                                        double k_sol, double b_sol) {
      auto f_mask_ = f_mask.view();
      if (f_mask_.shape(0) != self.asu.size())
        throw std::runtime_error("bad f_mask size");
      auto ret = make_numpy_array<double>({self.n_obs(), (size_t)3});
      double* ptr = ret.data();
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib)
        for (int io = 0; io < self.rb2o[ib].size(); ++io) {
          double i_calc_twin = 0, der1 = 0, der2 = 0;
          for (int ic = 0; ic < self.rbo2a[ib][io].size(); ++ic) {
            const size_t ia = self.rb2a[ib][self.rbo2a[ib][io][ic]];
            std::complex<double> fc = 0;
            const double temp_fac = std::exp(-b_sol * self.s2_array[ia] / 4.);
            for (int j = 0; j < self.n_models; ++j)
              if (j == self.n_models - 1)
                fc += k_sol * temp_fac * f_mask_(ia);
              else
                fc += self.f_calc(ia, j);
            const double alpha = self.alphas[self.rbo2c[ib][io][ic]];
            const double tmp = alpha * (fc * std::conj(f_mask_(ia)) * temp_fac).real();
            der1 += tmp;
            der2 += -tmp * k_sol * self.s2_array[ia] / 4.;
            i_calc_twin += alpha * std::norm(fc);
          }
          const double f_calc_twin = std::sqrt(i_calc_twin);
          ptr[3*self.rb2o[ib][io]] = f_calc_twin;
          ptr[3*self.rb2o[ib][io]+1] = der1 / f_calc_twin;
          ptr[3*self.rb2o[ib][io]+2] = der2 / f_calc_twin;
        }
      return ret;
    })
    // for stats calculation
    .def("i_calc_twin", [](const TwinData &self) {
      auto ret = make_numpy_array<double>({self.n_obs()});
      double* ptr = ret.data();
      for (size_t ib = 0; ib < self.rb2o.size(); ++ib)
        for (int io = 0; io < self.rb2o[ib].size(); ++io) {
          double i_calc_twin = 0;
          for (int ic = 0; ic < self.rbo2a[ib][io].size(); ++ic) {
            const size_t ia = self.rb2a[ib][self.rbo2a[ib][io][ic]];
            std::complex<double> fc = 0;
            for (int j = 0; j < self.n_models; ++j)
              fc += self.f_calc(ia, j);
            i_calc_twin += self.alphas[self.rbo2c[ib][io][ic]] * std::norm(fc);
          }
          ptr[self.rb2o[ib][io]] = i_calc_twin;
        }
      return ret;
    })
    .def("debye_waller_factors", [](const TwinData &self, double b_iso) {
      auto ret = make_numpy_array<double>({self.asu.size()});
      double* ptr = ret.data();
      for (int i = 0; i < ret.size(); ++i)
        ptr[i] = std::exp(-b_iso / 4 * self.s2_array[i]);
      return ret;
    }, nb::arg("b_iso"))
    .def("debug_open", [](TwinData &self, const std::string &fname) {
      self.ofs_debug.open(fname);
      self.ofs_debug << "[";
    })
    .def("debug_close", [](TwinData &self) {
      self.ofs_debug << "]\n";
      self.ofs_debug.close();
    })
    ;
}
