// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_REFINE_LL_HPP_
#define SERVALCAT_REFINE_LL_HPP_

#include "params.hpp"
#include "geom.hpp"
#include <memory>
#include <vector>
#include <gemmi/grid.hpp>
#include <gemmi/it92.hpp>
#include <gemmi/neutron92.hpp>
#include <gemmi/c4322.hpp> // CustomCoef
#include <gemmi/dencalc.hpp>
#include <Eigen/Sparse>

namespace servalcat {

// from Refmac subroutine SMOOTH_GAUSS_D in extra_eigen.f
std::pair<double, std::vector<double>> smooth_gauss_d(double kernel_width,
                                                      const std::vector<double> &x_points,
                                                      const std::vector<double> &y_points,
                                                      double x_current) {
  assert(x_points.size() == y_points.size());
  const int n_points = x_points.size();
  if (n_points == 1)
    return std::make_pair(y_points[0], std::vector<double>(1, 1.));

  const double kernel_width2 = kernel_width * kernel_width * 2.0;

  // return values
  double y_current = 0;
  std::vector<double> y_derivs(n_points);

  if (x_current <= x_points.back() && x_current >= x_points.front()) {
    double an = 0.0;
    double fn = 0.0;
    double dx = 0, dx0 = 1.0, dx1 = 0;
    for (int i = 0; i < n_points; ++i) {
      dx = gemmi::sq(x_current - x_points[i]) / kernel_width2;
      dx0 = std::min(std::abs(dx), dx0);
    }
    for (int i = 0; i < n_points; ++i) {
      dx = gemmi::sq(x_current - x_points[i]) / kernel_width2;
      dx1 = dx - dx0;
      if (dx1 <= 120.0) {
        const double expdx = std::exp(-dx1);
        an += expdx;
        fn += y_points[i] * expdx;
        y_derivs[i] = expdx;
      }
    }
    if (an <= 0.0)
      gemmi::fail("===> Error in smooth gauss. Width might be too small: 1",
                  std::to_string(n_points), " ", std::to_string(kernel_width), " ",
                  std::to_string(dx), " ", std::to_string(dx0), " ", std::to_string(dx1));

    y_current = fn / an;

    // calculate derivatives
    for (int i = 0; i < n_points; ++i) {
      const double dx = gemmi::sq(x_current - x_points[i]) / kernel_width2;
      const double dx1 = dx - dx0;
      if (dx1 <= 120.0)
        y_derivs[i] /= an;
      else
        y_derivs[i] = 0.0;
    }
  } else if (x_current > x_points.back()) {
    const double dx1 = gemmi::sq(x_current - x_points.back()) / kernel_width2;
    double an = 1.0;
    double fn = y_points.back();
    double dx;
    y_derivs.back() = 1.0;
    for (int i = 0; i < n_points-1; ++i) {
      dx = gemmi::sq(x_current - x_points[i]) / kernel_width2 - dx1;
      if (dx <= 120.0) {
        const double expdx = std::exp(-dx);
        an += expdx;
        fn += y_points[i] * expdx;
        y_derivs[i] = expdx;
      }
    }
    if (an <= 0.0)
      gemmi::fail("===> Error in smooth gauss. Width might be too small: 2 ",
                  std::to_string(n_points), " ", std::to_string(kernel_width), " ",
                  std::to_string(dx));

    y_current = fn / an;

    // calculate derivatives
    for (int i = 0; i < n_points; ++i) {
      const double dx = gemmi::sq(x_current - x_points[i]) / kernel_width2 - dx1;
      if (dx <= 120.0)
        y_derivs[i] /= an;
      else
        y_derivs[i] = 0.0;
    }
  } else if (x_current < x_points.front()) {
    const double dx1 = gemmi::sq(x_current-x_points.front()) / kernel_width2;
    double an = 1.0;
    double fn = y_points.front();
    double dx = 0;
    y_derivs[0] = 1.0;
    for (int i = 1; i < n_points; ++i) {
      dx = gemmi::sq(x_current - x_points[i]) / kernel_width2 - dx1;
      if (dx <= 120.0) {
        const double expdx = std::exp(-dx);
        an += expdx;
        fn += y_points[i]*expdx;
        y_derivs[i] = expdx;
      }
    }
    if (an <= 0.0)
      gemmi::fail("===> Error in smooth gauss. Width might be too small: 3 ",
                  std::to_string(n_points), " ", std::to_string(kernel_width), " ",
                  std::to_string(dx));

    y_current = fn/an;

    // calculate derivatives
    for (int i = 0; i < n_points; ++i) {
      const double dx = gemmi::sq(x_current - x_points[i]) / kernel_width2 - dx1;
      if (dx <= 120.0)
        y_derivs[i] /= an;
      else
        y_derivs[i] = 0.0;
    }
  }
  return std::make_pair(y_current, y_derivs);
}

struct TableS3 {
  double s_min, s_max;
  double delta_s3;
  int n_points;
  std::vector<double> s3_values;
  std::vector<double> y_values;
  int maxbin = 500;
  TableS3(double d_min, double d_max) {
    s_min = 1. / d_max;
    s_max = 1. / d_min;
    const double smax3 = s_max*s_max*s_max, smin3 = s_min*s_min*s_min;
    delta_s3 = 0.0005;
    n_points = (int) ((smax3 - smin3) / delta_s3);
    n_points = std::max(20, std::min(2000, n_points));
    delta_s3 = (smax3 - smin3) / n_points;
    s3_values.reserve(n_points+1);
    y_values.reserve(n_points+1);
    for (int i = 0; i <= n_points; ++i)
      s3_values.push_back(smin3 + i * delta_s3);
  }

  // From Refmac SUBROUTINE D2DA_RADIAL_BIN in hkon_secder_tch.f
  void make_table(const std::vector<double> &svals, const std::vector<double> &yvals) {
    assert(svals.size() == yvals.size());

    // define bin
    const double smax_ml = *std::max_element(svals.begin(), svals.end()) * 1.0001;
    const double smin_ml = *std::min_element(svals.begin(), svals.end()) * 0.9999;
    const double smin_ml3 = smin_ml * smin_ml * smin_ml;
    const double smax_ml3 = smax_ml * smax_ml * smax_ml;

    // from Refmac SUBROUTINE DEFINE_BINS_FOR_ML in oppro_allocate.f
    const double binsize_ml = 0.0005;
    int nbin_ml =  std::max(1, std::min(maxbin,
                                        (int)((smax_ml*smax_ml - smin_ml*smin_ml)/binsize_ml)+1
                                        ));
    int nbin_rad = nbin_ml;
    const double ds3 = (smax_ml3 - smin_ml3) / nbin_rad;

    std::vector<double> smeanb_rad(nbin_rad+1);
    for (int i = 0; i <= nbin_rad; ++i)
      smeanb_rad[i] = smin_ml3 + i * ds3;

    std::vector<double> sec_der_bin(nbin_rad+1);
    std::vector<int> nref_sec_bin(nbin_rad+1);
    for (size_t i = 0; i < svals.size(); ++i) {
      const double s = svals[i];
      const int ibin = std::upper_bound(smeanb_rad.begin(), smeanb_rad.end(), s*s*s) - smeanb_rad.begin() - 1;
      // int ibin;
      // for (int j = 0; j < nbin_rad; ++j)
      //   if (s3 > smeanb_rad[j] && s3 <= smeanb_rad[j+1]) {
      //     ibin = j;
      //     int ibin2 = std::upper_bound(smeanb_rad.begin(), smeanb_rad.end(), s*s*s) - smeanb_rad.begin();
      //     std::cout << "ibin= " << j << " diff= " << ibin2-j << "\n";
      //     break;
      //   }

      sec_der_bin[ibin] += yvals[i];
      nref_sec_bin[ibin] += 1;
    }

    for (int i = 0; i < nbin_rad; ++i)
      if (sec_der_bin[i] > 0 && nref_sec_bin[i] > 0)
        sec_der_bin[i] = std::log(sec_der_bin[i] / nref_sec_bin[i]);
      else
        nref_sec_bin[i] = 0;

    // sort out bins with no (usable) reflections
    for (int i = 1; i < nbin_rad; ++i)
      if (nref_sec_bin[i] <= 0 && nref_sec_bin[i-1] > 0)
        sec_der_bin[i] = sec_der_bin[i-1];
    for (int i = nbin_rad-2; i >= 0; --i)
      if (nref_sec_bin[i] <= 0 && nref_sec_bin[i+1] > 0)
        sec_der_bin[i] = sec_der_bin[i+1];

    // Start refining parameters of smoothining function
    double tmp1 = sec_der_bin[0];
    sec_der_bin[nbin_rad] = sec_der_bin[nbin_rad-1];
    for (int i = 1; i < nbin_rad; ++i) {
      double tmp2 = sec_der_bin[i];
      sec_der_bin[i] = (tmp1 + tmp2)/2.0;
      tmp1 = tmp2;
    }

    // TODO smooth curve here?

    const double kernel_g_rad = (smeanb_rad[1] - smeanb_rad[0]) / 2.0;
    for (int i = 0; i <= n_points; ++i) {
      const double yval = smooth_gauss_d(kernel_g_rad, smeanb_rad, sec_der_bin, s3_values[i]).first;
      y_values.push_back(std::exp(yval));
    }
  }

  double get_value(double s) const {
    const double s3 = s*s*s;
    const int i =  std::max(0, std::min(n_points,
                                        (int)std::round((s3 - s3_values.front()) / delta_s3)));
    return y_values[i];
  }
};

struct LL{
  gemmi::UnitCell cell;
  const gemmi::SpaceGroup *sg;
  std::vector<gemmi::Transform> ncs;
  bool mott_bethe;
  std::shared_ptr<RefineParams> params;
  // table (distances x b values)
  std::vector<double> table_bs;
  std::vector<std::vector<double>> pp1; // for x-x diagonal
  std::vector<std::vector<double>> bb;  // for B-B diagonal
  std::vector<std::vector<double>> aa; // for B-B diagonal, aniso
  std::vector<std::vector<double>> qq; // for q-q diagonal
  std::vector<std::vector<double>> qb; // for q-B
  std::vector<double> vn; // first derivatives
  std::vector<double> am; // second derivative sparse matrix

  LL(const gemmi::Structure &st, std::shared_ptr<RefineParams> params, bool mott_bethe)
    : cell(st.cell), sg(st.find_spacegroup()), params(params), mott_bethe(mott_bethe) {
    set_ncs({});
  }
  void set_ncs(const std::vector<gemmi::Transform> &trs) {
    ncs.clear();
    ncs.push_back(gemmi::Transform()); // make sure first is the identity op
    for (const auto &tr : trs)
      if (!tr.is_identity())
        ncs.push_back(tr);
  }
  // FFT-based gradient calculation: Murshudov et al. (1997) 10.1107/S0907444996012255
  // if cryo-EM SPA, den is the Fourier transform of (dLL/dAc-i dLL/dBc)*mott_bethe_factor/s^2
  // When b_add is given, den must have been sharpened
  template <typename Table, bool is_neutron=false>
  void calc_grad(gemmi::Grid<float> &den, double b_add) { // needs <double>?
    const size_t n_atoms = params->atoms.size();
    using CoefType = typename Table::Coef;
    if (params->is_refined(RefineParams::Type::D) && !is_neutron)
      gemmi::fail("Deuterium fraction refinement only works for neutron data.");
    const double d_minus_h = [&]() {
      if constexpr (is_neutron)
        return Table::get_(gemmi::El::D) - Table::get_(gemmi::El::H);
      return 1.; // never used
    }();
    vn.assign(params->n_params(), 0.);
    for (size_t i = 0; i < n_atoms; ++i) {
      const gemmi::Atom &atom = *params->atoms[i];
      if (!params->is_atom_refined(i)) continue;
      if (params->is_atom_excluded_ll(i)) continue;
      const gemmi::Element &el = atom.element;
      const double neutron_h_f_correction = [&](){
        if constexpr (is_neutron)
          if (atom.is_hydrogen())
            return (atom.fraction * Table::get_(gemmi::El::D) + (1 - atom.fraction) * Table::get_(gemmi::El::H)) / d_minus_h;
        return 1.;
      }();
      const auto coef = [&](){
        if constexpr (is_neutron)
          if (atom.is_hydrogen())
            return CoefType{{d_minus_h}}; // corrected later
        return Table::get(el, atom.charge, atom.serial);
      }();
      using precal_aniso_t = decltype(coef.precalculate_density_aniso_b(gemmi::SMat33<double>()));
      const int pos_x = params->get_pos_vec_ll(i, RefineParams::Type::X);
      const int pos_b = params->get_pos_vec_ll(i, RefineParams::Type::B);
      const int pos_q = params->get_pos_vec_ll(i, RefineParams::Type::Q);
      const int pos_d = params->get_pos_vec_ll(i, RefineParams::Type::D);
      const bool has_aniso = atom.aniso.nonzero();
      const int adp_mode = pos_b >= 0 ? (params->aniso ? 2 : 1) : 0;
      if (adp_mode == 1 && has_aniso) gemmi::fail("bad adp_mode");
      for (const gemmi::Transform &tr : ncs) { //TODO to use cell images?
        const gemmi::Fractional fpos = cell.fractionalize(gemmi::Position(tr.apply(atom.pos)));
        const gemmi::SMat33<double> b_aniso = atom.aniso.scaled(gemmi::u_to_b()).added_kI(b_add).transformed_by(tr.mat);
        double b_max = atom.b_iso + b_add;
        if (has_aniso) {
          const auto eig = b_aniso.calculate_eigenvalues();
          b_max = std::max(std::max(eig[0], eig[1]), eig[2]);
        }
        const auto precal = coef.precalculate_density_iso(b_max,
                                                          mott_bethe ? -el.atomic_number() : 0.);
        const precal_aniso_t precal_aniso = has_aniso ? coef.precalculate_density_aniso_b(b_aniso,
                                                                                          mott_bethe ? -el.atomic_number() : 0.)
          : precal_aniso_t();

        // DensityCalculator::estimate_radius() in gemmi/dencalc.hpp
        const double radius = is_neutron ? std::sqrt(std::log(1e-7 / std::abs(precal.a[0])) / precal.b[0])
                                         : gemmi::determine_cutoff_radius(gemmi::it92_radius_approx(b_max),
                                                                          precal, 1e-7); // TODO cutoff?
        const int N = sizeof(precal.a) / sizeof(precal.a[0]);
        const int du = (int) std::ceil(radius / den.spacing[0]);
        const int dv = (int) std::ceil(radius / den.spacing[1]);
        const int dw = (int) std::ceil(radius / den.spacing[2]);
        gemmi::Position gx;
        double gb = 0.;
        double gb_aniso[6] = {0,0,0,0,0,0};
        double gq = 0.;
        den.template use_points_in_box<true>(fpos, du, dv, dw,
                                             [&](float& point, double r2, const gemmi::Position& delta, int, int, int) {
                                               if (point == 0) return;
                                               if (!has_aniso) { // isotropic
                                                 double for_x = 0., for_b = 0., for_q = 0.;
                                                 for (int j = 0; j < N; ++j) {
                                                   double tmp = precal.a[j] * std::exp(precal.b[j] * r2);
                                                   if (pos_q >= 0 || pos_d >= 0) for_q += tmp;
                                                   tmp *= precal.b[j];
                                                   for_x += tmp;
                                                   if (adp_mode == 1) for_b += tmp * (1.5 + r2 * precal.b[j]);
                                                 }
                                                 gx += for_x * 2 * delta * point;
                                                 if (adp_mode == 1) gb += for_b * point;
                                                 if (pos_q >= 0 || pos_d >= 0) gq += for_q * point;
                                               } else { // anisotropic
                                                 for (int j = 0; j < N; ++j) {
                                                   const double tmp = precal_aniso.a[j] * std::exp(precal_aniso.b[j].r_u_r(delta));
                                                   const auto tmp2 = precal_aniso.b[j].multiply(delta); // -4pi^2 * (B+b)^-1 . delta
                                                   gx += 2 * tmp * gemmi::Position(tmp2) * point;
                                                   if (adp_mode == 2) {
                                                     // d/dp |B| = |B| B^-T
                                                     const auto tmp3 = precal_aniso.b[j].scaled(0.5 * tmp * point).elements_pdb();
                                                     // d/dp r^T B^-1 r = ..
                                                     gb_aniso[0] += tmp3[0] + tmp2.x * tmp2.x * tmp * point;
                                                     gb_aniso[1] += tmp3[1] + tmp2.y * tmp2.y * tmp * point;
                                                     gb_aniso[2] += tmp3[2] + tmp2.z * tmp2.z * tmp * point;
                                                     gb_aniso[3] += 2 * tmp3[3] + 2 * tmp2.x * tmp2.y * tmp * point;
                                                     gb_aniso[4] += 2 * tmp3[4] + 2 * tmp2.x * tmp2.z * tmp * point;
                                                     gb_aniso[5] += 2 * tmp3[5] + 2 * tmp2.y * tmp2.z * tmp * point;
                                                   }
                                                   if (pos_q >= 0 || pos_d >= 0)
                                                     gq += tmp * point;
                                                 }
                                               }
                                             }, false /* fail_on_too_large_radius */,
                                             radius);
        gx *= atom.occ * neutron_h_f_correction;
        if (adp_mode == 1)
          gb *= atom.occ * 0.25 / gemmi::sq(gemmi::pi()) * neutron_h_f_correction;
        else if (adp_mode == 2)
          for (int i = 0; i < 6; ++i)
            gb_aniso[i] *= atom.occ * 0.25 / gemmi::sq(gemmi::pi()) * neutron_h_f_correction;

        if (pos_x >= 0) {
          const auto gx2 = tr.mat.transpose().multiply(gx);
          vn[pos_x  ] += gx2.x;
          vn[pos_x+1] += gx2.y;
          vn[pos_x+2] += gx2.z;
        }
        if (adp_mode == 1)
          vn[pos_b] += gb;
        else if (adp_mode == 2) { // added as B (not U)
          for (int j = 0; j < 6; ++j) {
            const auto m = gemmi::SMat33<double>({double(j==0), double(j==1), double(j==2), double(j==3), double(j==4), double(j==5)}).transformed_by(tr.mat);
            vn[pos_b + j] += (gb_aniso[0] * m.u11 + gb_aniso[1] * m.u22 + gb_aniso[2] * m.u33 +
                              gb_aniso[3] * m.u12 + gb_aniso[4] * m.u13 + gb_aniso[5] * m.u23);
          }
        }
        if (pos_q >= 0)
          vn[pos_q] += gq * neutron_h_f_correction;
        if (pos_d >= 0)
          vn[pos_d] += gq * atom.occ;
      }
    }
    for (auto &v : vn) // to match scale of hessian
      v *= (mott_bethe ? -1 : 1) / (double) ncs.size();
  }

  /*
  void add_fisher_diagonal_naive(const Vec3 &svec, double d2ll) {
    // TODO symmetry
    // TODO aniso
    const double s2 = svec.length_sq();
    for (size_t i = 0; i < atoms.size(); ++i) {
      const gemmi::Atom &atom = *atoms[i];
      const auto coef = IT92<double>::get(atom.element); // as Table::?
      const double f = (atom.element.atomic_number() - coef.calculate_sf(s2/4))/s2; // only for mott_bethe
      const double w = atom.occ*atom.occ*f*f*std::exp(-atom.b_iso*s2/2)*d2ll * 4 * gemmi::pi()*gemmi::pi();
      const int ipos = i*6;
      for (int sign = 0; sign < 2; ++sign) { // Friedel pair
        const Vec3 s = sign ? -svec : svec;
        am[ipos]   += w * s.x * s.x;
        am[ipos+1] += w * s.y * s.y;
        am[ipos+2] += w * s.z * s.z;
        am[ipos+3] += w * s.y * s.x;
        am[ipos+4] += w * s.z * s.x;
        am[ipos+5] += w * s.z * s.y;
      }
    }
  }
  */

  template <typename Table>
  double b_sf_max() const {
    double ret = 0.;
    for (auto atom : params->atoms) {
      const auto &coef = Table::get(atom->element, atom->charge, atom->serial);
      for (int i = 0; i < Table::Coef::ncoeffs; ++i)
        if (coef.b(i) > ret)
          ret = coef.b(i);
    }
    return ret;
  }
  std::pair<double, double> b_min_max() const {
    double b_min = INFINITY, b_max = 0;
    for (const auto atom : params->atoms) {
      const double b_iso = atom->aniso.nonzero() ? gemmi::u_to_b() * atom->aniso.trace() / 3 : atom->b_iso;
      if (b_iso < b_min) b_min = b_iso;
      if (b_iso > b_max) b_max = b_iso;
    }
    return std::make_pair(b_min, b_max);
  }
  // preparation for fisher_diag_from_table()
  // Steiner et al. (2003) doi: 10.1107/S0907444903018675
  template <typename Table>
  void make_fisher_table_diag_fast(const TableS3 &d2dfw_table, const double b_step=5) {
    double b_min, b_max;
    std::tie(b_min, b_max) = b_min_max();
    b_max = 2 * (b_max + b_sf_max<Table>());
    //printf("b_min= %.2f b_max= %.2f\n", b_min, b_max);

    const double s_min = d2dfw_table.s_min, s_max = d2dfw_table.s_max;
    const int s_dim = 120; // actually +1 is allocated
    const double s_step = (s_max - s_min) / s_dim;
    int b_dim = static_cast<int>((b_max - b_min) / b_step) + 2;
    if (b_dim % 2 == 0) ++b_dim; // TODO: need to set maximum b_dim?
    pp1.assign(1, std::vector<double>(b_dim));
    bb.assign(1, std::vector<double>(b_dim));
    aa.assign(1, std::vector<double>(b_dim));
    qq.assign(1, std::vector<double>(b_dim));
    qb.assign(1, std::vector<double>(b_dim));

    table_bs.clear();
    table_bs.reserve(b_dim);

    // only for D = 0 (same atoms) for now
    for (int ib = 0; ib < b_dim; ++ib) {
      const double b = b_min + b_step * ib;
      table_bs.push_back(b);

      std::vector<double> tpp(s_dim+1), tbb(s_dim+1), taa(s_dim+1), tqq(s_dim+1), tqb(s_dim+1);
      for (int i = 0; i <= s_dim; ++i) {
        const double s = s_min + s_step * i;
        const double s2 = s * s;
        const double w_c = d2dfw_table.get_value(s); // average of weight
        const double w_c_ft_c = w_c * std::exp(-b*s2/4.);
        tpp[i] = 16. * gemmi::pi() * gemmi::pi() * gemmi::pi() * w_c_ft_c / 3.; // (2pi)^2 * 4pi/3
        tbb[i] = gemmi::pi() / 4 * w_c_ft_c * s2; // 1/16 * 4pi
        taa[i] = gemmi::pi() / 20 * w_c_ft_c * s2; // 1/16 * 4pi/5 (later *1, *1/3, *4/3)
        tqq[i] = gemmi::pi() * 4 * w_c_ft_c; // 4pi
        tqb[i] = -gemmi::pi() * w_c_ft_c; // -pi
        if (mott_bethe) {
          tqq[i] /= s2;
        } else {
          const double s4 = s2 * s2;
          tpp[i] *= s4;
          tbb[i] *= s4;
          taa[i] *= s4;
          tqq[i] *= s2;
          tqb[i] *= s4;
        }
      }

      // Numerical integration by Simpson's rule
      double sum_tpp1 = 0, sum_tpp2 = 0, sum_tbb1 = 0, sum_tbb2 = 0, sum_taa1 = 0, sum_taa2 = 0, sum_tqq1 = 0, sum_tqq2 = 0;
      double sum_tqb1 = 0, sum_tqb2 = 0;
      for (int i = 1; i < s_dim; i+=2) {
        sum_tpp1 += tpp[i];
        sum_tbb1 += tbb[i];
        sum_taa1 += taa[i];
        sum_tqq1 += tqq[i];
        sum_tqb1 += tqb[i];
      }
      for (int i = 2; i < s_dim; i+=2) {
        sum_tpp2 += tpp[i];
        sum_tbb2 += tbb[i];
        sum_taa2 += taa[i];
        sum_tqq2 += tqq[i];
        sum_tqb2 += tqb[i];
      }

      pp1[0][ib] = (tpp[0] + tpp.back() + 4 * sum_tpp1 + 2 * sum_tpp2) * s_step / 3.;
      bb[0][ib] = (tbb[0] + tbb.back() + 4 * sum_tbb1 + 2 * sum_tbb2) * s_step / 3.;
      aa[0][ib] = (taa[0] + taa.back() + 4 * sum_taa1 + 2 * sum_taa2) * s_step / 3.;
      qq[0][ib] = (tqq[0] + tqq.back() + 4 * sum_tqq1 + 2 * sum_tqq2) * s_step / 3.;
      qb[0][ib] = (tqb[0] + tqb.back() + 4 * sum_tqb1 + 2 * sum_tqb2) * s_step / 3.;
    }
  }

  template <typename Table>
  void make_fisher_table_diag_direct(const std::vector<double> &svals, const std::vector<double> &yvals,
                                     const double b_step=5) {
    assert(svals.size() == yvals.size());
    double b_min, b_max;
    std::tie(b_min, b_max) = b_min_max();
    b_max = 2 * (b_max + b_sf_max<Table>());
    //printf("b_min= %.2f b_max= %.2f\n", b_min, b_max);

    int b_dim = static_cast<int>((b_max - b_min) / b_step) + 2;
    if (b_dim % 2 == 0) ++b_dim; // TODO: need to set maximum b_dim?
    pp1.assign(1, std::vector<double>(b_dim));
    bb.assign(1, std::vector<double>(b_dim));
    aa.assign(1, std::vector<double>(b_dim));
    qq.assign(1, std::vector<double>(b_dim));
    qb.assign(1, std::vector<double>(b_dim));
    table_bs.clear();
    table_bs.reserve(b_dim);

    // assumes data cover asu
    //const double fac = sg->operations().sym_ops.size() * (sg->is_centrosymmetric() ? 1 : 2);
    const double fac = sg->operations().order() * (sg->is_centrosymmetric() ? 1 : 2);
    // only for D = 0 (same atoms) for now
    for (int ib = 0; ib < b_dim; ++ib) {
      const double b = b_min + b_step * ib;
      table_bs.push_back(b);
      for (size_t i = 0; i < svals.size(); ++i) {
        const double s = svals[i], w_c = yvals[i];
        if (std::isnan(w_c)) continue;
        const double s2 = s * s;
        const double s4 = s2 * s2;
        const double w_c_ft_c = w_c * std::exp(-b*s2/4.);
        double tpp = 4. * gemmi::pi() * gemmi::pi() * w_c_ft_c / 3.; // (2pi)^2 /3
        double tbb = 1. / 16 * w_c_ft_c; // 1/16
        double taa = 1. / 80 * w_c_ft_c; // 1/16 /5 (later *1, *1/3, *4/3)
        double tqq = w_c_ft_c; // 1
        double tqb = -w_c_ft_c / 4;
        if (mott_bethe) {
          tpp /= s2;
          tqq /= s4;
          tqb /= s2;
        } else {
          tpp *= s2;
          tbb *= s4;
          taa *= s4;
          tqb *= s2;
        }
        pp1[0][ib] += tpp;
        bb[0][ib] += tbb;
        aa[0][ib] += taa;
        qq[0][ib] += tqq;
        qb[0][ib] += tqb;
      }
      pp1[0][ib] *= fac;
      bb[0][ib] *= fac;
      aa[0][ib] *= fac;
      qq[0][ib] *= fac;
      qb[0][ib] *= fac;
    }
  }

  // no need to be a member of this class
  double interp_1d(const std::vector<double> &x_points,
                   const std::vector<double> &y_points,
                   double x) const {
    assert(x_points.size() == y_points.size());
    assert(!x_points.empty());
    if (x < x_points.front() || x > x_points.back()) gemmi::fail("bad x: " + std::to_string(x));

    if (x_points.size() == 1)
      return y_points.front();

    const int k1 = std::min((size_t) (std::lower_bound(x_points.begin(), x_points.end(), x) - x_points.begin()),
                            x_points.size()-1);
    if (k1 == 0)
      return y_points.front();
    const double a = (y_points[k1] - y_points[k1-1]) / (x_points[k1] - x_points[k1-1]);
    const double dx = x - x_points[k1-1];
    return a * dx + y_points[k1-1];
  }

  template <typename Table, bool is_neutron=false>
  void fisher_diag_from_table() {
    using CoefType = typename Table::Coef;
    if (params->is_refined(RefineParams::Type::D) && !is_neutron)
      gemmi::fail("Deuterium fraction refinement only works for neutron data.");
    const double d_minus_h = [&]() {
      if constexpr (is_neutron)
        return Table::get_(gemmi::El::D) - Table::get_(gemmi::El::H);
      return 1.; // never used
    }();
    const size_t n_atoms = params->atoms.size();
    const size_t n_a = params->n_fisher_ll();
    const int N = Table::Coef::ncoeffs;
    am.assign(n_a, 0.);
    auto get_pos = [&](size_t idx, RefineParams::Type t) {
      if (params->is_excluded_ll(idx, t))
        return -1;
      return params->get_pos_mat_ll(idx, t);
    };
    for (size_t i = 0; i < n_atoms; ++i) {
      const gemmi::Atom &atom = *params->atoms[i];
      if (!params->is_atom_refined(i)) continue;
      if (params->is_atom_excluded_ll(i)) continue;
      const double neutron_h_f_correction = [&](){
        if constexpr (is_neutron)
          if (atom.is_hydrogen())
            return (atom.fraction * Table::get_(gemmi::El::D) + (1 - atom.fraction) * Table::get_(gemmi::El::H)) / d_minus_h;
        return 1.; // never used
      }();
      const int pos_x = get_pos(i, RefineParams::Type::X);
      const int pos_b = get_pos(i, RefineParams::Type::B);
      const int pos_q = get_pos(i, RefineParams::Type::Q);
      const int pos_d = get_pos(i, RefineParams::Type::D);
      const int adp_mode = pos_b >= 0 ? (params->aniso ? 2 : 1) : 0;
      const auto coef = [&](){
        if constexpr (is_neutron)
          if (atom.is_hydrogen())
            return CoefType{{d_minus_h}}; // corrected later
        return Table::get(atom.element, atom.charge, atom.serial);
      }();
      const double w = atom.occ * atom.occ;
      const double c = mott_bethe ? coef.c() - atom.element.atomic_number(): coef.c();
      const double b_iso = atom.aniso.nonzero() ? gemmi::u_to_b() * atom.aniso.trace() / 3 : atom.b_iso;
      double fac_x = 0., fac_b = 0., fac_a = 0., fac_q = 0., fac_qb = 0.;

      // TODO can be reduced for the same elements
      for (int j = 0; j < N + 1; ++j)
        for (int k = j; k < N + 1; ++k) {
          // * -1 is needed for mott_bethe case, but we only need aj * ak so they cancel.
          const double aj = j < N ? coef.a(j) : c;
          const double ak = k < N ? coef.a(k) : c;
          if (aj == 0 || ak == 0)
            continue;
          const double b = 2 * b_iso + (j < N ? coef.b(j) : 0) + (k < N ? coef.b(k) : 0);
          const double ajak = j == k ? sq(aj) : 2 * aj * ak;
          fac_x += ajak * interp_1d(table_bs, pp1[0], b);
          fac_b += ajak * interp_1d(table_bs, bb[0], b);
          fac_a += ajak * interp_1d(table_bs, aa[0], b);
          fac_q += ajak * interp_1d(table_bs, qq[0], b);
          fac_qb += ajak * interp_1d(table_bs, qb[0], b);
        }
      fac_x *= gemmi::sq(neutron_h_f_correction);
      fac_a *= gemmi::sq(neutron_h_f_correction);
      if (pos_x >= 0)
        for (int j = 0; j < 3; ++j)
          am[pos_x + j] += w * fac_x;
      if (adp_mode == 1)
        am[pos_b] += w * fac_b;
      else if (adp_mode == 2) {
        for (int j = 0; j < 3; ++j) am[pos_b + j] += w * fac_a;     // 11-11, 22-22, 33-33
        for (int j = 3; j < 6; ++j) am[pos_b + j] += w * fac_a * 4; // 12-12, 13-13, 23-23
        for (int j : {6, 7, 11})    am[pos_b + j] += w * fac_a / 3; // 11-22, 11-33, 22-33
      }
      if (pos_q >= 0) {
        am[pos_q] += fac_q * gemmi::sq(neutron_h_f_correction); // w = q^2 not needed
        const int pos_qb = params->get_pos_mat_mixed_ll(i, RefineParams::Type::B, RefineParams::Type::Q);
        if (pos_qb >= 0) {
          if (adp_mode == 1)
            am[pos_qb] += fac_qb * atom.occ * gemmi::sq(neutron_h_f_correction);
          else if (adp_mode == 2)
            for (int j = 0; j < 3; ++j)
              am[pos_qb + j] += fac_qb * atom.occ / 3 * gemmi::sq(neutron_h_f_correction);
        }
      }
      if (pos_d >= 0) {
        am[pos_d] += w * fac_q;
        const int pos_bd = params->get_pos_mat_mixed_ll(i, RefineParams::Type::B, RefineParams::Type::D);
        const int pos_qd = params->get_pos_mat_mixed_ll(i, RefineParams::Type::Q, RefineParams::Type::D);
        if (pos_bd >= 0) {
          if (adp_mode == 1)
            am[pos_bd] += w * fac_qb * neutron_h_f_correction;
          else if (adp_mode == 2) {
            for (int j = 0; j < 3; ++j)
              am[pos_bd + j] += w * fac_qb * neutron_h_f_correction / 3;
          }
        }
        if (pos_qd >= 0)
          am[pos_qd] += fac_q * atom.occ * neutron_h_f_correction;
      }
    }
  }

  Eigen::SparseMatrix<double> make_spmat() const {
    const size_t n_atoms = params->atoms.size();
    const size_t n_a = params->n_fisher_ll();
    const size_t n_v = params->n_params();
    Eigen::SparseMatrix<double> spmat(n_v, n_v);
    std::vector<Eigen::Triplet<double>> data;
    std::vector<bool> am_flag(am.size());
    auto add_data = [&](size_t i, size_t j, size_t apos) {
      if (am_flag[apos]) return; // avoid overwriting
      am_flag[apos] = true;
      const double v = am[apos];
      if (i != j && v == 0) return; // we need all diagonals
      data.emplace_back(i, j, v);
      if (i != j)
        data.emplace_back(j, i, v);
    };
    for (int j : params->param_to_atom(RefineParams::Type::X)) {
      const int pos = params->get_pos_vec(j, RefineParams::Type::X);
      const int apos = params->get_pos_mat_ll(j, RefineParams::Type::X);
      if (pos >= 0 && apos >= 0) { // should be always ok
        add_data(pos,   pos,   apos);
        add_data(pos+1, pos+1, apos+1);
        add_data(pos+2, pos+2, apos+2);
        add_data(pos,   pos+1, apos+3);
        add_data(pos,   pos+2, apos+4);
        add_data(pos+1, pos+2, apos+5);
      }
    }
    for (int j : params->param_to_atom(RefineParams::Type::B)) {
      const int pos = params->get_pos_vec(j, RefineParams::Type::B);
      int apos = params->get_pos_mat_ll(j, RefineParams::Type::B);
      if (pos >= 0 && apos >= 0) { // should be always ok
        if (params->aniso) {
          for (size_t k = 0; k < 6; ++k)
            add_data(pos + k, pos + k, apos++);
          for (size_t k = 0; k < 6; ++k)
            for (size_t l = k + 1; l < 6; ++l)
              add_data(pos + k, pos + l, apos++);
        } else
          add_data(pos, pos, apos);
      }
    }
    for (int j : params->param_to_atom(RefineParams::Type::Q)) {
      const int pos = params->get_pos_vec(j, RefineParams::Type::Q);
      const int apos = params->get_pos_mat_ll(j, RefineParams::Type::Q);
      if (pos >= 0 && apos >= 0) // should be always ok
        add_data(pos, pos, apos);
    }
    for (int j : params->param_to_atom(RefineParams::Type::D)) {
      const int pos = params->get_pos_vec(j, RefineParams::Type::D);
      const int apos = params->get_pos_mat_ll(j, RefineParams::Type::D);
      if (pos >= 0 && apos >= 0) // should be always ok
        add_data(pos, pos, apos);
    }
    const std::vector<std::tuple<RefineParams::Type, RefineParams::Type, const std::vector<int> &>> mixvecs = {
      {RefineParams::Type::B, RefineParams::Type::Q, params->bq_mix_atoms},
      {RefineParams::Type::B, RefineParams::Type::D, params->bd_mix_atoms},
      {RefineParams::Type::Q, RefineParams::Type::D, params->qd_mix_atoms}};
    for (const auto &ttvec : mixvecs)
      for (size_t j : std::get<2>(ttvec)) {
        const int pos1 = params->get_pos_vec(j, std::get<0>(ttvec));
        const int pos2 = params->get_pos_vec(j, std::get<1>(ttvec));
        const int apos = params->get_pos_mat_mixed_ll(j, std::get<0>(ttvec), std::get<1>(ttvec));
        if (pos1 < 0 || pos2 < 0 || apos < 0) continue; // should not happen
        if (std::get<0>(ttvec) == RefineParams::Type::B && params->aniso)
          for (size_t k = 0; k < 6; ++k)
            add_data(pos1 + k, pos2, apos + k);
        else
          add_data(pos1, pos2, apos);
      }
    spmat.setFromTriplets(data.begin(), data.end());
    return spmat;
  }

  void spec_correction(const std::vector<Geometry::Special> &specials, double alpha=1e-3, bool use_rr=true) {
    for (const auto &spec : specials) {
      // TODO inefficient?
      const auto it = std::find(params->atoms.begin(), params->atoms.end(), spec.atom);
      if (it == params->atoms.end()) continue;
      const int j = std::distance(params->atoms.begin(), it);
      if (params->is_refined(RefineParams::Type::X)) {
        const int idx = params->get_pos_vec(j, RefineParams::Type::X);
        if (idx >= 0) {
          // correct gradient
          Eigen::Map<Eigen::Vector3d> v(&vn[idx], 3);
          v = (spec.Rspec_pos.transpose() * v).eval();
          // correct diagonal block
          double* a = am.data() + params->get_pos_mat_ll(j, RefineParams::Type::X);
          Eigen::Matrix3d m {{a[0], a[3], a[4]},
                             {a[3], a[1], a[5]},
                             {a[4], a[5], a[2]}};
          const double hmax = m.maxCoeff() * spec.n_mult;
          m = (spec.Rspec_pos.transpose() * m * spec.Rspec_pos * spec.n_mult).eval();
          if (use_rr)
            m += (hmax * alpha * (Eigen::Matrix3d::Identity()
                                  - spec.Rspec_pos * spec.Rspec_pos)).eval();
          else
            m += (hmax * alpha * Eigen::Matrix3d::Identity()).eval();
          a[0] = m(0,0);
          a[1] = m(1,1);
          a[2] = m(2,2);
          a[3] = m(0,1);
          a[4] = m(0,2);
          a[5] = m(1,2);
        }
      }
      if (params->is_refined(RefineParams::Type::B)) {
        const int idx = params->get_pos_vec(j, RefineParams::Type::B);
        const int pos = params->get_pos_mat_ll(j, RefineParams::Type::B);
        if (idx >= 0) {
          if (!params->aniso)
            am[pos] *= spec.n_mult;
          else {
            // correct gradient
            Eigen::Map<Eigen::VectorXd> v(&vn[idx], 6);
            v = (spec.Rspec_aniso.transpose() * v).eval();
            // correct diagonal block
            double* a = am.data() + pos;
            Eigen::MatrixXd m {{ a[0],  a[6],  a[7],  a[8],  a[9], a[10]},
                               { a[6],  a[1], a[11], a[12], a[13], a[14]},
                               { a[7], a[11],  a[2], a[15], a[16], a[17]},
                               { a[8], a[12], a[15],  a[3], a[18], a[19]},
                               { a[9], a[13], a[16], a[18],  a[4], a[20]},
                               {a[10], a[14], a[17], a[19], a[20],  a[5]}};
            const double hmax = m.maxCoeff() * spec.n_mult;
            m = (spec.Rspec_aniso.transpose() * m * spec.Rspec_aniso * spec.n_mult).eval();
            if (use_rr)
              m += (hmax * alpha * (Eigen::Matrix<double,6,6>::Identity()
                                    - spec.Rspec_aniso * spec.Rspec_aniso)).eval();
            else
              m += (hmax * alpha * Eigen::Matrix<double,6,6>::Identity()).eval();
            for (int i = 0; i < 6; ++i)
              a[i] = m(i, i);
            for (int j = 0, i = 6; j < 6; ++j)
              for (int k = j + 1; k < 6; ++k, ++i)
                a[i] = m(j, k);
          }
        }
      }
    }
  }
};



} // namespace servalcat
#endif
