// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_REFINE_GEOM_HPP_
#define SERVALCAT_REFINE_GEOM_HPP_

#include "params.hpp"
#include "ncsr.hpp"
#include <set>
#include <memory>
#include <algorithm>
#include <gemmi/model.hpp>    // for Structure, Atom
#include <gemmi/contact.hpp>  // for NeighborSearch, ContactSearch
#include <gemmi/topo.hpp>     // for Topo
#include <gemmi/select.hpp>   // for count_atom_sites
#include <gemmi/eig3.hpp>     // for eigen_decomposition
#include <gemmi/bond_idx.hpp> // for BondIndex
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace servalcat {

Eigen::Matrix<double,6,6> mat33_as66(const Eigen::Matrix<double,3,3> &m) {
  // suppose R is a transformation matrix that is applied to 3x3 symmetric matrix U: R U R^T
  // this function constructs equivalent transformation for 6-element vector: R' u
  Eigen::Matrix<double,6,6> r;
  const std::vector<std::pair<int,int>> idxes = {{0,0}, {1,1}, {2,2}, {0,1}, {0,2}, {1,2}};
  for (int k = 0; k < 6; ++k) {
    const int i = idxes[k].first, j = idxes[k].second;
    r(k, Eigen::all) <<
      m(i,0) * m(j,0),
      m(i,1) * m(j,1),
      m(i,2) * m(j,2),
      m(i,0) * m(j,1) + m(i,1) * m(j,0),
      m(i,0) * m(j,2) + m(i,2) * m(j,0),
      m(i,1) * m(j,2) + m(i,2) * m(j,1);
  }
  return r;
}

Eigen::Matrix<double,6,6> mat33_as66(const gemmi::Mat33 &m) {
  return mat33_as66(Eigen::Matrix<double,3,3>(&m.a[0][0]));
}

gemmi::Mat33 eigen_decomp_inv(const gemmi::SMat33<double> &m, double e, bool for_precond) {
  // good e = 1.e-9 for plane and ~1e-6 or 1e-4 for precondition
  auto f = [&](double v){
    if (std::abs(v) < e) return 0.;
    return for_precond ? 1. / std::sqrt(v) : 1. / v;
  };
  double eig[3] = {};
  const gemmi::Mat33 Q = gemmi::eigen_decomposition(m, eig);
  const gemmi::Vec3 l{f(eig[0]), f(eig[1]), f(eig[2])};
  if (for_precond)
    return Q.multiply_by_diagonal(l);
  else
    return Q.multiply_by_diagonal(l).multiply(Q.transpose());
}

// https://doi.org/10.48550/arXiv.1701.03077
struct Barron2019 {
  Barron2019(double alpha, double y) {
    if (std::abs(alpha - 2) < 1e-3) { // least square
      f = 0.5 * y * y;
      dfdy = y;
      d2fdy = 1.0;
    } else if (std::abs(alpha) < 1e-3) { // cauchy or lorentz
      f = std::log(0.5 * y * y + 1.0);
      dfdy = y / (0.5 * y * y + 1.0);
      d2fdy = 1.0 / (0.5 * y * y + 1.0);
    } else if (alpha < -1000) { // -inf. welch
      const double expy = std::exp(-0.5 * y * y);
      f = 1.0 - expy;
      dfdy = y * expy;
      d2fdy = expy;
    } else { // other alpha
      const double alpha2 = std::abs(alpha - 2.0);
      f = alpha2 / alpha * (std::pow(y * y / alpha2 + 1, 0.5 * alpha) - 1.0);
      dfdy = y * std::pow(y * y / alpha2 + 1, 0.5 * alpha - 1.0);
      d2fdy = std::pow(y * y / alpha2 + 1, 0.5 * alpha - 1.0);
    }
  }
  double f, dfdy, d2fdy;
};

struct PlaneDeriv {
  PlaneDeriv(const std::vector<gemmi::Atom*> &atoms)
    : dvmdx(atoms.size(), std::vector<gemmi::Vec3>(3)), dDdx(atoms.size()) {
    // centroid
    for (const gemmi::Atom* atom : atoms) xs += atom->pos;
    xs /= (double) atoms.size();

    // from gemmi/calculate.hpp find_best_plane()
    // for plane normal
    gemmi::SMat33<double> A{0, 0, 0, 0, 0, 0};
    for (const gemmi::Atom* atom : atoms) {
      const gemmi::Vec3 p = gemmi::Vec3(atom->pos) - xs;
      A.u11 += p.x * p.x;
      A.u22 += p.y * p.y;
      A.u33 += p.z * p.z;
      A.u12 += p.x * p.y;
      A.u13 += p.x * p.z;
      A.u23 += p.y * p.z;
    }
    double eig[3] = {};
    const gemmi::Mat33 V = gemmi::eigen_decomposition(A, eig);
    int smallest_idx = std::fabs(eig[0]) < std::fabs(eig[1]) ? 0 : 1;
    if (std::fabs(eig[2]) < std::fabs(eig[smallest_idx]))
      smallest_idx = 2;
    const double eigmin = eig[smallest_idx];
    vm = V.column_copy(smallest_idx);
    D = xs.dot(vm);

    // derivatives
    const gemmi::Mat33 pinv = eigen_decomp_inv(gemmi::SMat33<double>{eigmin,eigmin,eigmin,0,0,0}-A, 1e-9, false);
    for (size_t i = 0; i < atoms.size(); ++i) {
      const gemmi::Vec3 p = gemmi::Vec3(atoms[i]->pos) - xs;
      const gemmi::SMat33<double> dAdx[3] = {{2 * p.x, 0.,      0.,      p.y, p.z, 0.},   // dA/dx
                                             {0.,      2 * p.y, 0.,      p.x, 0.,  p.z},  // dA/dy
                                             {0.,      0.,      2 * p.z, 0.,  p.x, p.y}}; // dA/dz
      for (size_t j = 0; j < 3; ++j)
        dvmdx[i][j] = pinv.multiply(dAdx[j].multiply(vm));
    }
    for (size_t i = 0; i < atoms.size(); ++i) {
      dDdx[i] = vm / (double) atoms.size();
      for (size_t j = 0; j < 3; ++j)
        dDdx[i].at(j) += xs.dot(dvmdx[i][j]);
    }
  }

  void flip() {
    vm *= -1;
    D *= -1;
    for (auto &x : dvmdx)
      for (auto &y : x)
        y *= -1;
    for (auto &x : dDdx)
      x *= -1;
  }

  gemmi::Vec3 xs; // centroid
  gemmi::Vec3 vm; // normal vector
  double D; // xs dot vm
  std::vector<std::vector<gemmi::Vec3>> dvmdx; // derivative of vm wrt positions
  std::vector<gemmi::Vec3> dDdx; // derivative of D wrt positions
};

inline double chiral_abs_volume_sigma(double bond1, double bond2, double bond3,
                                      double angle1, double angle2, double angle3,
                                      double sigb1, double sigb2, double sigb3,
                                      double siga1, double siga2, double siga3) {
  using gemmi::sq;
  using gemmi::rad;
  const double mult = bond1 * bond2 * bond3;
  auto cosine = [](double a) {return a == 90. ? 0. : std::cos(rad(a));};
  const double cosa1 = cosine(angle1);
  const double cosa2 = cosine(angle2);
  const double cosa3 = cosine(angle3);
  double x_y = 1 + 2 * cosa1 * cosa2 * cosa3 - sq(cosa1) - sq(cosa2) - sq(cosa3);
  double varv = x_y * (sq(bond2 * bond3) * sq(sigb1) +
                       sq(bond1 * bond3) * sq(sigb2) +
                       sq(bond1 * bond2) * sq(sigb3));
  varv += sq(mult) / x_y * ((1 - sq(cosa1)) * sq(cosa1 - cosa2 * cosa3) * sq(rad(siga1)) +
                            (1 - sq(cosa2)) * sq(cosa2 - cosa1 * cosa3) * sq(rad(siga2)) +
                            (1 - sq(cosa3)) * sq(cosa3 - cosa1 * cosa2) * sq(rad(siga3)));
  return std::sqrt(varv);
}

std::pair<double,double>
ideal_chiral_abs_volume_sigma(const gemmi::Topo &topo, const gemmi::Topo::Chirality &ch) {
  const gemmi::Restraints::Bond* bond_c1 = topo.take_bond(ch.atoms[0], ch.atoms[1]);
  const gemmi::Restraints::Bond* bond_c2 = topo.take_bond(ch.atoms[0], ch.atoms[2]);
  const gemmi::Restraints::Bond* bond_c3 = topo.take_bond(ch.atoms[0], ch.atoms[3]);
  const gemmi::Restraints::Angle* angle_1c2 = topo.take_angle(ch.atoms[1], ch.atoms[0], ch.atoms[2]);
  const gemmi::Restraints::Angle* angle_2c3 = topo.take_angle(ch.atoms[2], ch.atoms[0], ch.atoms[3]);
  const gemmi::Restraints::Angle* angle_3c1 = topo.take_angle(ch.atoms[3], ch.atoms[0], ch.atoms[1]);
  if (bond_c1 && bond_c2 && bond_c3 && angle_1c2 && angle_2c3 && angle_3c1)
    return std::make_pair(gemmi::chiral_abs_volume(bond_c1->value, bond_c2->value, bond_c3->value,
                                                   angle_1c2->value, angle_2c3->value, angle_3c1->value),
                          chiral_abs_volume_sigma(bond_c1->value, bond_c2->value, bond_c3->value,
                                                  angle_1c2->value, angle_2c3->value, angle_3c1->value,
                                                  bond_c1->esd, bond_c2->esd, bond_c3->esd,
                                                  angle_1c2->esd, angle_2c3->esd, angle_3c1->esd));
  return std::make_pair(std::numeric_limits<double>::quiet_NaN(), 0);
}

struct GeomTarget {
  struct MatPos {
    int ipos;
    int imode;
  };

  GeomTarget(std::shared_ptr<RefineParams> params)
    : params(params) {}

  void setup(bool use_occr) {
    this->use_occr = use_occr;
    const size_t n_atoms = params->atoms.size(); // all atoms
    target = 0.;
    vn.clear();
    am.clear();
    rest_per_atom.clear();
    rest_pos_per_atom.clear();
    if (params->is_refined(RefineParams::Type::X))
      params->set_pairs(RefineParams::Type::X, pairs);
    if (params->is_refined(RefineParams::Type::B))
      params->set_pairs(RefineParams::Type::B, pairs);
    if (params->is_refined(RefineParams::Type::Q) &&
        (use_occr || !params->occ_group_constraints.empty()))
      params->set_pairs(RefineParams::Type::Q, pairs);
    const size_t qqv = params->n_params();
    const size_t qqm = params->n_fisher_geom();
    vn.assign(qqv, 0.);
    am.assign(qqm, 0.);

    std::vector<size_t> nrest_per_atom(n_atoms, 0);
    for (auto & p : pairs) {
      ++nrest_per_atom[p.first];
      if (p.first != p.second)
        ++nrest_per_atom[p.second];
    }

    rest_pos_per_atom.assign(n_atoms+1, 0);
    for (size_t ia = 0; ia < n_atoms; ++ia)
      rest_pos_per_atom[ia+1] = rest_pos_per_atom[ia] + nrest_per_atom[ia];

    rest_per_atom.assign(std::accumulate(nrest_per_atom.begin(), nrest_per_atom.end(), (size_t)0), 0);
    for (size_t i = 0; i < n_atoms; ++i) nrest_per_atom[i] = 0;
    for (size_t i = 0; i < rest_per_atom.size(); ++i) rest_per_atom[i] = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
      int ia1 = pairs[i].first, ia2 = pairs[i].second;
      nrest_per_atom[ia1] += 1;
      int ip1 = rest_pos_per_atom[ia1] + nrest_per_atom[ia1] - 1;
      rest_per_atom[ip1] = i;
      if (ia1 != ia2) {
        nrest_per_atom[ia2] += 1;
        int ip2 = rest_pos_per_atom[ia2] + nrest_per_atom[ia2] - 1;
        rest_per_atom[ip2] = i;
      }
    }
  }

  std::shared_ptr<RefineParams> params;
  bool use_occr; // occupancy restraint
  double target = 0.; // target function value
  std::vector<double> vn; // first derivatives
  std::vector<double> am; // second derivative sparse matrix
  std::vector<size_t> rest_per_atom;
  std::vector<size_t> rest_pos_per_atom;
  std::vector<std::pair<int,int>> pairs;
  std::vector<int> pairs_kind; // refmac's nw_uval. minimum of (bond=1, angle=2, torsion=3, chir=4, plane=5, vdw=6, stack=8, ncsr=10)
  size_t n_pairs() const { return pairs.size(); }
  MatPos find_restraint(int ia1, int ia2, RefineParams::Type type = RefineParams::Type::X) const {
    MatPos matpos;
    int idist = -1;
    for (size_t irest = rest_pos_per_atom[ia1]; irest < rest_pos_per_atom[ia1+1]; ++irest) {
      int ir1 = rest_per_atom[irest];
      if (pairs[ir1].first == ia2) {
        // atom ia1 is target atom and atom ia2 is object
        idist = ir1;
        matpos.imode = 0;
        break;
      }
      else if (pairs[ir1].second == ia2) {
        // atom ia1 is object atom and atom ia2 is target
        idist = ir1;
        matpos.imode = 1;
        break;
      }
    }
    if (idist < 0) gemmi::fail("cannot find atom pair");
    matpos.ipos = params->get_pos_mat_pair_geom(idist, type);
    return matpos;
  }
  void incr_vn(size_t ipos, double w, const gemmi::Vec3 &deriv) {
    assert(ipos+2 < vn.size());
    vn[ipos]   += w * deriv.x;
    vn[ipos+1] += w * deriv.y;
    vn[ipos+2] += w * deriv.z;
  }
  void incr_am_diag(size_t ipos, double w, const gemmi::Vec3 &deriv) {
    assert(ipos+5 < am.size());
    am[ipos]   += w * deriv.x * deriv.x;
    am[ipos+1] += w * deriv.y * deriv.y;
    am[ipos+2] += w * deriv.z * deriv.z;
    am[ipos+3] += w * deriv.y * deriv.x;
    am[ipos+4] += w * deriv.z * deriv.x;
    am[ipos+5] += w * deriv.z * deriv.y;
  }
  void incr_am_diag12(size_t ipos, double w, const gemmi::Vec3 &deriv1, const gemmi::Vec3 &deriv2) {
    // used when atoms are related with each other for example through symmetry
    assert(ipos+5 < am.size());
    am[ipos]   += w *  deriv1.x * deriv2.x * 2.;
    am[ipos+1] += w *  deriv1.y * deriv2.y * 2.;
    am[ipos+2] += w *  deriv1.z * deriv2.z * 2.;
    am[ipos+3] += w * (deriv1.x * deriv2.y + deriv2.x * deriv1.y);
    am[ipos+4] += w * (deriv1.x * deriv2.z + deriv2.x * deriv1.z);
    am[ipos+5] += w * (deriv1.y * deriv2.z + deriv2.y * deriv1.z);
  }
  // TODO to take matpos instead of ipos
  void incr_am_ndiag(size_t ipos, double w, const gemmi::Vec3 &deriv1, const gemmi::Vec3 &deriv2) {
    assert(ipos+8 < am.size());
    am[ipos]   += w * deriv1.x * deriv2.x;
    am[ipos+1] += w * deriv1.y * deriv2.x;
    am[ipos+2] += w * deriv1.z * deriv2.x;
    am[ipos+3] += w * deriv1.x * deriv2.y;
    am[ipos+4] += w * deriv1.y * deriv2.y;
    am[ipos+5] += w * deriv1.z * deriv2.y;
    am[ipos+6] += w * deriv1.x * deriv2.z;
    am[ipos+7] += w * deriv1.y * deriv2.z;
    am[ipos+8] += w * deriv1.z * deriv2.z;
  }
  Eigen::SparseMatrix<double> make_spmat() const {
    const size_t n_atoms = params->atoms.size();
    const size_t n_v = params->n_params();
    Eigen::SparseMatrix<double> spmat(n_v, n_v);
    std::vector<Eigen::Triplet<double>> data;
    std::vector<bool> am_flag(am.size());
    auto add_data = [&](size_t i, size_t j, size_t apos) {
      if (am_flag[apos]) return; // avoid overwriting
      am_flag[apos] = true;
      const double v = am[apos];
      if (i != j && v == 0.) return; // we need all diagonals
      data.emplace_back(i, j, v);
      if (i != j)
        data.emplace_back(j, i, v);
    };
    for (int j : params->param_to_atom(RefineParams::Type::X)) {
      const int pos = params->get_pos_vec(j, RefineParams::Type::X);
      const int apos = params->get_pos_mat_geom(j, RefineParams::Type::X);
      if (pos >= 0 && apos >= 0) { // should be always ok
        add_data(pos,   pos,   apos);
        add_data(pos+1, pos+1, apos+1);
        add_data(pos+2, pos+2, apos+2);
        add_data(pos,   pos+1, apos+3);
        add_data(pos,   pos+2, apos+4);
        add_data(pos+1, pos+2, apos+5);
      }
    }
    if (params->is_refined(RefineParams::Type::X)) {
      for (int j = 0; j < pairs.size(); ++j)
        if (params->pairs_refine(RefineParams::Type::X)[j] >= 0) {
          const auto &p = pairs[j]; // assumes sequential order..
          const int pos1 = params->get_pos_vec(p.second, RefineParams::Type::X);
          const int pos2 = params->get_pos_vec(p.first, RefineParams::Type::X);
          int apos = params->get_pos_mat_pair_geom(j, RefineParams::Type::X);
          for (size_t k = 0; k < 3; ++k)
            for (size_t l = 0; l < 3; ++l)
              add_data(pos1 + l, pos2 + k, apos++);
        }
    }
    for (int j : params->param_to_atom(RefineParams::Type::B)) {
      const int pos = params->get_pos_vec(j, RefineParams::Type::B);
      int apos = params->get_pos_mat_geom(j, RefineParams::Type::B);
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
    if (params->is_refined(RefineParams::Type::B)) {
      for (int j = 0; j < pairs.size(); ++j)
        if (params->pairs_refine(RefineParams::Type::B)[j] >= 0) {
          const auto &p = pairs[j];
          const int pos1 = params->get_pos_vec(p.second, RefineParams::Type::B);
          const int pos2 = params->get_pos_vec(p.first, RefineParams::Type::B);
          int apos = params->get_pos_mat_pair_geom(j, RefineParams::Type::B);
          if (params->aniso) {
            for (size_t k = 0; k < 6; ++k)
              for (size_t l = 0; l < 6; ++l)
                add_data(pos1 + l, pos2 + k, apos++);
          } else
            add_data(pos1, pos2, apos);
        }
    }
    if (params->is_refined(RefineParams::Type::Q)) {
      for (int j : params->param_to_atom(RefineParams::Type::Q)) {
        const int pos = params->get_pos_vec(j, RefineParams::Type::Q);
        const int apos = params->get_pos_mat_geom(j, RefineParams::Type::Q);
        if (pos >= 0 && apos >= 0)
          add_data(pos, pos, apos);
      }
      for (int j = 0; j < pairs.size(); ++j)
        if (params->pairs_refine(RefineParams::Type::Q)[j] >= 0) {
          const auto &p = pairs[j];
          const int pos1 = params->get_pos_vec(p.second, RefineParams::Type::Q);
          const int pos2 = params->get_pos_vec(p.first, RefineParams::Type::Q);
          const int apos = params->get_pos_mat_pair_geom(j, RefineParams::Type::Q);
          if (pos1 >= 0 && pos2 >= 0 && apos >= 0)
            add_data(pos1, pos2, apos);
        }
    }
    spmat.setFromTriplets(data.begin(), data.end());
    return spmat;
  }
};

struct Geometry {
  struct Reporting;
  struct Bond {
    struct Value {
      Value(double v, double s, double vn, double sn)
        : value(v), sigma(s),
          value_nucleus(std::isnan(vn) ? v : vn),
          sigma_nucleus(std::isnan(sn) ? s : sn) {}
      double value;
      double sigma;
      double value_nucleus;
      double sigma_nucleus;
      // alpha should be here?
    };
    Bond(gemmi::Atom* atom1, gemmi::Atom* atom2) : atoms({atom1, atom2}) {
      if (atoms[0]->serial > atoms[1]->serial)
        std::reverse(atoms.begin(), atoms.end());
    }
    void set_image(const gemmi::UnitCell& cell, gemmi::Asu asu) {
      const gemmi::NearestImage im = cell.find_nearest_image(atoms[0]->pos, atoms[1]->pos, asu);
      sym_idx = im.sym_idx;
      std::copy(std::begin(im.pbc_shift), std::end(im.pbc_shift), std::begin(pbc_shift));
    }
    bool same_asu() const {
      return (sym_idx == 0 || sym_idx == -1) && pbc_shift[0]==0 && pbc_shift[1]==0 && pbc_shift[2]==0;
    }
    std::tuple<int,int,int,int,int,int> sort_key() const {
      return std::tie(atoms[0]->serial, atoms[1]->serial, sym_idx, pbc_shift[0], pbc_shift[1], pbc_shift[2]);
    }
    const Value* find_closest_value(double dist, bool use_nucleus) const {
      double db = std::numeric_limits<double>::infinity();
      const Value* ret = nullptr; // XXX safer to initialise with first item
      for (const auto &v : values) {
        double tmp = std::abs((use_nucleus ? v.value_nucleus : v.value) - dist);
        if (tmp < db) {
          db = tmp;
          ret = &v;
        }
      }
      return ret;
    }
    double calc(const gemmi::UnitCell& cell, bool use_nucleus, double wdskal,
                GeomTarget* target, Reporting *reporting) const;

    int type = 1; // 0-2
    double alpha = 1; // only effective for type=2
    int sym_idx = 0;
    std::array<int, 3> pbc_shift = {{0,0,0}};
    std::array<gemmi::Atom*, 2> atoms;
    std::vector<Value> values;
  };
  struct Angle {
    struct Value {
      Value(double v, double s) : value(v), sigma(s) {}
      double value;
      double sigma;
    };
    Angle(gemmi::Atom* atom1, gemmi::Atom* atom2, gemmi::Atom* atom3) : atoms({atom1, atom2, atom3}) {
      if (atoms[0]->serial > atoms[2]->serial)
        std::reverse(atoms.begin(), atoms.end());
    }
    void set_images(const gemmi::UnitCell& cell, gemmi::Asu asu1, gemmi::Asu asu3) {
      const gemmi::NearestImage im1 = cell.find_nearest_image(atoms[1]->pos, atoms[0]->pos, asu1);
      const gemmi::NearestImage im2 = cell.find_nearest_image(atoms[1]->pos, atoms[2]->pos, asu3);
      sym_idx_1 = im1.sym_idx;
      sym_idx_2 = im2.sym_idx;
      std::copy(std::begin(im1.pbc_shift), std::end(im1.pbc_shift), std::begin(pbc_shift_1));
      std::copy(std::begin(im2.pbc_shift), std::end(im2.pbc_shift), std::begin(pbc_shift_2));
    }
    bool same_asu(int i) const {
      switch (i) {
      case 0:
        return sym_idx_1 == 0 && pbc_shift_1[0]==0 && pbc_shift_1[1]==0 && pbc_shift_1[2]==0;
      case 1:
        return true;
      case 2:
        return sym_idx_2 == 0 && pbc_shift_2[0]==0 && pbc_shift_2[1]==0 && pbc_shift_2[2]==0;
      default:
        throw std::runtime_error("invalid index in Angle::same_asu()");
      }
    }
    std::tuple<int,int,int,int,int,int,int,int,int,int,int> sort_key() const {
      return std::tie(atoms[0]->serial, atoms[1]->serial, atoms[2]->serial,
                      sym_idx_1, pbc_shift_1[0], pbc_shift_1[1], pbc_shift_1[2],
                      sym_idx_2, pbc_shift_2[0], pbc_shift_2[1], pbc_shift_2[2]);
    }
    const Value* find_closest_value(double v) const {
      double db = std::numeric_limits<double>::infinity();
      const Value* ret = nullptr;
      for (const auto &value : values) {
        double tmp = gemmi::angle_abs_diff(value.value, v);
        if (tmp < db) {
          db = tmp;
          ret = &value;
        }
      }
      return ret;
    }
    void normalize_ideal() {
      for (auto &value : values)
        value.value = gemmi::angle_abs_diff(value.value, 0.); // limit to [0,180]
    }
    double calc(const gemmi::UnitCell& cell, double waskal, bool von_mises, GeomTarget* target, Reporting *reporting) const;
    int type = 1; // 0 or not
    int sym_idx_1 = 0, sym_idx_2 = 0;
    std::array<int, 3> pbc_shift_1 = {{0,0,0}}, pbc_shift_2 = {{0,0,0}};
    std::array<gemmi::Atom*, 3> atoms;
    std::vector<Value> values;
  };
  struct Torsion {
    struct Value {
      Value(double v, double s, int p): value(v), sigma(s), period(p) {}
      double value;
      double sigma;
      int period;
      std::string label;
    };
    Torsion(gemmi::Atom* atom1, gemmi::Atom* atom2, gemmi::Atom* atom3, gemmi::Atom* atom4) : atoms({atom1, atom2, atom3, atom4}) {
      if (atoms[0]->serial > atoms[3]->serial)
        std::reverse(atoms.begin(), atoms.end());
    }
    std::tuple<int,int,int,int> sort_key() const {
      return std::tie(atoms[0]->serial, atoms[1]->serial, atoms[2]->serial, atoms[3]->serial);
    }
    const Value* find_closest_value(double v) const {
      double db = std::numeric_limits<double>::infinity();
      const Value* ret = nullptr;
      for (const auto &value : values) {
        double tmp = gemmi::angle_abs_diff(value.value, v, 360. / std::max(1, value.period));
        if (tmp < db) {
          db = tmp;
          ret = &value;
        }
      }
      return ret;
    }
    double calc(double wtskal, GeomTarget* target, Reporting *reporting) const;
    int type = 1; // 0 or not
    std::array<gemmi::Atom*, 4> atoms;
    std::vector<Value> values;
  };
  struct Chirality {
    Chirality(gemmi::Atom* atomc, gemmi::Atom* atom1, gemmi::Atom* atom2, gemmi::Atom* atom3) : atoms({atomc, atom1, atom2, atom3}) {}
    double calc(double wchiral, GeomTarget* target, Reporting *reporting) const;
    double value;
    double sigma;
    gemmi::ChiralityType sign;
    std::array<gemmi::Atom*, 4> atoms;
  };
  struct Plane {
    Plane(std::vector<gemmi::Atom*> a) : atoms(a) {}
    double calc(double wplane, GeomTarget* target, Reporting *reporting) const;
    double sigma;
    std::string label;
    std::vector<gemmi::Atom*> atoms;
  };
  struct Interval {
    Interval(gemmi::Atom* atom1, gemmi::Atom* atom2) : atoms({atom1, atom2}) {}
    void set_image(const gemmi::UnitCell& cell, gemmi::Asu asu) {
      const gemmi::NearestImage im = cell.find_nearest_image(atoms[0]->pos, atoms[1]->pos, asu);
      sym_idx = im.sym_idx;
      std::copy(std::begin(im.pbc_shift), std::end(im.pbc_shift), std::begin(pbc_shift));
    }
    bool same_asu() const {
      return sym_idx == 0 && pbc_shift[0]==0 && pbc_shift[1]==0 && pbc_shift[2]==0;
    }
    std::tuple<int,int,int,int,int,int> sort_key() const {
      return std::tie(atoms[0]->serial, atoms[1]->serial, sym_idx, pbc_shift[0], pbc_shift[1], pbc_shift[2]);
    }
    double calc(const gemmi::UnitCell& cell, double w, GeomTarget* target, Reporting *reporting) const;
    double dmin;
    double dmax;
    double smin;
    double smax;
    int sym_idx = 0;
    std::array<int, 3> pbc_shift = {{0,0,0}};
    std::array<gemmi::Atom*, 2> atoms;
  };
  struct Harmonic {
    Harmonic(gemmi::Atom* a) : atom(a) {}
    void calc(GeomTarget* target) const;
    double sigma;
    gemmi::Atom* atom;
  };
  struct Special {
    using Mat33 = Eigen::Matrix<double, 3, 3>;
    using Mat66 = Eigen::Matrix<double, 6, 6>;
    Special(gemmi::Atom* a, const Mat33 &mat_pos, const Mat66 &mat_aniso, int n_mult)
      : Rspec_pos(mat_pos), Rspec_aniso(mat_aniso), n_mult(n_mult), atom(a) {}
    Mat33 Rspec_pos;
    Mat66 Rspec_aniso;
    int n_mult;
    gemmi::Atom* atom;
  };
  struct Stacking {
    Stacking(std::vector<gemmi::Atom*> plane1, std::vector<gemmi::Atom*> plane2) : planes({plane1, plane2}) {}
    double calc(double wstack, bool use_dist, GeomTarget* target, Reporting *reporting) const;
    double dist;
    double sd_dist;
    double angle;
    double sd_angle;
    std::array<std::vector<gemmi::Atom*>, 2> planes;
  };
  struct Vdw {
    Vdw(gemmi::Atom* atom1, gemmi::Atom* atom2) : atoms({atom1, atom2}) {
      if (atoms[0]->serial > atoms[1]->serial)
        std::reverse(atoms.begin(), atoms.end());
    }
    void set_image(const gemmi::UnitCell& cell, gemmi::Asu asu) {
      const gemmi::NearestImage im = cell.find_nearest_image(atoms[0]->pos, atoms[1]->pos, asu);
      sym_idx = im.sym_idx;
      std::copy(std::begin(im.pbc_shift), std::end(im.pbc_shift), std::begin(pbc_shift));
    }
    void set_image(int sym_idx, const gemmi::Fractional &shift) {
      this->sym_idx = sym_idx;
      pbc_shift[0] = std::round(shift.x);
      pbc_shift[1] = std::round(shift.y);
      pbc_shift[2] = std::round(shift.z);
    }
    bool same_asu() const {
      return sym_idx == 0 && pbc_shift[0]==0 && pbc_shift[1]==0 && pbc_shift[2]==0;
    }
    std::tuple<int,int,int,int,int,int> sort_key() const {
      return std::tie(atoms[0]->serial, atoms[1]->serial, sym_idx, pbc_shift[0], pbc_shift[1], pbc_shift[2]);
    }
    double calc(const gemmi::UnitCell& cell, double wvdw, GeomTarget* target, Reporting *reporting) const;
    int type = 0; // 1: vdw, 2: torsion, 3: hbond, 4: metal, 5: dummy-nondummy, 6: dummy-dummy
    double value; // critical distance
    double sigma = 0.;
    int sym_idx = 0;
    std::array<int, 3> pbc_shift = {{0,0,0}};
    std::array<gemmi::Atom*, 2> atoms;
  };
  struct Ncsr {
    Ncsr(const Vdw *vdw1, const Vdw *vdw2, int idx) : pairs({vdw1, vdw2}), idx(idx) {}
    double calc(const gemmi::UnitCell& cell, double wncsr, GeomTarget* target, Reporting *reporting, double, double) const;
    std::array<const Vdw*, 2> pairs;
    double alpha;
    double sigma;
    int idx;
  };
  struct Reporting {
    using bond_reporting_t = std::tuple<const Bond*, const Bond::Value*, double>;
    using angle_reporting_t = std::tuple<const Angle*, const Angle::Value*, double>;
    using torsion_reporting_t = std::tuple<const Torsion*, const Torsion::Value*, double, double>; // delta, tors
    using chiral_reporting_t = std::tuple<const Chirality*, double, double>; // delta, ideal
    using plane_reporting_t = std::tuple<const Plane*, std::vector<double>>;
    using stacking_reporting_t = std::tuple<const Stacking*, double, double, double>; // delta_angle, delta_dist1, delta_dist2
    using vdw_reporting_t = std::tuple<const Vdw*, double>;
    using adp_reporting_t = std::tuple<const gemmi::Atom*, const gemmi::Atom*, int, float, float, float>; // atom1, atom2, type, dist, sigma, delta
    using occ_reporting_t = std::tuple<const gemmi::Atom*, const gemmi::Atom*, int, float, float, float>; // atom1, atom2, type, dist, sigma, delta
    using interval_reporting_t = std::tuple<const Interval*, float, bool>; // delta_dist, lt_dmin
    using ncsr_reporting_t = std::tuple<const Ncsr*, float, float>; // dist1, dist2
    std::vector<bond_reporting_t> bonds;
    std::vector<angle_reporting_t> angles;
    std::vector<torsion_reporting_t> torsions;
    std::vector<chiral_reporting_t> chirs;
    std::vector<plane_reporting_t> planes;
    std::vector<stacking_reporting_t> stackings;
    std::vector<vdw_reporting_t> vdws;
    std::vector<interval_reporting_t> intervals;
    std::vector<ncsr_reporting_t> ncsrs;
    std::vector<adp_reporting_t> adps;
    std::vector<occ_reporting_t> occs;
  };
  Geometry(gemmi::Structure& s, std::shared_ptr<RefineParams> params, const gemmi::EnerLib* ener_lib)
    : st(s), bondindex(s.first_model()), ener_lib(ener_lib), target(params) {}
  void load_topo(const gemmi::Topo& topo);
  void finalize_restraints(); // sort_restraints?
  void setup_nonbonded(bool skip_critical_dist, bool repulse_undefined_angles);
  void setup_ncsr(const NcsList &ncslist);
  bool in_same_plane(const gemmi::Atom *a1, const gemmi::Atom *a2) const {
    return std::binary_search(plane_pairs.begin(), plane_pairs.end(),
                              a1 < a2 ? std::make_pair(a1, a2) : std::make_pair(a2, a1));
  }
  static gemmi::Position apply_transform(const gemmi::UnitCell& cell, int sym_idx, const std::array<int, 3>& pbc_shift, const gemmi::Position &v) {
    gemmi::FTransform ft = sym_idx == 0 ? gemmi::FTransform() : cell.images[sym_idx-1];
    ft.vec += gemmi::Vec3(pbc_shift);
    return gemmi::Position(cell.orth.combine(ft).combine(cell.frac).apply(v));
  }
  static gemmi::Transform get_transform(const gemmi::UnitCell& cell, int sym_idx, const std::array<int, 3>& pbc_shift) {
    gemmi::FTransform ft = sym_idx == 0 ? gemmi::FTransform() : cell.images[sym_idx-1];
    ft.vec += gemmi::Vec3(pbc_shift);
    return cell.orth.combine(ft).combine(cell.frac);
  }

  void setup_target(bool use_occr);
  void clear_target() {
    target.target = 0.;
    std::fill(target.vn.begin(), target.vn.end(), 0.);
    std::fill(target.am.begin(), target.am.end(), 0.);
  }
  double calc(bool use_nucleus, bool check_only, double wbond, double wangle, double wtors,
              double wchir, double wplane, double wstack, double wvdw, double wncs);
  double calc_adp_restraint(bool check_only, double wbskal);
  double calc_occ_constraint(bool check_only, const std::vector<double> &ls, const std::vector<double> & u);
  double calc_occ_restraint(bool check_only, double wocc);
  void calc_jellybody();
  void spec_correction(double alpha=1e-3, bool use_rr=true);
  std::vector<Bond> bonds;
  std::vector<Angle> angles;
  std::vector<Torsion> torsions;
  std::vector<Chirality> chirs;
  std::vector<Plane> planes;
  std::vector<Interval> intervals;
  std::vector<Harmonic> harmonics;
  std::vector<Special> specials;
  std::vector<Stacking> stackings;
  std::vector<Vdw> vdws;
  std::vector<Ncsr> ncsrs;
  gemmi::Structure& st;
  gemmi::BondIndex bondindex;
  std::vector<std::pair<const gemmi::Atom*, const gemmi::Atom*>> plane_pairs;
  const gemmi::EnerLib* ener_lib = nullptr;
  std::map<int, std::string> chemtypes;
  std::map<int, char> hbtypes; // hydrogen bond types that override ener_lib
  Reporting reporting;
  GeomTarget target;

  // vdw parameters
  double vdw_sdi_vdw     = 0.2; // VDWR SIGM VDW val
  double vdw_sdi_torsion = 0.2; // VDWR SIGM TORS val
  double vdw_sdi_hbond   = 0.2; // VDWR SIGM HBON val
  double vdw_sdi_metal   = 0.2; // VDWR SIGM META val
  double hbond_dinc_ad   = -0.3; // VDWR INCR ADHB val
  double hbond_dinc_ah   = 0.1; // VDWR INCR AHHB val
  //double dinc_torsion    = -0.3; // not used? // // VDWR INCR TORS val
  double dinc_torsion_o  = -0.1;
  double dinc_torsion_n  = -0.1;
  double dinc_torsion_c  = -0.15; // VDWR INCR TORS val (copied)
  double dinc_torsion_all= -0.15; // VDWR INCR TORS val (copied)
  double dinc_dummy      = -0.7; // VDWR INCR DUMM val
  double vdw_sdi_dummy   = 0.3; // VDWR SIGM DUMM val
  //double dvdw_cut_min    = 1.75; // no need? // VDWR VDWC val
  //double dvdw_cut_min_x  = 1.75; // used as twice in fast_hessian_tabulation.f // VDWR VDWC val
  double max_vdw_radius = 2.0;

  // angle
  bool angle_von_mises = false;

  // torsion
  bool use_hydr_tors = true;
  std::map<std::string, std::vector<std::string>> link_tors_names;
  std::map<std::string, std::vector<std::string>> mon_tors_names;

  // ADP restraints
  float adpr_max_dist = 4.;
  double adpr_d_power = 4;
  double adpr_exp_fac = 0.011271; //1 ./ (2*4*4*4*std::log(2.));
  bool adpr_long_range = true;
  std::array<float, 8> adpr_kl_sigs = {0.1f, 0.15f, 0.3f, 0.5f, 0.7f, 0.7f, 0.7f, 1.0f};
  std::array<float, 8> adpr_diff_sigs = {5.f, 7.5f, 15.f, 25.f, 35.f, 35.f, 35.f, 50.f};
  int adpr_mode = 0; // 0: diff, 1: KLdiv

  // Occupancy restraints
  double occr_max_dist = 4.;
  bool occr_long_range = true;
  std::array<float, 8> occr_sigs = {0.1f, 0.15f, 0.3f, 0.5f, 0.7f, 0.7f, 0.7f, 1.0f};

  // Jelly body
  float ridge_dmax = 0;
  double ridge_sigma = 0.02;
  bool ridge_symm = false; // inter-symmetry
  bool ridge_exclude_short_dist = true;

  // NCS local
  double ncsr_alpha = -2; // alpha in the robust function
  double ncsr_sigma = 0.05;
  double ncsr_diff_cutoff = 10.0;
  float ncsr_max_dist = 4.2;

  bool use_stack_dist = false;

private:
  void set_vdw_values(Geometry::Vdw &vdw, int d_1_2) const;
}; // struct Geometry

inline void Geometry::load_topo(const gemmi::Topo& topo) {
  auto add = [&](const gemmi::Topo::Rule& rule, gemmi::Asu asu = gemmi::Asu::Same) {
    if (asu != gemmi::Asu::Same && rule.rkind != gemmi::Topo::RKind::Bond) return; // not supported
    if (rule.rkind == gemmi::Topo::RKind::Bond) {
      const gemmi::Topo::Bond& t = topo.bonds[rule.index];
      if (t.restr->esd <= 0) return;
      bonds.emplace_back(t.atoms[0], t.atoms[1]);
      bonds.back().values.emplace_back(t.restr->value, t.restr->esd,
                                       t.restr->value_nucleus, t.restr->esd_nucleus);
      if (asu != gemmi::Asu::Same)
        bonds.back().set_image(st.cell, asu);
    } else if (rule.rkind == gemmi::Topo::RKind::Angle) {
      const gemmi::Topo::Angle& t = topo.angles[rule.index];
      if (t.restr->esd <= 0) return;
      angles.emplace_back(t.atoms[0], t.atoms[1], t.atoms[2]);
      angles.back().values.emplace_back(t.restr->value, t.restr->esd);
      if (asu != gemmi::Asu::Same) {
        // Not implemented yet. Need to identify asu - which atom is in symmetry
        // angles.back().set_images(st.cell, asu1, asu3);
      }
    } else if (rule.rkind == gemmi::Topo::RKind::Torsion) {
      const gemmi::Topo::Torsion& t = topo.torsions[rule.index];
      if (t.restr->esd <= 0) return;
      torsions.emplace_back(t.atoms[0], t.atoms[1], t.atoms[2], t.atoms[3]);
      torsions.back().values.emplace_back(t.restr->value, t.restr->esd, t.restr->period);
      torsions.back().values.back().label = t.restr->label;
    } else if (rule.rkind == gemmi::Topo::RKind::Chirality) {
      const gemmi::Topo::Chirality& t = topo.chirs[rule.index];
      const auto val_sigma = ideal_chiral_abs_volume_sigma(topo, t);
      if (val_sigma.second <= 0 ||
          !std::isfinite(val_sigma.first) || !std::isfinite(val_sigma.second)) return;
      chirs.emplace_back(t.atoms[0], t.atoms[1], t.atoms[2], t.atoms[3]);
      chirs.back().value = val_sigma.first;
      chirs.back().sigma = val_sigma.second;
      chirs.back().sign = t.restr->sign;
    } else if (rule.rkind == gemmi::Topo::RKind::Plane) {
      const gemmi::Topo::Plane& t = topo.planes[rule.index];
      if (t.restr->esd <= 0) return;
      planes.emplace_back(t.atoms);
      planes.back().sigma = t.restr->esd;
      planes.back().label = t.restr->label;
    }
  };

  auto test_hydr_tors = [&](const gemmi::Topo::Torsion &t) {
    return use_hydr_tors && (t.atoms[0]->is_hydrogen() || t.atoms[3]->is_hydrogen());
  };
  auto test_r = [&topo,&test_hydr_tors](const gemmi::Topo::Rule& rule, const std::string& id,
                                        const std::map<std::string, std::vector<std::string>> &tors_names) {
    if (rule.rkind != gemmi::Topo::RKind::Torsion)
      return true;
    const gemmi::Topo::Torsion& t = topo.torsions[rule.index];
    if (test_hydr_tors(t))
      return true;
    const auto it = tors_names.find(id);
    if (it == tors_names.end())
      return false;
    return std::find(it->second.begin(), it->second.end(), t.restr->label) != it->second.end();
  };

  for (const gemmi::Topo::ChainInfo& chain_info : topo.chain_infos)
    for (const gemmi::Topo::ResInfo& ri : chain_info.res_infos) {
      // 1. link related
      for (const gemmi::Topo::Link& prev : ri.prev)
        if (!prev.link_rules.empty())
          for (const gemmi::Topo::Rule& rule : prev.link_rules)
            if (test_r(rule, prev.link_id, link_tors_names))
              add(rule);

      // 2. monomer related
      if (!ri.monomer_rules.empty())
        for (const gemmi::Topo::Rule& rule : ri.monomer_rules) {
          if (test_r(rule, ri.orig_chemcomp->name, mon_tors_names))
            add(rule);
        }

      // collect chem_types
      for (const gemmi::Atom& atom : ri.res->atoms) {
        const gemmi::ChemComp& cc = ri.get_final_chemcomp(atom.altloc);
        auto it = cc.find_atom(atom.name);
        if (it != cc.atoms.end()) {
          const std::string &chem_type = it->is_hydrogen() ? "H" : it->chem_type;
          if (ener_lib && ener_lib->atoms.find(chem_type) == ener_lib->atoms.end())
            throw std::runtime_error("Energy type " + chem_type + " of " + ri.res->name + " not found in ener_lib");
          chemtypes.emplace(atom.serial, chem_type);
        }
      }
    }

  for (const gemmi::Topo::Link& extra : topo.extras)
    for (const gemmi::Topo::Rule& rule : extra.link_rules)
      if (test_r(rule, extra.link_id, link_tors_names))
        add(rule, extra.asu);
}

inline void Geometry::finalize_restraints() {
  for (const auto& b : bonds)
    if (b.type < 2)
      bondindex.add_link(*b.atoms[0], *b.atoms[1], b.same_asu());

  // sort out type 0 or 1 bonds
  // type = 0: replace it
  // type = 1: add it
  // check type 2. remove if type 0 (or 1?) bonds defined

  if (bonds.size() > 1)
    std::stable_sort(bonds.begin(), bonds.end(),
                     [](const Bond& l, const Bond& r) { return l.sort_key() < r.sort_key(); });
  // remove duplicated type 0 bonds
  // remove type 2 bonds if bond/angles defined
  std::vector<size_t> to_remove;
  for (size_t i = 0; i < bonds.size(); ++i) {
    if (bonds[i].type == 2 && bondindex.graph_distance(*bonds[i].atoms[0], *bonds[i].atoms[1], bonds[i].same_asu()) < 3) {
      //std::cout << "remove type 2: " << bonds[i].atoms[0]->name << "-" <<  bonds[i].atoms[1]->name << "\n";
      to_remove.emplace_back(i);
    } else if (i < bonds.size() - 1 && bonds[i].sort_key() == bonds[i+1].sort_key()) {
      if (bonds[i+1].type > 0) // merge. overwrite if next is type 0.
        bonds[i+1].values.insert(bonds[i+1].values.end(), bonds[i].values.begin(), bonds[i].values.end());
      //std::cout << "remove/merge: " << bonds[i].atoms[0]->name << "-" <<  bonds[i].atoms[1]->name << " d="
      //          << bonds[i].atoms[0]->pos.dist(bonds[i].atoms[1]->pos)
      //          << " target0=" << bonds[i].values[0].value << "\n";
      to_remove.emplace_back(i);
    }
  }
  for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it)
    bonds.erase(bonds.begin() + (*it));

  // sort angles
  for (auto& t : angles)
    t.normalize_ideal();
  if (angles.size() > 1) {
    std::stable_sort(angles.begin(), angles.end(),
                     [](const Angle& l, const Angle& r) { return l.sort_key() < r.sort_key(); });
    // remove duplicated angles
    to_remove.clear();
    for (size_t i = 0; i < angles.size() - 1; ++i)
      if (angles[i].sort_key() == angles[i+1].sort_key()) {
        if (angles[i+1].type > 0) // should we always do this?
          angles[i+1].values.insert(angles[i+1].values.end(), angles[i].values.begin(), angles[i].values.end());
        to_remove.emplace_back(i);
      }
    for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it)
      angles.erase(angles.begin() + (*it));
  }

  // sort torsions
  if (torsions.size() > 1) {
    std::stable_sort(torsions.begin(), torsions.end(),
                     [](const Torsion& l, const Torsion& r) { return l.sort_key() < r.sort_key(); });
    // remove duplicated torsions
    to_remove.clear();
    for (size_t i = 0; i < torsions.size() - 1; ++i)
      if (torsions[i].sort_key() == torsions[i+1].sort_key()) {
        if (torsions[i+1].type > 0)
          torsions[i+1].values.insert(torsions[i+1].values.end(), torsions[i].values.begin(), torsions[i].values.end());
        to_remove.emplace_back(i);
      }
    for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it)
      torsions.erase(torsions.begin() + (*it));
  }

  // make plane_pairs
  for (const auto &plane : planes)
    for (int i = 0; i < plane.atoms.size(); ++i)
      for (int j = i+1; j < plane.atoms.size(); ++j)
        if (plane.atoms[i] < plane.atoms[j])
          plane_pairs.emplace_back(plane.atoms[i], plane.atoms[j]);
        else
          plane_pairs.emplace_back(plane.atoms[j], plane.atoms[i]);
  std::sort(plane_pairs.begin(), plane_pairs.end());

  // no care needed for others?
}

inline void Geometry::set_vdw_values(Geometry::Vdw &vdw, int d_1_2) const {
  if (ener_lib == nullptr) gemmi::fail("set ener_lib");
  double vdw_rad[2];
  double ion_rad[2];
  char hb_type[2];
  for (int i = 0; i < 2; ++i) {
    const std::string& chem_type = chemtypes.at(vdw.atoms[i]->serial);
    const auto& libatom = ener_lib->atoms.at(chem_type);
    vdw_rad[i] = std::min(max_vdw_radius,
                          // XXX needs switch. check hydrogen is there?
                          std::isnan(libatom.vdwh_radius) ? libatom.vdw_radius : libatom.vdwh_radius);
    ion_rad[i] = libatom.ion_radius;
    auto it = hbtypes.find(vdw.atoms[i]->serial);
    hb_type[i] = (it == hbtypes.end()) ? libatom.hb_type : it->second;
  }

  // this only happens when the ideal angle is not defined
  if (d_1_2 == 2) {
    vdw.type = 0;
    vdw.value = vdw.atoms[0]->element.covalent_r() + vdw.atoms[1]->element.covalent_r();
    vdw.sigma = vdw_sdi_vdw;
    return;
  }
  // check torsion related atoms XXX what if within ring? we can remove if torsion.period<3?
  if (d_1_2 == 3) { // for hydrogen also??
    double dinc_curr[2];
    for (int i = 0; i < 2; ++i) {
      switch (vdw.atoms[i]->element) {
      case gemmi::El::O: dinc_curr[i] = dinc_torsion_o; break;
      case gemmi::El::N: dinc_curr[i] = dinc_torsion_n; break;
      case gemmi::El::C: dinc_curr[i] = dinc_torsion_c; break;
      default:    dinc_curr[i] = dinc_torsion_all;
      }
    }
    vdw.type = 2;
    vdw.value = vdw_rad[0] + vdw_rad[1] + dinc_curr[0] + dinc_curr[1];
    vdw.sigma = vdw_sdi_torsion;
    return;
  }

  // check hydrogen bond
  if ((hb_type[0] == 'A' && (hb_type[1] == 'D' || hb_type[1] == 'B')) ||
      (hb_type[0] == 'D' && (hb_type[1] == 'A' || hb_type[1] == 'B')) ||
      (hb_type[0] == 'B' && (hb_type[1] == 'A' || hb_type[1] == 'D' || hb_type[1] == 'B'))) {
    vdw.value = vdw_rad[0] + vdw_rad[1] + hbond_dinc_ad;
    vdw.type = 3;
  }
  else if ((hb_type[0] == 'A' && hb_type[1] == 'H') ||
           (hb_type[0] == 'B' && hb_type[1] == 'H')) {
    vdw.value = vdw_rad[0] + hbond_dinc_ah;
    vdw.type = 3;
  }
  else if (hb_type[0] == 'H' && (hb_type[1] == 'A' || hb_type[1] == 'B')) {
    vdw.value = vdw_rad[1] + hbond_dinc_ah;
    vdw.type = 3;
  }
  if (vdw.type == 3) {
    vdw.sigma = vdw_sdi_hbond;
    return;
  }

  // check metal bond?
  if (vdw.atoms[0]->element.is_metal() || vdw.atoms[1]->element.is_metal()) { // XXX should be xor?
    vdw.value = ion_rad[0] + ion_rad[1];
    vdw.type = 4;
    vdw.sigma = vdw_sdi_metal;
    if (!std::isnan(vdw.value))
      return;
  }

  // check dummy XXX we should not depend on atom names?
  bool is_dum_1 = vdw.atoms[0]->name.rfind("DUM", 0) == 0;
  bool is_dum_2 = vdw.atoms[1]->name.rfind("DUM", 0) == 0;
  if ((is_dum_1 && !is_dum_2) || (!is_dum_1 && is_dum_2)) {
    vdw.value = std::max(0.7, vdw_rad[0] + vdw_rad[1] + dinc_dummy);
    vdw.type = 5;
    vdw.sigma = vdw_sdi_dummy;
    return;
  }
  if (is_dum_1 && is_dum_2) {
    vdw.value = vdw_rad[0] + vdw_rad[1];
    vdw.type = 6;
    vdw.sigma = vdw_sdi_dummy;
    return;
  }

  // otherwise
  vdw.value = vdw_rad[0] + vdw_rad[1];
  vdw.type = 1;
  vdw.sigma = vdw_sdi_vdw;
}

// sets up nonbonded interactions for vdwr, ADP restraints, and jellybody
// group_idxes from occupancy group definition. 0 means not belongig to any group
inline void Geometry::setup_nonbonded(bool skip_critical_dist,
                                      bool repulse_undefined_angles) {
  if (!skip_critical_dist && ener_lib == nullptr) gemmi::fail("set ener_lib");
  // set hbtypes for hydrogen
  if (!skip_critical_dist && hbtypes.empty()) {
    for (auto& b : bonds)
      if (b.atoms[0]->is_hydrogen() != b.atoms[1]->is_hydrogen()) {
        int p = b.atoms[0]->is_hydrogen() ? 1 : 0; // parent
        int h = b.atoms[0]->is_hydrogen() ? 0 : 1; // hydrogen
        const std::string& p_chem_type = chemtypes.at(b.atoms[p]->serial);
        const char p_hb_type = ener_lib->atoms.at(p_chem_type).hb_type;
        hbtypes.emplace(b.atoms[h]->serial, p_hb_type == 'D' || p_hb_type == 'B' ? 'H' : 'N');
      }
  }

  auto angle_defined = [&](const gemmi::Atom *a1, const gemmi::Atom *a2) {
    const auto key = a1->serial < a2->serial
      ? std::make_pair(a1->serial, a2->serial)
      : std::make_pair(a2->serial, a1->serial);
    // ignore symmetry and pbc
    struct cmp {
      bool operator()(const Angle &lhs, decltype(key) &rhs) const {
        return std::make_pair(lhs.atoms[0]->serial, lhs.atoms[2]->serial) < rhs;
      }
      bool operator()(decltype(key) &lhs, const Angle &rhs) const {
        return lhs < std::make_pair(rhs.atoms[0]->serial, rhs.atoms[2]->serial);
      }
    };
    return std::binary_search(angles.begin(), angles.end(), key, cmp());
  };
  auto test_skip_vdwr = [&](const gemmi::Atom *a1, const gemmi::Atom *a2, bool same_res, bool sym_related) {
    const bool sum_q_le1 = a1->occ + a2->occ < 1.0001;

    // 1. if symmetry related, just check sum of occ
    if (sym_related)
      return sum_q_le1;

    // 2. check occ group
    const int idx_1 = a1->serial - 1, idx_2 = a2->serial - 1;
    int gr_1 = -1, gr_2 = -1;
    for (int i = 0; i < target.params->occ_groups.size(); ++i) {
      const auto &gr = target.params->occ_groups[i];
      if (std::find(gr.begin(), gr.end(), idx_1) != gr.end())
        gr_1 = i;
      if (std::find(gr.begin(), gr.end(), idx_2) != gr.end())
        gr_2 = i;
    }
    // if defined, use it for decision
    if (gr_1 >=0 || gr_2 >= 0) {
      if (gr_1 >= 0 && gr_2 >= 0) {
        if (gr_1 == gr_2)
          return false; // do not skip if in the same group
        for (const auto &gc : target.params->occ_group_constraints)
          if (std::find(gc.second.begin(), gc.second.end(), gr_1) != gc.second.end() &&
              std::find(gc.second.begin(), gc.second.end(), gr_2) != gc.second.end())
            return true; // skip if belong to the constrained group
      }
      return false; // do not skip otherwise
    }

    // 3. check altlocs
    const bool same_conf = gemmi::is_same_conformer(a1->altloc, a2->altloc);
    if (same_res)
      return !same_conf; // skip if not same conf
    // if (a1->altloc == '\0' && a2->altloc == '\0') // not sure this is a good idea..
    //   return sum_q_le1; // follow sum of occ if both are non-alts
    return !same_conf;
  };

  vdws.clear();

  // Reference: Refmac vdw_and_contacts.f
  gemmi::NeighborSearch ns(st.first_model(), st.cell, 4);
  ns.populate();
  const float max_vdwr = 2.98f; // max from ener_lib, Cs.
  // we can reduce memory usage when ridge/adpr/ncsr not used - these max_dist parameters should be set zero
  const float max_other_sq = gemmi::sq(std::max(ridge_dmax, std::max(adpr_max_dist, ncsr_max_dist)));
  const float max_dist_sq = std::max(max_other_sq, gemmi::sq(max_vdwr * 2));
  // XXX Refmac uses intervals for distances as well? vdw_and_contacts.f remove_bonds_and_angles()
  // ref: gemmi/contact.hpp ContactSearch::for_each_contact
  for (int n_ch = 0; n_ch != (int) ns.model->chains.size(); ++n_ch) {
    gemmi::Chain &chain = ns.model->chains[n_ch];
    for (int n_res = 0; n_res != (int) chain.residues.size(); ++n_res) {
      gemmi::Residue &res = chain.residues[n_res];
      for (int n_atom = 0; n_atom != (int) res.atoms.size(); ++n_atom) {
        gemmi::Atom &atom = res.atoms[n_atom];
        ns.for_each_cell(atom.pos,
                         [&](std::vector<gemmi::NeighborSearch::Mark>& marks, const gemmi::Fractional &fr) {
                           const gemmi::Position &p = ns.use_pbc ? ns.grid.unit_cell.orthogonalize(fr) : atom.pos;
                           for (gemmi::NeighborSearch::Mark& m : marks) {
                             const gemmi::CRA cra2 = m.to_cra(*ns.model);
                             // avoid reporting connections twice
                             if (m.chain_idx < n_ch || (m.chain_idx == n_ch &&
                                                        (m.residue_idx < n_res || (m.residue_idx == n_res &&
                                                                                   m.atom_idx < n_atom))))
                               continue;
                             const double dist_sq = m.pos.dist_sq(p);
                             if (dist_sq < max_dist_sq) {
                               // do not include itself; special positions should have been sorted beforehand
                               if (m.chain_idx == n_ch && m.residue_idx == n_res &&
                                   m.atom_idx == n_atom && dist_sq < 0.1*0.1)
                                 continue;
                               if (test_skip_vdwr(&atom, cra2.atom, &res == cra2.residue, m.image_idx != 0))
                                 continue;
                               vdws.emplace_back(&atom, cra2.atom);
                               { // find pbc shift
                                 auto fpos = ns.grid.unit_cell.fractionalize(cra2.atom->pos);
                                 ns.grid.unit_cell.apply_transform(fpos, m.image_idx, false);
                                 const auto dvec = m.pos - p - ns.grid.unit_cell.orthogonalize(fpos) + atom.pos;
                                 vdws.back().set_image(m.image_idx, ns.grid.unit_cell.fractionalize(dvec));
                               }
                               int d_1_2 = bondindex.graph_distance(atom, *cra2.atom, vdws.back().same_asu());
                               if ((d_1_2 == 3 && in_same_plane(&atom, cra2.atom)) ||
                                   ((repulse_undefined_angles && d_1_2 == 2) ? angle_defined(&atom, cra2.atom) : d_1_2 < 3)) {
                                 vdws.pop_back();
                                 continue;
                               }
                               if (!skip_critical_dist) {
                                 set_vdw_values(vdws.back(), d_1_2);
                                 assert(!std::isnan(vdws.back().value) && vdws.back().value > 0);
                                 if (!vdws.back().same_asu())
                                   vdws.back().type += 6;
                               }
                               // don't include if too far. x1.5 is too large?
                               if (skip_critical_dist ? (dist_sq > max_other_sq)
                                   : (dist_sq > std::min((double)max_other_sq, gemmi::sq(vdws.back().value * 1.5))))
                                 vdws.pop_back();
                             }
                           }
                         }, ns.sufficient_k(std::sqrt(max_dist_sq)));
      }
    }
  }
}

inline void Geometry::setup_ncsr(const NcsList &ncslist) {
  ncsrs.clear();
  // vdws should be sorted.
  std::sort(vdws.begin(), vdws.end(),
            [](const Vdw &l, const Vdw &r) { return l.sort_key() < r.sort_key(); });
  struct CompVdwAndPair {
    using pair_t = std::pair<const gemmi::Atom*, const gemmi::Atom*>;
    static pair_t sorted_pair(const gemmi::Atom* a1, const gemmi::Atom* a2) {
      return a1->serial < a2->serial ? std::make_pair(a1, a2) : std::make_pair(a2, a1);
    }
    std::pair<int,int> make_key(const pair_t &p) const {
      return std::make_pair(p.first->serial, p.second->serial);
    }
    bool operator()(const Vdw &lhs, const pair_t &rhs) const {
      return make_key(sorted_pair(lhs.atoms[0], lhs.atoms[1])) < make_key(rhs);
    }
    bool operator()(const pair_t &lhs, const Vdw &rhs) const {
      return make_key(lhs) < make_key(sorted_pair(rhs.atoms[0], rhs.atoms[1]));
    }
  };
  for (const auto &vdw : vdws) {
    if (vdw.atoms[0]->is_hydrogen() || vdw.atoms[1]->is_hydrogen())
      continue;
    for (int i = 0; i < ncslist.all_pairs.size(); ++i) {
      auto it1 = ncslist.all_pairs[i].find(vdw.atoms[0]);
      if (it1 == ncslist.all_pairs[i].end()) continue;
      auto it2 = ncslist.all_pairs[i].find(vdw.atoms[1]);
      if (it2 == ncslist.all_pairs[i].end()) continue;
      const auto pair = CompVdwAndPair::sorted_pair(it1->second, it2->second);
      auto p = std::equal_range(vdws.begin(), vdws.end(), pair, CompVdwAndPair());
      if (p.first != p.second) {
        ncsrs.emplace_back(&vdw, &(*p.first), i);
        ncsrs.back().alpha = ncsr_alpha;
        ncsrs.back().sigma = ncsr_sigma;
      }
    }
  }
}

inline void Geometry::setup_target(bool use_occr) {
  std::map<std::pair<int,int>, int> all_pairs;
  auto add = [&](gemmi::Atom *a1, gemmi::Atom *a2, int n) {
    // should be called from smaller n so that smallest restraint kind will be kept
    const int i1 = a1->serial - 1;
    const int i2 = a2->serial - 1;
    if (i1 == i2) return;
    if (target.params->is_atom_refined(i1) || target.params->is_atom_refined(i2))
      all_pairs.emplace(std::minmax(i1, i2), n);
  };
  for (const auto &t : bonds)
    add(t.atoms[0], t.atoms[1], 1);

  for (const auto &t : angles)
    for (int i = 0; i < 2; ++i)
      for (int j = i+1; j < 3; ++j)
        add(t.atoms[i], t.atoms[j], 2);

  for (const auto &t : torsions)
    for (int i = 0; i < 3; ++i)
      for (int j = i+1; j < 4; ++j)
        add(t.atoms[i], t.atoms[j], 3);

  for (const auto &t : chirs)
    for (int i = 0; i < 3; ++i)
      for (int j = i+1; j < 4; ++j)
        add(t.atoms[i], t.atoms[j], 4);

  for (const auto &t : planes)
    for (size_t i = 1; i < t.atoms.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        add(t.atoms[i], t.atoms[j], 5);

  for (const auto &t : vdws)
    add(t.atoms[0], t.atoms[1], 6);

  for (const auto &t : stackings) {
    for (size_t i = 0; i < 2; ++i)
      for (size_t j = 1; j < t.planes[i].size(); ++j)
        for (size_t k = 0; k < j; ++k)
          add(t.planes[i][j], t.planes[i][k], 8);

    for (size_t j = 0; j < t.planes[0].size(); ++j)
      for (size_t k = 0; k < t.planes[1].size(); ++k)
        add(t.planes[0][j], t.planes[1][k], 8);
  }

  for (const auto &t : intervals)
    add(t.atoms[0], t.atoms[1], 9);

  for (const auto &t : ncsrs)
    for (const auto &a1 : t.pairs[0]->atoms)
      for (const auto &a2 : t.pairs[1]->atoms)
        add(a1, a2, 10);

  for (int i = 0; i < target.params->occ_group_constraints.size(); ++i) {
    const auto &group_idxes = target.params->occ_group_constraints[i].second;
    for (size_t j = 0; j < group_idxes.size(); ++j) {
      for (int ia1 : target.params->occ_groups[group_idxes[j]])
        if (target.params->is_atom_refined(ia1, RefineParams::Type::Q)) {
          for (size_t k = j + 1; k < group_idxes.size(); ++k) {
            for (int ia2 : target.params->occ_groups[group_idxes[k]])
              if (target.params->is_atom_refined(ia2, RefineParams::Type::Q)) {
                add(target.params->atoms[ia1], target.params->atoms[ia2], 11);
                break;
              }
          }
          break;
        }
    }
  }

  // sort_and_compress_distances
  target.pairs.clear();
  target.pairs_kind.clear();
  target.pairs.reserve(all_pairs.size());
  target.pairs_kind.reserve(all_pairs.size());
  for (const auto &p : all_pairs) {
    target.pairs.emplace_back(p.first.first, p.first.second);
    target.pairs_kind.push_back(p.second);
  }
  target.setup(use_occr);
}

inline double Geometry::calc(bool use_nucleus, bool check_only,
                             double wbond, double wangle, double wtors,
                             double wchir, double wplane, double wstack,
                             double wvdw, double wncs) {
  if (check_only)
    reporting = {}; // also deletes adp. is it ok?
  else
    assert(target.params->is_refined(RefineParams::Type::X)); // otherwise vector and matrix not ready

  auto target_ptr = check_only ? nullptr : &target;
  auto rep_ptr = check_only ? &reporting : nullptr;
  double ret = 0.;

  auto has_selected = [&](const auto &atoms) {
    for (const auto &a : atoms)
      if (target.params->is_atom_refined(a->serial - 1, RefineParams::Type::X))
        return true;
    return false;
  };

  for (const auto &t : bonds)
    if (has_selected(t.atoms))
      ret += t.calc(st.cell, use_nucleus, wbond, target_ptr, rep_ptr);
  for (const auto &t : angles)
    if (has_selected(t.atoms))
      ret += t.calc(st.cell, wangle, angle_von_mises, target_ptr, rep_ptr);
  for (const auto &t : torsions)
    if (has_selected(t.atoms))
      ret += t.calc(wtors, target_ptr, rep_ptr);
  for (const auto &t : chirs)
    if (has_selected(t.atoms))
      ret += t.calc(wchir, target_ptr, rep_ptr);
  for (const auto &t : planes)
    if (has_selected(t.atoms))
      ret += t.calc(wplane, target_ptr, rep_ptr);
  for (const auto &t : harmonics)
    t.calc(target_ptr);
  for (const auto &t : stackings)
    if (has_selected(t.planes[0]) || has_selected(t.planes[1]))
      ret += t.calc(wstack, use_stack_dist, target_ptr, rep_ptr);
  for (const auto &t : vdws)
    if (has_selected(t.atoms))
      ret += t.calc(st.cell, wvdw, target_ptr, rep_ptr);
  for (const auto &t : intervals)
    ret += t.calc(st.cell, wbond, target_ptr, rep_ptr);
  for (const auto &t : ncsrs)
    if (has_selected(t.pairs[0]->atoms) || has_selected(t.pairs[1]->atoms))
      ret += t.calc(st.cell, wncs, target_ptr, rep_ptr, ncsr_diff_cutoff, ncsr_max_dist);
  if (!check_only && ridge_dmax > 0)
    calc_jellybody(); // no contribution to target

  if (std::isnan(ret))
    gemmi::fail("geom became NaN");
  return ret;
}

inline double Geometry::calc_adp_restraint(bool check_only, double wbskal) {
  if (wbskal <= 0) return 0.;
  if (!check_only)
    assert(target.params->is_refined(RefineParams::Type::B));
  reporting.adps.clear();
  double ret = 0.;
  // TODO this misses self-pairs
  const int adp_mode = target.params->is_refined(RefineParams::Type::B) ? (target.params->aniso ? 2 : 1) : 0;
  const size_t pairs_size = target.pairs.size();
  for (int i = 0; i < pairs_size; ++i) {
    const int ia1 = target.pairs[i].first;
    const int ia2 = target.pairs[i].second;
    const int pair_kind = target.pairs_kind[i];
    const int pos1 = target.params->get_pos_vec_geom(ia1, RefineParams::Type::B);
    const int pos2 = target.params->get_pos_vec_geom(ia2, RefineParams::Type::B);
    if (pos1 < 0 && pos2 < 0) continue; // both atoms fixed
    if (!adpr_long_range && pair_kind > 4) continue;
    if (pair_kind > 9) continue; // don't use adpr for ncsr
    const gemmi::Atom* atom1 = target.params->atoms[ia1];
    const gemmi::Atom* atom2 = target.params->atoms[ia2];
    // calculate minimum distance - expensive?
    const gemmi::NearestImage im = st.cell.find_nearest_image(atom1->pos, atom2->pos, gemmi::Asu::Any);
    const double dsq = im.dist_sq;
    if (dsq > gemmi::sq(adpr_max_dist)) continue;
    double w = 0;
    const int apos1 = target.params->get_pos_mat_geom(ia1, RefineParams::Type::B);
    const int apos2 = target.params->get_pos_mat_geom(ia2, RefineParams::Type::B);
    if (adpr_mode == 0) {
      const float sig = adpr_diff_sigs.at(pair_kind-1);
      const bool bonded = pair_kind < 3; // bond and angle related
      w = gemmi::sq(wbskal / sig) * (bonded ? 1 : std::exp(-std::pow(dsq, 0.5 * adpr_d_power) * adpr_exp_fac));
    } else {
      const float sig = adpr_kl_sigs.at(pair_kind-1);
      w = gemmi::sq(wbskal / sig) / (std::max(4., dsq) / 4.);
    };
    if (adp_mode == 2) w /= 3;

    if (adp_mode == 1) {
      const double b_diff = atom1->b_iso - atom2->b_iso;
      double delta = 0;
      if (adpr_mode == 0)
        delta = b_diff;
      else // KL divergence
        delta = b_diff / std::sqrt(atom1->b_iso * atom2->b_iso);
      const double f = 0.5 * w * gemmi::sq(delta);
      ret += f;
      if (!check_only) {
        target.target += f;
        double df1 = 0, df2 = 0;
        if (adpr_mode == 0) {
          df1 = 1.;
          df2 = -1.;
        } else { // KL divergence
          df1 =  (std::sqrt(atom2->b_iso) / std::pow(atom1->b_iso, 1.5) + 1. / std::sqrt(atom1->b_iso * atom2->b_iso)) * 0.5;
          df2 = -(std::sqrt(atom1->b_iso) / std::pow(atom2->b_iso, 1.5) + 1. / std::sqrt(atom1->b_iso * atom2->b_iso)) * 0.5;
        }
        // gradient and diagonal
        if (pos1 >= 0) {
          target.vn[pos1] += w * delta * df1;
          target.am[apos1] += w * gemmi::sq(df1);
        }
        if (pos2 >= 0) {
          target.vn[pos2] += w * delta * df2;
          target.am[apos2] += w * gemmi::sq(df2);
        }
        // non-diagonal
        if (pos1 >= 0 && pos2 >= 0) {
          auto mp = target.find_restraint(ia1, ia2, RefineParams::Type::B);
          target.am[mp.ipos] += w * df1 * df2;
        }
      } else {
        if (!atom1->is_hydrogen() && !atom2->is_hydrogen()) {
          double report_sigma = wbskal / std::sqrt(w);
          if (adpr_mode == 1) report_sigma *= std::sqrt(atom1->b_iso * atom2->b_iso);
          // atom1, atom2, type, dist, sigma, delta
          reporting.adps.emplace_back(atom1, atom2, pair_kind, std::sqrt(dsq),
                                      report_sigma, b_diff);
        }
      }
    } else if (adp_mode == 2) { // Aniso
      const gemmi::Transform tr = get_transform(st.cell, im.sym_idx, {0,0,0}); // shift does not matter
      const Eigen::Matrix<double,6,6> R = mat33_as66(tr.mat);
      auto amat_ctor = [](const gemmi::Atom* a) {
        if (a->aniso.nonzero())
          return Eigen::Matrix<double,6,1>(a->aniso.scaled(gemmi::u_to_b()).elements_pdb().data()); // safe?
        else
          return Eigen::Matrix<double,6,1>({a->b_iso, a->b_iso, a->b_iso, 0., 0., 0.});
      };
      const Eigen::Matrix<double,6,1> a1 = amat_ctor(atom1);
      Eigen::Matrix<double,6,1> a2 = amat_ctor(atom2);
      a2 = R * a2;
      const auto a_diff = a1 - a2;
      double f = 0;
      Eigen::Matrix<double,6,1> der1, der2;
      Eigen::Matrix<double,6,6> am11, am22, am12; // diagonal and non-diagonal blocks
      double B1_B2 = 0;
      if (adpr_mode == 0) {
        f = 0.5 * w * (a_diff.transpose() * a_diff).value();
        if (!check_only) {
          der1 = w * a_diff;
          der2 = R.transpose() * (-der1);
          am11 = w * Eigen::Matrix<double,6,6>::Identity();
          am22 = R.transpose() * am11 * R;
          am12 = R.transpose() * (-am11);
        }
      } else { // KL divergence (not exactly)
        const double B1 = a1(Eigen::seq(0,2)).sum() / 3;
        const double B2 = a2(Eigen::seq(0,2)).sum() / 3;
        B1_B2 = B1 * B2;
        const Eigen::Matrix<double,6,1> B = {1./3, 1./3, 1./3, 0, 0, 0};
        const Eigen::Matrix<double,6,6> B_B = B * B.transpose();
        const Eigen::DiagonalMatrix<double, 6> A(2,2,2,4,4,4);
        f = 0.5 * w * (a_diff.transpose() * (A * 0.5) * a_diff).value() / B1_B2;
        if (!check_only) {
          const auto v1 = A * a_diff / B1_B2;
          const auto v2 = (a_diff.transpose() * (A * 0.5) * a_diff).value() / B1_B2 * B;
          der1 = 0.5 * w * (v1 - v2 / B1);
          der2 = 0.5 * w * R.transpose() * (-v1 - v2 / B2);
          const Eigen::Matrix<double,6,6> tmp = 0.5 * w / B1_B2 * A;
          am11 = tmp + 2 * f / gemmi::sq(B1) * B_B;
          am22 = R.transpose() * (tmp + 2 * f / gemmi::sq(B2) * B_B) * R;
          am12 = R.transpose() * (-tmp + f / B1_B2 * B_B);
        }
      }
      ret += f;
      if (!check_only) {
        target.target += f;
        for (int j = 0; j < 6; ++j) {
          if (pos1 >= 0) target.vn[pos1 + j] += der1[j];
          if (pos2 >= 0) target.vn[pos2 + j] += der2[j];
        }
        // diagonal blocks (6 x 6 symmetric)
        for (int j = 0; j < 6; ++j) { // diagonals
          if (pos1 >= 0) target.am[apos1 + j] += am11(j, j);
          if (pos2 >= 0) target.am[apos2 + j] += am22(j, j);
        }
        for (int j = 0, l = 6; j < 6; ++j) // non-diagonals
          for (int k = j + 1; k < 6; ++k, ++l) {
            if (pos1 >= 0) target.am[apos1 + l] += am11(j, k);
            if (pos2 >= 0) target.am[apos2 + l] += am22(j, k);
          }
        // non-diagonal block (6 x 6)
        if (pos1 >= 0 && pos2 >= 0) {
          auto mp = target.find_restraint(ia1, ia2, RefineParams::Type::B);
          for (int j = 0, l = 0; j < 6; ++j)
            for (int k = 0; k < 6; ++k, ++l)
              target.am[mp.ipos + l] += am12(k, j);
        }
      } else {
        if (!atom1->is_hydrogen() && !atom2->is_hydrogen()) {
          double report_sigma = wbskal / std::sqrt(w);
          if (adpr_mode == 1) report_sigma *= std::sqrt(B1_B2);
          // atom1, atom2, type, dist, sigma, delta
          reporting.adps.emplace_back(atom1, atom2, pair_kind, std::sqrt(dsq),
                                      report_sigma, a_diff.norm());
        }
      }
    }
  }
  return ret;
}
inline double Geometry::calc_occ_constraint(bool check_only, const std::vector<double> &ls, const std::vector<double> &u) {
  if (target.params->occ_group_constraints.size() != ls.size())
    gemmi::fail("calc_occ_constraint: size mismatch");
  double ret = 0.;
  const std::vector<double> consts = target.params->occ_constraints();
  for (int i = 0; i < target.params->occ_group_constraints.size(); ++i) {
    const auto &group_idxes = target.params->occ_group_constraints[i].second;
    double sum_occ = 0.;
    const double c = consts[i];
    if (c == 0) // otherwise gradient will be affected - should be?
      continue;
    ret += 0.5 * u[i] * gemmi::sq(c) - ls[i] * c;
    if (!check_only) {
      for (size_t j = 0; j < group_idxes.size(); ++j) {
        for (int ia : target.params->occ_groups[group_idxes[j]])
          if (target.params->is_atom_refined(ia, RefineParams::Type::Q)) {
            const int vpos = target.params->get_pos_vec(ia, RefineParams::Type::Q);
            const int apos = target.params->get_pos_mat_geom(ia, RefineParams::Type::Q);
            if (vpos >= 0)
              target.vn[vpos] += u[i] * c - ls[i];
            if (apos >= 0)
              target.am[apos] += u[i];

            // non-diagonal
            for (size_t k = j + 1; k < group_idxes.size(); ++k) {
              for (int ia2 : target.params->occ_groups[group_idxes[k]])
                if (target.params->is_atom_refined(ia2, RefineParams::Type::Q)) {
                  const int vpos2 = target.params->get_pos_vec_geom(ia2, RefineParams::Type::Q);
                  if (vpos >= 0 && vpos2 >= 0) {
                    auto mp = target.find_restraint(ia, ia2, RefineParams::Type::Q);
                    target.am[mp.ipos] += u[i];
                  }
                  break;
                }
            }
            break; // parameter is common
          }
      }
    }
  }
  return ret;
}
inline double Geometry::calc_occ_restraint(bool check_only, double wocc) {
  if (wocc <= 0) return 0.;
  if (!check_only)
    assert(target.params->is_refined(RefineParams::Type::Q) && target.use_occr);
  reporting.occs.clear();

  double ret = 0.;
  const size_t pairs_size = target.pairs.size();
  for (int i = 0; i < pairs_size; ++i) {
    const int ia1 = target.pairs[i].first;
    const int ia2 = target.pairs[i].second;
    const int pair_kind = target.pairs_kind[i];
    const int pos1 = target.params->get_pos_vec_geom(ia1, RefineParams::Type::Q);
    const int pos2 = target.params->get_pos_vec_geom(ia2, RefineParams::Type::Q);
    if (pos1 < 0 && pos2 < 0) continue; // both atoms fixed
    if (!occr_long_range && pair_kind > 4) continue;
    if (pair_kind > 9) continue; // don't use occr for ncsr
    const gemmi::Atom* atom1 = target.params->atoms[ia1];
    const gemmi::Atom* atom2 = target.params->atoms[ia2];
    // calculate minimum distance - expensive?
    const gemmi::NearestImage im = st.cell.find_nearest_image(atom1->pos, atom2->pos, gemmi::Asu::Any);
    const double dsq = im.dist_sq;
    if (dsq > gemmi::sq(occr_max_dist)) continue;
    const float sig = occr_sigs.at(pair_kind-1);
    const bool bonded = pair_kind < 3; // bond and angle related
    const double w = gemmi::sq(wocc / sig) / (std::max(4., dsq) / 4.);
    const double delta = atom1->occ - atom2->occ;
    const double f = 0.5 * w * gemmi::sq(delta);
    ret += f;
    if (!check_only) {
      target.target += f;
      const int apos1 = target.params->get_pos_mat_geom(ia1, RefineParams::Type::Q);
      const int apos2 = target.params->get_pos_mat_geom(ia2, RefineParams::Type::Q);
      // gradient and diagonal
      if (ia1 >= 0) {
        target.vn[pos1] += w * delta;
        target.am[apos1] += w;
      }
      if (ia2 >= 0) {
        target.vn[pos2] += -w * delta;
        target.am[apos2] += w;
      }
      // non-diagonal
      if (pos1 >= 0 && pos2 >= 0) {
        auto mp = target.find_restraint(ia1, ia2, RefineParams::Type::Q);
        target.am[mp.ipos] += -w;
      }
    } else {
      if (!atom1->is_hydrogen() && !atom2->is_hydrogen()) {
        const double report_sigma = wocc / std::sqrt(w);
        // atom1, atom2, type, dist, sigma, delta
        reporting.occs.emplace_back(atom1, atom2, pair_kind, std::sqrt(dsq),
                                    report_sigma, delta);
      }
    }
  }
  return ret;
}

inline void Geometry::calc_jellybody() {
  if (ridge_sigma <= 0) return;
  const double weight = 1 / (ridge_sigma * ridge_sigma);
  // TODO main chain / side chain check?
  // TODO B value filter?
  // TODO intra-chain only, residue gap filter

  for (const auto &t : vdws) {
    if (!ridge_symm && !t.same_asu()) continue;
    const gemmi::Atom& atom1 = *t.atoms[0];
    const gemmi::Atom& atom2 = *t.atoms[1];
    if (atom1.is_hydrogen() || atom2.is_hydrogen()) continue;
    const int pos1 = target.params->get_pos_vec_geom(atom1.serial-1, RefineParams::Type::X);
    const int pos2 = target.params->get_pos_vec_geom(atom2.serial-1, RefineParams::Type::X);
    if (pos1 < 0 && pos2 < 0) continue; // both fixed
    const gemmi::Transform tr = get_transform(st.cell, t.sym_idx, t.pbc_shift);
    const gemmi::Position& x1 = atom1.pos;
    const gemmi::Position& x2 = t.same_asu() ? atom2.pos : gemmi::Position(tr.apply(atom2.pos));
    const double b = x1.dist(x2);
    if (b > ridge_dmax) continue;
    if (ridge_exclude_short_dist && b < std::max(2., t.value * 0.95)) continue;
    const gemmi::Position dbdx1 = (x1 - x2) / std::max(b, 0.02);
    const gemmi::Position dbdx2 = t.same_asu() ? -dbdx1 : gemmi::Position(tr.mat.transpose().multiply(-dbdx1));

    const int apos1 = target.params->get_pos_mat_geom(atom1.serial-1, RefineParams::Type::X);
    const int apos2 = target.params->get_pos_mat_geom(atom2.serial-1, RefineParams::Type::X);
    if (pos1 >= 0) target.incr_am_diag(apos1, weight, dbdx1);
    if (pos2 >= 0) target.incr_am_diag(apos2, weight, dbdx2);
    if (pos1 >=0 && pos2 >= 0) {
      if (pos1 != pos2) {
        auto mp = target.find_restraint(atom1.serial-1, atom2.serial-1);
        if (mp.imode == 0)
          target.incr_am_ndiag(mp.ipos, weight, dbdx1, dbdx2);
        else
          target.incr_am_ndiag(mp.ipos, weight, dbdx2, dbdx1);
      } else
        target.incr_am_diag12(apos1, weight, dbdx1, dbdx2);
    }
  }
}

inline double Geometry::Bond::calc(const gemmi::UnitCell& cell, bool use_nucleus, double wdskal,
                                   GeomTarget* target, Reporting *reporting) const {
  assert(!values.empty());
  const gemmi::Atom* atom1 = atoms[0];
  const gemmi::Atom* atom2 = atoms[1];
  const gemmi::Transform tr = get_transform(cell, sym_idx, pbc_shift);
  const gemmi::Position& x1 = atom1->pos;
  const gemmi::Position& x2 = same_asu() ? atom2->pos : gemmi::Position(tr.apply(atom2->pos));
  const double b = x1.dist(x2);
  auto closest = find_closest_value(b, use_nucleus);
  const double ideal = use_nucleus ? closest->value_nucleus : closest->value;
  const double db = b - ideal;
  const double sigma = (use_nucleus ? closest->sigma_nucleus : closest->sigma);
  const double weight = wdskal / sigma;
  const double y = db * weight;
  Barron2019 robustf(type < 2 ? 2. : alpha, y);

  // note that second derivative is not exact in some alpha
  if (target != nullptr) {
    const gemmi::Position dydx1 = weight * (x1 - x2) / std::max(b, 0.02);
    const gemmi::Position dydx2 = same_asu() ? -dydx1 : gemmi::Position(tr.mat.transpose().multiply(-dydx1));
    const int ia1 = atom1->serial - 1;
    const int ia2 = atom2->serial - 1;
    const int pos1 = target->params->get_pos_vec_geom(ia1, RefineParams::Type::X);
    const int pos2 = target->params->get_pos_vec_geom(ia2, RefineParams::Type::X);
    const int apos1 = target->params->get_pos_mat_geom(ia1, RefineParams::Type::X);
    const int apos2 = target->params->get_pos_mat_geom(ia2, RefineParams::Type::X);
    if (pos1 >= 0) {
      target->incr_vn(pos1, robustf.dfdy, dydx1);
      target->incr_am_diag(apos1, robustf.d2fdy, dydx1);
    }
    if (pos2 >= 0) {
      target->incr_vn(pos2, robustf.dfdy, dydx2);
      target->incr_am_diag(apos2, robustf.d2fdy, dydx2);
    }
    if (pos1 >= 0 && pos2 >= 0) {
      if (pos1 != pos2) {
        auto mp = target->find_restraint(ia1, ia2);
        if (mp.imode == 0)
          target->incr_am_ndiag(mp.ipos, robustf.d2fdy, dydx1, dydx2);
        else
          target->incr_am_ndiag(mp.ipos, robustf.d2fdy, dydx2, dydx1);
      } else
        target->incr_am_diag12(apos1, robustf.d2fdy, dydx1, dydx2);
    }
    target->target += robustf.f;
  }
  if (reporting != nullptr)
    reporting->bonds.emplace_back(this, closest, db);
  return robustf.f;
}

inline double Geometry::Angle::calc(const gemmi::UnitCell& cell, double waskal, bool von_mises,
                                    GeomTarget* target, Reporting *reporting) const {
  // target functions:
  //  when ideal close to 180: 0.5 * w * h^T h = w * (1 + cosa) where h = v1/|v1| + v2/|v2|
  //  if von_mises: w * (1 - cos(a - a0))
  //  otherwise: 0.5 * w * (a - a0)**2
  const gemmi::Transform tr1 = get_transform(cell, sym_idx_1, pbc_shift_1);
  const gemmi::Transform tr2 = get_transform(cell, sym_idx_2, pbc_shift_2);
  const gemmi::Position& x1 = same_asu(0) ? atoms[0]->pos : gemmi::Position(tr1.apply(atoms[0]->pos));
  const gemmi::Position& x2 = atoms[1]->pos;
  const gemmi::Position& x3 = same_asu(2) ? atoms[2]->pos : gemmi::Position(tr2.apply(atoms[2]->pos));
  const gemmi::Position v1 = x2 - x1;
  const gemmi::Position v2 = x2 - x3;
  const double v1n = std::max(v1.length(), 0.02);
  const double v2n = std::max(v2.length(), 0.02);
  const double v12 = v1.dot(v2);
  const double cosa = std::max(-1., std::min(1., v12 / v1n / v2n));
  const double sina = std::min(1., std::max(std::sqrt(1 - cosa * cosa), 0.001));
  const double a_rad = std::acos(cosa);
  const double a = gemmi::deg(a_rad);
  auto closest = find_closest_value(a);
  const double da = a - closest->value;
  const double a0_rad = gemmi::rad(closest->value);
  const bool close_to_180 = std::abs(closest->value - 180.0) < 0.5;
  const double weight = gemmi::sq(waskal / closest->sigma * ((von_mises || close_to_180) ? gemmi::deg(1) : 1));
  const double ret = close_to_180 ? (weight * (1. + cosa)) : von_mises ? ((1-std::cos(gemmi::rad(da))) * weight) : (da * da * weight * 0.5);
  if (target != nullptr) {
    int ia[3], pos[3], apos[3];
    for (int i = 0; i < 3; ++i) {
      ia[i] = atoms[i]->serial - 1;
      pos[i] = target->params->get_pos_vec_geom(ia[i], RefineParams::Type::X);
      apos[i] = target->params->get_pos_mat_geom(ia[i], RefineParams::Type::X);
    }
    if (close_to_180) { // a special case.
      const gemmi::Vec3 h = v1 / v1n + v2 / v2n;
      gemmi::Vec3 dhdx[9]; // dh/dx11, dx12, dx13, dx21, ...
      for (int i = 0; i < 3; ++i) {
        dhdx[i]   = -(gemmi::Vec3(i==0, i==1, i==2) - v1 * v1.at(i) / gemmi::sq(v1n)) / v1n;
        dhdx[6+i] = -(gemmi::Vec3(i==0, i==1, i==2) - v2 * v2.at(i) / gemmi::sq(v2n)) / v2n;
        dhdx[3+i] = -dhdx[i] - dhdx[6+i];
      }
      gemmi::Mat33 trs[3] = {tr1.mat, {}, tr2.mat};
      for(int i = 0; i < 3; ++i)
        if (pos[i] >= 0) {
          gemmi::Vec3 v(h.dot(dhdx[3*i]), h.dot(dhdx[3*i+1]), h.dot(dhdx[3*i+2]));
          if (!same_asu(i)) // same_asu(1) will always return true
            v = trs[i].transpose().multiply(v);
          target->incr_vn(pos[i], weight, v);
          gemmi::SMat33<double> smat{dhdx[3*i].length_sq(), dhdx[3*i+1].length_sq(), dhdx[3*i+2].length_sq(),
                                     dhdx[3*i].dot(dhdx[3*i+1]), dhdx[3*i].dot(dhdx[3*i+2]), dhdx[3*i+1].dot(dhdx[3*i+2])};
          if (!same_asu(i))
            smat = smat.transformed_by<double>(trs[i].transpose());
          const int ia6 = ia[i] * 6;
          target->am[apos[i]]   += weight * smat.u11;
          target->am[apos[i]+1] += weight * smat.u22;
          target->am[apos[i]+2] += weight * smat.u33;
          target->am[apos[i]+3] += weight * smat.u12;
          target->am[apos[i]+4] += weight * smat.u13;
          target->am[apos[i]+5] += weight * smat.u23;
        }
      for (int i = 0; i < 2; ++i)
        for (int j = i+1; j < 3; ++j)
          if (pos[i] >= 0 && pos[j] >= 0) { // shouldn't we make sure pos[i] != pos[j]?
            auto mp = target->find_restraint(ia[i], ia[j]);
            gemmi::Mat33 mat;
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l)
                mat[l][k] = dhdx[3*i+l].dot(dhdx[3*j+k]);
            mat = trs[mp.imode == 0 ? i : j].transpose().multiply(mat).multiply(trs[mp.imode == 0 ? j : i]); // correct?
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l)
                target->am[mp.ipos+3*k+l] += weight * mat[l][k];
          }
    } else {
      gemmi::Vec3 dcosadx[3]; // d/dx cosa
      dcosadx[0] = (v2 / (v1n * v2n) - v1 * cosa / (v1n * v1n));
      dcosadx[2] = (v1 / (v1n * v2n) - v2 * cosa / (v2n * v2n));
      dcosadx[1] = -dcosadx[0] - dcosadx[2];
      if (!same_asu(0))
        dcosadx[0] = tr1.mat.transpose().multiply(dcosadx[0]);
      if (!same_asu(2))
        dcosadx[2] = tr2.mat.transpose().multiply(dcosadx[2]);
      const double deriv_fac = von_mises
        // sin(a-a0) / sina
        ? (weight * (std::cos(a0_rad) - cosa * (std::abs(a_rad - a0_rad) < 1e-4
                                                ? 1. : (std::sin(a0_rad) / sina))))
        : (weight * da * gemmi::deg(1) / sina);
      const double secder_fac = von_mises
        ? (weight / gemmi::sq(sina))
        : (weight * gemmi::sq(gemmi::deg(1) / sina));
      for(int i = 0; i < 3; ++i)
        if (pos[i] >= 0) {
          target->incr_vn(pos[i], deriv_fac, dcosadx[i]);
          target->incr_am_diag(apos[i], secder_fac, dcosadx[i]);
        }
      for (int i = 0; i < 2; ++i)
        for (int j = i+1; j < 3; ++j)
          if (pos[i] >= 0 && pos[j] >= 0) {
            auto mp = target->find_restraint(ia[i], ia[j]);
            if (mp.imode == 0) // ia[i] > ia[j]
              target->incr_am_ndiag(mp.ipos, secder_fac, dcosadx[i], dcosadx[j]);
            else
              target->incr_am_ndiag(mp.ipos, secder_fac, dcosadx[j], dcosadx[i]);
          }
    }
    target->target += ret;
  }
  if (reporting != nullptr)
    reporting->angles.emplace_back(this, closest, da);
  return ret;
}

inline double Geometry::Torsion::calc(double wtskal, GeomTarget* target, Reporting *reporting) const {
  const gemmi::Position& x1 = atoms[0]->pos;
  const gemmi::Position& x2 = atoms[1]->pos;
  const gemmi::Position& x3 = atoms[2]->pos;
  const gemmi::Position& x4 = atoms[3]->pos;
  const gemmi::Vec3 u = x1 - x2;
  const gemmi::Vec3 v = x4 - x3;
  const gemmi::Vec3 w = x3 - x2;
  const gemmi::Vec3 a = u.cross(w);
  const gemmi::Vec3 b = v.cross(w);
  const double s = a.cross(b).dot(w);
  const double wl = std::max(0.0001, w.length());
  const double t = wl * a.dot(b);
  const double theta = gemmi::deg(std::atan2(s, t));
  auto closest = find_closest_value(theta);
  const int period = std::max(1, closest->period);
  const double weight = wtskal * wtskal / (closest->sigma * closest->sigma);
  const double dtheta1 = gemmi::rad(period * (theta - closest->value));
  const double dtheta2 = gemmi::deg(std::atan2(std::sin(dtheta1), std::cos(dtheta1)));
  const double dtheta = dtheta2 / period;
  const double ret = dtheta * dtheta * weight * 0.5;

  if (target != nullptr) {
    int ia[4], pos[4], apos[4];
    for (int i = 0; i < 4; ++i) {
      ia[i] = atoms[i]->serial - 1;
      pos[i] = target->params->get_pos_vec_geom(ia[i], RefineParams::Type::X);
      apos[i] = target->params->get_pos_mat_geom(ia[i], RefineParams::Type::X);
    }
    const double denom = gemmi::rad(std::max(0.0001, s * s + t * t));
    gemmi::Vec3 dadx[3][3], dbdx[3][3], dwdx[3][2];
    double dwldx[3][2];
    for (int i = 0; i < 3; ++i) {
      gemmi::Vec3 drdx; drdx.at(i) = 1.;
      const gemmi::Vec3 d1 = drdx.cross(w);
      const gemmi::Vec3 d2 = u.cross(drdx);
      const gemmi::Vec3 d3 = v.cross(drdx);
      dadx[i][0] = d1;     // da/dx1
      dadx[i][1] = -d1-d2; // da/dx2
      dadx[i][2] = d2;     // da/dx3
      dbdx[i][0] = -d3;    // db/dx2
      dbdx[i][1] = d3-d1;  // db/dx3
      dbdx[i][2] = d1;     // db/dx4
      dwdx[i][0] = -drdx;  // dw/dx2
      dwdx[i][1] = drdx;   // dw/dx3
      dwldx[i][1] = w.dot(drdx)/wl; // dwl/dx3
      dwldx[i][0] = -dwldx[i][1];   // dwl/dx2
    }
    gemmi::Vec3 dthdx[4];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        double dsdx = 0.;
        double dtdx = 0.;
        if (j != 3) { // only for x1,x2,x3
          dsdx = dadx[i][j].cross(b).dot(w);
          dtdx = dadx[i][j].dot(b) * wl;
          if (j == 0) { // only for x1
            dthdx[j].at(i) = (t * dsdx - s * dtdx) / denom;
            continue;
          }
        }
        // only for x2,x3,x4
        dsdx += a.cross(dbdx[i][j-1]).dot(w);
        dtdx += a.dot(dbdx[i][j-1]) * wl;
        if (j != 3) { // only for x2,x3
          dsdx += a.cross(b).dot(dwdx[i][j-1]);
          dtdx += t / wl * dwldx[i][j-1];
        }
        dthdx[j].at(i) = (t * dsdx - s * dtdx)/denom;
      }
    }

    for(int i = 0; i < 4; ++i)
      if (pos[i] >= 0) {
        target->incr_vn(pos[i], dtheta * weight, dthdx[i]);
        target->incr_am_diag(apos[i], weight, dthdx[i]);
      }

    for (int i = 0; i < 3; ++i)
      for (int j = i+1; j < 4; ++j)
        if (pos[i] >= 0 && pos[j] >= 0) {
          auto mp = target->find_restraint(ia[i], ia[j]);
          if (mp.imode == 0)
            target->incr_am_ndiag(mp.ipos, weight, dthdx[i], dthdx[j]);
          else
            target->incr_am_ndiag(mp.ipos, weight, dthdx[j], dthdx[i]);
        }
    target->target += ret;
  }
  if (reporting != nullptr)
    reporting->torsions.emplace_back(this, closest, dtheta, theta);
  return ret;
}

inline double Geometry::Chirality::calc(double wchiral, GeomTarget* target, Reporting *reporting) const {
  const double weight = wchiral * wchiral / (sigma * sigma);
  const gemmi::Position& xc = atoms[0]->pos;
  const gemmi::Position& x1 = atoms[1]->pos;
  const gemmi::Position& x2 = atoms[2]->pos;
  const gemmi::Position& x3 = atoms[3]->pos;
  const gemmi::Vec3 a1 = x1 - xc;
  const gemmi::Vec3 a2 = x2 - xc;
  const gemmi::Vec3 a3 = x3 - xc;
  const gemmi::Vec3 a1xa2 = a1.cross(a2);
  const double v = a1xa2.dot(a3);
  const bool isneg = (sign == gemmi::ChiralityType::Negative || (sign == gemmi::ChiralityType::Both && v < 0));
  const double ideal = (isneg ? -1 : 1) * value;
  const double dv = v - ideal;
  const double ret = dv * dv * weight * 0.5;

  if (target != nullptr) {
    int ia[4], pos[4], apos[4];
    for (int i = 0; i < 4; ++i) {
      ia[i] = atoms[i]->serial - 1;
      pos[i] = target->params->get_pos_vec_geom(ia[i], RefineParams::Type::X);
      apos[i] = target->params->get_pos_mat_geom(ia[i], RefineParams::Type::X);
    }
    gemmi::Vec3 dcdx[4];
    for (int i = 0; i < 3; ++i) {
      gemmi::Vec3 drdx; drdx.at(i) = 1.;
      dcdx[1].at(i) = drdx.cross(a2).dot(a3); // atom1
      dcdx[2].at(i) = a1.cross(drdx).dot(a3); // atom2
      dcdx[3].at(i) = a1xa2.dot(drdx);        // atom3
      dcdx[0].at(i) = -dcdx[1].at(i) - dcdx[2].at(i) - dcdx[3].at(i); //atomc
    }

    for(int i = 0; i < 4; ++i)
      if (pos[i] >= 0) {
        target->incr_vn(pos[i], dv * weight, dcdx[i]);
        target->incr_am_diag(apos[i], weight, dcdx[i]);
      }

    for (int i = 0; i < 3; ++i)
      for (int j = i+1; j < 4; ++j)
        if (pos[i] >= 0 && pos[j] >= 0) {
          auto mp = target->find_restraint(ia[i], ia[j]);
          if (mp.imode == 0)
            target->incr_am_ndiag(mp.ipos, weight, dcdx[i], dcdx[j]);
          else
            target->incr_am_ndiag(mp.ipos, weight, dcdx[j], dcdx[i]);
        }
    target->target += ret;
  }
  if (reporting != nullptr)
    reporting->chirs.emplace_back(this, dv, ideal);
  return ret;
}

inline double Geometry::Plane::calc(double wplane, GeomTarget* target, Reporting *reporting) const {
  const double weight = wplane * wplane / (sigma * sigma);
  const int natoms = atoms.size();
  const PlaneDeriv pder(atoms);

  double ret = 0.;
  std::vector<double> deltas(natoms);
  for (int j = 0; j < natoms; ++j) {
    deltas[j] = pder.D - pder.vm.dot(atoms[j]->pos);
    ret += deltas[j] * deltas[j] * weight * 0.5;
  }

  if (target != nullptr) {
    for (int j = 0; j < natoms; ++j) {
      const gemmi::Position &xj = atoms[j]->pos;
      for (int l = 0; l < natoms; ++l) {
        const int ial = atoms[l]->serial-1;
        const int posl = target->params->get_pos_vec_geom(ial, RefineParams::Type::X);
        const int aposl = target->params->get_pos_mat_geom(ial, RefineParams::Type::X);
        gemmi::Position dpdx1;
        for (int m = 0; m < 3; ++m)
          dpdx1.at(m) = pder.dDdx[l].at(m) - xj.dot(pder.dvmdx[l][m]) - (j==l ? pder.vm.at(m) : 0);

        if (posl >= 0) target->incr_vn(posl, deltas[j] * weight, dpdx1);

        for (int k = l; k < natoms; ++k) {
          const int iak = atoms[k]->serial-1;
          const int posk = target->params->get_pos_vec_geom(iak, RefineParams::Type::X);
          gemmi::Position dpdx2;
          for (int m = 0; m < 3; ++m)
            dpdx2.at(m) = pder.dDdx[k].at(m) - xj.dot(pder.dvmdx[k][m]) - (k==j ? pder.vm.at(m) : 0);

          if (posl >= 0 && posk >= 0) {
            if (k == l)
              target->incr_am_diag(aposl, weight, dpdx1);
            else {
              auto mp = target->find_restraint(ial, iak);
              if (mp.imode == 0)
                target->incr_am_ndiag(mp.ipos, weight, dpdx1, dpdx2);
              else
                target->incr_am_ndiag(mp.ipos, weight, dpdx2, dpdx1);
            }
          }
        }
      }
    }
    target->target += ret;
  }
  if (reporting != nullptr)
    reporting->planes.emplace_back(this, deltas);
  return ret;
}

inline void Geometry::Harmonic::calc(GeomTarget* target) const {
  if (target != nullptr) {
    // Refmac style - only affects second derivatives
    const int apos = target->params->get_pos_mat_geom(atom->serial-1, RefineParams::Type::X);
    if (apos < 0) return;
    const double w = 1. / (sigma * sigma);
    for (size_t i = 0; i < 3; ++i)
      target->am[apos+i] += w;
  }
}

inline double Geometry::Stacking::calc(double wstack, bool use_dist, GeomTarget* target, Reporting *reporting) const {
  double ret = 0;
  PlaneDeriv pder[2] = {planes[0], planes[1]};
  double vm1vm2 = pder[0].vm.dot(pder[1].vm);
  if (vm1vm2 < 0) {
    pder[1].flip();
    vm1vm2 *= -1;
  }

  // angle
  const double wa = wstack * wstack / (sd_angle * sd_angle);
  const double cosa = std::min(1., vm1vm2);
  const double a = gemmi::deg(std::acos(std::max(-1., std::min(1., cosa))));
  const double deltaa = a - angle;
  const double deltaa2 = deltaa * deltaa;
  ret += 0.5 * wa * deltaa2;
  if (target != nullptr) {
    const double inv_sina = 1. / std::min(1., std::max(std::sqrt(1 - cosa * cosa), 0.1));
    std::vector<std::vector<gemmi::Vec3>> dpdx;
    for (size_t i = 0; i < 2; ++i) { // plane index
      dpdx.emplace_back(planes[i].size());
      for (size_t j = 0; j < planes[i].size(); ++j) { // atom index of plane i
        const int iaij = planes[i][j]->serial-1;
        const int posij = target->params->get_pos_vec_geom(iaij, RefineParams::Type::X);
        const int aposij = target->params->get_pos_mat_geom(iaij, RefineParams::Type::X);
        for (size_t m = 0; m < 3; ++m)
          dpdx[i][j].at(m) = -gemmi::deg(1) * pder[i].dvmdx[j][m].dot(pder[1-i].vm) * inv_sina;
        if (posij < 0) continue;
        target->incr_vn(posij, wa * deltaa, dpdx[i][j]);

        // second derivatives in the same plane
        for (size_t k = 0; k <= j; ++k) {
          const int iaik = planes[i][k]->serial-1;
          const int posik = target->params->get_pos_vec_geom(iaik, RefineParams::Type::X);
          if (posik < 0) continue;
          if (k == j)
            target->incr_am_diag(aposij, wa, dpdx[i][j]);
          else {
            auto mp = target->find_restraint(iaij, iaik);
            if (mp.imode == 0)
              target->incr_am_ndiag(mp.ipos, wa, dpdx[i][j], dpdx[i][k]);
            else
              target->incr_am_ndiag(mp.ipos, wa, dpdx[i][k], dpdx[i][j]);
          }
        }
      }
    }
    // second derivatives between two planes
    for (size_t j = 0; j < planes[0].size(); ++j)
      for (size_t k = 0; k < planes[1].size(); ++k) {
        const int ia0j = planes[0][j]->serial-1;
        const int ia1k = planes[1][k]->serial-1;
        const int pos0j = target->params->get_pos_vec_geom(ia0j, RefineParams::Type::X);
        const int pos1k = target->params->get_pos_vec_geom(ia1k, RefineParams::Type::X);
        if (pos0j >= 0 && pos1k >= 0) {
          auto mp = target->find_restraint(ia0j, ia1k);
          if (mp.imode == 0)
            target->incr_am_ndiag(mp.ipos, wa, dpdx[0][j], dpdx[1][k]);
          else
            target->incr_am_ndiag(mp.ipos, wa, dpdx[1][k], dpdx[0][j]);
        }
      }
  }

  // distance; turned off by default in Refmac
  double deltad[2] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  if (use_dist && dist > 0) { // skip if ideal dist < 0
    const double wd = wstack * wstack / (sd_dist * sd_dist);
    for (size_t i = 0; i < 2; ++i) {
      double d = pder[i].xs.dot(pder[1-i].vm) - pder[1-i].D; // distance from i to the other
      if (d < 0) {
        d *= -1;
        pder[1-i].flip();
      }
      deltad[i] = d - dist;
      const double deltad2 = deltad[i] * deltad[i];
      ret += 0.5 * wd * deltad2; // distance between planes is not symmetric, so we add both
    }
    if (target != nullptr) {
      std::vector<std::vector<gemmi::Vec3>> dpdx;
      dpdx.emplace_back(planes[0].size());
      dpdx.emplace_back(planes[1].size());
      for (size_t i = 0; i < 2; ++i) {
        // for the atoms of this plane
        for (size_t j = 0; j < planes[i].size(); ++j) {
          const int iaij = planes[i][j]->serial-1;
          const int posij = target->params->get_pos_vec_geom(iaij, RefineParams::Type::X);
          const int aposij = target->params->get_pos_mat_geom(iaij, RefineParams::Type::X);
          dpdx[i][j] = pder[1-i].vm / planes[i].size();
          if (posij < 0) continue;
          target->incr_vn(posij, wd * deltad[i], dpdx[i][j]);
          // second derivatives
          for (size_t k = 0; k <= j; ++k) {
            const int iaik = planes[i][k]->serial-1;
            const int posik = target->params->get_pos_vec_geom(iaik, RefineParams::Type::X);
            if (posik < 0) continue;
            if (k == j)
              target->incr_am_diag(aposij, wd, dpdx[i][j]);
            else {
              auto mp = target->find_restraint(iaij, iaik);
              if (mp.imode == 0)
                target->incr_am_ndiag(mp.ipos, wd, dpdx[i][j], dpdx[i][k]);
              else
                target->incr_am_ndiag(mp.ipos, wd, dpdx[i][k], dpdx[i][j]);
            }
          }
        }
        // for the atoms of the other plane
        for (size_t j = 0; j < planes[1-i].size(); ++j) {
          const int iaj = planes[1-i][j]->serial-1;
          const int posj = target->params->get_pos_vec_geom(iaj, RefineParams::Type::X);
          const int aposj = target->params->get_pos_mat_geom(iaj, RefineParams::Type::X);
          for (size_t m = 0; m < 3; ++m)
            dpdx[1-i][j].at(m) = pder[1-i].dvmdx[j][m].dot(pder[i].xs) - pder[1-i].dDdx[j].at(m);
          if (posj < 0) continue;
          target->incr_vn(posj, wd * deltad[i], dpdx[1-i][j]);
          // second derivatives
          for (size_t k = 0; k <= j; ++k) {
            const int iak = planes[1-i][k]->serial-1;
            const int posk = target->params->get_pos_vec_geom(iak, RefineParams::Type::X);
            if (posk < 0) continue;
            if (k == j)
              target->incr_am_diag(aposj, wd, dpdx[1-i][j]);
            else {
              auto mp = target->find_restraint(iaj, iak);
              if (mp.imode == 0)
                target->incr_am_ndiag(mp.ipos, wd, dpdx[1-i][j], dpdx[1-i][k]);
              else
                target->incr_am_ndiag(mp.ipos, wd, dpdx[1-i][k], dpdx[1-i][j]);
            }
          }
        }
        // second derivatives between two planes
        for (size_t j = 0; j < planes[0].size(); ++j)
          for (size_t k = 0; k < planes[1].size(); ++k) {
            const int ia0j = planes[0][j]->serial-1;
            const int ia1k = planes[1][k]->serial-1;
            const int pos0j = target->params->get_pos_vec_geom(ia0j, RefineParams::Type::X);
            const int pos1k = target->params->get_pos_vec_geom(ia1k, RefineParams::Type::X);
            if (pos0j >= 0 && pos1k >= 0) {
              auto mp = target->find_restraint(ia0j, ia1k);
              if (mp.imode == 0)
                target->incr_am_ndiag(mp.ipos, wd, dpdx[0][j], dpdx[1][k]);
              else
                target->incr_am_ndiag(mp.ipos, wd, dpdx[1][k], dpdx[0][j]);
            }
          }
      }
    }
  }
  if (target != nullptr)
    target->target += ret;
  if (reporting != nullptr)
    reporting->stackings.emplace_back(this, deltaa, deltad[0], deltad[1]);
  return ret;
}

inline double
Geometry::Vdw::calc(const gemmi::UnitCell& cell, double wvdw, GeomTarget* target, Reporting *reporting) const {
  if (sigma <= 0) return 0.;
  const double weight = wvdw * wvdw / (sigma * sigma);
  const gemmi::Atom& atom1 = *atoms[0];
  const gemmi::Atom& atom2 = *atoms[1];
  const gemmi::Transform tr = get_transform(cell, sym_idx, pbc_shift);
  const gemmi::Position& x1 = atom1.pos;
  const gemmi::Position& x2 = same_asu() ? atom2.pos : gemmi::Position(tr.apply(atom2.pos));
  const double b = x1.dist(x2);
  const double db = b - value;
  if (db > 0)
    return 0.;

  const double ret = db * db * weight * 0.5;
  if (target != nullptr) {
    const int ia1 = atom1.serial - 1;
    const int ia2 = atom2.serial - 1;
    const int pos1 = target->params->get_pos_vec_geom(ia1, RefineParams::Type::X);
    const int pos2 = target->params->get_pos_vec_geom(ia2, RefineParams::Type::X);
    const int apos1 = target->params->get_pos_mat_geom(ia1, RefineParams::Type::X);
    const int apos2 = target->params->get_pos_mat_geom(ia2, RefineParams::Type::X);
    const gemmi::Position dbdx1 = (x1 - x2) / std::max(b, 0.02);
    const gemmi::Position dbdx2 = same_asu() ? -dbdx1 : gemmi::Position(tr.mat.transpose().multiply(-dbdx1));
    if (pos1 >= 0) {
      target->incr_vn(pos1, weight * db, dbdx1);
      target->incr_am_diag(apos1, weight, dbdx1);
    }
    if (pos2 >= 0) {
      target->incr_vn(pos2, weight * db, dbdx2);
      target->incr_am_diag(apos2, weight, dbdx2);
    }
    if (pos1 >= 0 && pos2 >= 0) {
      if (ia1 != ia2) {
        auto mp = target->find_restraint(ia1, ia2);
        if (mp.imode == 0)
          target->incr_am_ndiag(mp.ipos, weight, dbdx1, dbdx2);
        else
          target->incr_am_ndiag(mp.ipos, weight, dbdx2, dbdx1);
      } else
        target->incr_am_diag12(apos1, weight, dbdx1, dbdx2);
    }
    target->target += ret;
  }
  if (reporting != nullptr)
    reporting->vdws.emplace_back(this, db);
  return ret;
}

inline double
Geometry::Interval::calc(const gemmi::UnitCell& cell, double wb, GeomTarget* target, Reporting *reporting) const {
  if (smin <= 0 || smax <= 0) return 0.;
  const gemmi::Atom& atom1 = *atoms[0];
  const gemmi::Atom& atom2 = *atoms[1];
  const gemmi::Transform tr = get_transform(cell, sym_idx, pbc_shift);
  const gemmi::Position& x1 = atom1.pos;
  const gemmi::Position& x2 = same_asu() ? atom2.pos : gemmi::Position(tr.apply(atom2.pos));
  const double b = x1.dist(x2);
  double db = 0;
  double weight = wb * wb;
  if (b < dmin) {
    db = b - dmin;
    weight /= smin * smin;
  } else if (b > dmax) {
    db = b - dmax;
    weight /= smax * smax;
  } else // do we need this?
    weight *= 2 / (smin * smin + smax * smax);

  const double ret = db * db * weight * 0.5;
  if (target != nullptr) {
    const int ia1 = atom1.serial - 1;
    const int ia2 = atom2.serial - 1;
    const int pos1 = target->params->get_pos_vec_geom(ia1, RefineParams::Type::X);
    const int pos2 = target->params->get_pos_vec_geom(ia2, RefineParams::Type::X);
    const int apos1 = target->params->get_pos_mat_geom(ia1, RefineParams::Type::X);
    const int apos2 = target->params->get_pos_mat_geom(ia2, RefineParams::Type::X);
    const gemmi::Position dbdx1 = (x1 - x2) / std::max(b, 0.02);
    const gemmi::Position dbdx2 = same_asu() ? -dbdx1 : gemmi::Position(tr.mat.transpose().multiply(-dbdx1));
    if (pos1 >= 0) {
      target->incr_vn(pos1, weight * db, dbdx1);
      target->incr_am_diag(apos1, weight, dbdx1);
    }
    if (pos2 >= 0) {
      target->incr_vn(pos2, weight * db, dbdx2);
      target->incr_am_diag(apos2, weight, dbdx2);
    }
    if (pos1 >= 0 && pos2 >= 0) {
      if (ia1 != ia2) {
        auto mp = target->find_restraint(ia1, ia2);
        if (mp.imode == 0)
          target->incr_am_ndiag(mp.ipos, weight, dbdx1, dbdx2);
        else
          target->incr_am_ndiag(mp.ipos, weight, dbdx2, dbdx1);
      } else
        target->incr_am_diag12(apos1, weight, dbdx1, dbdx2);
    }
    target->target += ret;
  }
  if (reporting != nullptr && db != 0)
    reporting->intervals.emplace_back(this, db, b < dmin);
  return ret;
}

inline double
Geometry::Ncsr::calc(const gemmi::UnitCell& cell, double wncsr, GeomTarget* target, Reporting *reporting,
                     double ncsr_diff_cutoff, double ncsr_max_dist) const {
  if (sigma <= 0) return 0.;
  const Vdw &vdw1 = *pairs[0], &vdw2 = *pairs[1];
  const gemmi::Atom* atoms[4] = {vdw1.atoms[0], vdw1.atoms[1], vdw2.atoms[0], vdw2.atoms[1]};
  const gemmi::Transform tr1 = get_transform(cell, vdw1.sym_idx, vdw1.pbc_shift);
  const gemmi::Transform tr2 = get_transform(cell, vdw2.sym_idx, vdw2.pbc_shift);
  const gemmi::Position& x1 = atoms[0]->pos;
  const gemmi::Position& x2 = vdw1.same_asu() ? atoms[1]->pos : gemmi::Position(tr1.apply(atoms[1]->pos));
  const gemmi::Position& x3 = atoms[2]->pos;
  const gemmi::Position& x4 = vdw2.same_asu() ? atoms[3]->pos : gemmi::Position(tr2.apply(atoms[3]->pos));
  const double b1 = x1.dist(x2);
  const double b2 = x3.dist(x4);
  const double db = b1 - b2;
  const double weight = (std::abs(db) > ncsr_diff_cutoff || b1 > ncsr_max_dist || b2 > ncsr_max_dist) ? 0 : wncsr / sigma;
  const double y = db * weight;
  Barron2019 robustf(alpha, y);

  // note that second derivative is not exact in some alpha
  if (target != nullptr && weight > 0) {
    gemmi::Position dydx[4];
    dydx[0] = weight * (x1 - x2) / std::max(b1, 0.02);
    dydx[1] = vdw1.same_asu() ? -dydx[0] : gemmi::Position(tr1.mat.transpose().multiply(-dydx[0]));
    dydx[2] = -weight * (x3 - x4) / std::max(b2, 0.02);
    dydx[3] = vdw2.same_asu() ? -dydx[2] : gemmi::Position(tr2.mat.transpose().multiply(-dydx[2]));
    for (int i = 0; i < 4; ++i) {
      const int iai = atoms[i]->serial-1;
      const int posi = target->params->get_pos_vec_geom(iai, RefineParams::Type::X);
      const int aposi = target->params->get_pos_mat_geom(iai, RefineParams::Type::X);
      if (posi < 0) continue;
      target->incr_vn(posi, robustf.dfdy, dydx[i]);
      target->incr_am_diag(aposi, robustf.d2fdy, dydx[i]);
      for (int j = 0; j < i; ++j) {
        const int iaj = atoms[j]->serial-1;
        const int posj = target->params->get_pos_vec_geom(iaj, RefineParams::Type::X);
        if (posj < 0) continue;
        auto mp = target->find_restraint(iai, iaj);
        if (mp.imode == 0)
          target->incr_am_ndiag(mp.ipos, robustf.d2fdy, dydx[i], dydx[j]);
        else
          target->incr_am_ndiag(mp.ipos, robustf.d2fdy, dydx[j], dydx[i]);
        // could atoms[i] == atoms[j] happen?
      }
    }
    target->target += robustf.f;
  }
  if (reporting != nullptr)
    reporting->ncsrs.emplace_back(this, b1, b2);
  return robustf.f;
}

void Geometry::spec_correction(double alpha, bool use_rr) {
  const int n_pairs = target.pairs.size();
  for (const auto &spec : specials) {
    const int idx = spec.atom->serial - 1;
    if (target.params->is_refined(RefineParams::Type::X)) {
      const int pos = target.params->get_pos_vec(idx, RefineParams::Type::X);
      if (pos < 0) continue;
      // correct gradient
      Eigen::Map<Eigen::Vector3d> v(&target.vn[pos], 3);
      v = (spec.Rspec_pos.transpose() * v).eval();
      // correct diagonal block
      const int apos = target.params->get_pos_mat_geom(idx, RefineParams::Type::X);
      double* a = target.am.data() + apos;
      Eigen::Matrix3d m {{a[0], a[3], a[4]},
                         {a[3], a[1], a[5]},
                         {a[4], a[5], a[2]}};
      const double hmax = m.maxCoeff();
      m = (spec.Rspec_pos.transpose() * m * spec.Rspec_pos).eval();
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
      // correct non-diagonal block
      for (int i = 0; i < n_pairs; ++i) {
        if (target.pairs[i].first == idx || target.pairs[i].second == idx) {
          const int mpos = target.params->get_pos_mat_pair_geom(i, RefineParams::Type::X);
          if (mpos < 0) continue;
          Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> m(&target.am[mpos]);
          if (target.pairs[i].first == idx)
            m = (spec.Rspec_pos.transpose() * m).eval();
          else
            m = (m * spec.Rspec_pos).eval();
        }
      }
    }
    if (target.params->is_refined(RefineParams::Type::B) && target.params->aniso) {
      const int pos = target.params->get_pos_vec(idx, RefineParams::Type::B);
      if (pos < 0) continue;
      // correct gradient
      Eigen::Map<Eigen::VectorXd> v(&target.vn[pos], 6);
      v = (spec.Rspec_aniso.transpose() * v).eval();
      // correct diagonal block
      const int apos = target.params->get_pos_mat_geom(idx, RefineParams::Type::B);
      double* a = target.am.data() + apos;
      Eigen::MatrixXd m {{ a[0],  a[6],  a[7],  a[8],  a[9], a[10]},
                         { a[6],  a[1], a[11], a[12], a[13], a[14]},
                         { a[7], a[11],  a[2], a[15], a[16], a[17]},
                         { a[8], a[12], a[15],  a[3], a[18], a[19]},
                         { a[9], a[13], a[16], a[18],  a[4], a[20]},
                         {a[10], a[14], a[17], a[19], a[20],  a[5]}};
      const double hmax = m.maxCoeff();
      m = (spec.Rspec_aniso.transpose() * m * spec.Rspec_aniso).eval();
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
      // correct non-diagonal block
      for (int i = 0; i < n_pairs; ++i) {
        if (target.pairs[i].first == idx || target.pairs[i].second == idx) {
          const int mpos = target.params->get_pos_mat_pair_geom(i, RefineParams::Type::B);
          if (mpos < 0) continue;
          Eigen::Map<Eigen::Matrix<double,6,6, Eigen::RowMajor>> m(&target.am[mpos]);
          if (target.pairs[i].first == idx)
            m = (spec.Rspec_aniso.transpose() * m).eval();
          else
            m = (m * spec.Rspec_aniso).eval();
        }
      }
    }
  }
}

} // namespace servalcat
#endif
