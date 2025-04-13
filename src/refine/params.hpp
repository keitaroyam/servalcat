// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_REFINE_PARAMS_HPP_
#define SERVALCAT_REFINE_PARAMS_HPP_

#include <bitset>
#include <vector>
#include <gemmi/model.hpp>     // for Atom
#include <gemmi/calculate.hpp> // for count_atom_sites
#include <gemmi/eig3.hpp>      // eigen_decomposition
namespace servalcat {

struct RefineParams {
  // Manage data structure
  //  grad vector: ordered by xyz, B, occ, dfrac
  //  Fisher matrix for geometry:
  //    xyz diagonal blocks (6 per atom), non-diagonal blocks (9 per pair),
  //    ADP diag (iso=1 or aniso=21), non-diag (iso=1 or aniso=36),
  //    occ diag (1), non-diag (1),
  //    dfrac diag (1),
  //  Fisher matrix for -LL:
  //    xyz diagonal blocks (6 per atom)
  //    ADP diag (iso=1 or aniso=21)
  //    occ diag (1)
  //    dfrac diag (1),
  //    ADP-occ mixed (iso=1 or aniso=6),
  //    ADP-dfrac mixed (iso=1 or aniso=6)
  //    occ-dfrac mixed (1)
  enum class Type : unsigned char { X=0, B, Q, D, END }; // pos, adp, occ, dfrac
  static Type num2type(unsigned char x) { return static_cast<Type>(x); }
  static unsigned char type2num(Type t) { return static_cast<unsigned char>(t); }
  static constexpr size_t N = static_cast<size_t>(Type::END);
  static constexpr std::array<Type, N> Types = {Type::X, Type::B, Type::Q, Type::D};

  std::bitset<N> flag_global; // what parameters to be refined
  bool aniso;
  bool use_q_b_mixed_derivatives;
  std::vector<gemmi::Atom*> atoms;
  std::array<std::vector<int>, N> atom_to_param_; // atom index to parameter index (-1 indicates fixed atoms)
  std::array<std::vector<int>, N> param_to_atom_; // parameter index to atom index
  std::array<std::vector<int>, N> pairs_refine_; // should we have param_to_atom equivalent for this?
  std::vector<bool> ll_exclusion; // TODO. should be Type dependent?
  std::array<size_t, N> npairs_refine_{};
  std::vector<int> bq_mix_atoms; // B-Q
  std::vector<int> bd_mix_atoms; // B-D
  std::vector<int> qd_mix_atoms; // Q-D
  RefineParams(bool refine_xyz, int adp_mode, bool refine_occ, bool refine_dfrac, bool use_q_b_mixed)
    : use_q_b_mixed_derivatives(use_q_b_mixed) {
    // decide which parameters are refined
    if (refine_xyz)
      flag_global.set(type2num(Type::X));
    if (adp_mode > 0) {
      flag_global.set(type2num(Type::B));
      aniso = (adp_mode == 2);
    }
    if (refine_occ)
      flag_global.set(type2num(Type::Q));
    if (refine_dfrac)
      flag_global.set(type2num(Type::D));
  }
  std::vector<int> &atom_to_param(Type t) { return atom_to_param_[type2num(t)]; }
  const std::vector<int> &atom_to_param(Type t) const { return const_cast<RefineParams*>(this)->atom_to_param(t); }
  std::vector<int> &param_to_atom(Type t) { return param_to_atom_[type2num(t)]; }
  const std::vector<int> &param_to_atom(Type t) const { return const_cast<RefineParams*>(this)->param_to_atom(t); }
  std::vector<int> &pairs_refine(Type t) { return pairs_refine_[type2num(t)]; }
  const std::vector<int> &pairs_refine(Type t) const { return const_cast<RefineParams*>(this)->pairs_refine(t); }
  size_t &npairs_refine(Type t) { return npairs_refine_[type2num(t)]; }
  size_t npairs_refine(Type t) const { return const_cast<RefineParams*>(this)->npairs_refine(t); }
  size_t npar_per_atom(Type t) const {
    switch (t) {
    case Type::X: return 3;
    case Type::B: return aniso ? 6 : 1;
    case Type::Q: return 1;
    case Type::D: return 1;
    default: gemmi::fail("npar_per_atom: bad t");
    }
  }
  size_t nfisher_geom_per_atom(Type t) const {
    switch (t) {
    case Type::X: return 6;
    case Type::B: return aniso ? 21 : 1;
    case Type::Q: return 1;
    case Type::D: return 1;
    default: gemmi::fail("nfisher_geom_per_atom: bad t");
    }
  }
  size_t nfisher_ll_per_atom(Type t) const {
    switch (t) {
    case Type::X: return 6; // we actually need 3
    case Type::B: return aniso ? 21 : 1;
    case Type::Q: return 1;
    case Type::D: return 1;
    default: gemmi::fail("nfisher_ll_per_atom: bad t");
    }
  }
  size_t nfisher_geom_per_pair(Type t) const {
    switch (t) {
    case Type::X: return 9;
    case Type::B: return aniso ? 36 : 1;
    case Type::Q: return 1;
    case Type::D: return 1;
    default: gemmi::fail("nfisher_geom_per_pair: bad t");
    }
  }
  size_t n_refined_atoms(Type t) const {
    if (t == Type::END) return 0;
    return param_to_atom(t).size();
  }
  size_t n_refined_pairs(Type t) const {
    if (t == Type::END) return 0;
    return npairs_refine(t);
  }
  size_t n_params() const { // number of params (grad vector size)
    size_t ret = 0;
    for (Type t : Types)
      ret += n_refined_atoms(t) * npar_per_atom(t);
    return ret;
  }
  // fisher sparse matrix size
  size_t n_fisher_geom() const {
    size_t ret = 0;
    for (Type t : Types) {
      ret += n_refined_atoms(t) * nfisher_geom_per_atom(t);
      ret += n_refined_pairs(t) * nfisher_geom_per_pair(t);
    }
    // no mixed derivatives
    return ret;
  }
  size_t n_fisher_ll() const {
    size_t ret = 0;
    for (Type t : Types)
      ret += n_refined_atoms(t) * nfisher_ll_per_atom(t);
    // mixed derivatives
    ret += bq_mix_atoms.size() * npar_per_atom(Type::B) * npar_per_atom(Type::Q);
    ret += bd_mix_atoms.size() * npar_per_atom(Type::B) * npar_per_atom(Type::D);
    ret += qd_mix_atoms.size() * npar_per_atom(Type::Q) * npar_per_atom(Type::D);
    return ret;
  }
  bool is_refined(Type t) const {
    return t != Type::END && flag_global.test(type2num(t));
  }
  bool is_atom_refined(size_t idx, Type t) const {
    if (!is_refined(t)) return false;
    return atom_to_param(t)[idx] >= 0;
  }
  bool is_atom_refined(size_t idx) const { // for any t
    for (Type t : Types)
      if (is_refined(t) && atom_to_param(t)[idx] >= 0)
        return true;
    return false;
  }
  void set_pairs(Type t, const std::vector<std::pair<int,int>> &pairs) {
    if (!is_refined(t))
      gemmi::fail("set_pairs: bad t");
    auto &pp = pairs_refine(t);
    const auto &atp = atom_to_param(t);
    pp.clear();
    pp.assign(pairs.size(), -1);
    size_t count = 0;
    for (int i = 0; i < pairs.size(); ++i) {
      const auto &p = pairs[i];
      if (atp[p.first] >= 0 && atp[p.second] >= 0)
        pp[i] = count++;
    }
    npairs_refine(t) = count;
  }
  int get_pos_vec(int atom_idx, Type t) const {
    if (!is_refined(t))
      return -1;
    int ret = atom_to_param(t)[atom_idx] * npar_per_atom(t);
    if (ret < 0)
      return -1; //gemmi::fail("unrefined atom");
    for (Type tt : Types) {
      if (tt == t)
        return ret;
      ret += n_refined_atoms(tt) * npar_per_atom(tt);
    }
    gemmi::fail("get_pos_vec: bad t");
  }
  int get_pos_mat_geom(int atom_idx, Type t) const {
    if (!is_refined(t))
      return -1;
    int ret = atom_to_param(t)[atom_idx] * nfisher_geom_per_atom(t);
    if (ret < 0)
      return -1;
    for (Type tt : Types) {
      if (tt == t)
        return ret;
      ret += n_refined_atoms(tt) * nfisher_geom_per_atom(tt);
      ret += n_refined_pairs(tt) * nfisher_geom_per_pair(tt);
    }
    gemmi::fail("get_pos_mat_geom: bad t");
  }
  int get_pos_mat_ll(int idx, Type t) const {
    if (!is_refined(t))
      return -1;
    int ret = atom_to_param(t)[idx] * nfisher_ll_per_atom(t);
    if (ret < 0)
      return -1;
    for (Type tt : Types) {
      if (tt == t)
        return ret;
      ret += n_refined_atoms(tt) * nfisher_ll_per_atom(tt);
    }
    gemmi::fail("get_pos_mat_ll: bad t");
  }
  int get_pos_mat_pair_geom(size_t pair_idx, Type t) const {
    if (pairs_refine(t).empty())
      return -1;
    int ret = pairs_refine(t)[pair_idx] * nfisher_geom_per_pair(t);
    if (ret < 0)
      return -1;
    for (Type tt : Types) {
      ret += n_refined_atoms(tt) * nfisher_geom_per_atom(tt);
      if (t == tt)
        return ret;
      ret += n_refined_pairs(tt) * nfisher_geom_per_pair(tt);
    }
    gemmi::fail("get_pos_mat_pair_geom: bad t");
  }
  int get_pos_mat_mixed_ll(int atom_idx, Type t1, Type t2) const {
    auto test = [&atom_idx](const std::vector<int> &vec) -> int {
      const auto it = std::lower_bound(vec.begin(), vec.end(), atom_idx);
      if (it == vec.end() || *it != atom_idx)
        return -1;
      return std::distance(vec.begin(), it);
    };
    const std::vector<std::tuple<Type, Type, const std::vector<int> &>> mixvecs = {
      {Type::B, Type::Q, bq_mix_atoms},
      {Type::B, Type::D, bd_mix_atoms},
      {Type::Q, Type::D, qd_mix_atoms}};
    int ret = [&]() {
      for (const auto &t : mixvecs)
        if (t1 == std::get<0>(t) && t2 == std::get<1>(t))
          return test(std::get<2>(t)) * npar_per_atom(std::get<0>(t)) * npar_per_atom(std::get<1>(t));
      gemmi::fail("get_pos_mat_mixed_ll: bad types");
    }();
    if (ret < 0)
      return -1;
    for (Type t : Types)
      ret += n_refined_atoms(t) * nfisher_ll_per_atom(t);
    for (const auto &t : mixvecs) {
      if (t1 == std::get<0>(t) && t2 == std::get<1>(t))
        return ret;
      ret += std::get<2>(t).size() * npar_per_atom(std::get<0>(t)) * npar_per_atom(std::get<1>(t));
    }
    gemmi::fail("get_pos_mat_mixed_ll: bad types");
  }
  void set_model(gemmi::Model &model) {
    atoms.clear();
    atoms.assign(gemmi::count_atom_sites(model), nullptr);
    for (gemmi::CRA cra : model.all()) {
      atoms[cra.atom->serial - 1] = cra.atom;
    }
    for (const auto &atom : atoms)
      if (atom == nullptr)
        gemmi::fail("set_model: invalid atom serial");
  }
  void clear_params() {
    bq_mix_atoms.clear();
    bd_mix_atoms.clear();
    qd_mix_atoms.clear();
    for (Type t : Types) {
      atom_to_param(t).clear();
      param_to_atom(t).clear();
      pairs_refine(t).clear();
      npairs_refine(t) = 0;
      if (is_refined(t))
        atom_to_param(t).assign(atoms.size(), -1);
    }
  }
  void add_atom_to_params(const gemmi::Atom *atom, std::bitset<N> flag = {~0ULL}) {
    const int idx = atom->serial - 1;
    for (Type t : Types) {
      if (is_refined(t) && flag.test(type2num(t)) &&
          (t != Type::D || atom->is_hydrogen())) {
        atom_to_param(t)[idx] = param_to_atom(t).size();
        param_to_atom(t).push_back(idx);
      }
    }
    if (use_q_b_mixed_derivatives && is_refined(Type::B) && is_refined(Type::Q) &&
        flag.test(type2num(Type::B)) && flag.test(type2num(Type::Q)))
      bq_mix_atoms.push_back(idx);
    if (use_q_b_mixed_derivatives && is_refined(Type::B) && is_refined(Type::D) &&
        flag.test(type2num(Type::B)) && flag.test(type2num(Type::D)))
      bd_mix_atoms.push_back(idx);
    if (is_refined(Type::Q) && is_refined(Type::D) &&
        flag.test(type2num(Type::Q)) && flag.test(type2num(Type::D)))
      qd_mix_atoms.push_back(idx);
  }
  void set_params_default() {
    clear_params();
    for (const auto atom : atoms)
      add_atom_to_params(atom);
  }
  void set_params_selected(const std::vector<int> &indices) {
    clear_params();
    for (int j : indices)
      add_atom_to_params(atoms[j]);
  }
  void set_params_from_flags() {
    clear_params();
    for (const auto atom : atoms)
      add_atom_to_params(atom, std::bitset<N>(atom->flag));
  }
  std::vector<double> get_x() const {
    std::vector<double> x;
    x.reserve(n_params());
    // xyz
    for (int i : param_to_atom(Type::X))
      for (int j = 0; j < 3; ++j)
        x.push_back(atoms[i]->pos.at(j));

    // ADP
    for (int i : param_to_atom(Type::B))
      if (aniso)
        for (auto u : atoms[i]->aniso.elements_pdb())
          x.push_back(u * gemmi::u_to_b());
      else
        x.push_back(atoms[i]->b_iso);

    // occ
    for (int i : param_to_atom(Type::Q))
      x.push_back(atoms[i]->occ);
    // dfrac
    for (int i : param_to_atom(Type::D))
      x.push_back(atoms[i]->fraction);

    return x;
  }
  void set_x(const std::vector<double>& x, double min_b = 0.5) {
    assert(x.size() == n_params());
    int k = 0;
    // xyz
    for (int i : param_to_atom(Type::X))
      for (int j = 0; j < 3; ++j)
        atoms[i]->pos.at(j) = x[k++];

    // ADP
    for (int i :  param_to_atom(Type::B))
      if (aniso) {
        // TODO eig to set minimum
        gemmi::SMat33<double> m = {x[k], x[k+1], x[k+2], x[k+3], x[k+4], x[k+5]};
        double eig[3] = {};
        const gemmi::Mat33 V = gemmi::eigen_decomposition(m, eig);
        for (int j = 0; j < 3; ++j)
          eig[j] = std::max(eig[j], min_b);
        const gemmi::SMat33<double> m2 = {eig[0], eig[1], eig[2], 0, 0, 0};
        atoms[i]->aniso = m2.transformed_by<float>(V).scaled<float>(1. / gemmi::u_to_b());
        atoms[i]->b_iso = atoms[i]->aniso.trace() / 3.;
        k+= 6;
      } else
        atoms[i]->b_iso = std::max(x[k++], min_b);

    // occ
    for (int i : param_to_atom(Type::Q))
      atoms[i]->occ = gemmi::clamp(x[k++], 1e-3, 1.); // max occ depends!

    // dfrac
    for (int i : param_to_atom(Type::D))
      atoms[i]->fraction = gemmi::clamp(x[k++], 1e-3, 1.);

    assert(k == n_params());
  }
};

} // namespace servalcat
#endif
