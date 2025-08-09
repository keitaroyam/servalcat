// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_REFINE_PARAMS_HPP_
#define SERVALCAT_REFINE_PARAMS_HPP_

#include <bitset>
#include <set>
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
  static std::string type2str(Type t) {
    switch (t) {
    case Type::X: return "xyz";
    case Type::B: return "adp";
    case Type::Q: return "occ";
    case Type::D: return "dfrac";
    default: gemmi::fail("type2str: bad t");
    }
  }
  static constexpr size_t N = static_cast<size_t>(Type::END);
  static constexpr std::array<Type, N> Types = {Type::X, Type::B, Type::Q, Type::D};
  static std::bitset<N> make_flag(bool refine_xyz, bool refine_adp, bool refine_occ, bool refine_dfrac) {
    std::bitset<N> flag;
    if (refine_xyz)
      flag.set(type2num(Type::X));
    if (refine_adp)
      flag.set(type2num(Type::B));
    if (refine_occ)
      flag.set(type2num(Type::Q));
    if (refine_dfrac)
      flag.set(type2num(Type::D));
    return flag;
  }

  bool aniso;
  bool use_q_b_mixed_derivatives;
  std::vector<gemmi::Atom*> atoms;
  std::array<std::vector<int>, N> atom_to_param_; // atom index to parameter index (-1 indicates fixed atoms)
  std::array<std::vector<int>, N> param_to_atom_; // parameter index to atom index
  std::array<std::vector<int>, N> pairs_refine_; // should we have param_to_atom equivalent for this?
  std::array<size_t, N> npairs_refine_{};
  std::vector<int> bq_mix_atoms; // B-Q
  std::vector<int> bd_mix_atoms; // B-D
  std::vector<int> qd_mix_atoms; // Q-D
  std::array<std::set<int>, N> ll_exclusion; // set of atom indices
  std::array<std::set<int>, N> geom_exclusion; // set of atom indices
  std::vector<std::vector<int>> occ_groups;  // vector of vector of atom indices
  std::vector<std::pair<bool, std::vector<size_t>>> occ_group_constraints; // vector of pair(is_complete, group_indices)
  RefineParams(bool use_aniso, bool use_q_b_mixed)
    : aniso(use_aniso), use_q_b_mixed_derivatives(use_q_b_mixed) { }
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
    case Type::D: return 0; // no restraints for deuterium fractions
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
    case Type::D: return 0; // no restraints for deuterium fractions
    default: gemmi::fail("nfisher_geom_per_pair: bad t");
    }
  }
  size_t n_refined_atoms(Type t) const { // not atoms, params actually
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
    for (const int i : atom_to_param(t))
      if (i >= 0)
        return true;
    return false;
  }
  bool is_refined_any() const { // will any parameter be refined
    for (Type t : Types)
      if (is_refined(t))
        return true;
    return false;
  }
  bool is_atom_refined(size_t idx, Type t) const {
    return atom_to_param(t)[idx] >= 0;
  }
  bool is_atom_refined(size_t idx) const { // for any t
    for (Type t : Types)
      if (atom_to_param(t)[idx] >= 0)
        return true;
    return false;
  }
  bool is_excluded_ll(size_t idx, Type t) const {
    return ll_exclusion[type2num(t)].count(idx) > 0;
  }
  bool is_atom_excluded_ll(size_t idx) const { // for all t
    for (Type t : Types)
      if (!is_excluded_ll(idx, t))
        return false;
    return true;
  }
  bool is_excluded_geom(size_t idx, Type t) const {
    return geom_exclusion[type2num(t)].count(idx) > 0;
  }
  void add_ll_exclusion(size_t idx, Type t) {
    ll_exclusion[type2num(t)].insert(idx);
  }
  void add_ll_exclusion(size_t idx) {
    for (Type t : Types)
      if (is_atom_refined(idx, t))
        add_ll_exclusion(idx, t);
  }
  void add_geom_exclusion(size_t idx, Type t) {
    geom_exclusion[type2num(t)].insert(idx);
  }
  void set_pairs(Type t, const std::vector<std::pair<int,int>> &pairs) {
    if (!is_refined(t))
      gemmi::fail("set_pairs: bad t");
    auto &pp = pairs_refine(t);
    const auto &atp = atom_to_param(t);
    pp.clear();
    pp.assign(pairs.size(), -1);
    size_t count = 0;
    // TODO for group constrained Q, add only single pair
    for (int i = 0; i < pairs.size(); ++i) {
      const auto &p = pairs[i];
      if (atp[p.first] >= 0 && atp[p.second] >= 0)
        pp[i] = count++;
    }
    npairs_refine(t) = count;
  }
  int get_pos_vec(int atom_idx, Type t) const {
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
  int get_pos_vec_geom(int atom_idx, Type t) const {
    if (is_excluded_geom(atom_idx, t))
      return -1;
    return get_pos_vec(atom_idx, t);
  }
  int get_pos_vec_ll(int atom_idx, Type t) const {
    if (is_excluded_ll(atom_idx, t))
      return -1;
    return get_pos_vec(atom_idx, t);
  }
  int get_pos_mat_geom(int atom_idx, Type t) const {
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
  int get_pos_mat_ll(int idx, Type t) const { // should check is_excluded_ll()?
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
      atom_to_param(t).assign(atoms.size(), -1);
    }
  }
  void add_atom_to_params(const gemmi::Atom *atom, std::bitset<N> flag = {~0ULL}) {
    const int idx = atom->serial - 1;
    const int group_idx = [&]() {
      for (int i = 0; i < occ_groups.size(); ++i) {
        const auto it = std::find(occ_groups[i].begin(), occ_groups[i].end(), idx);
        if (it != occ_groups[i].end())
          return i;
      }
      return -1;
    }();
    for (Type t : Types) {
      if (flag.test(type2num(t)) &&
          (t != Type::D || atom->is_hydrogen())) {
        if (t == Type::Q && group_idx >= 0) {
          // if the atom belongs to the occupancy group,
          // atom_to_param should point the same indexes
          // and param_to_atom should point the first atom of the group
          // so the occ_groups should be vector<vector<int>>

          // check if any atom of the group is registered
          const int found = [&](){
            for (int i = 0; i < occ_groups[group_idx].size(); ++i)
              if (atom_to_param(t)[occ_groups[group_idx][i]] >= 0)
                return occ_groups[group_idx][i]; // return atom index
            return -1;
          }();
          if (found < 0) { // this is the first atom to be registered
            atom_to_param(t)[idx] = param_to_atom(t).size();
            param_to_atom(t).push_back(idx);
          } else
            atom_to_param(t)[idx] = atom_to_param(t)[found];
        } else {
          atom_to_param(t)[idx] = param_to_atom(t).size();
          param_to_atom(t).push_back(idx);
        }
      }
    }
    if (use_q_b_mixed_derivatives && flag.test(type2num(Type::B)) && flag.test(type2num(Type::Q)))
      bq_mix_atoms.push_back(idx);
    if (use_q_b_mixed_derivatives && flag.test(type2num(Type::B)) && flag.test(type2num(Type::D))
        && atom->is_hydrogen())
      bd_mix_atoms.push_back(idx);
    if (flag.test(type2num(Type::Q)) && flag.test(type2num(Type::D)) && atom->is_hydrogen())
      qd_mix_atoms.push_back(idx);
  }
  void set_params(bool refine_xyz, bool refine_adp, bool refine_occ, bool refine_dfrac) {
    clear_params();
    const auto flag = make_flag(refine_xyz, refine_adp, refine_occ, refine_dfrac);
    for (const auto atom : atoms)
      add_atom_to_params(atom, flag);
  }
  void set_params_selected(const std::vector<int> &indices,
                           bool refine_xyz, bool refine_adp, bool refine_occ, bool refine_dfrac) {
    clear_params();
    const auto flag = make_flag(refine_xyz, refine_adp, refine_occ, refine_dfrac);
    for (int j : indices)
      add_atom_to_params(atoms[j], flag);
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
    if (x.size() != n_params())
      gemmi::fail("RefineParams::set_x: wrong x size ", std::to_string(x.size()), " ", std::to_string(n_params()));
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
        atoms[i]->b_iso = atoms[i]->aniso.trace() / 3. * gemmi::u_to_b();
        k+= 6;
      } else
        atoms[i]->b_iso = std::max(x[k++], min_b);

    // occ
    for (int i : param_to_atom(Type::Q)) {
      const double occ = gemmi::clamp(x[k++], 1e-3, 1.); // max occ depends!
      // check groups
      for (const auto &gr : occ_groups)
        if (std::find(gr.begin(), gr.end(), i) != gr.end())
          for (const auto j : gr)
            atoms[j]->occ = occ;
      atoms[i]->occ = occ;
    }

    // dfrac
    for (int i : param_to_atom(Type::D))
      atoms[i]->fraction = gemmi::clamp(x[k++], 0., 1.);

    if (k != n_params())
      gemmi::fail("RefineParams::set_x: wrong k ", std::to_string(k), " ", std::to_string(n_params()));
  }
  std::vector<std::vector<double>> constrained_occ_values() const {
    std::vector<std::vector<double>> ret;
    for (auto const &con : occ_group_constraints) {
      ret.emplace_back();
      for (size_t j : con.second)
        if (!occ_groups[j].empty())
          ret.back().push_back(atoms[occ_groups[j].front()]->occ);
    }
    return ret;
  }
  std::vector<double> occ_constraints() const { // constraint violations
    const size_t n_consts = occ_group_constraints.size();
    std::vector<double> ret(n_consts, 0.);
    for (int i = 0; i < n_consts; ++i) {
      const bool is_comp = occ_group_constraints[i].first;
      const auto &group_idxes = occ_group_constraints[i].second;
      double sum_occ = 0.;
      for (size_t j : group_idxes)
        if (!occ_groups[j].empty()) {
          const int atom_idx = occ_groups[j].front();
          sum_occ += atoms[atom_idx]->occ;
        }
      if (is_comp || sum_occ > 1)
        ret[i] = sum_occ - 1;
    }
    return ret;
  }
  void ensure_occ_constraints() {
    const size_t n_consts = occ_group_constraints.size();
    for (int i = 0; i < n_consts; ++i) {
      const bool is_comp = occ_group_constraints[i].first;
      const auto &group_idxes = occ_group_constraints[i].second;
      double sum_occ = 0.;
      for (size_t j : group_idxes)
        if (!occ_groups[j].empty()) {
          const int atom_idx = occ_groups[j].front();
          sum_occ += atoms[atom_idx]->occ;
        }
      const double fac = (is_comp || sum_occ > 1) ? 1./sum_occ : 1.;
      for (size_t j : group_idxes)
        for (int ia : occ_groups[j])
          atoms[ia]->occ *= fac;
    }
  }
};

} // namespace servalcat
#endif
