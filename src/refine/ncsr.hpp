// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_REFINE_NCSR_HPP_
#define SERVALCAT_REFINE_NCSR_HPP_

#include <set>
#include <gemmi/model.hpp>    // for Structure, Atom
#include <gemmi/seqalign.hpp> // for AlignmentResult
#include "../math.hpp"
#include <Eigen/Dense>

namespace servalcat {

struct NcsList {
  struct Ncs {
    Ncs(const gemmi::AlignmentResult &al,
        gemmi::ConstResidueSpan fixed, gemmi::ConstResidueSpan movable,
        const std::string &chain_fixed, const std::string &chain_movable)
      : chains(std::make_pair(chain_fixed, chain_movable)) {
      auto it1 = fixed.first_conformer().begin();
      auto it2 = movable.first_conformer().begin();
      n_atoms.push_back(0);
      for (const auto &item : al.cigar) {
        char op = item.op();
        for (uint32_t i = 0; i < item.len(); ++i) {
          if (op == 'M' && it1->name == it2->name) {
            for (const gemmi::Atom& a1 : it1->atoms)
              if (const gemmi::Atom* a2 = it2->find_atom(a1.name, a1.altloc, a1.element))
                atoms.emplace_back(&a1, a2);
            seqids.emplace_back(it1->seqid, it2->seqid);
            n_atoms.push_back(atoms.size());
          }
          if (op == 'M' || op == 'I')
            ++it1;
          if (op == 'M' || op == 'D')
            ++it2;
        }
      }
    }

    void calculate_local_rms(int nlen) {
      local_rms.clear();
      if (seqids.size() < nlen)
        return;
      for (int i = 0; i < seqids.size() - nlen; ++i) {
        const int n = n_atoms.at(i + nlen) - n_atoms.at(i);
        Eigen::MatrixXd x(n, 3), y(n, 3);
        for (int j = 0, m = 0; j < nlen; ++j)
          for (int k = n_atoms.at(i+j); k < n_atoms.at(i+j+1); ++k, ++m)
            for (int l = 0; l < 3; ++l) {
              x(m, l) = atoms[k].first->pos.at(l);
              y(m, l) = atoms[k].second->pos.at(l);
            }
        local_rms.push_back(procrust_dist(x, y));
      }
    }

    std::vector<std::pair<const gemmi::Atom*, const gemmi::Atom*>> atoms;
    std::vector<std::pair<gemmi::SeqId, gemmi::SeqId>> seqids;
    std::pair<std::string, std::string> chains;
    std::vector<size_t> n_atoms;
    std::vector<double> local_rms;
  };

  void set_pairs() {
    all_pairs.clear();
    for (const auto &ncs : ncss) {
      all_pairs.emplace_back();
      for (const auto &p : ncs.atoms)
        all_pairs.back().emplace(p.first, p.second);
    }
  }

  std::vector<Ncs> ncss;
  std::vector<std::map<const gemmi::Atom*, const gemmi::Atom*>> all_pairs;
};
} // namespace servalcat
#endif
