// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <gemmi/symmetry.hpp>
#include <vector>
namespace py = pybind11;

struct TwinData {
  std::vector<gemmi::Op::Miller> asu;
  std::vector<double> alphas;

  // References
  // this may be slow. should we use 1d array and access via function?
  std::vector<std::vector<size_t>> rb2o; // [i_block][i_obs] -> index of iobs
  std::vector<std::vector<size_t>> rb2a; // [i_block][i] -> index of asu (for P(F; Fc))
  std::vector<std::vector<std::vector<size_t>>> rbo2a; // [i_block][i_obs][i] -> index of asu
  std::vector<std::vector<std::vector<size_t>>> rbo2c; // [i_block][i_obs][i] -> index of alphas

  void clear() {
    asu.clear();
    alphas.clear();
    rb2o.clear();
    rb2a.clear();
    rbo2a.clear();
    rbo2c.clear();
  }

  void setup(const std::vector<gemmi::Op::Miller> &hkls,
             const gemmi::SpaceGroup &sg,
             const std::vector<gemmi::Op> &operators) {
    clear();
    const gemmi::GroupOps gops = sg.operations();
    const gemmi::ReciprocalAsu rasu(&sg);
    auto apply_and_asu = [&rasu, &gops](const gemmi::Op &op, const gemmi::Miller &h) {
                           return rasu.to_asu(op.apply_to_hkl(h), gops).first;
                         };
    alphas.assign(operators.size() + 1, 0.);
    // Set asu
    for (const auto &h : hkls) {
      // assuming hkl is in ASU - but may not be complete?
      asu.push_back(h);
      for (const auto &op : operators)
        asu.push_back(apply_and_asu(op, h));
    }
    std::sort(asu.begin(), asu.end());
    asu.erase(std::unique(asu.begin(), asu.end()), asu.end());
    auto idx_of_asu = [&](const gemmi::Op::Miller &h) {
                        auto it = std::lower_bound(asu.begin(), asu.end(), h);
                        if (it != asu.end() && *it == h)
                          return std::distance(asu.begin(), it);
                        throw std::runtime_error("hkl not found in asu");
                      };
    // Permutation sort
    std::vector<int> perm(hkls.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](int lhs, int rhs) {return hkls[lhs] < hkls[rhs];});
    // Loop over hkls
    std::vector<bool> done(hkls.size());
    for (int i = 0; i < perm.size(); ++i) {
      const auto &h = hkls[perm[i]];
      if (done[perm[i]]) continue;
      rbo2a.emplace_back();
      rbo2c.emplace_back();
      rb2o.emplace_back();
      rb2a.emplace_back(1, idx_of_asu(h));
      // loop over same hkls (would not happen if unique set was given)
      for (int j = i; j < perm.size() && hkls[perm[j]] == h; ++j) {
        rb2o.back().push_back(perm[j]);
        done[perm[j]] = true;
      }
      // loop over twin related
      for (const auto &op : operators) {
        const auto hr = apply_and_asu(op, h);
        for (auto it = std::lower_bound(perm.begin(), perm.end(), hr,
                                        [&](int lhs, const gemmi::Miller &rhs) {return hkls[lhs] < rhs;});
             it != perm.end() && hkls[*it] == hr; ++it) {
          size_t j = std::distance(perm.begin(), it);
          if (done[perm[j]]) continue;
          rb2o.back().push_back(perm[j]);
          done[perm[j]] = true;
        }
        rb2a.back().push_back(idx_of_asu(hr));
      }
      std::sort(rb2a.back().begin(), rb2a.back().end());
      rb2a.back().erase(std::unique(rb2a.back().begin(), rb2a.back().end()), rb2a.back().end());
      for (auto j : rb2o.back()) {
        const auto &h2 = hkls[j];
        rbo2a.back().emplace_back(1, idx_of_asu(h2));
        rbo2c.back().emplace_back(1, 0);
        for (int k = 0; k < operators.size(); ++k) {
          const auto h2r = apply_and_asu(operators[k], h2);
          rbo2a.back().back().push_back(idx_of_asu(h2r));
          rbo2c.back().back().push_back(k + 1);
        }
      }
    }
  }
};

void add_twin(py::module& m) {
  py::class_<TwinData>(m, "TwinData")
    .def(py::init<>())
    .def_readonly("rb2o", &TwinData::rb2o)
    .def_readonly("rb2a", &TwinData::rb2a)
    .def_readonly("rbo2a", &TwinData::rbo2a)
    .def_readonly("rbo2c", &TwinData::rbo2c)
    .def_readonly("asu", &TwinData::asu)
    .def_readonly("alphas", &TwinData::alphas)
    .def("setup", [](TwinData &self, py::array_t<int> hkl,
                     const gemmi::SpaceGroup &sg, const std::vector<gemmi::Op> &operators) {
                    auto h = hkl.mutable_unchecked<2>();
                    if (h.shape(1) < 3)
                      throw std::domain_error("error: the size of the second dimension < 3");
                    std::vector<gemmi::Op::Miller> hkls;
                    hkls.reserve(h.shape(0));
                    for (py::ssize_t i = 0; i < h.shape(0); ++i)
                      hkls.push_back({h(i, 0), h(i, 1), h(i, 2)});
                    self.setup(hkls, sg, operators);
                  })
    .def("pairs", [](const TwinData &self, int i_op) {
                    if (i_op < 0 || i_op >= self.alphas.size())
                      throw std::runtime_error("bad i_op");
                    std::vector<std::array<int, 2>> idxes;
                    idxes.reserve(self.rb2o.size());
                    for (int ib = 0; ib < self.rb2o.size(); ++ib)
                      for (int io = 0; io < self.rb2o[ib].size(); ++io)
                        for (int io2 = io+1; io2 < self.rb2o[ib].size(); ++io2)
                          if (self.rbo2a[ib][io2][0] == self.rbo2a[ib][io][i_op+1] &&
                              self.rb2o[ib][io] != self.rb2o[ib][io2])
                            idxes.push_back({self.rb2o[ib][io], self.rb2o[ib][io2]});
                    return idxes;
                  })
    ;
}
