// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include "refine/geom.hpp"    // for Geometry
#include "refine/ll.hpp"      // for LL
#include "refine/cgsolve.hpp" // for CgSolve
#include "refine/ncsr.hpp"    // for
#include <gemmi/it92.hpp>
#include <gemmi/neutron92.hpp>
#include <gemmi/monlib.hpp>
#include <gemmi/unitcell.hpp>
#include <gemmi/model.hpp>

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>  // for detail::pythonbuf
#include <pybind11/eigen.h>
namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal
using namespace servalcat;

PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Bond>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Bond::Value>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Angle>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Angle::Value>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Torsion>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Torsion::Value>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Chirality>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Plane>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Interval>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Stacking>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Harmonic>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Special>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Vdw>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Ncsr>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Reporting::bond_reporting_t>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Reporting::angle_reporting_t>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Reporting::torsion_reporting_t>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Reporting::chiral_reporting_t>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Reporting::plane_reporting_t>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Reporting::stacking_reporting_t>)
PYBIND11_MAKE_OPAQUE(std::vector<Geometry::Reporting::vdw_reporting_t>)
PYBIND11_MAKE_OPAQUE(std::vector<NcsList::Ncs>)

py::tuple precondition_eigen_coo(py::array_t<double> am, py::array_t<int> rows,
                                 py::array_t<int> cols, int N, double cutoff) {
  int* colp = (int*) cols.request().ptr;
  int* rowp = (int*) rows.request().ptr;
  double* amp = (double*) am.request().ptr;
  auto len = cols.shape(0);

  //std::vector<gemmi::SMat33<double>> blocks(N);
  std::vector<double> blocks(2*N);
  for(int i = 0; i < len; ++i) {
    const int c = colp[i], r = rowp[i];
    const int b = c % 3, j = c / 3;
    int k;
    if (r < c - b || r > c) continue;
    if (c == r) k = b;
    else if (b == 1 && r == c - 1) k = 3;
    else if (b == 2 && r == c - 2) k = 4;
    else k = 5; //if (b == 2 && r == c - 1) k = 5;
    blocks[j*6+k] = amp[i];
  }

  std::vector<double> ret(N * 3);
  std::vector<int> retrow(N * 3), retcol(N * 3);
  for (int i = 0; i < N / 3; ++i) {
    const gemmi::SMat33<double> m{blocks[i*6], blocks[i*6+1], blocks[i*6+2], blocks[i*6+3], blocks[i*6+4], blocks[i*6+5]};
    const gemmi::Mat33 pinv = eigen_decomp_inv(m, cutoff, true);
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        ret[i*9+3*j+k] = pinv[j][k];
        retrow[i*9+3*j+k] = 3 * i + j;
        retcol[i*9+3*j+k] = 3 * i + k;
      }
  }

  return py::make_tuple(ret, py::make_tuple(retrow, retcol));
}

void add_refine(py::module& m) {
  py::class_<GeomTarget> geomtarget(m, "GeomTarget");
  py::class_<Geometry> geom(m, "Geometry");

  py::class_<Geometry::Reporting>(geom, "Reporting")
    .def_readonly("bonds", &Geometry::Reporting::bonds)
    .def_readonly("angles", &Geometry::Reporting::angles)
    .def_readonly("torsions", &Geometry::Reporting::torsions)
    .def_readonly("chirs", &Geometry::Reporting::chirs)
    .def_readonly("planes", &Geometry::Reporting::planes)
    .def_readonly("stackings", &Geometry::Reporting::stackings)
    .def_readonly("vdws", &Geometry::Reporting::vdws)
    .def_readonly("adps", &Geometry::Reporting::adps)
    .def_readonly("occs", &Geometry::Reporting::occs)
    .def("get_summary_table", [](const Geometry::Reporting& self, bool use_nucleus) {
      std::vector<std::string> keys;
      std::vector<int> nrest;
      std::vector<double> rmsd, rmsz, msigma;
      auto append = [&](const std::string& k, const std::vector<double>& delsq,
                        const std::vector<double>& zsq, const std::vector<double> &sigmas) {
        keys.emplace_back(k);
        nrest.push_back(delsq.size());
        rmsd.push_back(std::sqrt(std::accumulate(delsq.begin(), delsq.end(), 0.) / nrest.back()));
        rmsz.push_back(std::sqrt(std::accumulate(zsq.begin(), zsq.end(), 0.) / nrest.back()));
        msigma.push_back(std::accumulate(sigmas.begin(), sigmas.end(), 0.) / nrest.back());
      };
      // Bond
      std::map<int, std::vector<double>> delsq, zsq, sigmas;
      for (const auto& b : self.bonds) {
        const auto& restr = std::get<0>(b);
        const auto& val = std::get<1>(b);
        const double sigma = use_nucleus ? val->sigma_nucleus : val->sigma;
        const double d2 = gemmi::sq(std::get<2>(b)), z2 = gemmi::sq(std::get<2>(b) / sigma);
        const int k = (restr->type == 2 ? 2 :
                       (restr->atoms[0]->is_hydrogen() || restr->atoms[1]->is_hydrogen()) ? 1 : 0);
        delsq[k].push_back(d2);
        zsq[k].push_back(z2);
        sigmas[k].push_back(sigma);
      }
      for (const auto& p : delsq)
        if (!p.second.empty())
          append(p.first == 2 ? "External distances" :
                 p.first == 1 ? "Bond distances, H" :
                 "Bond distances, non H", p.second, zsq[p.first], sigmas[p.first]);

      // Angle
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& a : self.angles) {
        const auto& restr = std::get<0>(a);
        const auto& val = std::get<1>(a);
        const double d2 = gemmi::sq(std::get<2>(a)), z2 = gemmi::sq(std::get<2>(a) / val->sigma);
        const int k = (restr->atoms[0]->is_hydrogen() || restr->atoms[1]->is_hydrogen() || restr->atoms[2]->is_hydrogen()) ? 1 : 0;
        delsq[k].push_back(d2);
        zsq[k].push_back(z2);
        sigmas[k].push_back(val->sigma);
      }
      for (const auto& p : delsq)
        if (!p.second.empty())
          append(p.first == 1 ? "Bond angles, H" : "Bond angles, non H",
                 p.second, zsq[p.first], sigmas[p.first]);

      // Torsion
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& t : self.torsions) {
        const auto& val = std::get<1>(t);
        const double d2 = gemmi::sq(std::get<2>(t)), z2 = gemmi::sq(std::get<2>(t) / val->sigma);
        const int period = std::max(1, val->period);
        delsq[period].push_back(d2);
        zsq[period].push_back(z2);
        sigmas[period].push_back(val->sigma);
      }
      for (const auto& p : delsq)
        if (!p.second.empty())
          append("Torsion angles, period " + std::to_string(p.first), p.second, zsq[p.first], sigmas[p.first]);

      // Chiral
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& c : self.chirs) {
        const auto& val = std::get<0>(c);
        const double d2 = gemmi::sq(std::get<1>(c)), z2 = gemmi::sq(std::get<1>(c) / val->sigma);
        delsq[0].push_back(d2);
        zsq[0].push_back(z2);
        sigmas[0].push_back(val->sigma);
      }
      if (!delsq[0].empty())
        append("Chiral centres", delsq[0], zsq[0], sigmas[0]);

      // Plane
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& p : self.planes) {
        const auto& val = std::get<0>(p);
        for (double d : std::get<1>(p)) {
          const double d2 = gemmi::sq(d), z2 = gemmi::sq(d / val->sigma);
          delsq[0].push_back(d2);
          zsq[0].push_back(z2);
          sigmas[0].push_back(val->sigma);
        }
      }
      if (!delsq[0].empty())
        append("Planar groups", delsq[0], zsq[0], sigmas[0]);

      // Stack
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& s : self.stackings) {
        const auto& restr = std::get<0>(s);
        const double da2 = gemmi::sq(std::get<1>(s)), za2 = gemmi::sq(std::get<1>(s) / restr->sd_angle);
        const double dd2 = 0.5 * (gemmi::sq(std::get<2>(s)) + gemmi::sq(std::get<3>(s)));
        const double zd2 = dd2 / gemmi::sq(restr->sd_dist);
        delsq[0].push_back(da2);
        zsq[0].push_back(za2);
        sigmas[0].push_back(restr->sd_angle);
        delsq[1].push_back(dd2);
        zsq[1].push_back(zd2);
        sigmas[1].push_back(restr->sd_dist);
      }
      if (!delsq[0].empty())
        append("Stacking angles", delsq[0], zsq[0], sigmas[0]);
      if (!delsq[1].empty())
        append("Stacking distances", delsq[1], zsq[1], sigmas[1]);

      // VDW
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& v : self.vdws) {
        const auto& restr = std::get<0>(v);
        const double d2 = gemmi::sq(std::get<1>(v)), z2 = gemmi::sq(std::get<1>(v) / restr->sigma);
        delsq[restr->type].push_back(d2);
        zsq[restr->type].push_back(z2);
        sigmas[restr->type].push_back(restr->sigma);
      }
      for (const auto& p : delsq)
        if (!p.second.empty()) {
          const int i = p.first > 6 ? p.first - 6 : p.first;
          append((i == 1 ? "VDW nonbonded" :
                  i == 2 ? "VDW torsion" :
                  i == 3 ? "VDW hbond" :
                  i == 4 ? "VDW metal" :
                  i == 5 ? "VDW dummy" :
                  "VDW dummy-dummy") + std::string(p.first > 6 ? ", symmetry" : ""),
                 p.second, zsq[p.first], sigmas[p.first]);
        }

      // NCSR
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& v : self.ncsrs) {
        const auto& restr = std::get<0>(v);
        const double delta = std::get<1>(v) - std::get<2>(v);
        const double d2 = gemmi::sq(delta), z2 = gemmi::sq(delta / restr->sigma);
        delsq[restr->idx].push_back(d2);
        zsq[restr->idx].push_back(z2);
        sigmas[restr->idx].push_back(restr->sigma);
      }
      for (const auto& p : delsq)
        if (!p.second.empty()) {
          append("ncsr local: group " + std::to_string(p.first + 1),
                 p.second, zsq[p.first], sigmas[p.first]);
        }

      // ADP
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& a : self.adps) {
        const int rkind = std::get<2>(a);
        const float delta_b = std::get<5>(a);
        const float sigma = std::get<4>(a);
        const double d2 = gemmi::sq(delta_b), z2 = gemmi::sq(delta_b / sigma);
        const int k = std::min(rkind, 3);
        delsq[k].push_back(d2);
        zsq[k].push_back(z2);
        sigmas[k].push_back(sigma);
      }
      for (const auto& p : delsq)
        if (!p.second.empty())
          append(p.first == 1 ? "B values (bond)" :
                 p.first == 2 ? "B values (angle)" :
                 "B values (others)", p.second, zsq[p.first], sigmas[p.first]);

      // Occupancies
      delsq.clear(); zsq.clear(); sigmas.clear();
      for (const auto& a : self.occs) {
        const int rkind = std::get<2>(a);
        const float delta_b = std::get<5>(a);
        const float sigma = std::get<4>(a);
        const double d2 = gemmi::sq(delta_b), z2 = gemmi::sq(delta_b / sigma);
        const int k = std::min(rkind, 3);
        delsq[k].push_back(d2);
        zsq[k].push_back(z2);
        sigmas[k].push_back(sigma);
      }
      for (const auto& p : delsq)
        if (!p.second.empty())
          append(p.first == 1 ? "Occupancies (bond)" :
                 p.first == 2 ? "Occupancies (angle)" :
                 "Occupancies (others)", p.second, zsq[p.first], sigmas[p.first]);

      return py::dict("Restraint type"_a=keys, "N restraints"_a=nrest,
                      "r.m.s.d."_a=rmsd, "r.m.s.Z"_a=rmsz, "Mn(sigma)"_a=msigma);
    })
    .def("get_bond_outliers", [](const Geometry::Reporting& self, bool use_nucleus, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, zs, alphas;
      std::vector<int> types;
      for (const auto& b : self.bonds) {
        const auto& restr = std::get<0>(b);
        const auto& val = std::get<1>(b);
        const double ideal = use_nucleus ? val->value_nucleus : val->value;
        const double sigma = use_nucleus ? val->sigma_nucleus : val->sigma;
        const double z = std::get<2>(b) / sigma; // value - ideal
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->atoms[0]);
          atom2.push_back(restr->atoms[1]);
          values.push_back(std::get<2>(b) + ideal);
          ideals.push_back(ideal);
          zs.push_back(z);
          types.push_back(restr->type);
          alphas.push_back(restr->alpha);
        }
      }
      return py::dict("atom1"_a=atom1, "atom2"_a=atom2, "value"_a=values,
                      "ideal"_a=ideals, "z"_a=zs, "type"_a=types, "alpha"_a=alphas);
    }, py::arg("use_nucleus"), py::arg("min_z"))
    .def("get_angle_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3;
      std::vector<double> values, ideals, zs;
      for (const auto& t : self.angles) {
        const auto& restr = std::get<0>(t);
        const auto& val = std::get<1>(t);
        const double z = std::get<2>(t) / val->sigma; // value - ideal
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->atoms[0]);
          atom2.push_back(restr->atoms[1]);
          atom3.push_back(restr->atoms[2]);
          values.push_back(std::get<2>(t) + val->value);
          ideals.push_back(val->value);
          zs.push_back(z);
        }
      }
      return py::dict("atom1"_a=atom1, "atom2"_a=atom2, "atom3"_a=atom3,
                      "value"_a=values, "ideal"_a=ideals, "z"_a=zs);
    }, py::arg("min_z"))
    .def("get_torsion_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3, atom4;
      std::vector<double> values, ideals, zs;
      std::vector<int> pers;
      std::vector<std::string> labels;
      for (const auto& t : self.torsions) {
        const auto& restr = std::get<0>(t);
        const auto& val = std::get<1>(t);
        const double z = std::get<2>(t) / val->sigma; // value - ideal
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->atoms[0]);
          atom2.push_back(restr->atoms[1]);
          atom3.push_back(restr->atoms[2]);
          atom4.push_back(restr->atoms[3]);
          labels.push_back(val->label);
          values.push_back(std::get<3>(t));
          ideals.push_back(val->value);
          pers.push_back(val->period);
          zs.push_back(z);
        }
      }
      return py::dict("label"_a=labels, "atom1"_a=atom1, "atom2"_a=atom2, "atom3"_a=atom3, "atom4"_a=atom4,
                      "value"_a=values, "ideal"_a=ideals, "per"_a=pers, "z"_a=zs);
    }, py::arg("min_z"))
    .def("get_chiral_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3, atom4;
      std::vector<double> values, ideals, zs;
      std::vector<bool> signs;
      for (const auto& t : self.chirs) {
        const auto& restr = std::get<0>(t);
        const double z = std::get<1>(t) / restr->sigma; // value - ideal
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->atoms[0]);
          atom2.push_back(restr->atoms[1]);
          atom3.push_back(restr->atoms[2]);
          atom4.push_back(restr->atoms[3]);
          values.push_back(std::get<1>(t) + std::get<2>(t));
          ideals.push_back(std::get<2>(t));
          signs.push_back(restr->sign == gemmi::ChiralityType::Both);
          zs.push_back(z);
        }
      }
      return py::dict("atomc"_a=atom1, "atom1"_a=atom2, "atom2"_a=atom3, "atom3"_a=atom4,
                      "value"_a=values, "ideal"_a=ideals, "both"_a=signs, "z"_a=zs);
    }, py::arg("min_z"))
    .def("get_plane_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atoms;
      std::vector<double> values, zs;
      std::vector<std::string> labels;
      for (const auto& t : self.planes) {
        const auto& restr = std::get<0>(t);
        for (size_t i = 0; i < restr->atoms.size(); ++i) {
          const double z = std::get<1>(t)[i] / restr->sigma;
          if (std::abs(z) >= min_z) {
            atoms.push_back(restr->atoms[i]);
            labels.push_back(restr->label);
            values.push_back(std::get<1>(t)[i]);
            zs.push_back(z);
          }
        }
      }
      return py::dict("label"_a=labels, "atom"_a=atoms, "dev"_a=values, "z"_a=zs);
    }, py::arg("min_z"))
    .def("get_stacking_angle_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, zs;
      for (const auto& t : self.stackings) {
        const auto& restr = std::get<0>(t);
        const double za = std::get<1>(t) / restr->sd_angle;
        if (std::abs(za) >= min_z) {
          atom1.push_back(restr->planes[0][0]); // report only first atom
          atom2.push_back(restr->planes[1][0]);
          values.push_back(std::get<1>(t) + restr->angle);
          ideals.push_back(restr->angle);
          zs.push_back(za);
        }
      }
      return py::dict("plane1"_a=atom1, "plane2"_a=atom2, "value"_a=values, "ideal"_a=ideals, "z"_a=zs);
    }, py::arg("min_z"))
    .def("get_stacking_dist_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, zs;
      for (const auto& t : self.stackings) {
        const auto& restr = std::get<0>(t);
        const double zd1 = std::get<2>(t) / restr->sd_dist;
        const double zd2 = std::get<3>(t) / restr->sd_dist;
        if (std::min(std::abs(zd1), std::abs(zd2)) >= min_z) {
          const double zd = std::abs(zd1) > std::abs(zd2) ? zd1 : zd2;
          atom1.push_back(restr->planes[0][0]); // report only first atom
          atom2.push_back(restr->planes[1][0]);
          values.push_back(zd * restr->sd_dist + restr->dist);
          ideals.push_back(restr->dist);
          zs.push_back(zd);
        }
      }
      return py::dict("plane1"_a=atom1, "plane2"_a=atom2, "value"_a=values, "ideal"_a=ideals,  "z"_a=zs);
    }, py::arg("min_z"))
    .def("get_vdw_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, zs;
      std::vector<int> types;
      for (const auto& t : self.vdws) {
        const auto& restr = std::get<0>(t);
        const double z = std::get<1>(t) / restr->sigma;
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->atoms[0]);
          atom2.push_back(restr->atoms[1]);
          values.push_back(std::get<1>(t) + restr->value);
          ideals.push_back(restr->value);
          zs.push_back(z);
          types.push_back(restr->type);
        }
      }
      return py::dict("atom1"_a=atom1, "atom2"_a=atom2, "value"_a=values,
                      "ideal"_a=ideals, "z"_a=zs, "type"_a=types);
    }, py::arg("min_z"))
    .def("get_ncsr_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3, atom4;
      std::vector<double> dist1, dist2, devs, zs;
      for (const auto& t : self.ncsrs) {
        const auto& restr = std::get<0>(t);
        const double d1 = std::get<1>(t), d2 = std::get<2>(t);
        const double z = (d1 - d2) / restr->sigma;
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->pairs[0]->atoms[0]);
          atom2.push_back(restr->pairs[0]->atoms[1]);
          atom3.push_back(restr->pairs[1]->atoms[0]);
          atom4.push_back(restr->pairs[1]->atoms[1]);
          dist1.push_back(d1);
          dist2.push_back(d2);
          devs.push_back(d1 - d2);
          zs.push_back(z);
        }
      }
      return py::dict("1_atom1"_a=atom1, "1_atom2"_a=atom2, "2_atom1"_a=atom3, "2_atom2"_a=atom4,
                      "dist_1"_a=dist1, "dist_2"_a=dist2, "del_dist"_a=devs, "z"_a=zs);
    }, py::arg("min_z"))
    .def("per_atom_score", [](const Geometry::Reporting& self, int n_atoms,
                              bool use_nucleus, const std::string& metric) {
      if (metric != "max" && metric != "mean" && metric != "sum")
        gemmi::fail("invalid metric");
      const int imet = metric == "max" ? 0 : metric == "sum" ? 1 : 2;
      std::vector<std::vector<double>> ret(7);
      std::vector<std::vector<int>> num(7);
      for (auto& v : ret)
        v.assign(n_atoms, 0.);
      for (auto& v : num)
        v.assign(n_atoms, 0);

      auto add_residual = [&](int idx, int i, double x) {
        if (i < 0 || i >= n_atoms) gemmi::fail("invalid atom index");
        if (imet == 0) // max
          ret[idx][i] = std::max(ret[idx][i], std::abs(x));
        else if (imet == 1) // sum
          ret[idx][i] += std::abs(x);
        else { // mean
          ret[idx][i] += gemmi::sq(x);
          num[idx][i] += 1;
        }
      };
      auto add = [&](int i, double x, int idx) {
        add_residual(0, i, x); // total
        add_residual(idx, i, x);
      };

      for (const auto& t : self.bonds) { // check restr->type?
        const auto& restr = std::get<0>(t);
        const auto& val = std::get<1>(t);
        const double sigma = use_nucleus ? val->sigma_nucleus : val->sigma;
        for (const auto& a : restr->atoms)
          add(a->serial - 1, std::get<2>(t) / sigma, 1);
      }
      for (const auto& t : self.angles) {
        const auto& restr = std::get<0>(t);
        const auto& val = std::get<1>(t);
        for (const auto& a : restr->atoms)
          add(a->serial - 1, std::get<2>(t) / val->sigma, 2);
      }
      for (const auto& t : self.torsions) {
        const auto& restr = std::get<0>(t);
        const auto& val = std::get<1>(t);
        for (const auto& a : restr->atoms)
          add(a->serial - 1, std::get<2>(t) / val->sigma, 3);
      }
      for (const auto& t : self.chirs) {
        const auto& restr = std::get<0>(t);
        for (const auto& a : restr->atoms)
          add(a->serial - 1, std::get<1>(t) / restr->sigma, 4);
      }
      for (const auto& t : self.planes) {
        const auto& restr = std::get<0>(t);
        for (size_t i = 0; i < restr->atoms.size(); ++i)
          add(restr->atoms[i]->serial - 1, std::get<1>(t)[i] / restr->sigma, 5);
      }
      // include stac?
      for (const auto& t : self.vdws) {
        const auto& restr = std::get<0>(t);
        for (const auto& a : restr->atoms)
          add(a->serial - 1, std::get<1>(t) / restr->sigma, 6);
      }
      if (imet == 2) { // mean
        for (size_t j = 0; j < ret.size(); ++j)
          for (size_t i = 0; i < ret[j].size(); ++i)
            ret[j][i] = std::sqrt(ret[j][i] / num[j][i]);
      }
      return py::dict("total"_a=ret[0], "bonds"_a=ret[1], "angles"_a=ret[2],
                      "torsions"_a=ret[3], "chirs"_a=ret[4], "planes"_a=ret[5],
                      "vdws"_a=ret[6]);
    })
    ;
  py::class_<Geometry::Bond> bond(geom, "Bond");
  py::class_<Geometry::Angle> angle(geom, "Angle");
  py::class_<Geometry::Torsion> torsion(geom, "Torsion");
  py::class_<Geometry::Chirality> chirality(geom, "Chirality");
  py::class_<Geometry::Plane> plane(geom, "Plane");
  py::class_<Geometry::Vdw> vdw(geom, "Vdw");
  py::class_<Geometry::Ncsr> ncsr(geom, "Ncsr");
  py::class_<Geometry::Bond::Value>(bond, "Value")
    .def(py::init<double,double,double,double>())
    .def_readwrite("value", &Geometry::Bond::Value::value)
    .def_readwrite("sigma", &Geometry::Bond::Value::sigma)
    .def_readwrite("value_nucleus", &Geometry::Bond::Value::value_nucleus)
    .def_readwrite("sigma_nucleus", &Geometry::Bond::Value::sigma_nucleus)
    ;
  py::class_<Geometry::Angle::Value>(angle, "Value")
    .def(py::init<double,double>())
    .def_readwrite("value", &Geometry::Angle::Value::value)
    .def_readwrite("sigma", &Geometry::Angle::Value::sigma)
    ;
  py::class_<Geometry::Torsion::Value>(torsion, "Value")
    .def(py::init<double,double,int>())
    .def_readwrite("value", &Geometry::Torsion::Value::value)
    .def_readwrite("sigma", &Geometry::Torsion::Value::sigma)
    .def_readwrite("period", &Geometry::Torsion::Value::period)
    .def_readwrite("label", &Geometry::Torsion::Value::label)
    ;
  bond
    .def(py::init<gemmi::Atom*,gemmi::Atom*>())
    .def("set_image", &Geometry::Bond::set_image)
    .def_readwrite("type", &Geometry::Bond::type)
    .def_readwrite("alpha", &Geometry::Bond::alpha)
    .def_readwrite("sym_idx", &Geometry::Bond::sym_idx)
    .def_readwrite("pbc_shift", &Geometry::Bond::pbc_shift)
    .def_readwrite("atoms", &Geometry::Bond::atoms)
    .def_readwrite("values", &Geometry::Bond::values)
    ;
  angle
    .def(py::init<gemmi::Atom*,gemmi::Atom*,gemmi::Atom*>())
    .def_readwrite("atoms", &Geometry::Angle::atoms)
    .def_readwrite("values", &Geometry::Angle::values)
    ;
  torsion
    .def(py::init<gemmi::Atom*,gemmi::Atom*,gemmi::Atom*,gemmi::Atom*>())
    .def_readwrite("atoms", &Geometry::Torsion::atoms)
    .def_readwrite("values", &Geometry::Torsion::values)
    ;
  chirality
    .def(py::init<gemmi::Atom*,gemmi::Atom*,gemmi::Atom*,gemmi::Atom*>())
    .def_readwrite("value", &Geometry::Chirality::value)
    .def_readwrite("sigma", &Geometry::Chirality::sigma)
    .def_readwrite("sign", &Geometry::Chirality::sign)
    .def_readwrite("atoms", &Geometry::Chirality::atoms)
    ;
  plane
    .def(py::init<std::vector<gemmi::Atom*>>())
    .def_readwrite("sigma", &Geometry::Plane::sigma)
    .def_readwrite("label", &Geometry::Plane::label)
    .def_readwrite("atoms", &Geometry::Plane::atoms)
    ;
  py::class_<Geometry::Interval>(geom, "Interval")
    .def(py::init<gemmi::Atom*,gemmi::Atom*>())
    .def_readwrite("dmin", &Geometry::Interval::dmin)
    .def_readwrite("dmax", &Geometry::Interval::dmax)
    .def_readwrite("smin", &Geometry::Interval::smin)
    .def_readwrite("smax", &Geometry::Interval::smax)
    .def_readwrite("atoms", &Geometry::Interval::atoms)
    ;
  py::class_<Geometry::Harmonic>(geom, "Harmonic")
    .def(py::init<gemmi::Atom*>())
    .def_readwrite("sigma", &Geometry::Harmonic::sigma)
    .def_readwrite("atom", &Geometry::Harmonic::atom)
    ;
  py::class_<Geometry::Special>(geom, "Special")
    .def(py::init<gemmi::Atom*, const Geometry::Special::Mat33&, const Geometry::Special::Mat66&, int>())
    .def_readwrite("Rspec_pos", &Geometry::Special::Rspec_pos)
    .def_readwrite("Rspec_aniso", &Geometry::Special::Rspec_aniso)
    .def_readwrite("n_mult", &Geometry::Special::n_mult)
    .def_readwrite("atom", &Geometry::Special::atom)
    ;
  py::class_<Geometry::Stacking>(geom, "Stacking")
    .def(py::init<std::vector<gemmi::Atom*>,std::vector<gemmi::Atom*>>())
    .def_readwrite("dist", &Geometry::Stacking::dist)
    .def_readwrite("sd_dist", &Geometry::Stacking::sd_dist)
    .def_readwrite("angle", &Geometry::Stacking::angle)
    .def_readwrite("sd_angle", &Geometry::Stacking::sd_angle)
    .def_readwrite("planes", &Geometry::Stacking::planes)
    ;
  vdw
    .def(py::init<gemmi::Atom*,gemmi::Atom*>())
    .def("set_image", &Geometry::Vdw::set_image)
    .def("same_asu", &Geometry::Vdw::same_asu)
    .def_readwrite("type", &Geometry::Vdw::type)
    .def_readwrite("value", &Geometry::Vdw::value)
    .def_readwrite("sigma", &Geometry::Vdw::sigma)
    .def_readwrite("sym_idx", &Geometry::Vdw::sym_idx)
    .def_readwrite("pbc_shift", &Geometry::Vdw::pbc_shift)
    .def_readwrite("atoms", &Geometry::Vdw::atoms)
    ;
  ncsr
    .def(py::init<const Geometry::Vdw*, const Geometry::Vdw*, int>())
    .def_readwrite("pairs", &Geometry::Ncsr::pairs)
    .def_readwrite("alpha", &Geometry::Ncsr::alpha)
    .def_readwrite("sigma", &Geometry::Ncsr::sigma)
    ;

  py::bind_vector<std::vector<Geometry::Reporting::bond_reporting_t>>(geom, "ReportingBonds");
  py::bind_vector<std::vector<Geometry::Reporting::angle_reporting_t>>(geom, "ReportingAngles");
  py::bind_vector<std::vector<Geometry::Reporting::torsion_reporting_t>>(geom, "ReportingTorsions");
  py::bind_vector<std::vector<Geometry::Reporting::chiral_reporting_t>>(geom, "ReportingChirals");
  py::bind_vector<std::vector<Geometry::Reporting::plane_reporting_t>>(geom, "ReportingPlanes");
  py::bind_vector<std::vector<Geometry::Reporting::stacking_reporting_t>>(geom, "ReportingStackings");
  py::bind_vector<std::vector<Geometry::Reporting::vdw_reporting_t>>(geom, "ReportingVdws");
  py::bind_vector<std::vector<Geometry::Reporting::ncsr_reporting_t>>(geom, "ReportingNcsrs");
  py::bind_vector<std::vector<Geometry::Bond>>(geom, "Bonds");
  py::bind_vector<std::vector<Geometry::Angle>>(geom, "Angles");
  py::bind_vector<std::vector<Geometry::Chirality>>(geom, "Chiralitys");
  py::bind_vector<std::vector<Geometry::Torsion>>(geom, "Torsions");
  py::bind_vector<std::vector<Geometry::Plane>>(geom, "Planes");
  py::bind_vector<std::vector<Geometry::Interval>>(geom, "Intervals");
  py::bind_vector<std::vector<Geometry::Stacking>>(geom, "Stackings");
  py::bind_vector<std::vector<Geometry::Harmonic>>(geom, "Harmonics");
  py::bind_vector<std::vector<Geometry::Special>>(geom, "Specials");
  py::bind_vector<std::vector<Geometry::Vdw>>(geom, "Vdws");
  py::bind_vector<std::vector<Geometry::Ncsr>>(geom, "Ncsrs");
  py::bind_vector<std::vector<Geometry::Bond::Value>>(bond, "Values");
  py::bind_vector<std::vector<Geometry::Angle::Value>>(angle, "Values");
  py::bind_vector<std::vector<Geometry::Torsion::Value>>(torsion, "Values");

  geomtarget
    .def_readonly("target", &GeomTarget::target)
    .def_readonly("vn", &GeomTarget::vn)
    .def_readonly("am", &GeomTarget::am)
    .def_property_readonly("am_spmat", &GeomTarget::make_spmat)
  ;
  geom
    .def(py::init<gemmi::Structure&, const std::vector<int> &, const gemmi::EnerLib*>(),
         py::arg("st"), py::arg("atom_pos"), py::arg("ener_lib")=nullptr)
    .def_readonly("bonds", &Geometry::bonds)
    .def_readonly("angles", &Geometry::angles)
    .def_readonly("chirs", &Geometry::chirs)
    .def_readonly("torsions", &Geometry::torsions)
    .def_readonly("planes", &Geometry::planes)
    .def_readonly("intervals", &Geometry::intervals)
    .def_readonly("stackings", &Geometry::stackings)
    .def_readonly("harmonics", &Geometry::harmonics)
    .def_readonly("specials", &Geometry::specials)
    .def_readonly("vdws", &Geometry::vdws)
    .def_readonly("ncsrs", &Geometry::ncsrs)
    .def_readonly("target", &Geometry::target)
    .def_readonly("reporting", &Geometry::reporting)
    .def("load_topo", &Geometry::load_topo)
    .def("finalize_restraints", &Geometry::finalize_restraints)
    .def("setup_target", &Geometry::setup_target)
    .def("clear_target", &Geometry::clear_target)
    .def("setup_nonbonded", &Geometry::setup_nonbonded,
         py::arg("skip_critical_dist")=false,
         py::arg("group_idxes")=std::vector<int>{})
    .def("setup_ncsr", &Geometry::setup_ncsr)
    .def("calc", &Geometry::calc, py::arg("use_nucleus"), py::arg("check_only"),
         py::arg("wbond")=1, py::arg("wangle")=1, py::arg("wtors")=1,
         py::arg("wchir")=1, py::arg("wplane")=1, py::arg("wstack")=1, py::arg("wvdw")=1,
         py::arg("wncs")=1)
    .def("calc_adp_restraint", &Geometry::calc_adp_restraint)
    .def("calc_occ_restraint", &Geometry::calc_occ_restraint)
    .def("spec_correction", &Geometry::spec_correction, py::arg("alpha")=1e-3, py::arg("use_rr")=true)
    .def_readonly("bondindex", &Geometry::bondindex)
    // vdw parameters
    .def_readwrite("vdw_sdi_vdw", &Geometry::vdw_sdi_vdw)
    .def_readwrite("vdw_sdi_torsion", &Geometry::vdw_sdi_torsion)
    .def_readwrite("vdw_sdi_hbond", &Geometry::vdw_sdi_hbond)
    .def_readwrite("vdw_sdi_metal", &Geometry::vdw_sdi_metal)
    .def_readwrite("hbond_dinc_ad", &Geometry::hbond_dinc_ad)
    .def_readwrite("hbond_dinc_ah", &Geometry::hbond_dinc_ah)
    .def_readwrite("dinc_torsion_o", &Geometry::dinc_torsion_o)
    .def_readwrite("dinc_torsion_n", &Geometry::dinc_torsion_n)
    .def_readwrite("dinc_torsion_c", &Geometry::dinc_torsion_c)
    .def_readwrite("dinc_torsion_all", &Geometry::dinc_torsion_all)
    .def_readwrite("dinc_dummy", &Geometry::dinc_dummy)
    .def_readwrite("vdw_sdi_dummy", &Geometry::vdw_sdi_dummy)
    .def_readwrite("max_vdw_radius", &Geometry::max_vdw_radius)
    // torsion parameters
    .def_readwrite("use_hydr_tors", &Geometry::use_hydr_tors)
    .def_readwrite("link_tors_names", &Geometry::link_tors_names)
    .def_readwrite("mon_tors_names", &Geometry::mon_tors_names)
    // ADP restraint parameters
    .def_readwrite("adpr_max_dist", &Geometry::adpr_max_dist)
    .def_readwrite("adpr_d_power", &Geometry::adpr_d_power)
    .def_readwrite("adpr_exp_fac", &Geometry::adpr_exp_fac)
    .def_readwrite("adpr_long_range", &Geometry::adpr_long_range)
    .def_readwrite("adpr_kl_sigs", &Geometry::adpr_kl_sigs)
    .def_readwrite("adpr_diff_sigs", &Geometry::adpr_diff_sigs)
    .def_property("adpr_mode",
                  [](const Geometry& self) {
                    switch(self.adpr_mode) {
                      case 0: return "diff";
                      case 1: return "kldiv";
                    }
                    return "unknown"; // should not happen
                  },
                  [](Geometry& self, const std::string& mode) {
                    if (mode == "diff")
                      self.adpr_mode = 0;
                    else if (mode == "kldiv")
                      self.adpr_mode = 1;
                    else
                      throw std::runtime_error("unknown adpr mode");
                  })
    // Occupancy restraint parameters
    .def_readwrite("occr_max_dist", &Geometry::occr_max_dist)
    .def_readwrite("occr_long_range", &Geometry::occr_long_range)
    .def_readwrite("occr_sigs", &Geometry::occr_sigs)
    // jelly body parameters
    .def_readwrite("ridge_dmax", &Geometry::ridge_dmax)
    .def_readwrite("ridge_sigma", &Geometry::ridge_sigma)
    .def_readwrite("ridge_symm", &Geometry::ridge_symm)
    .def_readwrite("ridge_exclude_short_dist", &Geometry::ridge_exclude_short_dist)
    // NCS restraint parameters
    .def_readwrite("ncsr_alpha", &Geometry::ncsr_alpha)
    .def_readwrite("ncsr_sigma", &Geometry::ncsr_sigma)
    .def_readwrite("ncsr_diff_cutoff", &Geometry::ncsr_diff_cutoff)
  ;

  py::class_<TableS3>(m, "TableS3")
    .def(py::init<double, double>(), py::arg("d_min"), py::arg("d_max"))
    .def_readonly("s3_values", &TableS3::s3_values)
    .def_readonly("y_values", &TableS3::y_values)
    .def("make_table",&TableS3::make_table)
    .def("get_value", &TableS3::get_value)
    .def_readwrite("maxbin", &TableS3::maxbin)
    ;
  py::class_<LL>(m, "LL")
    .def(py::init<const gemmi::Structure &, const std::vector<int> &, bool, bool, int, bool, bool>(),
         py::arg("st"), py::arg("atom_pos"), py::arg("mott_bethe"),
         py::arg("refine_xyz"), py::arg("adp_mode"), py::arg("refine_occ"), py::arg("refine_h"))
    .def("set_ncs", &LL::set_ncs)
    .def("calc_grad_it92", &LL::calc_grad<gemmi::IT92<double>>)
    .def("calc_grad_n92", &LL::calc_grad<gemmi::Neutron92<double>>)
    .def("make_fisher_table_diag_fast_it92", &LL::make_fisher_table_diag_fast<gemmi::IT92<double>>)
    .def("make_fisher_table_diag_fast_n92", &LL::make_fisher_table_diag_fast<gemmi::Neutron92<double>>)
    .def("make_fisher_table_diag_direct_it92", &LL::make_fisher_table_diag_direct<gemmi::IT92<double>>)
    .def("make_fisher_table_diag_direct_n92", &LL::make_fisher_table_diag_direct<gemmi::Neutron92<double>>)
    .def("fisher_diag_from_table_it92", &LL::fisher_diag_from_table<gemmi::IT92<double>>)
    .def("fisher_diag_from_table_n92", &LL::fisher_diag_from_table<gemmi::Neutron92<double>>)
    .def("spec_correction", &LL::spec_correction,
         py::arg("specials"), py::arg("alpha")=1e-3, py::arg("use_rr")=true)
    .def_property_readonly("fisher_spmat", &LL::make_spmat)
    .def_readonly("table_bs", &LL::table_bs)
    .def_readonly("pp1", &LL::pp1)
    .def_readonly("bb", &LL::bb)
    .def_readonly("aa", &LL::aa)
    .def_readonly("vn", &LL::vn)
    .def_readonly("am", &LL::am)
    ;
  m.def("precondition_eigen_coo", &precondition_eigen_coo);
  py::class_<CgSolve>(m, "CgSolve")
    .def(py::init<const GeomTarget *, const LL *>(),
         py::arg("geom"), py::arg("ll")=nullptr)
    .def("solve", [](CgSolve &self, double weight, const py::object& pystream,
                     bool use_ic) {
      std::ostream os(nullptr);
      std::unique_ptr<py::detail::pythonbuf> buffer;
      buffer.reset(new py::detail::pythonbuf(pystream));
      os.rdbuf(buffer.get());
      if (use_ic)
        return self.solve<Eigen::IncompleteCholesky<double>>(weight, &os);
      else
        return self.solve<>(weight, &os);
    })
    .def_readwrite("gamma", &CgSolve::gamma)
    .def_readwrite("toler", &CgSolve::toler)
    .def_readwrite("ncycle", &CgSolve::ncycle)
    .def_readwrite("max_gamma_cyc", &CgSolve::max_gamma_cyc)
    ;

  m.def("smooth_gauss", [](py::array_t<double> bin_centers, py::array_t<double> bin_values,
                           py::array_t<double> s_array, int n, double kernel_width) {
    // assume all values are sorted by resolution
    if (bin_centers.size() != (size_t)bin_values.shape(0)) throw std::runtime_error("bin_centers and bin_values shape mismatch");
    if (s_array.ndim() != 1) throw std::runtime_error("s_array dimension != 1");
    if (n < 1) throw std::runtime_error("non positive n");
    const double krn2 = gemmi::sq(kernel_width) * 2;
    const size_t n_par = bin_values.shape(1);
    const size_t n_bin = bin_values.shape(0);
    const size_t n_ref = s_array.size();
    auto s_ = s_array.unchecked<1>();
    auto bin_cen_ = bin_centers.unchecked<1>();
    auto bin_val_ = bin_values.unchecked<2>();
    auto ret = py::array_t<double>({n_ref, n_par});
    double* ptr = (double*) ret.request().ptr;

    // setup new (finer) binning
    const double s_step = (s_(n_ref-1) - s_(0)) / n;
    std::vector<std::vector<double>> smoothed(n_par);
    for (int i = 0; i < n_par; ++i) smoothed[i].resize(n);
    for (int i = 0; i < n; ++i) {
      const double s_i = s_(0) + s_step * (i + 0.5);
      double an = 0.;
      for (int j = 0; j < n_bin; ++j) {
        double dx = gemmi::sq(s_i - bin_cen_(j)) / krn2;
        if (dx > 30) continue; // skip small contribution
        double expdx = std::exp(-dx);
        an += expdx;
        for (int k = 0; k < n_par; ++k)
          smoothed[k][i] += expdx * bin_val_(j, k);
      }
      for (int k = 0; k < n_par; ++k)
        smoothed[k][i] /= an; // FIXME an may be zero
    }

    // apply to array with the nearest neighbour
    for (int i = 0; i < n_ref; ++i) {
      int nearest = std::max(0, std::min(n - 1, static_cast<int>((s_(i) - s_(0)) / s_step)));
      for (int j = 0; j < n_par; ++j)
        ptr[i * n_par + j] = smoothed[j][nearest];
    }
    return ret;
  });


  py::class_<NcsList> ncslist(m, "NcsList");
  py::class_<NcsList::Ncs>(ncslist, "Ncs")
    .def(py::init([](const gemmi::AlignmentResult &al, const gemmi::ResidueSpan &fixed, const gemmi::ResidueSpan &movable) {
      return NcsList::Ncs(al, fixed, movable);
    }))
    .def("calculate_local_rms", &NcsList::Ncs::calculate_local_rms)
    .def_readonly("atoms", &NcsList::Ncs::atoms)
    .def_readonly("seqids", &NcsList::Ncs::seqids)
    .def_readonly("n_atoms", &NcsList::Ncs::n_atoms)
    .def_readonly("local_rms", &NcsList::Ncs::local_rms)
    ;
  ncslist
    .def(py::init<>())
    .def("set_pairs", &NcsList::set_pairs)
    .def_readonly("ncss", &NcsList::ncss)
    .def_readonly("all_pairs", &NcsList::all_pairs)
    ;
  py::bind_vector<std::vector<NcsList::Ncs>>(ncslist, "Ncs_vector");
}
