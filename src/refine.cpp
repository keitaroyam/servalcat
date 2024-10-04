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

#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/sparse.h>
namespace nb = nanobind;
using namespace nanobind::literals; // to bring in the `_a` literal
using namespace servalcat;

NB_MAKE_OPAQUE(std::vector<Geometry::Bond>)
NB_MAKE_OPAQUE(std::vector<Geometry::Bond::Value>)
NB_MAKE_OPAQUE(std::vector<Geometry::Angle>)
NB_MAKE_OPAQUE(std::vector<Geometry::Angle::Value>)
NB_MAKE_OPAQUE(std::vector<Geometry::Torsion>)
NB_MAKE_OPAQUE(std::vector<Geometry::Torsion::Value>)
NB_MAKE_OPAQUE(std::vector<Geometry::Chirality>)
NB_MAKE_OPAQUE(std::vector<Geometry::Plane>)
NB_MAKE_OPAQUE(std::vector<Geometry::Interval>)
NB_MAKE_OPAQUE(std::vector<Geometry::Stacking>)
NB_MAKE_OPAQUE(std::vector<Geometry::Harmonic>)
NB_MAKE_OPAQUE(std::vector<Geometry::Special>)
NB_MAKE_OPAQUE(std::vector<Geometry::Vdw>)
NB_MAKE_OPAQUE(std::vector<Geometry::Ncsr>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::bond_reporting_t>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::angle_reporting_t>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::torsion_reporting_t>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::chiral_reporting_t>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::plane_reporting_t>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::stacking_reporting_t>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::vdw_reporting_t>)
NB_MAKE_OPAQUE(std::vector<Geometry::Reporting::ncsr_reporting_t>)
NB_MAKE_OPAQUE(std::vector<NcsList::Ncs>)

#if 0
nb::tuple precondition_eigen_coo(py::array_t<double> am, py::array_t<int> rows,
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

  return nb::make_tuple(ret, nb::make_tuple(retrow, retcol));
}
#endif

void add_refine(nb::module_& m) {
  nb::class_<GeomTarget> geomtarget(m, "GeomTarget");
  nb::class_<Geometry> geom(m, "Geometry");

  nb::class_<Geometry::Bond> bond(geom, "Bond");
  nb::class_<Geometry::Angle> angle(geom, "Angle");
  nb::class_<Geometry::Torsion> torsion(geom, "Torsion");
  nb::class_<Geometry::Chirality> chirality(geom, "Chirality");
  nb::class_<Geometry::Plane> plane(geom, "Plane");
  nb::class_<Geometry::Vdw> vdw(geom, "Vdw");
  nb::class_<Geometry::Ncsr> ncsr(geom, "Ncsr");

  nb::class_<Geometry::Reporting>(geom, "Reporting")
    .def_ro("bonds", &Geometry::Reporting::bonds)
    .def_ro("angles", &Geometry::Reporting::angles)
    .def_ro("torsions", &Geometry::Reporting::torsions)
    .def_ro("chirs", &Geometry::Reporting::chirs)
    .def_ro("planes", &Geometry::Reporting::planes)
    .def_ro("stackings", &Geometry::Reporting::stackings)
    .def_ro("vdws", &Geometry::Reporting::vdws)
    .def_ro("adps", &Geometry::Reporting::adps)
    .def_ro("occs", &Geometry::Reporting::occs)
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

      nb::dict d;
      d["Restraint type"] = keys;
      d["N restraints"] = nrest;
      d["r.m.s.d."] = rmsd;
      d["r.m.s.Z"] = rmsz;
      d["Mn(sigma)"]=msigma;
      return d;
    })
#if 0
    .def("get_bond_outliers", [](const Geometry::Reporting& self, bool use_nucleus, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, sigmas, zs, alphas;
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
          sigmas.push_back(sigma);
          zs.push_back(z);
          types.push_back(restr->type);
          alphas.push_back(restr->alpha);
        }
      }
      return nb::dict("atom1"_a=atom1, "atom2"_a=atom2, "value"_a=values,
                      "ideal"_a=ideals, "sigma"_a=sigmas, "z"_a=zs, "type"_a=types, "alpha"_a=alphas);
    }, nb::arg("use_nucleus"), nb::arg("min_z"))
    .def("get_angle_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3;
      std::vector<double> values, ideals, sigmas, zs;
      for (const auto& t : self.angles) {
        const auto& restr = std::get<0>(t);
        const auto& val = std::get<1>(t);
        const double z = std::get<2>(t) / val->sigma; // value - ideal
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->atoms[0]);
          atom2.push_back(restr->atoms[1]);
          atom3.push_back(restr->atoms[2]);
          values.push_back(std::get<2>(t) + val->value);
          sigmas.push_back(val->sigma);
          ideals.push_back(val->value);
          zs.push_back(z);
        }
      }
      return nb::dict("atom1"_a=atom1, "atom2"_a=atom2, "atom3"_a=atom3,
                      "value"_a=values, "ideal"_a=ideals, "sigma"_a=sigmas, "z"_a=zs);
    }, nb::arg("min_z"))
    .def("get_torsion_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3, atom4;
      std::vector<double> values, ideals, sigmas, zs;
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
          sigmas.push_back(val->sigma);
          pers.push_back(val->period);
          zs.push_back(z);
        }
      }
      return nb::dict("label"_a=labels, "atom1"_a=atom1, "atom2"_a=atom2, "atom3"_a=atom3, "atom4"_a=atom4,
                      "value"_a=values, "ideal"_a=ideals, "sigma"_a=sigmas, "per"_a=pers, "z"_a=zs);
    }, nb::arg("min_z"))
    .def("get_chiral_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3, atom4;
      std::vector<double> values, ideals, sigmas, zs;
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
          sigmas.push_back(restr->sigma);
          signs.push_back(restr->sign == gemmi::ChiralityType::Both);
          zs.push_back(z);
        }
      }
      return nb::dict("atomc"_a=atom1, "atom1"_a=atom2, "atom2"_a=atom3, "atom3"_a=atom4,
                      "value"_a=values, "ideal"_a=ideals, "sigma"_a=sigmas, "both"_a=signs, "z"_a=zs);
    }, nb::arg("min_z"))
    .def("get_plane_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atoms;
      std::vector<double> values, sigmas, zs;
      std::vector<std::string> labels;
      for (const auto& t : self.planes) {
        const auto& restr = std::get<0>(t);
        for (size_t i = 0; i < restr->atoms.size(); ++i) {
          const double z = std::get<1>(t)[i] / restr->sigma;
          if (std::abs(z) >= min_z) {
            atoms.push_back(restr->atoms[i]);
            labels.push_back(restr->label);
            values.push_back(std::get<1>(t)[i]);
            sigmas.push_back(restr->sigma);
            zs.push_back(z);
          }
        }
      }
      return nb::dict("label"_a=labels, "atom"_a=atoms, "dev"_a=values, "sigma"_a=sigmas, "z"_a=zs);
    }, nb::arg("min_z"))
    .def("get_stacking_angle_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, sigmas, zs;
      for (const auto& t : self.stackings) {
        const auto& restr = std::get<0>(t);
        const double za = std::get<1>(t) / restr->sd_angle;
        if (std::abs(za) >= min_z) {
          atom1.push_back(restr->planes[0][0]); // report only first atom
          atom2.push_back(restr->planes[1][0]);
          values.push_back(std::get<1>(t) + restr->angle);
          ideals.push_back(restr->angle);
          sigmas.push_back(restr->sd_angle);
          zs.push_back(za);
        }
      }
      return nb::dict("plane1"_a=atom1, "plane2"_a=atom2, "value"_a=values,
                      "ideal"_a=ideals, "sigma"_a=sigmas, "z"_a=zs);
    }, nb::arg("min_z"))
    .def("get_stacking_dist_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, sigmas, zs;
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
          sigmas.push_back(restr->sd_dist);
          zs.push_back(zd);
        }
      }
      return nb::dict("plane1"_a=atom1, "plane2"_a=atom2, "value"_a=values,
                      "ideal"_a=ideals, "sigma"_a=sigmas, "z"_a=zs);
    }, nb::arg("min_z"))
    .def("get_vdw_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2;
      std::vector<double> values, ideals, sigmas, zs;
      std::vector<int> types;
      for (const auto& t : self.vdws) {
        const auto& restr = std::get<0>(t);
        const double z = std::get<1>(t) / restr->sigma;
        if (std::abs(z) >= min_z) {
          atom1.push_back(restr->atoms[0]);
          atom2.push_back(restr->atoms[1]);
          values.push_back(std::get<1>(t) + restr->value);
          ideals.push_back(restr->value);
          sigmas.push_back(restr->sigma);
          zs.push_back(z);
          types.push_back(restr->type);
        }
      }
      return nb::dict("atom1"_a=atom1, "atom2"_a=atom2, "value"_a=values,
                      "ideal"_a=ideals, "sigma"_a=sigmas, "z"_a=zs, "type"_a=types);
    }, nb::arg("min_z"))
    .def("get_ncsr_outliers", [](const Geometry::Reporting& self, double min_z) {
      std::vector<const gemmi::Atom*> atom1, atom2, atom3, atom4;
      std::vector<double> dist1, dist2, devs, sigmas, zs;
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
          sigmas.push_back(restr->sigma);
          zs.push_back(z);
        }
      }
      return nb::dict("1_atom1"_a=atom1, "1_atom2"_a=atom2, "2_atom1"_a=atom3, "2_atom2"_a=atom4,
                      "dist_1"_a=dist1, "dist_2"_a=dist2, "del_dist"_a=devs, "sigma"_a=sigmas, "z"_a=zs);
    }, nb::arg("min_z"))
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
      return nb::dict("total"_a=ret[0], "bonds"_a=ret[1], "angles"_a=ret[2],
                      "torsions"_a=ret[3], "chirs"_a=ret[4], "planes"_a=ret[5],
                      "vdws"_a=ret[6]);
    })
#endif
    ;
  nb::class_<Geometry::Bond::Value>(bond, "Value")
    .def(nb::init<double,double,double,double>())
    .def_rw("value", &Geometry::Bond::Value::value)
    .def_rw("sigma", &Geometry::Bond::Value::sigma)
    .def_rw("value_nucleus", &Geometry::Bond::Value::value_nucleus)
    .def_rw("sigma_nucleus", &Geometry::Bond::Value::sigma_nucleus)
    ;
  nb::class_<Geometry::Angle::Value>(angle, "Value")
    .def(nb::init<double,double>())
    .def_rw("value", &Geometry::Angle::Value::value)
    .def_rw("sigma", &Geometry::Angle::Value::sigma)
    ;
  nb::class_<Geometry::Torsion::Value>(torsion, "Value")
    .def(nb::init<double,double,int>())
    .def_rw("value", &Geometry::Torsion::Value::value)
    .def_rw("sigma", &Geometry::Torsion::Value::sigma)
    .def_rw("period", &Geometry::Torsion::Value::period)
    .def_rw("label", &Geometry::Torsion::Value::label)
    ;
  bond
    .def(nb::init<gemmi::Atom*,gemmi::Atom*>())
    .def("set_image", &Geometry::Bond::set_image)
    .def_rw("type", &Geometry::Bond::type)
    .def_rw("alpha", &Geometry::Bond::alpha)
    .def_rw("sym_idx", &Geometry::Bond::sym_idx)
    .def_rw("pbc_shift", &Geometry::Bond::pbc_shift)
    .def_rw("atoms", &Geometry::Bond::atoms)
    .def_rw("values", &Geometry::Bond::values)
    ;
  angle
    .def(nb::init<gemmi::Atom*,gemmi::Atom*,gemmi::Atom*>())
    .def("set_images", &Geometry::Angle::set_images)
    .def_rw("sym_idx_1", &Geometry::Angle::sym_idx_1)
    .def_rw("sym_idx_2", &Geometry::Angle::sym_idx_2)
    .def_rw("pbc_shift_1", &Geometry::Angle::pbc_shift_1)
    .def_rw("pbc_shift_2", &Geometry::Angle::pbc_shift_2)
    .def_rw("atoms", &Geometry::Angle::atoms)
    .def_rw("values", &Geometry::Angle::values)
    ;
  torsion
    .def(nb::init<gemmi::Atom*,gemmi::Atom*,gemmi::Atom*,gemmi::Atom*>())
    .def_rw("atoms", &Geometry::Torsion::atoms)
    .def_rw("values", &Geometry::Torsion::values)
    ;
  chirality
    .def(nb::init<gemmi::Atom*,gemmi::Atom*,gemmi::Atom*,gemmi::Atom*>())
    .def_rw("value", &Geometry::Chirality::value)
    .def_rw("sigma", &Geometry::Chirality::sigma)
    .def_rw("sign", &Geometry::Chirality::sign)
    .def_rw("atoms", &Geometry::Chirality::atoms)
    ;
  plane
    .def(nb::init<std::vector<gemmi::Atom*>>())
    .def_rw("sigma", &Geometry::Plane::sigma)
    .def_rw("label", &Geometry::Plane::label)
    .def_rw("atoms", &Geometry::Plane::atoms)
    ;
  nb::class_<Geometry::Interval>(geom, "Interval")
    .def(nb::init<gemmi::Atom*,gemmi::Atom*>())
    .def_rw("dmin", &Geometry::Interval::dmin)
    .def_rw("dmax", &Geometry::Interval::dmax)
    .def_rw("smin", &Geometry::Interval::smin)
    .def_rw("smax", &Geometry::Interval::smax)
    .def_rw("atoms", &Geometry::Interval::atoms)
    ;
  nb::class_<Geometry::Harmonic>(geom, "Harmonic")
    .def(nb::init<gemmi::Atom*>())
    .def_rw("sigma", &Geometry::Harmonic::sigma)
    .def_rw("atom", &Geometry::Harmonic::atom)
    ;
  nb::class_<Geometry::Special>(geom, "Special")
    .def(nb::init<gemmi::Atom*, const Geometry::Special::Mat33&, const Geometry::Special::Mat66&, int>())
    .def_rw("Rspec_pos", &Geometry::Special::Rspec_pos)
    .def_rw("Rspec_aniso", &Geometry::Special::Rspec_aniso)
    .def_rw("n_mult", &Geometry::Special::n_mult)
    .def_rw("atom", &Geometry::Special::atom)
    ;
  nb::class_<Geometry::Stacking>(geom, "Stacking")
    .def(nb::init<std::vector<gemmi::Atom*>,std::vector<gemmi::Atom*>>())
    .def_rw("dist", &Geometry::Stacking::dist)
    .def_rw("sd_dist", &Geometry::Stacking::sd_dist)
    .def_rw("angle", &Geometry::Stacking::angle)
    .def_rw("sd_angle", &Geometry::Stacking::sd_angle)
    .def_rw("planes", &Geometry::Stacking::planes)
    ;
  vdw
    .def(nb::init<gemmi::Atom*,gemmi::Atom*>())
    .def("set_image", &Geometry::Vdw::set_image)
    .def("same_asu", &Geometry::Vdw::same_asu)
    .def_rw("type", &Geometry::Vdw::type)
    .def_rw("value", &Geometry::Vdw::value)
    .def_rw("sigma", &Geometry::Vdw::sigma)
    .def_rw("sym_idx", &Geometry::Vdw::sym_idx)
    .def_rw("pbc_shift", &Geometry::Vdw::pbc_shift)
    .def_rw("atoms", &Geometry::Vdw::atoms)
    ;
  ncsr
    .def(nb::init<const Geometry::Vdw*, const Geometry::Vdw*, int>())
    .def_rw("pairs", &Geometry::Ncsr::pairs)
    .def_rw("alpha", &Geometry::Ncsr::alpha)
    .def_rw("sigma", &Geometry::Ncsr::sigma)
    ;

  nb::bind_vector<std::vector<Geometry::Reporting::bond_reporting_t>>(geom, "ReportingBonds");
  nb::bind_vector<std::vector<Geometry::Reporting::angle_reporting_t>>(geom, "ReportingAngles");
  nb::bind_vector<std::vector<Geometry::Reporting::torsion_reporting_t>>(geom, "ReportingTorsions");
  nb::bind_vector<std::vector<Geometry::Reporting::chiral_reporting_t>>(geom, "ReportingChirals");
  nb::bind_vector<std::vector<Geometry::Reporting::plane_reporting_t>>(geom, "ReportingPlanes");
  nb::bind_vector<std::vector<Geometry::Reporting::stacking_reporting_t>>(geom, "ReportingStackings");
  nb::bind_vector<std::vector<Geometry::Reporting::vdw_reporting_t>>(geom, "ReportingVdws");
  nb::bind_vector<std::vector<Geometry::Reporting::ncsr_reporting_t>>(geom, "ReportingNcsrs");
  nb::bind_vector<std::vector<Geometry::Bond>>(geom, "Bonds");
  nb::bind_vector<std::vector<Geometry::Angle>>(geom, "Angles");
  nb::bind_vector<std::vector<Geometry::Chirality>>(geom, "Chiralitys");
  nb::bind_vector<std::vector<Geometry::Torsion>>(geom, "Torsions");
  nb::bind_vector<std::vector<Geometry::Plane>>(geom, "Planes");
  nb::bind_vector<std::vector<Geometry::Interval>>(geom, "Intervals");
  nb::bind_vector<std::vector<Geometry::Stacking>>(geom, "Stackings");
  nb::bind_vector<std::vector<Geometry::Harmonic>>(geom, "Harmonics");
  nb::bind_vector<std::vector<Geometry::Special>>(geom, "Specials");
  nb::bind_vector<std::vector<Geometry::Vdw>>(geom, "Vdws");
  nb::bind_vector<std::vector<Geometry::Ncsr>>(geom, "Ncsrs");
  nb::bind_vector<std::vector<Geometry::Bond::Value>>(bond, "Values");
  nb::bind_vector<std::vector<Geometry::Angle::Value>>(angle, "Values");
  nb::bind_vector<std::vector<Geometry::Torsion::Value>>(torsion, "Values");

  geomtarget
    .def_ro("target", &GeomTarget::target)
    .def_ro("vn", &GeomTarget::vn)
    .def_ro("am", &GeomTarget::am)
    .def_prop_ro("am_spmat", &GeomTarget::make_spmat)
    .def("n_atoms", &GeomTarget::n_atoms)
    .def("n_pairs", &GeomTarget::n_pairs)
  ;
  geom
    .def(nb::init<gemmi::Structure&, const std::vector<int> &, const gemmi::EnerLib*>(),
         nb::arg("st"), nb::arg("atom_pos"), nb::arg("ener_lib")=nb::none())
    .def_ro("bonds", &Geometry::bonds)
    .def_ro("angles", &Geometry::angles)
    .def_ro("chirs", &Geometry::chirs)
    .def_ro("torsions", &Geometry::torsions)
    .def_ro("planes", &Geometry::planes)
    .def_ro("intervals", &Geometry::intervals)
    .def_ro("stackings", &Geometry::stackings)
    .def_ro("harmonics", &Geometry::harmonics)
    .def_ro("specials", &Geometry::specials)
    .def_ro("vdws", &Geometry::vdws)
    .def_ro("ncsrs", &Geometry::ncsrs)
    .def_ro("target", &Geometry::target)
    .def_ro("reporting", &Geometry::reporting)
    .def("load_topo", &Geometry::load_topo)
    .def("finalize_restraints", &Geometry::finalize_restraints)
    .def("setup_target", &Geometry::setup_target)
    .def("clear_target", &Geometry::clear_target)
    .def("setup_nonbonded", &Geometry::setup_nonbonded,
         nb::arg("skip_critical_dist")=false,
         nb::arg("group_idxes")=std::vector<int>{})
    .def("setup_ncsr", &Geometry::setup_ncsr)
    .def("calc", &Geometry::calc, nb::arg("use_nucleus"), nb::arg("check_only"),
         nb::arg("wbond")=1, nb::arg("wangle")=1, nb::arg("wtors")=1,
         nb::arg("wchir")=1, nb::arg("wplane")=1, nb::arg("wstack")=1, nb::arg("wvdw")=1,
         nb::arg("wncs")=1)
    .def("calc_adp_restraint", &Geometry::calc_adp_restraint)
    .def("calc_occ_restraint", &Geometry::calc_occ_restraint)
    .def("spec_correction", &Geometry::spec_correction, nb::arg("alpha")=1e-3, nb::arg("use_rr")=true)
    .def_ro("bondindex", &Geometry::bondindex)
    // vdw parameters
    .def_rw("vdw_sdi_vdw", &Geometry::vdw_sdi_vdw)
    .def_rw("vdw_sdi_torsion", &Geometry::vdw_sdi_torsion)
    .def_rw("vdw_sdi_hbond", &Geometry::vdw_sdi_hbond)
    .def_rw("vdw_sdi_metal", &Geometry::vdw_sdi_metal)
    .def_rw("hbond_dinc_ad", &Geometry::hbond_dinc_ad)
    .def_rw("hbond_dinc_ah", &Geometry::hbond_dinc_ah)
    .def_rw("dinc_torsion_o", &Geometry::dinc_torsion_o)
    .def_rw("dinc_torsion_n", &Geometry::dinc_torsion_n)
    .def_rw("dinc_torsion_c", &Geometry::dinc_torsion_c)
    .def_rw("dinc_torsion_all", &Geometry::dinc_torsion_all)
    .def_rw("dinc_dummy", &Geometry::dinc_dummy)
    .def_rw("vdw_sdi_dummy", &Geometry::vdw_sdi_dummy)
    .def_rw("max_vdw_radius", &Geometry::max_vdw_radius)
    // angle parameters
    .def_rw("angle_von_mises", &Geometry::angle_von_mises)
    // torsion parameters
    .def_rw("use_hydr_tors", &Geometry::use_hydr_tors)
    .def_rw("link_tors_names", &Geometry::link_tors_names)
    .def_rw("mon_tors_names", &Geometry::mon_tors_names)
    // ADP restraint parameters
    .def_rw("adpr_max_dist", &Geometry::adpr_max_dist)
    .def_rw("adpr_d_power", &Geometry::adpr_d_power)
    .def_rw("adpr_exp_fac", &Geometry::adpr_exp_fac)
    .def_rw("adpr_long_range", &Geometry::adpr_long_range)
    .def_rw("adpr_kl_sigs", &Geometry::adpr_kl_sigs)
    .def_rw("adpr_diff_sigs", &Geometry::adpr_diff_sigs)
    .def_prop_rw("adpr_mode",
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
    .def_rw("occr_max_dist", &Geometry::occr_max_dist)
    .def_rw("occr_long_range", &Geometry::occr_long_range)
    .def_rw("occr_sigs", &Geometry::occr_sigs)
    // jelly body parameters
    .def_rw("ridge_dmax", &Geometry::ridge_dmax)
    .def_rw("ridge_sigma", &Geometry::ridge_sigma)
    .def_rw("ridge_symm", &Geometry::ridge_symm)
    .def_rw("ridge_exclude_short_dist", &Geometry::ridge_exclude_short_dist)
    // NCS restraint parameters
    .def_rw("ncsr_alpha", &Geometry::ncsr_alpha)
    .def_rw("ncsr_sigma", &Geometry::ncsr_sigma)
    .def_rw("ncsr_diff_cutoff", &Geometry::ncsr_diff_cutoff)
    .def_rw("ncsr_max_dist", &Geometry::ncsr_max_dist)
    ;

  nb::class_<TableS3>(m, "TableS3")
    .def(nb::init<double, double>(), nb::arg("d_min"), nb::arg("d_max"))
    .def_ro("s3_values", &TableS3::s3_values)
    .def_ro("y_values", &TableS3::y_values)
    .def("make_table",&TableS3::make_table)
    .def("get_value", &TableS3::get_value)
    .def_rw("maxbin", &TableS3::maxbin)
    ;
  nb::class_<LL>(m, "LL")
    .def(nb::init<const gemmi::Structure &, const std::vector<int> &, bool, bool, int, bool, bool>(),
         nb::arg("st"), nb::arg("atom_pos"), nb::arg("mott_bethe"),
         nb::arg("refine_xyz"), nb::arg("adp_mode"), nb::arg("refine_occ"), nb::arg("refine_h"))
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
         nb::arg("specials"), nb::arg("alpha")=1e-3, nb::arg("use_rr")=true)
    .def_prop_ro("fisher_spmat", &LL::make_spmat)
    .def_ro("table_bs", &LL::table_bs)
    .def_ro("pp1", &LL::pp1)
    .def_ro("bb", &LL::bb)
    .def_ro("aa", &LL::aa)
    .def_ro("vn", &LL::vn)
    .def_ro("am", &LL::am)
    .def_rw("use_q_b_mixed_derivatives", &LL::use_q_b_mixed_derivatives)
    ;
#if 0
  m.def("precondition_eigen_coo", &precondition_eigen_coo);
#endif
  nb::class_<CgSolve>(m, "CgSolve")
    .def(nb::init<const GeomTarget *, const LL *>(),
         nb::arg("geom"), nb::arg("ll")=nb::none())
    .def("solve", [](CgSolve &self, double weight, const nb::object& pystream,
                     bool use_ic) {
      std::ostream os(nullptr);
#if 0
      std::unique_ptr<py::detail::pythonbuf> buffer;
      buffer.reset(new py::detail::pythonbuf(pystream));
      os.rdbuf(buffer.get());
#endif
      if (use_ic)
        return self.solve<Eigen::IncompleteCholesky<double>>(weight, &os);
      else
        return self.solve<>(weight, &os);
    })
    .def_rw("gamma", &CgSolve::gamma)
    .def_rw("toler", &CgSolve::toler)
    .def_rw("ncycle", &CgSolve::ncycle)
    .def_rw("max_gamma_cyc", &CgSolve::max_gamma_cyc)
    ;

#if 0
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
#endif


  nb::class_<NcsList> ncslist(m, "NcsList");
  nb::class_<NcsList::Ncs>(ncslist, "Ncs")
    .def("__init__", [](NcsList::Ncs* p, const gemmi::AlignmentResult &al, const gemmi::ResidueSpan &fixed, const gemmi::ResidueSpan &movable) {
      new(p) NcsList::Ncs(al, fixed, movable);
    })
    .def("calculate_local_rms", &NcsList::Ncs::calculate_local_rms)
    .def_ro("atoms", &NcsList::Ncs::atoms)
    .def_ro("seqids", &NcsList::Ncs::seqids)
    .def_ro("n_atoms", &NcsList::Ncs::n_atoms)
    .def_ro("local_rms", &NcsList::Ncs::local_rms)
    ;
  ncslist
    .def(nb::init<>())
    .def("set_pairs", &NcsList::set_pairs)
    .def_ro("ncss", &NcsList::ncss)
    .def_ro("all_pairs", &NcsList::all_pairs)
    ;
  nb::bind_vector<std::vector<NcsList::Ncs>>(ncslist, "Ncs_vector");
}
