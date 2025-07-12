// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <gemmi/grid.hpp>
#include <gemmi/fourier.hpp>
#include <gemmi/neighbor.hpp>
#include <gemmi/solmask.hpp>
#include <gemmi/it92.hpp>
#include <gemmi/c4322.hpp> // CustomCoef

namespace nb = nanobind;
void add_refine(nb::module_& m); // refine.cpp
void add_intensity(nb::module_& m); // intensity.cpp
void add_amplitude(nb::module_& m); // amplitude.cpp
void add_twin(nb::module_& m); // twin.cpp

template<typename T> // radius in A^-1 unit
gemmi::FPhiGrid<T> hard_sphere_kernel_recgrid(std::tuple<int,int,int> size,
                                              const gemmi::UnitCell &unit_cell,
                                              T radius) {
  gemmi::Grid<T> grid;
  grid.set_size(std::get<0>(size), std::get<1>(size), std::get<2>(size));
  grid.set_unit_cell(unit_cell);
  grid.spacegroup = gemmi::find_spacegroup_by_number(1); // would not work in other space group
  // box should be sufficiently large.
  for (int w = -grid.nw/2; w < grid.nw/2; ++w)
    for (int v = -grid.nv/2; v < grid.nv/2; ++v)
      for (int u = -grid.nu/2; u < grid.nu/2; ++u) {
        const size_t idx = grid.index_near_zero(u, v, w);
        const gemmi::Position delta = grid.unit_cell.orthogonalize_difference(grid.get_fractional(u, v, w));
        const double t = 2 * gemmi::pi() * delta.length() * radius;
        if (t == 0)
          grid.data[idx] = 1;
        else
          grid.data[idx] = 3. * (-t * std::cos(t) + std::sin(t)) / (t * t * t);
      }
  auto rg = gemmi::transform_map_to_f_phi(grid, false);
  const T rg_sum = rg.sum().real();
  for (auto &x : rg.data)
    x /= rg_sum;
  return rg;
}

template<typename T>
void soft_mask_from_model(gemmi::Grid<T> &grid, gemmi::Model &model,
                          float radius, float soft_width) {
  gemmi::mask_points_in_constant_radius<T>(grid, model, radius, 1., false, false);
  const float max_r = radius + soft_width;
  gemmi::NeighborSearch ns(model, grid.unit_cell, max_r);
  ns.populate();
  for (int w = 0; w < grid.nw; ++w)
    for (int v = 0; v < grid.nv; ++v)
      for (int u = 0; u < grid.nu; ++u) {
        size_t idx = grid.index_q(u, v, w);
        if (grid.data[idx] > 0) continue;
        const gemmi::Position p = grid.get_position(u, v, w);
        const auto result = ns.find_nearest_atom_within_k(p, 1, max_r);
        const float d = std::sqrt(result.second) - radius;
        if (d < 0) continue; // should not happen
        grid.data[idx] = T(0.5 + 0.5 * std::cos(gemmi::pi() * d / soft_width));
      }
}

NB_MODULE(ext, m) {
  m.doc() = "Servalcat extension";
  // importing gemmi allows to output gemmi types from nanobind.stubgen.
  nb::module_::import_("gemmi");

  add_refine(m);
  add_intensity(m);
  add_amplitude(m);
  add_twin(m);
  m.def("hard_sphere_kernel_recgrid", hard_sphere_kernel_recgrid<float>);
  m.def("soft_mask_from_model", soft_mask_from_model<float>);
  // gemmi/python/elem.cpp
  using CustomCoef = gemmi::CustomCoef<double>;
  m.def("set_custom_form_factors", [](const std::vector<decltype(CustomCoef::Coef::coefs)>& pp) {
      CustomCoef::data.clear();
      CustomCoef::Coef item;
      for (const auto& p : pp) {
        item.set_coefs(p);
        CustomCoef::data.push_back(item);
      }
  });
  m.def("IT92_normalize_etc", [](const gemmi::Element &el_x){
      gemmi::IT92<double>::normalize();
      gemmi::IT92<double>::ignore_charge = false;
      const auto coefs = gemmi::IT92<double>::get(el_x, 0).coefs;
      gemmi::IT92<double>::get(gemmi::El::X, 0).set_coefs(coefs);
  });
}
