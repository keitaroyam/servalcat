Scattering source
=================

The *refine_xtal_norefmac* and *refine_spa_norefmac* programs accept the ``--source`` option, where the scattering source (``xray``, ``electron``, ``neutron``, or ``custom``) can be specified.
The following sections describe how this option affects refinement and map calculation.

X-ray
-----

X-ray scattering factor coefficients from the International Tables for Crystallography Volume C (1992 edition or later) are used.
These are implemented in `GEMMI <https://github.com/project-gemmi/gemmi/blob/master/include/gemmi/it92.hpp>`_ and are compatible with CCP4's atomsf.lib and cctbx's it1992 table.
This scheme tabulates four Gaussian coefficients and one constant value for each element:

.. math::
	f_X(s) = \sum_{j=1}^{4} a_j e^{-b_j s^2/4} + c.

Some ionised atoms are supported. For ideal bond lengths and their sigmas, values from _chem_comp_bond.value_dist and _chem_comp_bond.value_dist_esd from the monomer library are used.


Electron
--------

For electron scattering (Cryo-EM), the Mott-Bethe formula is used to convert X-ray scattering factors to electron scattering factors.

.. math::
	f_E(s) = \frac{r_{\rm Bohr}}{4} \frac{Z - f_X(s)}{s^2},

where :math:`Z` is the atomic number.
Because both the electron cloud and the nucleus contribute to the scattering, special treatment is required for hydrogen atoms, as their electron cloud centre and nucleus positions differ.
The _chem_comp_bond.value_dist and _chem_comp_bond.value_dist_nucleus values are used to adjust these positions.


Neutron
-------

Neutron coherent scattering lengths of the elements from Neutron News, Vol. 3, No. 3, 1992 are used (implemented in `GEMMI <https://github.com/project-gemmi/gemmi/blob/master/include/gemmi/neutron92.hpp>`_).
These consist of a single constant value for each element.
For ideal bond lengths and their sigmas, values from _chem_comp_bond.value_dist_nucleus and _chem_comp_bond.value_dist_nucleus_esd from the monomer library are used.


Custom
------

Atom-type-dependent custom scattering factors are available through gemmi::CustomCoef.
This was introduced in `Shtyrov et al. (2025) <https://doi.org/10.1101/2025.10.24.683059>`_.
In this scheme, users can define coefficients for a sum of five Gaussians:

.. math::
	f_X(s) = \sum_{j=1}^{5} a_j e^{-b_j s^2/4}.

To utilise this feature, it is necessary to define the _atom_site.scat_id and the _lmb_scat_coef table in the input mmCIF file.
In the example below, _atom_site.scat_id refers to _lmb_scat_coef.scat_id, where :math:`a` and :math:`b` values are defined.
There are no default scattering factors; all factors must be explicitly defined.
We do not expect users to enter these values manually; for example, `SFFIT <https://github.com/as2875/sffit>`_ can be used to prepare such mmCIF files.

::

  loop_
  _atom_site.group_PDB
  _atom_site.id
  _atom_site.type_symbol
  _atom_site.label_atom_id
  _atom_site.label_alt_id
  _atom_site.label_comp_id
  _atom_site.label_asym_id
  _atom_site.label_entity_id
  _atom_site.label_seq_id
  _atom_site.pdbx_PDB_ins_code
  _atom_site.Cartn_x
  _atom_site.Cartn_y
  _atom_site.Cartn_z
  _atom_site.occupancy
  _atom_site.B_iso_or_equiv
  _atom_site.pdbx_formal_charge 
  _atom_site.auth_seq_id 
  _atom_site.auth_asym_id
  _atom_site.pdbx_PDB_model_num
  _atom_site.scat_id
  ATOM 1 N N . PRO Axp A . ? 79.23 74.607 154.24 1 101.63 ? 5 A 1 24
  ATOM 2 C CA . PRO Axp A . ? 78.782 74.665 152.811 1 99.32 ? 5 A 1 11
  ATOM 3 C C . PRO Axp A . ? 79.357 75.901 152.12 1 85.23 ? 5 A 1 18
  ATOM 4 O O . PRO Axp A . ? 79.265 77.007 152.63 1 85.85 ? 5 A 1 29
  ATOM 5 C CB . PRO Axp A . ? 77.255 74.699 152.867 1 105.62 ? 5 A 1 5
  ATOM 6 C CG . PRO Axp A . ? 76.91 74.923 154.335 1 106.65 ? 5 A 1 5 
  ATOM 7 C CD . PRO Axp A . ? 78.185 75.309 155.039 1 106.89 ? 5 A 1 6
  ATOM 8 H H . PRO Axp A . ? 80.035 75.021 154.338 1 101.63 ? 5 A 1 2
  ...

  loop_
  _lmb_scat_coef.scat_id
  _lmb_scat_coef.coef_a1
  _lmb_scat_coef.coef_a2
  _lmb_scat_coef.coef_a3
  _lmb_scat_coef.coef_a4
  _lmb_scat_coef.coef_a5
  _lmb_scat_coef.coef_b1
  _lmb_scat_coef.coef_b2
  _lmb_scat_coef.coef_b3
  _lmb_scat_coef.coef_b4
  _lmb_scat_coef.coef_b5
  0 0.0349 0.1201 0.1970 0.0573 0.1195 0.5347 3.5867 12.3471 18.9525 38.6269
  1 -0.2309 0.5030 0.5916 -0.9023 -0.4652 0.6683 16.7729 9.7436 49.7746 87.8615
  2 -0.0277 0.4135 0.1836 -0.3466 0.0742 0.0000 8.7072 22.3761 44.3939 139.4651
  3 0.1531 0.5159 0.9095 0.7895 -0.1348 0.9874 5.3341 14.0160 29.2378 114.7447
  4 0.2307 0.5302 0.7519 0.6807 0.2067 0.7861 4.5306 13.1862 31.9342 74.9638
  5 0.2174 0.5207 0.9008 0.9918 0.2636 0.8352 4.9580 14.7037 34.0317 74.9807
  6 0.0577 0.9441 0.4127 1.1907 0.4078 0.0000 10.7793 2.6508 30.5821 80.0065
  ...
