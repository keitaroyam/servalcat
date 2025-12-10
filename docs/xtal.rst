Crystallographic refinement
===========================

Getting started
---------------
To perform crystallographic refinement with Servalcat, it is necessary to specify an input model (PDB, mmCIF or smCIF), diffraction data (MTZ or CIF format) and radiation source type (xray, neutron or electron). For example:

.. code-block:: console

    $ servalcat refine_xtal_norefmac \
     --model input.pdb --hklin ../data.mtz \
     -s xray \
     [-o prefix]

Output files:
   * ``prefix.pdb``: refined model (legacy PDB format)
   * ``prefix.mmcif``: refined model (mmCIF format)
   * ``prefix.mtz``: 2Fo-Fc and Fo-Fc maps which can be auto-opened with Coot.

Output logs:
   * ``servalcat.log``
   * ``prefix_stats.json``: refinement statistics per cycle in JSON format

Frequently used options
-----------------------

   * ``--ligand [LIGAND ...]``: Restraint dictionary CIF file(s)
   * ``--ncycle NCYCLE``: Number of refinement cycles. Default: 10
   * ``--weight WEIGHT``: Starting value of the weight for the experimental data term (default: automatically determined from resolution). By default, the weight is further adjusted to achieve bond length rmsZ in the range between 0.5 and 1.0. This can be changed using the option ``--target_bond_rmsz_range MIN_RMSZ MAX_RMSZ``.
   * ``--ncsr``: Use local restraints for non-crystallographic symmetry
   * ``--jellybody``: Use jelly body restraints
   * ``--adp {iso,aniso,fix}``: Atomic displacement parameter (isotropic: 1 parameter per atom, anisotropic: 6 parameters per atom, fixed B-values)
   * ``-d D_MIN, --d_min D_MIN``: High-resolution limit (in Å)
   * ``--d_max D_MAX``: Low-resolution limit (in Å)
   * ``--free FREE``: flag number for test set
   * ``--hydrogen {all,yes,no}``: Hydrogen atoms - ``all``: (re)generate riding hydrogen atoms, ``yes``: use hydrogen atoms if present in input structure model, ``no``: remove hydrogen atoms in input structure model. Default: all.
   * ``--hout``: Write hydrogen atoms in the output model
   * ``--twin``: Twin refinement
   * ``--randomize RANDOMIZE``: Shake coordinates with a specified rms (in Å)
   * ``--bfactor BFACTOR``: Reset all atomic B values to specified value
   * ``--keywords KEYWORDS [KEYWORDS ...]``: Keyword(s) in REFMAC5 syntax. See :doc:`supported Refmac keywords <refmac_keywords>`
   * ``--keyword_file KEYWORD_FILE [KEYWORD_FILE ...]``: File with keyword(s) in REFMAC5 syntax  (e.g. ProSMART restraint file)


Input columns for diffraction data
----------------------------------

In the required ``--hklin`` option, it is possible to provide merged or unmerged diffraction data (MTZ or CIF format).
If there are multiple columns available in the input file, mean amplitudes (Friedel pairs averaged) are used by default.

To specify which columns to use, use the ``--labin`` option. For example, the file ``data_merged.mtz`` contains the following columns with **merged** diffraction data:

``H K L FreeR_flag IMEAN SIGIMEAN I(+) SIGI(+) I(-) SIGI(-) FP SIGFP F(+) SIGF(+) F(-) SIGF(-)``

Servalcat would select to use the ``FP, SIGFP,FreeR_flag`` columns by default (refinement against mean structure factor amplitudes).
Anyway, we can specify to use intensities or separate Friedel pairs as follows:

  * ``--labin I(+),SIGI(+),I(-),SIGI(-),FreeR_flag`` (refinement against intensities, separate Friedel pairs)
  * ``--labin IMEAN,SIGIMEAN,FreeR_flag`` (refinement against mean intensities)
  * ``--labin F(+),SIGF(+),F(-),SIGF(-),FreeR_flag`` (refinement against amplitudes, separate Friedel pairs)
  * ``--labin FP, SIGFP,FreeR_flag`` (refinement against mean amplitudes, selected by default)

If the separate Friedel pairs are specified, anomalous difference density map (``FAN`` and ``PHAN``) columns will be present in the output MTZ file.
Note that the anomalous signal is used only for the map calculation but not for the actual refinement.

If the column for **unmerged** intensities is specified, Servalcat merges the data internally and refines against merged intensities. In the log files, the CC* statistic is also available (described below).
An MTZ or CIF file with free flags can be specified with the ``--hklin_free`` option. A particular column for free flags in this file can be specified with the ``--labin_free`` option.


Logs and statistics
-------------------

Refinement progress is monitored by several statistics that quantify the agreement of the refined model with the experimental diffraction data as well as its quality in respect to expected geometry.
The statistics are written in the logfile ``servalcat.log`` and are also available in the JSON format ``output_prefix_stats.json``.


Model agreement with data
~~~~~~~~~~~~~~~~~~~~~~~~~

Which model quality statistics are calculated depends on the nature of the input reflection data.

When amplitudes are used for refinement and free R flags are provided for the test set, the conventional ``Rwork`` and ``Rfree`` values are provided. Moreover, the correlation coefficients ``CCFwork`` and ``CCFfree`` between experimentally observed amplitudes and amplitudes calculated based on the refined model are calculated. If free R flags are not available, only ``R`` (sometimes called ``Rall``) and ``CCF`` are reported.

When intensities (or unmerged diffraction data) are given and free R flags are provided, ``R1work`` and ``R1free`` values are provided, as is common in crystallography of small molecules. These statistics are calculated as R-values between the square roots of observed and calculated intensities, considering only reflections with an intensity-to-sigma ratio above 2. The correlation coefficients ``CCIwork`` and ``CCIfree`` between experimentally observed intensities and intensities calculated based on the refined model are also calculated. If free R flags are not provided, only ``R1`` and ``CCI`` are available.

If unmerged data are used, ``CC*`` statistic is also reported. It estimates the data quality and represents an upper limit for the ``CCI`` statistics. See `Karplus and Diederichs (2012) <https://doi.org/10.1126/science.1218231>`_ or `Diederichs and Karplus (2013) <https://doi.org/10.1107/S0907444913001121>`_.


Model agreement with ideal geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The geometry of the refined model is assessed in the same way as in refinement against cryoEM SPA maps.
Root mean square deviations (RMSD) from expected bond lengths and angles are calculated, as well as  their Z-scores (RMSZ), which represent how many standard deviations the observed geometry deviates from the ideal values. 

Importantly, various kinds of individual model geometry outliers with a Z-score greater than 5 are reported in the last cycle of refinement in ``servalcat.log`` and ``output_prefix_stats.json``. It is recommended to inspect these outliers as they can indicate where the model may need improvement.


Radiation sources
-----------------

Radiation sources can be changed by using the ``-s`` or ``--source`` option (xray, neutron, electron).

When performing refinement against neutron diffraction data, it is possible to refine deuterium fraction using the option ``--refine_dfrac``.
In this case, an extra output file ``output_prefix_expanded.mmcif`` is created for the purpose of deposition to the PDB. The bond lengths and their sigmas from ``_chem_comp_bond.value_dist_nucleus`` and ``_chem_comp_bond.value_dist_nucleus_esd`` are used.

For electron data (MicroED), the scattering factors are calculated using the Mott-Bethe formula from X-ray scattering factors. Both the electron and nucleus positions for hydrogen are considered in the structure factor calculation. See `Yamashita et al. (2021) <https://doi.org/10.1107/S2059798321009475>`_


Small molecules
---------------

Servalcat can also refine small molecules against crystallographic data. A generally suggested protocol is as follows:

.. code-block:: console

    $ servalcat refine_xtal_norefmac \
     --model small_molecule.cif --hklin small_molecule.hkl \
     -s xray --unre --hydrogen no --adp aniso --no_solvent \
     [-o prefix]

This would run unrestrained refinement of the small molecule with anisotropic ADPs without bulk solvent correction. Note the Servalcat does not currently support riding hydrogen atoms for small molecules.

Servalcat provides a helper command which converts files between common formats for small molecule crystallography (.hkl, .cif, .res, .ins) to .pdb, .mmcif or .mtz:

.. code-block:: console

    $ servalcat util sm2mm structure_and_data.cif -o output


Complete list of options
------------------------

.. code-block:: console

    $ servalcat refine_xtal_norefmac --help
    usage: servalcat refine_xtal_norefmac [-h] --hklin HKLIN [--hklin_free HKLIN_FREE] [-d D_MIN] [--d_max D_MAX] [--nbins NBINS]
                                        [--nbins_ml NBINS_ML] [--labin LABIN] [--labin_free LABIN_FREE] [--free FREE] --model MODEL
                                        [--monlib MONLIB] [--ligand [LIGAND ...]] [--newligand_continue] [--hydrogen {all,yes,no}] [--hout]
                                        [--jellybody] [--jellybody_params sigma dmax] [--jellyonly] [--find_links]
                                        [--keywords KEYWORDS [KEYWORDS ...]] [--keyword_file KEYWORD_FILE [KEYWORD_FILE ...]]
                                        [--randomize RANDOMIZE] [--ncycle NCYCLE] [--weight WEIGHT] [--no_weight_adjust]
                                        [--target_bond_rmsz_range TARGET_BOND_RMSZ_RANGE TARGET_BOND_RMSZ_RANGE] [--ncsr]
                                        [--adpr_weight ADPR_WEIGHT] [--occr_weight OCCR_WEIGHT] [--bfactor BFACTOR] [--fix_xyz]
                                        [--adp {fix,iso,aniso}] [--refine_all_occ] [--max_dist_for_adp_restraint MAX_DIST_FOR_ADP_RESTRAINT]
                                        [--adp_restraint_power ADP_RESTRAINT_POWER] [--adp_restraint_exp_fac ADP_RESTRAINT_EXP_FAC]
                                        [--adp_restraint_no_long_range] [--adp_restraint_mode {diff,kldiv}] [--unrestrained] [--refine_h]
                                        [--refine_dfrac] [--twin] -s {electron,xray,neutron} [--no_solvent] [--use_in_est {all,work,test}]
                                        [--keep_charges] [--keep_entities] [--allow_unusual_occupancies] [-o OUTPUT_PREFIX]
                                        [--write_trajectory] [--vonmises] [--prefer_intensity] [--use_fw] [--config CONFIG]

    program to refine crystallographic structures

    optional arguments:
    -h, --help            show this help message and exit
    --hklin HKLIN
    --hklin_free HKLIN_FREE
                            Input MTZ file for test flags
    -d D_MIN, --d_min D_MIN
    --d_max D_MAX
    --nbins NBINS         Number of bins for statistics (default: auto)
    --nbins_ml NBINS_ML   Number of bins for ML parameters (default: auto)
    --labin LABIN         F,SIGF,FREE input
    --labin_free LABIN_FREE
                            MTZ column of --hklin_free
    --free FREE           flag number for test set
    --model MODEL         Input atomic model file
    --monlib MONLIB       Monomer library path. Default: $CLIBD_MON
    --ligand [LIGAND ...]
                            restraint dictionary cif file(s)
    --newligand_continue  Make ad-hoc restraints for unknown ligands (not recommended)
    --hydrogen {all,yes,no}
                            all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input. Default:
                            all
    --hout                write hydrogen atoms in the output model
    --jellybody           Use jelly body restraints
    --jellybody_params sigma dmax
                            Jelly body sigma and dmax (default: [0.01, 4.2])
    --jellyonly           Jelly body only (experimental, may not be useful)
    --find_links          Automatically add links
    --keywords KEYWORDS [KEYWORDS ...]
                            refmac keyword(s)
    --keyword_file KEYWORD_FILE [KEYWORD_FILE ...]
                            refmac keyword file(s)
    --randomize RANDOMIZE
                            Shake coordinates with specified rmsd
    --ncycle NCYCLE       number of CG cycles (default: 10)
    --weight WEIGHT       refinement weight (default: auto)
    --no_weight_adjust    Do not adjust weight during refinement
    --target_bond_rmsz_range TARGET_BOND_RMSZ_RANGE TARGET_BOND_RMSZ_RANGE
                            Bond rmsz range for weight adjustment (default: [0.5, 1.0])
    --ncsr                Use local NCS restraints
    --adpr_weight ADPR_WEIGHT
                            ADP restraint weight (default: 1.000000)
    --occr_weight OCCR_WEIGHT
                            Occupancy restraint weight (default: 0.000000)
    --bfactor BFACTOR     reset all atomic B values to specified value
    --fix_xyz
    --adp {fix,iso,aniso}
    --refine_all_occ
    --max_dist_for_adp_restraint MAX_DIST_FOR_ADP_RESTRAINT
    --adp_restraint_power ADP_RESTRAINT_POWER
    --adp_restraint_exp_fac ADP_RESTRAINT_EXP_FAC
    --adp_restraint_no_long_range
    --adp_restraint_mode {diff,kldiv}
    --unrestrained        No positional restraints
    --refine_h            Refine hydrogen (default: restraints only)
    --refine_dfrac        Refine deuterium fraction (neutron only)
    --twin                Turn on twin refinement
    -s {electron,xray,neutron}, --source {electron,xray,neutron}
    --no_solvent          Do not consider bulk solvent contribution
    --use_in_est {all,work,test}
                            Which set of reflections to use for the ML parameter estimation. Default: 'work' if --twin is set; otherwise
                            'test'.
    --keep_charges        Use scattering factor for charged atoms. Use it with care.
    --keep_entities       Do not override entities
    --allow_unusual_occupancies
                            Allow negative or more than one occupancies
    -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
    --write_trajectory    Write all output from cycles
    --vonmises            Experimental: von Mises type restraint for angles
    --prefer_intensity
    --use_fw              For debugging purpose; use F&W-converted amplitudes but use intensity for stats
    --config CONFIG       Config file (.yaml)

    $ servalcat --version
    Servalcat 0.4.123 with Python 3.9.18 (gemmi 0.7.3, scipy 1.7.3, numpy 1.19.5, pandas 1.3.5)