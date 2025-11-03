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

   * ``--ncycle NCYCLE``: Number of refinement cycles. Default: 10
   * ``--adp {iso,aniso,fix}``: Atomic displacement parameter (isotropic: 1 parameter per atom, anisotropic: 6 parameters per atom, fixed B-values)
   * ``-d D_MIN, --d_min D_MIN``: High-resolution limit (in Å)
   * ``--d_max D_MAX``: Low-resolution limit (in Å)
   * ``--free FREE``: flag number for test set
   * ``--hydrogen {all,yes,no}``: Hydrogen atoms - ``all``: add riding hydrogen atoms, ``yes``: use hydrogen atoms if present in input structure model, ``no``: remove hydrogen atoms in input structure model. Default: all.
   * ``--hout``: Write hydrogen atoms in the output model
   * ``--twin``: Twin refinement
   * ``--randomize RANDOMIZE``: Shake coordinates with a specified mean shift (in Å).
   * ``--bfactor BFACTOR``: Reset all ADPs to specified value (in Å)
   * ``--keywords KEYWORDS [KEYWORDS ...]``: Keyword(s) in REFMAC5 syntax
   * ``--keyword_file KEYWORD_FILE [KEYWORD_FILE ...]``: File with keyword(s) in REFMAC5 syntax  (e.g. ProSMART restraint file)

Restraints
----------
TBD

   * ``--ligand [LIGAND ...]``: Restraint dictionary CIF file(s)
   * ``--weight WEIGHT``: Weight for the experimental data term (default: automatic). The automatic mode adjusts the weight to achieve root mean square deviation Z-score in the range between 0.5 and 1.0. This can be changed using the option ``--target_bond_rmsz_range MIN_RMSZ MAX_RMSZ``.
   * ``--ncsr``: Use local restraints for non-crystallographic symmetry. TBD: How does this work?
   * ``--jellybody``: Use jelly body restraints
   * ``--adpr_weight ADPR_WEIGHT``: ADP restraint weight (default: 1.0)
   * Keyword argument ``vdwr VDWR_WEIGHT``: Van der Waals repulsion restraint weight, a higher number can help to reduce interatomic clashes (default: 1.0)
   * TBD external bond, angle and torsion restraints
   * TBD harmonic restraints for coordinates
   * TBD YAML file

Input columns for diffraction data
----------------------------------

In the required ``--hklin`` option, it is possible to provide merged or unmerged diffraction data (MTZ or CIF format).
If there are multiple columns available in the input file, mean amplitudes (Fridel pairs averaged) are used by default.

To specify which columns to use, use the ``--labin`` option. For example, the file ``data_merged.mtz`` contains the following columns with **merged** diffraction data:

``H K L FreeR_flag IMEAN SIGIMEAN I(+) SIGI(+) I(-) SIGI(-) FMEAN SIGFMEAN F(+) SIGF(+) F(-) SIGF(-)``

Servalcat would select to use the ``FMEAN,SIGFMEAN,FreeR_flag`` columns by default (refinement against mean amplitudes of structure factor).
Anyway, we can specify to use intensities or separate Fridel pairs as follows:

  * ``--labin I(+),SIGI(+),I(-),SIGI(-),FreeR_flag`` (refinement against intensities, separate Fridel pairs)
  * ``--labin IMEAN,SIGIMEAN,FreeR_flag`` (refinement against mean intensities)
  * ``--labin F(+),SIGF(+),F(-),SIGF(-),FreeR_flag`` (refinement against amplitudes, separate Fridel pairs)
  * ``--labin FMEAN,SIGFMEAN,FreeR_flag`` (refinement against mean amplitudes, selected by default)

If the separate Fridel pairs are specified, anomalous difference density map (``FAN`` and ``PHAN``) columns will be present in the output MTZ file.
Note that the anomalous signal is used only for the map calculation but not for the actual refinement.

If the input file contains **unmerged** data, Servalcat merges the data internally and refines against intensities.
An MTZ or CIF file with free flags can be specified with the ``--hklin_free`` option. A particular column for free flags in this file can be specified with the ``--labin_free`` option.
In the logfiles, the CC* statistic is then available which estimates the data quality and represents an upper limit for CCI which is correlation between experimentally observed intensities and intensities calculated based on the refined structure model.

TBD - How does intensity-based refinement work?

Radiation sources
-----------------
Apart from x-ray crystallography, it is possible to use atomic scattering factors for neutron and electron radiation:

.. code-block:: console

    -s {electron,xray,neutron}, --source {electron,xray,neutron}

When performing refinement against neutron diffraction data, it is possible to refine deuterium fraction using the option ``--refine_dfrac``.
In this case, an extra output file ``output_prefix_expanded.mmcif`` is created for the purpose of deposition to the PDB.

Logs and statistics
-------------------
TBD

Occupancy refinement
--------------------
TBD

Small molecules
---------------
TBD


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