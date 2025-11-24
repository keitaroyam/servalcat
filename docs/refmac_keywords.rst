Supported Refmac keywords
=========================

In Servalcat "No Refmac" refinement programs, some Refmac keywords are supported.

External restraints
-------------------

(Almost?) all the external restraint keywords are supported.

EXTErnal UNDEfined [IGNOre | STOP]
    Default: stop. When ``igno`` is given, the program continues even when it encounters external restraints defined for non-existent atoms.

EXTErnal WEIGht SCALe [<dist_sigma_scale>] [DISTance <dist_sigma_scale>] [ANGLe <angle_sigma_scale>] [TORSion <torsion_sigma_scale>] [CHIRal <chiral_sigma_scale>] [PLANe <plane_sigma_scale>] [INTErval <inte_sigma_scale>]
    Default: 1.0. The sigma values in subsequent keywords will be divided by the specified value.

EXTErnal WEIGht [SGMN <sigma_min>] [SGMX <sigma_max>]
    Default: no capping. Cap the sigma values in subsequent keywords (only for dist, angl, chir, tors).

EXTErnal ALPHall <alpha>
    Default: 1.0. Change the default alpha value for subsequent keywords.

EXTErnal TYPEAll <type>
    Default: 2. Change the default external restraint types (0, 1, 2) for subsequent keywords.

EXTErnal [SYMAll | SYMAll [Yes | No] [EXCLude [SELF]]]
    Default: no symmetry. Change whether symmetry should be considered or not for subsequent keywords.

EXTErnal [DMIN <dist_min>] [DMAX <dist_max>]
    Default: no capping. Cap the ideal distance values in subsequent keywords.

EXTErnal [DISTance | ANGLe | PLANe | CHIRal | TORSion] <atom_specs...> [TYPE <type>] [SYMM Y|N] [VALUe <value>] [SIGMa <sigma>] [ALPHa <alpha>]
    Define distance/angle/plane/chiral/torsion restraints. SYMM is only supported for distance and angle. ALPHA is only for distances. See `Barron (2019) <https://arxiv.org/abs/1701.03077>`_ for alpha.

EXTErnal INTErval <atom_specs...> [DMIN <dist_min>] [DMAX <dist_max>] [SMIN <sigma_min>] [SMAX <sigma_max>]
    Define an interval distance restraint.

EXTErnal STACking PLANe 1 <atom_specs...> PLANe 2 <atom_specs...> [DISTance <dist>] [SDDI <sigma_dist>] [ANGLe <angle>] [SDAN <sigma_angle>]
    Define a stacking restraint.

EXTErnal HARMonic <atom_spec> [SIGMa <sigma>]
    Define a positional harmonic restraint.


Restraint weights
-----------------

DISTance <wbond>
   Default: 1.0. The bond length sigma is effectively divided by this number.

ANGLe <wangle>
   Default: 1.0. The bond angle sigma is effectively divided by this number.

TORSion <wtorsion>
   Default: 1.0. The torsion angle sigma is effectively divided by this number.

PLANe <wplane>
   Default: 1.0. The planarity sigma is effectively divided by this number.

CHIRal <wchiral>
   Default: 1.0. The chirality sigma is effectively divided by this number.

[VDWr | VANDerwaal | NONBonding] <wvdwr>
   Default: 1.0. The nonbonding interaction sigma is effectively divided by this number.


Controlling restraints
----------------------

RESTraint TORSion [INCLude | EXCLude] [NONE | RESIdue <residue_name> | GROUp <group_name> | LINK <link_name>] [NAME <torsion_name>] [VALUe <ideal_value>] [SIGMa <sigma_value>] [PERIod <period_value>]
    Update torsion angle restraints. For example, ``restr tors include resi ARG name chi5 sigma 2.0`` will make ARG's chi5 torsion angle sigma 2.0 (ideal and period unchanged). 