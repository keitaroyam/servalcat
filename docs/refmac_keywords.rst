Supported Refmac keywords
=========================

In Servalcat "No Refmac" refinement programs, some Refmac keywords are supported.

External restraints
-------------------

(Almost?) all the external restraint keywords are supported.

EXTErnal UNDEfined [IGNOre | STOP]
    Default: stop. When ``igno`` is given, the program continues even when it encounters external restraints defined for non-existent atoms.

EXTErnal HYDRogen [IGNOre | USE]
    Default: ignore. Hydrogen atoms in external restraints are ignored.

EXTErnal WEIGht SCALe [<dist_sigma_scale>] [DISTance <dist_sigma_scale>] [ANGLe <angle_sigma_scale>] [TORSion <torsion_sigma_scale>] [CHIRal <chiral_sigma_scale>] [PLANe <plane_sigma_scale>] [INTErval <inte_sigma_scale>]
    Default: 1.0. The sigma values in subsequent keywords will be divided by the specified value.

EXTErnal WEIGht [SGMN <sigma_min>] [SGMX <sigma_max>]
    Default: no capping. Cap the sigma values in subsequent keywords (only for dist, angl, chir, tors).

EXTErnal ALPHall <alpha>
    Default: 1.0. Change the default alpha value for subsequent keywords.

EXTErnal TYPEAll <type>
    Default: 2. Change the default external restraint types (0, 1, 2) for subsequent keywords. Type 2 is only effective for distance restraints. When type 0 is specified, any existing restraints defined for the selected atoms will be removed and overridden by the new restraint. Otherwise, the specified restraint is added to existing ones, and the closest ideal value is selected in each refinement cycle.

EXTErnal [SYMAll | SYMAll [Yes | No] [EXCLude [SELF]]]
    Default: no symmetry. Change whether symmetry should be considered or not for subsequent keywords.

EXTErnal [DMIN <dist_min>] [DMAX <dist_max>]
    Default: no capping. Cap the ideal distance values in subsequent keywords.

EXTErnal DISTance FIRSt CHAIn <chain_1> RESI <residue_1> INSertion <ins_1> ATOM <atom_1> ALTecode <alt_1> SECOnd CHAIn <chain_2> RESI <residue_2> INSertion <ins_2> ATOM <atom_2> ALTecode <alt_2> [TYPE <type>] [SYMM Y|N] [VALUe <value>] [SIGMa <sigma>] [ALPHa <alpha>]
    Define an external distance restraint. When type 2 is specified, the function type is controlled by alpha; see `Barron (2019) <https://arxiv.org/abs/1701.03077>`_ for details regarding alpha. Note that type 2 restraints will be ignored if a type 0 or type 1 restraint has already been defined for the same atom pair. Example: ``exte dist firs chai E2 resi 451 ins . atom O seco chai E4 resi 461 ins . atom N value 2.8 sigma 0.1 alph 2``

EXTErnal ANGLe FIRSt <atom_spec_1> SECOnd <atom_spec_2> THIRd <atom_spec_3> [TYPE <type>] [SYMM Y|N] [VALUe <value>] [SIGMa <sigma>]
    Define an external angle restraint. For atom specs, please refer to the distance restraint. Example: ``exte angl firs chai D1 resi 1000 ins . atom ZN seco chai D1 resi 71 ins . atom SG thir chain D1 resi 71 ins . atom CB valu 109.0 sigm 3.0``

EXTErnal TORSion FIRSt <atom_spec_1> SECOnd <atom_spec_2> THIRd <atom_spec_3> FOURth <atom_spec_4> [TYPE <type>] [VALUe <value>] [SIGMa <sigma>]
    Define an external torsion angle restraint. For atom specs, please refer to the distance restraint.

EXTErnal CHIRal FIRSt <atom_spec_centre> SECOnd <atom_spec_1> THIRd <atom_spec_2> FOURth <atom_spec_3> [VALUe <value>] [SIGMa <sigma>]
    Define an external chirality restraint. For atom specs, please refer to the distance restraint. The ``value`` must be a real value, instead of a sign.

EXTErnal PLANe FIRSt <atom_spec_1> NEXT <atom_spec_2> NEXT <atom_spec_3> NEXT ... [SIGMa <sigma>]
    Define an external planarity restraint. In the atom spec, multiple atom names may be specified using the ``atom {CB CD1 CD2 CE1 ...}`` syntax.

EXTErnal INTErval FIRSt <atom_spec_1> SECOnd <atom_spec_2> [DMIN <dist_min>] [DMAX <dist_max>] [SMIN <sigma_min>] [SMAX <sigma_max>]
    Define an interval distance restraint.

EXTErnal STACking PLANe 1 <atom_specs...> PLANe 2 <atom_specs...> [DISTance <dist>] [SDDI <sigma_dist>] [ANGLe <angle>] [SDAN <sigma_angle>]
    Define a stacking restraint. For atom specs, please refer to the planarity restraint. Example: ``exte stac plan 1 firs resi 50 ins . chai B atoms { C2 C4 C5 C6 N1 N3 N4 O2 } plan 2 firs resi 51 ins . chai B atoms { C2 C4 C5 C6 C8 N1 N3 N6 N7 N9 } dist 3.4 sddi 0.2 sdan 6.0``

EXTErnal HARMonic [ATIN CHAIn <chain> RESI <residue> INSertion <ins> ATOM <atom> ALTecode <alt> | RESIdues FROM <resi_start> <chain_start> TO <resi_end> <chain_end> [ATOM <atom>] ] [SIGMa <sigma>]
    Define a positional harmonic restraint. The atoms will be restrained to their current position and movement from those positions will be slower than for other atoms. 


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