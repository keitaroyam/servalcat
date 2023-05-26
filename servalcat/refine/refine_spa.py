"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import argparse
import json
import numpy
from servalcat.utils import logger
from servalcat import utils
from servalcat.spa.run_refmac import check_args, process_input, calc_fsc, calc_fofc
from servalcat.spa import fofc
from servalcat.refine import spa
from servalcat.refine.refine import Geom, Refine
b_to_u = utils.model.b_to_u

def add_arguments(parser):
    parser.description = "EXPERIMENTAL program to refine cryo-EM SPA structures"
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--halfmaps", nargs=2, help="Input half map files")
    group.add_argument("--map", help="Use this only if you really do not have half maps.")
    group.add_argument("--hklin", help="Use mtz file. With limited functionality.")
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--labin', 
                        help='F,PHI for hklin')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument("-d", '--resolution', type=float, required=True)
    parser.add_argument('-r', '--mask_radius', type=float, default=3, help="mask radius")
    parser.add_argument('--padding',
                        type=float, 
                        help='Default: 2*mask_radius')
    parser.add_argument('--no_mask', action='store_true')
    parser.add_argument('--no_trim',
                        action='store_true',
                        help='Keep original box (not recommended)')
    parser.add_argument('--mask_soft_edge',
                        type=float, default=0,
                        help='Add soft edge to model mask. Should use with --no_sharpen_before_mask?')
    parser.add_argument('--no_sharpen_before_mask', action='store_true',
                        help='By default half maps are sharpened before masking by std of signal and unsharpened after masking. This option disables it.')
    parser.add_argument("--b_before_mask", type=float,
                        help="sharpening B value for sharpen-mask-unsharpen procedure. By default it is determined automatically.")
    parser.add_argument('--blur',
                        type=float, default=0,
                        help='Sharpening or blurring B')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--ligand', nargs="*", action="append",
                        help="restraint dictionary cif file(s)")
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no"],
                        help="all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input. "
                        "Default: %(default)s")
    parser.add_argument('--jellybody', action='store_true',
                        help="Use jelly body restraints")
    parser.add_argument('--jellybody_params', nargs=2, type=float,
                        metavar=("sigma", "dmax"), default=[0.01, 4.2],
                        help="Jelly body sigma and dmax (default: %(default)s)")
    parser.add_argument('--jellyonly', action='store_true',
                        help="Jelly body only (experimental, may not be useful)")
    utils.symmetry.add_symmetry_args(parser) # add --pg etc
    parser.add_argument('--contacting_only', action="store_true", help="Filter out non-contacting NCS")
    parser.add_argument('--ignore_symmetry',
                        help='Ignore symmetry information (MTRIX/_struct_ncs_oper) in the model file')
    parser.add_argument('--no_link_check', action='store_true', 
                        help='Do not find and fix link records in input model.')
    parser.add_argument('--no_check_ncs_overlaps', action='store_true', 
                        help='Disable model overlap (e.g. expanded model is used with --pg) test')
    parser.add_argument('--no_check_ncs_map', action='store_true', 
                        help='Disable map NCS consistency test')
    parser.add_argument('--no_check_mask_with_model', action='store_true', 
                        help='Disable mask test using model')
    parser.add_argument('--keywords', nargs='+', action="append",
                        help="refmac keyword(s)")
    parser.add_argument('--keyword_file', nargs='+', action="append",
                        help="refmac keyword file(s)")
    parser.add_argument('--randomize', type=float, default=0,
                        help='Shake coordinates with specified rmsd')
    parser.add_argument('--ncycle', type=int, default=10,
                        help="number of CG cycles (default: %(default)d)")
    parser.add_argument('--weight', type=float,
                        help="refinement weight. default: automatic")
    parser.add_argument('--sigma_b', type=float, default=10,
                        help="refinement ADP sigma in B (default: %(default)f)")
    parser.add_argument('--bfactor', type=float,
                        help="reset all atomic B values to specified value")
    parser.add_argument('--fix_xyz', action="store_true")
    parser.add_argument('--adp',  choices=["fix", "iso", "aniso"], default="iso")
    parser.add_argument('--max_dist_for_adp_restraint', type=float, default=4.)
    parser.add_argument('--adp_restraint_power', type=float)
    parser.add_argument('--adp_restraint_exp_fac', type=float)
    parser.add_argument('--refine_h', action="store_true", help="Refine hydrogen (default: restraints only)")
    parser.add_argument("--source", choices=["electron", "xray", "neutron"], default="electron")
    parser.add_argument('-o','--output_prefix', default="refined")
    parser.add_argument('--cross_validation', action='store_true',
                        help='Run cross validation. Only "throughout" mode is available (no "shake" mode)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mask_for_fofc', help="Mask file for Fo-Fc map calculation")
    group.add_argument('--mask_radius_for_fofc', type=float, help="Mask radius for Fo-Fc map calculation")
    parser.add_argument("--fsc_resolution", type=float,
                        help="High resolution limit for FSC calculation. Default: Nyquist")
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def main(args):
    args.mask = None
    args.invert_mask = False
    args.trim_fofc_mtz = args.mask_for_fofc is not None
    args.cross_validation_method = "throughout"
    check_args(args)    
    refmac_keywords = args.keywords + [l for f in args.keyword_file for l in open(f)]

    st = utils.fileio.read_structure(args.model)
    monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                   stop_for_unknowns=True)
    utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
    if args.hklin:
        assert not args.cross_validation
        mtz = utils.fileio.read_mmhkl(args.hklin)
        hkldata = utils.hkl.hkldata_from_mtz(mtz, args.labin.split(","),
                                             newlabels=["FP", ""],
                                             require_types=["F", "P"])
        hkldata.df = hkldata.df.dropna() # workaround for missing data
        #hkldata.setup_relion_binning()
        hkldata.setup_binning(n_bins=10) # need to sort out
        st.cell = hkldata.cell
        st.spacegroup_hm = hkldata.sg.xhm()
        st.setup_cell_images()
        info = {}
        if not args.no_link_check:
            utils.restraints.find_and_fix_links(st, monlib)
    else:
        if args.halfmaps:
            maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
        else:
            maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]
        hkldata, info = process_input(st, maps, resolution=args.resolution - 1e-6, monlib=monlib,
                                      mask_in=args.mask, args=args, use_refmac=False,
                                      fix_link=not args.no_link_check)
    h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                "yes":gemmi.HydrogenChange.NoChange,
                "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
    try:
        topo, metal_kws = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                            check_hydrogen=(args.hydrogen=="yes"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    refmac_keywords.extend(metal_kws)
    # initialize ADP
    if args.adp != "fix":
        utils.model.reset_adp(st[0], args.bfactor, args.adp == "aniso")

    # auto weight
    if args.weight is None:
        # from 230303_weight_test using 472 test cases
        reso = info["d_eff"] if "d_eff" in info else args.resolution
        if "vol_ratio" in info:
            if "d_eff" in info:
                rlmc = (-2.3976, 0.5933, 3.5160)
            else:
                rlmc = (-2.5151, 0.6681, 3.6467)
            logger.writeln("Estimating weight auto scale using resolution and volume ratio")
            ws = numpy.exp(rlmc[0] + reso*rlmc[1] +info["vol_ratio"]*rlmc[2])
        else:
            if "d_eff" in info:
                rlmc = (-1.6908, 0.5668)
            else:
                rlmc = (-1.7588, 0.6311)
            logger.writeln("Estimating weight auto scale using resolution")
            ws =  numpy.exp(rlmc[0] + args.resolution*rlmc[1])
        args.weight = max(0.2, min(18.0, ws))
        logger.writeln(" Will use weight= {:.2f}".format(args.weight))

    geom = Geom(st, topo, monlib, shake_rms=args.randomize, sigma_b=args.sigma_b,
                refmac_keywords=refmac_keywords, unrestrained=args.jellyonly)
    ll = spa.LL_SPA(hkldata, st, monlib,
                    lab_obs="F_map1" if args.cross_validation else "FP",
                    source=args.source)
    refiner = Refine(st, geom, ll,
                     refine_xyz=not args.fix_xyz,
                     adp_mode=dict(fix=0, iso=1, aniso=2)[args.adp],
                     refine_h=args.refine_h)

    geom.geom.adpr_max_dist = args.max_dist_for_adp_restraint
    if args.adp_restraint_power is not None: geom.geom.adpr_d_power = args.adp_restraint_power
    if args.adp_restraint_exp_fac is not None: geom.geom.adpr_exp_fac = args.adp_restraint_exp_fac
    if args.jellybody or args.jellyonly: geom.geom.ridge_sigma, geom.geom.ridge_dmax = args.jellybody_params
    if args.jellyonly: geom.geom.ridge_exclude_short_dist = False

    #logger.writeln("TEST: shift x+0.3 A")
    #for cra in st[0].all():
    #    cra.atom.pos += gemmi.Position(0.3,0,0)

    stats = refiner.run_cycles(args.ncycle, weight=args.weight)
    if not args.hklin and not args.no_trim:
        refiner.st.cell = maps[0][0].unit_cell
        refiner.st.setup_cell_images()

    refiner.st.name = args.output_prefix
    utils.fileio.write_model(refiner.st, args.output_prefix, pdb=True, cif=True)
    with open(args.output_prefix + "_stats.json", "w") as ofs:
        for s in stats:
            if "geom" in s: s["geom"] = s["geom"].to_dict()
        json.dump(stats, ofs, indent=2)
        logger.writeln("Refinement statistics saved: {}".format(ofs.name))

    if args.hklin:
        return
        
    # Expand sym here
    st_expanded = refiner.st.clone()
    if not all(op.given for op in st.ncs):
        utils.model.expand_ncs(st_expanded)
        utils.fileio.write_model(st_expanded, args.output_prefix+"_expanded", pdb=True, cif=True)

    # Calc FSC
    mask = utils.fileio.read_ccp4_map(args.mask)[0] if args.mask else None
    fscavg_text = calc_fsc(st_expanded, args.output_prefix, maps,
                           args.resolution, mask=mask, mask_radius=args.mask_radius if not args.no_mask else None,
                           soft_edge=args.mask_soft_edge,
                           b_before_mask=args.b_before_mask,
                           no_sharpen_before_mask=args.no_sharpen_before_mask,
                           make_hydrogen=args.hydrogen,
                           monlib=monlib, 
                           blur=args.blur,
                           d_min_fsc=args.fsc_resolution,
                           cross_validation=args.cross_validation,
                           cross_validation_method=args.cross_validation_method
                           )
    
    # Calc Fo-Fc (and updated) maps
    diffmap_prefix = "{}_diffmap".format(args.output_prefix)
    calc_fofc(refiner.st, st_expanded, maps, monlib, ".mmcif", args, diffmap_prefix=diffmap_prefix)
    
    # Final summary
    adpstats_txt = ""
    adp_stats = utils.model.adp_stats_per_chain(refiner.st[0])
    max_chain_len = max([len(x[0]) for x in adp_stats])
    max_num_len = max([len(str(x[1])) for x in adp_stats])
    for chain, natoms, qs in adp_stats:
        adpstats_txt += " Chain {0:{1}s}".format(chain, max_chain_len) if chain!="*" else " {0:{1}s}".format("All", max_chain_len+6)
        adpstats_txt += " ({0:{1}d} atoms) min={2:5.1f} median={3:5.1f} max={4:5.1f} A^2\n".format(natoms, max_num_len, qs[0],qs[2],qs[4])

    if "geom" in stats[-1]:
        rmsbond = stats[-1]["geom"]["r.m.s.d."]["Bond distances, non H"]
        rmsangle = stats[-1]["geom"]["r.m.s.d."]["Bond angles, non H"]
    else:
        rmsbond, rmsangle = numpy.nan, numpy.nan
    if args.mask_for_fofc:
        map_peaks_str = """\
List Fo-Fc map peaks in the ASU:
servalcat util map_peaks --map {diffmap_prefix}_normalized_fofc.mrc --model {prefix}.pdb --abs_level 4.0 \
""".format(prefix=args.output_prefix, diffmap_prefix=diffmap_prefix)
    else:
        map_peaks_str = "WARNING: --mask_for_fofc was not given, so the Fo-Fc map was not normalized."

    logger.writeln("""
=============================================================================
* Final Summary *

Rmsd from ideal
  bond lengths: {rmsbond:.4f} A
  bond  angles: {rmsangle:.3f} deg

{fscavgs}
 Run loggraph {fsclog} to see plots

ADP statistics
{adpstats}

Weight used: {final_weight:.3e}
             If you want to change the weight, give larger (looser restraints)
             or smaller (tighter) value to --weight=.
             
Open refined model and {diffmap_prefix}.mtz with COOT:
coot --script {prefix}_coot.py

{map_peaks_msg}
=============================================================================
""".format(rmsbond=rmsbond,
           rmsangle=rmsangle,
           fscavgs=fscavg_text.rstrip(),
           fsclog="{}_fsc.log".format(args.output_prefix),
           adpstats=adpstats_txt.rstrip(),
           final_weight=args.weight,
           prefix=args.output_prefix,
           diffmap_prefix=diffmap_prefix,
           map_peaks_msg=map_peaks_str))

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
