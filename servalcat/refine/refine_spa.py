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
from servalcat.utils import logger
from servalcat import utils
from servalcat.spa.run_refmac import check_args, prepare_files, calc_fsc, calc_fofc
from servalcat.spa import fofc
from servalcat.refine import spa
from servalcat.refine.refine import Geom, Refine
b_to_u = utils.model.b_to_u

def add_arguments(parser):
    parser.description = "EXPERIMENTAL program to refine cryo-EM SPA structures"
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--halfmaps", nargs=2, help="Input half map files")
    group.add_argument("--map", help="Use this only if you really do not have half maps.")
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
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
    utils.symmetry.add_symmetry_args(parser) # add --pg etc
    parser.add_argument('--contacting_only', action="store_true", help="Filter out non-contacting NCS")
    parser.add_argument('--ignore_symmetry',
                        help='Ignore symmetry information (MTRIX/_struct_ncs_oper) in the model file')
    parser.add_argument('--no_check_ncs_overlaps', action='store_true', 
                        help='Disable model overlap (e.g. expanded model is used with --pg) test')
    parser.add_argument('--keywords', nargs='+', action="append",
                        help="refmac keyword(s)")
    parser.add_argument('--keyword_file', nargs='+', action="append",
                        help="refmac keyword file(s)")
    parser.add_argument('--randomize', type=float, default=0,
                        help='Shake coordinates with specified rmsd')
    parser.add_argument('--ncycle', type=int, default=10,
                        help="number of CG cycles (default: %(default)d)")
    parser.add_argument('--weight', type=float, default=1,
                        help="refinement weight")
    parser.add_argument('--sigma_b', type=float, default=30,
                        help="refinement ADP sigma in B (default: %(default)f)")
    parser.add_argument('--bfactor', type=float,
                        help="reset all atomic B values to specified value")
    parser.add_argument('--fix_xyz', action="store_true")
    parser.add_argument('--adp',  choices=["fix", "iso", "aniso"], default="iso")
    parser.add_argument('--max_dist_for_adp_restraint', type=float, default=4.)
    parser.add_argument('--refine_h', action="store_true", help="Refine hydrogen (default: restraints only)")
    parser.add_argument("--source", choices=["electron", "xray", "neutron"], default="electron")
    parser.add_argument('-o','--output_prefix', default="refined")
    parser.add_argument('--cross_validation', action='store_true',
                        help='Run cross validation. Only "throughout" mode is available (no "shake" mode)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mask_for_fofc', help="Mask file for Fo-Fc map calculation")
    group.add_argument('--mask_radius_for_fofc', type=float, help="Mask radius for Fo-Fc map calculation")
    parser.add_argument('--trim_fofc_mtz', action="store_true", help="diffmap.mtz will have smaller cell (if --mask_for_fofc is given)")
    parser.add_argument("--fsc_resolution", type=float,
                        help="High resolution limit for FSC calculation. Default: Nyquist")
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def main(args):
    shifted_model_prefix = "shifted"
    args.mask = None
    args.invert_mask = False
    args.no_fix_microheterogeneity = True
    args.no_fix_resi9999 = True
    args.cross_validation_method = "throughout"
    check_args(args)    
    refmac_keywords = args.keywords + [l for f in args.keyword_file for l in open(f)]

    st = utils.fileio.read_structure(args.model)
    monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                   stop_for_unknowns=True)
    utils.restraints.find_and_fix_links(st, monlib)
    st.assign_cis_flags()
    if args.halfmaps:
        maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]
    file_info = prepare_files(st, maps, resolution=args.resolution - 1e-6, monlib=monlib,
                              mask_in=args.mask, args=args,
                              shifted_model_prefix=shifted_model_prefix,
                              no_refmac_fix=True)
    st.setup_cell_images()
    h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                "yes":gemmi.HydrogenChange.NoChange,
                "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
    try:
        topo = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                 check_hydrogen=(args.hydrogen=="yes"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if args.cross_validation:
        lab_f, lab_phi = "lab_f_half1", "lab_phi_half1"
    else:
        lab_f, lab_phi = "lab_f", "lab_phi"
    hkldata = utils.hkl.hkldata_from_mtz(gemmi.read_mtz_file(file_info["mtz_file"]), 
                                         labels=[file_info[lab_f], file_info[lab_phi]],
                                         newlabels=["FP", ""])
    hkldata.setup_relion_binning()
    
    # initialize ADP
    if args.adp != "fix":
        utils.model.reset_adp(st[0], args.bfactor, args.adp == "aniso")
            
    geom = Geom(st, topo, monlib, shake_rms=args.randomize, sigma_b=args.sigma_b,
                refmac_keywords=refmac_keywords)
    ll = spa.LL_SPA(hkldata, st, monlib, source=args.source)
    refiner = Refine(st, geom, ll,
                     refine_xyz=not args.fix_xyz,
                     adp_mode=dict(fix=0, iso=1, aniso=2)[args.adp],
                     refine_h=args.refine_h)

    geom.geom.adpr_max_dist = args.max_dist_for_adp_restraint
    if args.jellybody:
        geom.geom.ridge_sigma, geom.geom.ridge_dmax = args.jellybody_params

    #logger.writeln("TEST: shift x+0.3 A")
    #for cra in st[0].all():
    #    cra.atom.pos += gemmi.Position(0.3,0,0)

    stats = refiner.run_cycles(args.ncycle, weight=args.weight)
    if not args.no_trim: refiner.st.cell = maps[0][0].unit_cell
    utils.fileio.write_model(refiner.st, args.output_prefix, pdb=True, cif=True)
    with open(args.output_prefix + "_stats.json", "w") as ofs:
        for s in stats: s["geom"] = s["geom"].to_dict()
        json.dump(stats, ofs, indent=2)
        logger.writeln("Refinement statistics saved: {}".format(ofs.name))

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
    calc_fofc(refiner.st, st_expanded, maps, monlib, file_info["model_format"], args)
    
    # Final summary
    adpstats_txt = ""
    adp_stats = utils.model.adp_stats_per_chain(refiner.st[0])
    max_chain_len = max([len(x[0]) for x in adp_stats])
    max_num_len = max([len(str(x[1])) for x in adp_stats])
    for chain, natoms, qs in adp_stats:
        adpstats_txt += " Chain {0:{1}s}".format(chain, max_chain_len) if chain!="*" else " {0:{1}s}".format("All", max_chain_len+6)
        adpstats_txt += " ({0:{1}d} atoms) min={2:5.1f} median={3:5.1f} max={4:5.1f} A^2\n".format(natoms, max_num_len, qs[0],qs[2],qs[4])

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
             
Open refined model and diffmap.mtz with COOT:
coot --script {prefix}_coot.py

List Fo-Fc map peaks in the ASU:
servalcat util map_peaks --map diffmap_normalized_fofc.mrc --model {prefix}.pdb --abs_level 4.0
=============================================================================
""".format(rmsbond=stats[-1]["geom"]["r.m.s.d."]["Bond distances, non H"],
           rmsangle=stats[-1]["geom"]["r.m.s.d."]["Bond angles, non H"],
           fscavgs=fscavg_text.rstrip(),
           fsclog="{}_fsc.log".format(args.output_prefix),
           adpstats=adpstats_txt.rstrip(),
           final_weight=args.weight,
           prefix=args.output_prefix))

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
