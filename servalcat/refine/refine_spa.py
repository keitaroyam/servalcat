"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import json
import os
import shutil
import argparse
from servalcat.utils import logger
from servalcat import utils
from servalcat.spa.run_refmac import prepare_files
from servalcat.spa import fofc
from servalcat.refine import spa
from servalcat.refine.refine import Refine

logger.set_file("servalcat_test_refine.log")

def add_arguments(parser):
    parser.add_argument("--halfmaps", nargs=2, required=True)
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument("-d", '--resolution', type=float, required=True)
    parser.add_argument('-r', '--mask_radius', type=float, default=3, help="mask radius")
    parser.add_argument('--padding',
                        type=float, 
                        help='Default: 2*mask_radius')
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
    parser.add_argument('--adp_weight', type=float, default=1,
                        help="refinement ADP weight (experimental)")
    parser.add_argument('--bfactor', type=float,
                        help="reset all atomic B values to specified value")
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def main(args):
    if args.ligand: args.ligand = sum(args.ligand, [])

    if args.keywords:
        args.keywords = sum(args.keywords, [])
    else:
        args.keywords = []
    st = utils.fileio.read_structure(args.model)
    maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
    monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                   stop_for_unknowns=True,
                                                   check_hydrogen=(args.hydrogen=="yes"))

    shifted_model_prefix = "shifted"
    args.mask = None
    args.no_mask = False
    args.invert_mask = False
    args.gemmi_prep = False
    args.no_fix_microheterogeneity = False
    args.no_fix_resi9999 = False
    file_info = prepare_files(st, maps, resolution=args.resolution - 1e-6, monlib=monlib,
                              mask_in=args.mask, args=args,
                              shifted_model_prefix=shifted_model_prefix)

    hkldata = utils.hkl.hkldata_from_mtz(gemmi.read_mtz_file(file_info["mtz_file"]), 
                                         labels=["Fmap1", "Pmap1", "Fmap2", "Pmap2", "Fout", "Pout"],
                                         newlabels=["F_map1", "", "F_map2", "", "FP", ""])
    hkldata.setup_relion_binning()
    utils.maps.calc_noise_var_from_halfmaps(hkldata)
    h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                "yes":gemmi.HydrogenChange.NoChange,
                "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=logger,
                                  reorder=True, ignore_unknown_links=False) # we should remove logger here??

    if args.bfactor is not None:
        for cra in st[0].all():
            cra.atom.b_iso = args.bfactor
            cra.atom.aniso = gemmi.SMat33f(0,0,0,0,0,0)
    
    ll = spa.LL_SPA(hkldata, st, monlib)
    refiner = Refine(st, topo, monlib, ll)

    if args.randomize > 0:
        numpy.random.seed(0)
        from servalcat.utils import model
        utils.model.shake_structure(refiner.st, args.randomize, copy=False)
        utils.fileio.write_model(refiner.st, "shaken", pdb=True)#, cif=True)
        ll.update_fc()
        ll.calc_fsc()

    #logger.writeln("TEST: shift x+0.3 A")
    #for cra in st[0].all():
    #    cra.atom.pos += gemmi.Position(0.3,0,0)

    for i in range(args.ncycle):
        logger.writeln("==== CYCLE {:2d}".format(i))
        ll.update_ml_params()
        refiner.run_cycle(weight=args.weight, adp_weight=args.adp_weight)
        utils.fileio.write_model(refiner.st, "refined_{:02d}".format(i), pdb=True)#, cif=True)
        ll.update_fc()
        ll.calc_fsc()

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
