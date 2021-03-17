"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import os
from servalcat.utils import logger
from servalcat import utils
from servalcat import spa

def add_arguments(parser):
    parser.description = 'Run REFMAC5 for SPA'

    parser.add_argument('--exe', default="refmac5", help='refmac5 binary')
    # sfcalc options
    sfcalc_group = parser.add_argument_group("sfcalc")
    spa.sfcalc.add_sfcalc_args(sfcalc_group)

    # run_refmac options
    # TODO use group! like refmac options
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument('--mtz', help='Input mtz file')
    parser.add_argument('--mtz_half', nargs=2, help='Input mtz files for half maps')
    parser.add_argument('--lab_f')
    parser.add_argument('--lab_sigf')
    parser.add_argument('--lab_phi')
    parser.add_argument('--bfactor', type=float)
    parser.add_argument('--ncsr', default="local", choices=["local", "global"])
    parser.add_argument('--ncycle', type=int, default=10)
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no"])
    parser.add_argument('--jellybody', action='store_true')
    parser.add_argument('--jellybody_params', nargs=2, type=float,
                        metavar=("sigma", "dmax"), default=[0.01, 4.2])
    parser.add_argument('--hout', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--weight_auto_scale', type=float)
    group.add_argument('--weight', type=float)
    parser.add_argument('--keywords', nargs='+', action="append")
    parser.add_argument('--keyword_file', nargs='+', action="append")
    parser.add_argument('--external_restraints_json')
    parser.add_argument('--show_refmac_log', action='store_true')
    parser.add_argument('--output_prefix', default="refined",
                        help='output file name prefix')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Run cross validation')
    parser.add_argument('--shake_radius', default=0.5,
                        help='Shake rmsd')

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def main(args):
    if not args.model:
        logger.write("Error: give --model.")
        return

    if not (args.map or args.halfmaps) and not args.mtz:
        logger.write("Error: give --map | --halfmaps | --mtz.")
        return

    if args.ligand:
        args.ligand = sum(args.ligand, [])
        if len(args.ligand) > 1:
            mcif = "merged_ligands.cif"
            logger.write("Merging ligand cif inputs: {}".format(args.ligand))
            utils.fileio.merge_ligand_cif(libin, mcif)
            args.ligand = "merged_ligands.cif"
        else:
            args.ligand = args.ligand[0]

    model_format = utils.fileio.check_model_format(args.model)
    
    if args.map or args.halfmaps:
        args.output_model_prefix = "shifted_local"
        args.output_masked_prefix = "masked_fs"
        args.output_mtz_prefix = "starting_map"
        args.remove_multiple_models = True
        file_info = spa.sfcalc.main(args)
        args.mtz = file_info["mtz_file"]
        if args.halfmaps: # FIXME if no_mask?
            args.mtz_half = [file_info["mtz_file"], file_info["mtz_file"]]
        args.lab_phi = file_info["lab_phi"]  #"Pout0"
        args.lab_f = file_info["lab_f"]
        args.model = file_info["model_file"]
    else:
        file_info = {}
        # Not supported actually..

    if args.keyword_file:
        args.keyword_file = sum(args.keyword_file, [])
        for f in args.keyword_file:
            logger.write("Keyword file: {}".format(f))
            assert os.path.exists(f)
    else:
        args.keyword_file = []
            
    if args.keywords:
        args.keywords = sum(args.keywords, [])

    # FIXME if mtz is given and sfcalc() not ran?
    has_ncsc = "ncsc_file" in file_info
    if has_ncsc:
        args.keyword_file.append(file_info["ncsc_file"])

    if not args.no_shift:
        refmac_prefix = "local_" + args.output_prefix
    else:
        refmac_prefix = args.output_prefix

    refmac = utils.refmac.Refmac(prefix=refmac_prefix, args=args, global_mode="spa")
    refmac.run_refmac()

    if not args.no_shift:
        ncsc_in = ("ncsc_global.txt") if has_ncsc else None
        spa.shiftback.shift_back(xyz_in=refmac_prefix+model_format,
                                 refine_mtz=refmac_prefix+".mtz",
                                 shifts_json="shifts.json",
                                 ncsc_in=ncsc_in,
                                 out_prefix=args.output_prefix)

    # Expand sym here
    if has_ncsc:
        refined_xyz = args.output_prefix+model_format
        logger.write("Expanding {}".format(refined_xyz))
        st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(refined_xyz)
        st.expand_ncs(gemmi.HowToNameCopiedChain.Short)
        utils.fileio.write_model(st, file_name=args.output_prefix+"_expanded"+model_format,
                                 cif_ref=cif_ref)

    if args.cross_validation:
        logger.write("Cross validation is requested.")
        st = gemmi.read_structure(refmac_prefix+model_format)
        logger.write("  Shaking atomic coordinates with rms={}".format(args.shake_radius))
        st = utils.model.shake_structure(st, args.shake_radius)
        shaken_file = refmac_prefix+"_shaken"+model_format
        utils.fileio.write_model(st, file_name=shaken_file)
        refmac_prefix_shaken = refmac_prefix+"_shaken_refined"
        refmac_prefix_hm2 = refmac_prefix+"_shaken_refined_statshm2"

        logger.write("  Starting refinement using half map 1")
        refmac_hm1 = refmac.copy(hklin=args.mtz_half[0],
                                 xyzin=shaken_file,
                                 prefix=refmac_prefix_shaken)
        if "lab_f_half1" in file_info:
            refmac_hm1.lab_f = file_info["lab_f_half1"]
            refmac_hm1.lab_phi = file_info["lab_phi_half1"]
            # SIGMA?
            
        refmac_hm1.run_refmac()

        # TODO replace this part later
        logger.write("  Calculating stats using half map 2")
        refmac_hm2 = refmac.copy(hklin=args.mtz_half[1],
                                 xyzin=refmac_prefix_shaken+model_format,
                                 prefix=refmac_prefix_hm2,
                                 ncycle=0, bfactor=None)
        if "lab_f_half2" in file_info:
            refmac_hm2.lab_f = file_info["lab_f_half2"]
            refmac_hm2.lab_phi = file_info["lab_phi_half2"]
            # SIGMA?

        refmac_hm2.run_refmac()

        # TODO calc FSC

        if not args.no_shift:
            ncsc_in = ("ncsc_global.txt") if has_ncsc else None
            spa.shiftback.shift_back(xyz_in=refmac_prefix_shaken+model_format,
                                 refine_mtz=refmac_prefix_shaken+".mtz",
                                 shifts_json="shifts.json",
                                 ncsc_in=ncsc_in,
                                 out_prefix=args.output_prefix+"_shaken_refined")
        
        
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)

