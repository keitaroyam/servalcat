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
import subprocess
import pipes
from servalcat.utils import logger
from servalcat import utils
from servalcat import spa

def add_arguments(parser):
    parser.description = 'Run REFMAC5 for SPA'

    parser.add_argument('--exe', default="refmac5", help='refmac5 binary')
    # sfcalc options
    parser.add_argument('--map', help='Input map file')
    parser.add_argument('--halfmaps', nargs=2, help='Input half map files')
    parser.add_argument('--mapref', help='Reference map file')
    parser.add_argument('--mask', help='Mask file')
    parser.add_argument('--model', required=True, help="")
    parser.add_argument('--mask_radius', type=float, help='')
    parser.add_argument('--resolution', type=float, help='')
    parser.add_argument('--shift', action='store_true', help='')
    parser.add_argument('--blur', nargs="+", type=float, help='Sharpening or blurring B')
    parser.add_argument('--ligand', nargs="*")
    parser.add_argument('--relion_pg',
                        help='RELION point group symbol for strict symmetry')
    parser.add_argument('--ignore_symmetry', action='store_true',
                        help='Ignore symmetry information in the model file')
    parser.add_argument('--remove_multiple_models', action='store_true',
                        help='Keep 1st model only')
    # run_refmac options
    parser.add_argument('--mtz', help='Input mtz file')
    parser.add_argument('--mtz_half', nargs=2, help='Input mtz files for half maps')
    parser.add_argument('--lab_f')
    parser.add_argument('--lab_sigf')
    parser.add_argument('--lab_phi')
    parser.add_argument('--bfactor', type=float)
    parser.add_argument('--ncsr', default="local")
    parser.add_argument('--ncycle', type=int, default=10)
    parser.add_argument('--hydrogen', default="all")
    parser.add_argument('--jellybody', action='store_true')
    parser.add_argument('--hout', action='store_true')
    parser.add_argument('--weight_auto_scale', type=float)
    parser.add_argument('--keywords', nargs='+')
    parser.add_argument('--keyword_file', nargs='+')
    parser.add_argument('--output_prefix', default="refined",
                        help='output file name prefix')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Run cross validation')
    parser.add_argument('--shake_radius', default=0.3,
                        help='Shake rmsd')

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def make_cmd(xyzin, hklin, xyzout=None, hklout=None, prefix=None, exe="refmac5", libin=None):
    cmd = [exe]
    if prefix:
        if not xyzout: xyzout = prefix + ".pdb"
        if not hklout: hklout = prefix + ".mtz"
        
    cmd.extend(["hklin", pipes.quote(hklin)])
    cmd.extend(["hklout", pipes.quote(hklout)])
    cmd.extend(["xyzin", pipes.quote(xyzin)])
    cmd.extend(["xyzout", pipes.quote(xyzout)])
    if libin:
        if len(libin) > 1:
            mcif = "merged_ligands.cif"
            logger.write("Merging ligand cif inputs: {}".format(libin))
            utils.fileio.merge_ligand_cif(libin, mcif)
            cmd.extend(["libin", mcif])
        else:
            cmd.extend(["libin", libin[0]])

    return cmd
# make_cmd()

def make_keywords(lab_f, lab_sigf, lab_phi, ncycle, hydrogen="all", hout=False, bfactor=None, jellybody=True,
                  resolution=None, weight_fixed=None, weight_auto_scale=None, ncsr=None,
                  keyword_files=None, keywords=None):
    ret = ""
    labin = []
    if lab_f: labin.append("FP={}".format(lab_f))
    if lab_sigf: labin.append("SIGFP={}".format(lab_sigf))
    if lab_phi: labin.append("PHIB={}".format(lab_phi))
    ret += "labin {}\n".format(" ".join(labin))

    
    ret += "make hydr {}\n".format(hydrogen)
    ret += "make hout {}\n".format("yes" if hout else "no")
    ret += "solvent no\n"
    ret += "source em mb\n"
    ret += "ncycle {}\n".format(ncycle)
    if resolution is not None:
        ret += "reso {}\n".format(resolution)
    if weight_fixed is not None:
        ret += "weight matrix {}\n".format(weight_fixed)
    elif weight_auto_scale is not None:
        ret += "weight auto {}\n".format(weight_auto_scale)
    else:
        ret += "weight auto\n"

    if bfactor is not None:
        ret += "bfactor set {}\n".format(bfactor)
    if jellybody:
        ret += "ridge dist sigma 0.01\n"
        ret += "ridge dist dmax 4.2\n"
    if ncsr:
        ret += "ncsr {}\n".format(ncsr)
    
    if keyword_files:
        for f in keyword_files:
            ret += "@{}\n".format(f)

    if keywords:
        ret += keywords

    return ret
# make_keywords()

def run_refmac(mtz_in, model_in, ncycle, lab_f, lab_phi, lab_sigf=None, hydrogen="auto", hout=False,
               libin=None, bfactor=None, jellybody=None, resolution=None, weight_auto_scale=None,
               ncsr=None, keyword_files=None, keywords=None,
               prefix="refined", exe="refmac5"):
    cmd = make_cmd(model_in, mtz_in, libin=libin, prefix=prefix, exe=exe)
    
    logger.write("Running REFMAC5..")
    log = open(prefix+".log", "w")
    p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=log, stderr=log)
    stdin = make_keywords(lab_f=lab_f, lab_phi=lab_phi, lab_sigf=lab_sigf,
                          ncycle=ncycle, resolution=resolution,
                          hydrogen=hydrogen, hout=hout, ncsr=ncsr,
                          bfactor=bfactor, jellybody=jellybody,
                          weight_auto_scale=weight_auto_scale,
                          keyword_files=keyword_files,
                          keywords=keywords)
    open(prefix+".inp", "w").write(stdin)
    p.stdin.write(stdin.encode("utf-8"))
    p.stdin.close()
    ret = p.wait()
    logger.write("REFMAC5 finished with {}".format(ret))

# run_refmac()

def main(args):
    model_format = utils.fileio.check_model_format(args.model)
    if args.map or args.halfmaps:
        args.output_model_prefix = "shifted_local"
        args.output_mtz_prefix = "masked_fs"
        args.remove_multiple_models = True
        spa.sfcalc.main(args)
        args.mtz = "masked_fs_obs.mtz"
        if args.halfmaps:
            args.mtz_half = ["masked_fs_half1.mtz", "masked_fs_half2.mtz"]
        args.lab_phi = "Pout0"
        if args.blur:
            args.lab_f = "FoutBlur_{:.2f}".format(args.blur[0])
        else:
            args.lab_f = "Fout0"
        if args.shift:
            args.model = "shifted_local" + model_format
        else:
            args.model = "starting_model" + model_format

    keyword_files = []
    if args.keyword_file:
        for f in args.keyword_file:
            logger.write("Keyword file: {}".format(f))
            assert os.path.exists(f)
            keyword_files.append(f)
        
    keywords = ""
    if args.keywords:
        keywords = "\n".join(args.keywords)
            
    if args.shift:
        refmac_prefix = args.output_prefix + "_local"
        if os.path.isfile("ncsc_local.txt"):
            keyword_files.append("ncsc_local.txt")
    else:
        refmac_prefix = args.output_prefix
        if os.path.isfile("ncsc_global.txt"):
            keyword_files.append("ncsc_global.txt")


    run_refmac(mtz_in=args.mtz,
               model_in=args.model,
               ncycle=args.ncycle,
               lab_f=args.lab_f, lab_phi=args.lab_phi, lab_sigf=args.lab_sigf,
               hydrogen=args.hydrogen, hout=args.hout,
               ncsr=args.ncsr,
               libin=args.ligand,
               bfactor=args.bfactor,
               jellybody=args.jellybody,
               resolution=args.resolution,
               prefix=refmac_prefix,
               weight_auto_scale=args.weight_auto_scale,
               keyword_files=keyword_files,
               keywords=keywords,
               exe=args.exe)

    """
    # Calc diffmaps
    if 0:
        st = gemmi.read_structure(refmac_prefix+model_format)
        st.expand_ncs(gemmi.HowToNameCopiedChain.Short)
        grid_shape = 
        mask = gemmi.FloatGrid(*grid_shape)
        mask.set_unit_cell(st.cell)
        mask.spacegroup = gemmi.SpaceGroup(st.spacegroup_hm)
        mask.mask_points_in_constant_radius(st[0], args.mask_radius, 1.)
        spa.fofc.
    """

    if args.shift:
        ncsc_in = ("ncsc_global.txt") if os.path.isfile("ncsc_global.txt") else None
        spa.shiftback.shift_back(xyz_in=refmac_prefix+model_format,
                                 refine_mtz=refmac_prefix+".mtz",
                                 shifts_json="shifts.json",
                                 ncsc_in=ncsc_in,
                                 out_prefix=args.output_prefix)

    # Expand sym here
    if os.path.isfile("ncsc_global.txt"):
        refined_xyz = args.output_prefix+model_format
        logger.write("Expanding {}".format(refined_xyz))
        st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(refined_xyz)
        st.expand_ncs(gemmi.HowToNameCopiedChain.Short)
        utils.fileio.write_model(st, file_name=args.output_prefix+"_expanded"+model_format,
                                 cif_ref=cif_ref)


    if args.cross_validation:
        st = gemmi.read_structure(refmac_prefix+model_format)
        st = utils.model.shake_structure(st, args.shake_radius)
        shaken_file = refmac_prefix+"_shaken"+model_format
        utils.fileio.write_model(st, file_name=shaken_file)
        if args.shift:
            refmac_prefix_shaken = refmac_prefix+"_shaken_refined_local"
        else:
            refmac_prefix_shaken = refmac_prefix+"_shaken_refined"

        run_refmac(mtz_in=args.mtz_half[0],
                   model_in=shaken_file,
                   ncycle=args.ncycle,
                   lab_f=args.lab_f, lab_phi=args.lab_phi, lab_sigf=args.lab_sigf,
                   hydrogen=args.hydrogen, hout=args.hout,
                   bfactor=args.bfactor,
                   jellybody=args.jellybody,
                   resolution=args.resolution,
                   prefix=refmac_prefix_shaken,
                   keyword_files=keyword_files,
                   keywords=keywords)

        # TODO calc FSC

        if args.shift:
            ncsc_in = ("ncsc_global.txt") if os.path.isfile("ncsc_global.txt") else None
            spa.shiftback.shift_back(xyz_in=refmac_prefix_shaken+model_format,
                                 refine_mtz=refmac_prefix_shaken+".mtz",
                                 shifts_json="shifts.json",
                                 ncsc_in=ncsc_in,
                                 out_prefix=refmac_prefix+"_shaken_refined")
        
        
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)

