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
import argparse
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
    parser.add_argument('--cross_validation_method', default="shake", choices=["throughout", "shake"])
    parser.add_argument('--shake_radius', default=0.5,
                        help='Shake rmsd')
    parser.add_argument('--mask_for_fofc', help="Mask file for Fo-Fc map calculation")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_fsc(st, output_prefix, maps, d_min, mask_radius, b_before_mask, no_sharpen_before_mask, make_hydrogen, monlib):
    logger.write("Calculating map-model FSC..")

    st = st.clone()

    if make_hydrogen == "all":
        utils.restraints.add_hydrogens(st, monlib)

    if mask_radius is not None:
        mask = gemmi.FloatGrid(*maps[0][0].shape)
        mask.set_unit_cell(st.cell)
        mask.spacegroup = st.find_spacegroup()
        mask.mask_points_in_constant_radius(st[0], mask_radius, 1.)
        if no_sharpen_before_mask or len(maps) < 2:
            maps = [[gemmi.FloatGrid(numpy.array(ma[0])*mask, mask.unit_cell, mask.spacegroup)]+ma[1:]
                    for ma in maps]
        else:
            maps = utils.maps.sharpen_mask_unsharpen(maps, mask, d_min, b=b_before_mask)
        
    hkldata = utils.maps.mask_and_fft_maps(maps, d_min)
    fc_asu = utils.model.calc_fc_fft(st, d_min, cutoff=1e-7, monlib=monlib, source="electron")
    hkldata.merge_asu_data(fc_asu, "FC")
    hkldata.setup_relion_binning()

    fsc_logfile = "{}_fsc.log".format(output_prefix)
    ofs = open(fsc_logfile, "w")
    if len(maps) == 2:
        ofs.write("""$TABLE: Map-model FSC after refinement:
$GRAPHS: FSC :A:1,5,6,7,8,9:
: number of Fourier coeffs :A:1,2:
: ln(Mn(|F|)) :A:1,3,4:
$$ 1/resol^2 ncoef ln(Mn(|F_full|)) ln(Mn(|Fc|)) FSC(full,model) FSC(half1,model) FSC(half2,model) FSC_full FSC_full_sqrt$$
$$
""")
        F_map1 = hkldata.df.F_map1.to_numpy()
        F_map2 = hkldata.df.F_map2.to_numpy()
    else:
        ofs.write("""$TABLE: Map-model FSC after refinement:
$GRAPHS: FSC :A:1,5:
: number of Fourier coeffs :A:1,2:
: ln(Mn(|F|)) :A:1,3,4:
$$ 1/resol^2 ncoef ln(Mn(|F_full|)) ln(Mn(|Fc|)) FSC(full,model) $$
$$
""")

    FP = hkldata.df.FP.to_numpy()
    FC = hkldata.df.FC.to_numpy()
    fscvals = [[], [], []]
    ncoeffs = []

    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        Fo = FP[sel]
        Fc = FC[sel]
        fsc_model = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
        ncoeffs.append(Fo.size)
        fscvals[0].append(fsc_model)
        with numpy.errstate(divide="ignore"):
            ofs.write("{:.4f} {:6d} {:.3f} {:.3f} {: .4f}".format(1/bin_d_min**2, Fo.size,
                                                                  numpy.log(numpy.average(numpy.abs(Fo))),
                                                                  numpy.log(numpy.average(numpy.abs(Fc))),
                                                                  fsc_model))
        
        if len(maps) == 2:
            F1, F2 = F_map1[sel], F_map2[sel]
            fsc_half = numpy.real(numpy.corrcoef(F1, F2)[1,0])
            fsc_full = 2*fsc_half/(1+fsc_half)
            fsc1 = numpy.real(numpy.corrcoef(F1, Fc)[1,0])
            fsc2 = numpy.real(numpy.corrcoef(F2, Fc)[1,0])
            fscvals[1].append(fsc1)
            fscvals[2].append(fsc2)
            ofs.write(" {: .4f} {: .4f} {: .4f} {: .4f}\n".format(fsc1, fsc2, fsc_full,
                                                                numpy.sqrt(fsc_full) if fsc_full >= 0 else numpy.nan ))
        else:
            ofs.write("\n")

    ofs.write("$$\n")
    ofs.close()
    
    ncoeffs = numpy.array(ncoeffs)
    sum_n = sum(ncoeffs)
    logger.write("Map-model FSCaverages:")
    logger.write(" FSCaverage(full) = {: .4f}".format(sum(ncoeffs*fscvals[0])/sum_n))
    if len(maps) == 2:
        logger.write(" FSCaverage(half1)= {: .4f}".format(sum(ncoeffs*fscvals[1])/sum_n))
        logger.write(" FSCaverage(half2)= {: .4f}".format(sum(ncoeffs*fscvals[2])/sum_n))
    logger.write(" Run loggraph {} to see plots.".format(fsc_logfile))
# calc_fsc()

def modify_output(st, inscode_mods=None):
    if not inscode_mods: return
    
    if inscode_mods is not None:
        utils.model.modify_inscodes_back(st, inscode_mods)
# modify_output()

def main(args):
    if not args.model:
        logger.write("Error: give --model.")
        return

    if not (args.map or args.halfmaps):
        logger.write("Error: give --map | --halfmaps.")
        return

    if args.ligand: args.ligand = sum(args.ligand, [])

    st = utils.fileio.read_structure(args.model)
    resnames = st[0].get_all_residue_names()
    monlib = utils.restraints.load_monomer_library(resnames,
                                                   monomer_dir=args.monlib,
                                                   cif_files=args.ligand)
    # TODO exit if there are unknown residues
    
    args.output_model_prefix = "shifted_local"
    args.output_masked_prefix = "masked_fs"
    args.output_mtz_prefix = "starting_map"
    args.remove_multiple_models = True
    file_info = spa.sfcalc.main(args, monlib=monlib)
    args.mtz = file_info["mtz_file"]
    if args.halfmaps: # FIXME if no_mask?
        args.mtz_half = [file_info["mtz_file"], file_info["mtz_file"]]
    args.lab_phi = file_info["lab_phi"]  #"Pout0"
    args.lab_f = file_info["lab_f"]
    args.lab_sigf = None
    args.model = file_info["model_file"]
    model_format = file_info["model_format"]

    if args.cross_validation and args.cross_validation_method == "throughout":
        args.lab_f = file_info["lab_f_half1"]
        args.lab_phi = file_info["lab_phi_half1"]
        # XXX args.lab_sigf?

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

    # Weight auto scale
    if args.weight is None and args.weight_auto_scale is None:
        reso = file_info["d_eff"] if "d_eff" in file_info else args.resolution
        if "vol_ratio" in file_info:
            if "d_eff" in file_info:
                rlmc = (-9.503541, 3.129882, 15.439744)
            else:
                rlmc = (-8.329418, 3.032409, 14.381907)
            logger.write("Estimating weight auto scale using resolution and volume ratio")
            ws = rlmc[0] + reso*rlmc[1] +file_info["vol_ratio"]*rlmc[2]
        else:
            if "d_eff" in file_info:
                rlmc = (-5.903807, 2.870723)
            else:
                rlmc = (-4.891140, 2.746791)
            logger.write("Estimating weight auto scale using resolution")
            ws =  rlmc[0] + args.resolution*rlmc[1]
        args.weight_auto_scale = max(0.2, min(18.0, ws))
        logger.write(" Will use weight auto {:.2f}".format(args.weight_auto_scale))

    # Run Refmac
    refmac = utils.refmac.Refmac(prefix=refmac_prefix, args=args, global_mode="spa")
    refmac.set_libin(args.ligand)
    refmac.run_refmac()

    if args.halfmaps:
        maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]

    # Modify output
    st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(refmac_prefix+model_format)
    utils.model.adp_analysis(st)
    
    if not args.no_shift:
        st.cell = maps[0][0].unit_cell
        spa.shiftback.shift_back_model(st, file_info["shifts"]) # st is modified
    
    modify_output(st, file_info.get("inscode_mods"))
    utils.fileio.write_model(st, prefix=args.output_prefix,
                             pdb=True, cif=True, cif_ref=cif_ref)

    # Expand sym here
    st_expanded = st.clone()
    if has_ncsc:
        utils.model.expand_ncs(st_expanded)
        utils.fileio.write_model(st_expanded, file_name=args.output_prefix+"_expanded"+model_format,
                                 cif_ref=cif_ref)

    if args.cross_validation and args.cross_validation_method == "shake":
        logger.write("Cross validation is requested.")
        st = utils.fileio.read_structure(refmac_prefix+model_format)
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

        if not args.no_shift:
            spa.shiftback.shift_back(xyz_in=refmac_prefix_shaken+model_format,
                                     shifts_json="shifts.json",
                                     out_prefix=args.output_prefix+"_shaken_refined")
        
    if args.halfmaps:
        maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]

    # Calc FSC
    calc_fsc(st_expanded, args.output_prefix, maps,
             args.resolution, mask_radius=args.mask_radius if not args.no_mask else None,
             b_before_mask=args.b_before_mask,
             no_sharpen_before_mask=args.no_sharpen_before_mask,
             make_hydrogen=args.hydrogen,
             monlib=monlib)

    # Calc updated and Fo-Fc maps
    if args.halfmaps:
        logger.write("Starting Fo-Fc calculation..")
        logger.write(" model: {}".format(args.output_prefix+model_format))

        if args.mask_for_fofc:
            mask = numpy.array(utils.fileio.read_ccp4_map(args.mask_for_fofc)[0])
        else:
            mask = None

        hkldata, map_labs, stats_str = spa.fofc.calc_fofc(st_expanded, args.resolution, maps, mask=mask, monlib=monlib,
                                                          half1_only=(args.cross_validation and args.cross_validation_method == "throughout"))
        spa.fofc.write_files(hkldata, map_labs, maps[0][1], stats_str,
                             mask=mask, output_prefix="diffmap",
                             crop=mask is not None, normalize_map=mask is not None)
    else:
        logger.write("Will not calculate Fo-Fc map because half maps were not provided")
        
# main()
        
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)

