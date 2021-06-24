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
    parser.add_argument('--bfactor', type=float,
                        help="reset all atomic B values to specified value")
    parser.add_argument('--ncsr', default="local", choices=["local", "global"],
                        help="local or global NCS restrained")
    parser.add_argument('--ncycle', type=int, default=10)
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no"],
                        help="all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input")
    parser.add_argument('--jellybody', action='store_true')
    parser.add_argument('--jellybody_params', nargs=2, type=float,
                        metavar=("sigma", "dmax"), default=[0.01, 4.2])
    parser.add_argument('--hout', action='store_true', help="write hydrogen atoms in the output model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--weight_auto_scale', type=float,
                       help="'weight auto' scale value. automatically determined from resolution and mask/box volume ratio if unspecified")
    group.add_argument('--weight', type=float,
                       help="weight matrix value")
    parser.add_argument('--keywords', nargs='+', action="append",
                        help="refmac keyword(s)")
    parser.add_argument('--keyword_file', nargs='+', action="append",
                        help="refmac keyword file(s)")
    parser.add_argument('--external_restraints_json')
    parser.add_argument('--show_refmac_log', action='store_true')
    parser.add_argument('--output_prefix', default="refined",
                        help='output file name prefix')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Run cross validation')
    parser.add_argument('--cross_validation_method', default="shake", choices=["throughout", "shake"],
                        help="shake: randomize a model refined against a full map and then refine it against a half map, "
                        "throughout: use only a half map for refinement (another half map is used for error estimation)")
    parser.add_argument('--shake_radius', default=0.5,
                        help='Shake rmsd in case of --cross_validation_method=shake')
    parser.add_argument('--mask_for_fofc', help="Mask file for Fo-Fc map calculation")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_fsc(st, output_prefix, maps, d_min, mask_radius, b_before_mask, no_sharpen_before_mask, make_hydrogen, monlib,
             cross_validation=False, cross_validation_method=None, st_sr=None):
    # XXX blur is not considered
    # st_sr: shaken-and-refined st in case of cross_validation_method=="shake"
    if cross_validation:
        assert len(maps) == 2
        assert cross_validation_method in ("shake", "throughout")
    if cross_validation and cross_validation_method == "shake":
        assert st_sr is not None
    else:
        assert st_sr is None
    
    logger.write("Calculating map-model FSC..")

    st = st.clone()
    if st_sr is not None: st_sr = st_sr.clone()

    if make_hydrogen == "all":
        utils.restraints.add_hydrogens(st, monlib)
        if st_sr is not None: utils.restraints.add_hydrogens(st_sr, monlib)

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
    labs_fc = ["FC"]

    if st_sr is not None:
        fc_sr_asu = utils.model.calc_fc_fft(st_sr, d_min, cutoff=1e-7, monlib=monlib, source="electron")
        hkldata.merge_asu_data(fc_sr_asu, "FC_sr")
        labs_fc.append("FC_sr")
    
    hkldata.setup_relion_binning()
    stats = spa.fsc.calc_fsc(hkldata, labs_fc=labs_fc, lab_f="FP",
                             labs_half=["F_map1", "F_map2"] if len(maps)==2 else None)
    
    if "fsc_half" in stats:
        with numpy.errstate(invalid="ignore"): # XXX negative fsc results in nan!
            stats["fsc_full_sqrt"] = numpy.sqrt(2*stats.fsc_half/(1+stats.fsc_half))

    logger.write(stats.to_string()+"\n")

    # remove and rename columns
    stats.rename(columns=dict(fsc_FC_full="fsc_model"), inplace=True)
    if cross_validation:
        if cross_validation_method == "shake":
            stats.drop(columns=["fsc_FC_half1", "fsc_FC_half2", "fsc_FC_sr_full"], inplace=True)
            stats.rename(columns=dict(fsc_FC_sr_half1="fsc_model_half1",
                                      fsc_FC_sr_half2="fsc_model_half2"), inplace=True)
        else:
            stats.rename(columns=dict(fsc_FC_half1="fsc_model_half1",
                                      fsc_FC_half2="fsc_model_half2"), inplace=True)
    else:
        stats.drop(columns=[x for x in stats if x.startswith("fsc_FC") and x.endswith(("half1","half2"))], inplace=True)

    # FSCaverages
    fscavg_text  = "Map-model FSCaverages:\n"
    fscavg_text += " FSCaverage(full) = {: .4f}\n".format(spa.fsc.fsc_average(stats.ncoeffs, stats.fsc_model))
    if cross_validation:
        fscavg_text += "Cross-validated map-model FSCaverages:\n"
        fscavg_text += " FSCaverage(half1)= {: .4f}\n".format(spa.fsc.fsc_average(stats.ncoeffs, stats.fsc_model_half1))
        fscavg_text += " FSCaverage(half2)= {: .4f}\n".format(spa.fsc.fsc_average(stats.ncoeffs, stats.fsc_model_half2))
    
    # for loggraph
    fsc_logfile = "{}_fsc.log".format(output_prefix)
    with open(fsc_logfile, "w") as ofs:
        columns = "1/resol^2 ncoef ln(Mn(|Fo|^2)) ln(Mn(|Fc|^2)) FSC(full,model) FSC_half FSC_full_sqrt FSC(half1,model) FSC(half2,model)".split()
        
        ofs.write("$TABLE: Map-model FSC after refinement:\n")
        if len(maps) == 2:
            if cross_validation: fsc_cols = [5,6,7,8,9]
            else:                fsc_cols = [5,6,7]
        else: fsc_cols = [5]
        if len(maps) == 2: ofs.write("$GRAPHS: FSC :A:1,5,6,7:\n")
        else:              ofs.write("$GRAPHS: FSC :A:1,5:\n")
        if cross_validation: ofs.write(": cross-validated FSC :A:1,8,9:\n".format(",".join(map(str,fsc_cols))))
        ofs.write(": ln(Mn(|F|^2)) :A:1,3,4:\n")
        ofs.write(": number of Fourier coeffs :A:1,2:\n")
        ofs.write("$$ {}$$\n".format(" ".join(columns[:4]+[columns[i-1] for i in fsc_cols])))
        ofs.write("$$\n")

        plot_columns = ["d_min", "ncoeffs", "power_FP", "power_FC", "fsc_model"]
        if len(maps) == 2:
            plot_columns.extend(["fsc_half", "fsc_full_sqrt"])
            if cross_validation:
                plot_columns.extend(["fsc_model_half1", "fsc_model_half2"])

        with numpy.errstate(divide="ignore"):
            log_format = lambda x: "{:.3f}".format(numpy.log(x))
            ofs.write(stats.to_string(header=False, index=False, index_names=False, columns=plot_columns,
                                      formatters=dict(d_min=lambda x: "{:.4f}".format(1/x**2),
                                                      power_FP=log_format, power_FC=log_format)))
        ofs.write("\n")
        ofs.write("$$\n\n")
        ofs.write(fscavg_text)
    
    logger.write(fscavg_text, end="")
    logger.write("Run loggraph {} to see plots.".format(fsc_logfile))
    
    # write json
    json.dump(stats.to_dict("records"),
              open("{}_fsc.json".format(output_prefix), "w"),
              indent=True)
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
    
    args.shifted_model_prefix = "shifted"
    args.output_masked_prefix = "masked_fs"
    args.output_mtz_prefix = "starting_map"
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
        refmac_prefix = "{}_{}".format(args.shifted_model_prefix, args.output_prefix)
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
        
        # Modify output
        st_sr, cif_ref_sr = utils.fileio.read_structure_from_pdb_and_mmcif(refmac_prefix_shaken+model_format)
        if not args.no_shift:
            st_sr.cell = maps[0][0].unit_cell
            spa.shiftback.shift_back_model(st_sr, file_info["shifts"])
            
        modify_output(st_sr, file_info.get("inscode_mods"))
        utils.fileio.write_model(st_sr, prefix=args.output_prefix+"_shaken_refined",
                                 pdb=True, cif=True, cif_ref=cif_ref_sr)

        # Expand sym here
        st_sr_expanded = st_sr.clone()
        if has_ncsc:
            utils.model.expand_ncs(st_sr_expanded)
            utils.fileio.write_model(st_sr_expanded, file_name=args.output_prefix+"shaken_refined_expanded"+model_format,
                                     cif_ref=cif_ref_sr)

    else:
        st_sr_expanded = None
        
    # Calc FSC
    calc_fsc(st_expanded, args.output_prefix, maps,
             args.resolution, mask_radius=args.mask_radius if not args.no_mask else None,
             b_before_mask=args.b_before_mask,
             no_sharpen_before_mask=args.no_sharpen_before_mask,
             make_hydrogen=args.hydrogen,
             monlib=monlib, cross_validation=args.cross_validation,
             cross_validation_method=args.cross_validation_method, st_sr=st_sr_expanded)

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

