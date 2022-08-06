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
from servalcat import spa

def add_arguments(parser):
    parser.description = 'Run REFMAC5 for SPA'

    parser.add_argument('--exe', default="refmac5",
                        help='refmac5 binary (default: %(default)s)')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    # sfcalc options
    sfcalc_group = parser.add_argument_group("sfcalc")
    spa.sfcalc.add_sfcalc_args(sfcalc_group)

    # run_refmac options
    # TODO use group! like refmac options
    parser.add_argument('--ligand', nargs="*", action="append",
                        help="restraint dictionary cif file(s)")
    parser.add_argument('--bfactor', type=float,
                        help="reset all atomic B values to specified value")
    parser.add_argument('--ncsr', default="local", choices=["local", "global"],
                        help="local or global NCS restrained (default: %(default)s)")
    parser.add_argument('--ncycle', type=int, default=10,
                        help="number of cycles in Refmac (default: %(default)d)")
    parser.add_argument('--tlscycle', type=int, default=0,
                        help="number of TLS cycles in Refmac (default: %(default)d)")
    parser.add_argument('--tlsin',
                        help="TLS parameter input for Refmac")
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no"],
                        help="all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input. "
                        "Default: %(default)s")
    parser.add_argument('--jellybody', action='store_true',
                        help="Use jelly body restraints")
    parser.add_argument('--jellybody_params', nargs=2, type=float,
                        metavar=("sigma", "dmax"), default=[0.01, 4.2],
                        help="Jelly body sigma and dmax (default: %(default)s)")
    parser.add_argument('--hout', action='store_true', help="write hydrogen atoms in the output model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--weight_auto_scale', type=float,
                       help="'weight auto' scale value. automatically determined from resolution and mask/box volume ratio if unspecified")
    group.add_argument('--weight_matrix', type=float,
                       help="weight matrix value")
    parser.add_argument('--keywords', nargs='+', action="append",
                        help="refmac keyword(s)")
    parser.add_argument('--keyword_file', nargs='+', action="append",
                        help="refmac keyword file(s)")
    parser.add_argument('--external_restraints_json')
    parser.add_argument('--show_refmac_log', action='store_true',
                        help="show all Refmac log instead of summary")
    parser.add_argument('--output_prefix', default="refined",
                        help='output file name prefix (default: %(default)s)')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Run cross validation')
    parser.add_argument('--cross_validation_method', default="shake", choices=["throughout", "shake"],
                        help="shake: randomize a model refined against a full map and then refine it against a half map, "
                        "throughout: use only a half map for refinement (another half map is used for error estimation) "
                        "Default: %(default)s")
    parser.add_argument('--shake_radius', default=0.3,
                        help='Shake rmsd in case of --cross_validation_method=shake (default: %(default).1f)')
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

def calc_fsc(st, output_prefix, maps, d_min, mask, mask_radius, b_before_mask, no_sharpen_before_mask, make_hydrogen, monlib,
             blur=None, d_min_fsc=None, cross_validation=False, cross_validation_method=None, st_sr=None):
    # st_sr: shaken-and-refined st in case of cross_validation_method=="shake"
    if cross_validation:
        assert len(maps) == 2
        assert cross_validation_method in ("shake", "throughout")
    if cross_validation and cross_validation_method == "shake":
        assert st_sr is not None
    else:
        assert st_sr is None
    
    logger.write("Calculating map-model FSC..")

    if d_min_fsc is None:
        d_min_fsc = utils.maps.nyquist_resolution(maps[0][0])
        logger.write("  --fsc_resolution is not specified. Using Nyquist resolution: {:.2f}".format(d_min_fsc))
    
    st = st.clone()
    if st_sr is not None: st_sr = st_sr.clone()

    if make_hydrogen == "all":
        utils.restraints.add_hydrogens(st, monlib)
        if st_sr is not None: utils.restraints.add_hydrogens(st_sr, monlib)

    if mask is not None or mask_radius is not None:
        if mask is None:
            mask = gemmi.FloatGrid(*maps[0][0].shape)
            mask.set_unit_cell(st.cell)
            mask.spacegroup = st.find_spacegroup()
            mask.mask_points_in_constant_radius(st[0], mask_radius, 1.)
        if no_sharpen_before_mask or len(maps) < 2:
            for ma in maps: ma[0].array[:] *= mask
        else:
            # It seems we need different B for different resolution limit
            if b_before_mask is None: b_before_mask = spa.sfcalc.determine_b_before_mask(st, maps, maps[0][1], mask, d_min_fsc)
            maps = utils.maps.sharpen_mask_unsharpen(maps, mask, d_min_fsc, b=b_before_mask)
        
    hkldata = utils.maps.mask_and_fft_maps(maps, d_min_fsc)
    hkldata.df["FC"] = utils.model.calc_fc_fft(st, d_min_fsc - 1e-6, cutoff=1e-7, monlib=monlib, source="electron",
                                               miller_array=hkldata.miller_array())
    labs_fc = ["FC"]

    if st_sr is not None:
        hkldata.df["FC_sr"] = utils.model.calc_fc_fft(st_sr, d_min_fsc - 1e-6, cutoff=1e-7, monlib=monlib, source="electron",
                                                      miller_array=hkldata.miller_array())
        labs_fc.append("FC_sr")

    if blur is not None:
        logger.write(" Unblurring Fc with B={} for FSC calculation".format(blur))
        unblur = numpy.exp(blur/hkldata.d_spacings().to_numpy()**2/4.)
        for lab in labs_fc:
            hkldata.df[lab] *= unblur
    
    hkldata.setup_relion_binning()
    stats = spa.fsc.calc_fsc(hkldata, labs_fc=labs_fc, lab_f="FP",
                             labs_half=["F_map1", "F_map2"] if len(maps)==2 else None)

    hkldata2 = hkldata.copy(d_min=d_min) # for FSCaverage at resolution for refinement # XXX more efficient way
    hkldata2.setup_relion_binning()
    stats2 = spa.fsc.calc_fsc(hkldata2, labs_fc=labs_fc, lab_f="FP",
                              labs_half=["F_map1", "F_map2"] if len(maps)==2 else None)
    
    if "fsc_half" in stats:
        with numpy.errstate(invalid="ignore"): # XXX negative fsc results in nan!
            stats.loc[:,"fsc_full_sqrt"] = numpy.sqrt(2*stats.fsc_half/(1+stats.fsc_half))

    logger.write(stats.to_string()+"\n")

    # remove and rename columns
    for s in (stats, stats2):
        s.rename(columns=dict(fsc_FC_full="fsc_model", Rcmplx_FC_full="Rcmplx"), inplace=True)
        if cross_validation:
            if cross_validation_method == "shake":
                s.drop(columns=["fsc_FC_half1", "fsc_FC_half2", "fsc_FC_sr_full", "Rcmplx_FC_sr_full"], inplace=True)
                s.rename(columns=dict(fsc_FC_sr_half1="fsc_model_half1",
                                          fsc_FC_sr_half2="fsc_model_half2"), inplace=True)
            else:
                s.rename(columns=dict(fsc_FC_half1="fsc_model_half1",
                                          fsc_FC_half2="fsc_model_half2"), inplace=True)
        else:
            s.drop(columns=[x for x in s if x.startswith("fsc_FC") and x.endswith(("half1","half2"))], inplace=True)

    # FSCaverages
    fscavg_text  = "Map-model FSCaverages (at {:.2f} A):\n".format(d_min)
    fscavg_text += " FSCaverage(full) = {: .4f}\n".format(spa.fsc.fsc_average(stats2.ncoeffs, stats2.fsc_model))
    if cross_validation:
        fscavg_text += "Cross-validated map-model FSCaverages:\n"
        fscavg_text += " FSCaverage(half1)= {: .4f}\n".format(spa.fsc.fsc_average(stats2.ncoeffs, stats2.fsc_model_half1))
        fscavg_text += " FSCaverage(half2)= {: .4f}\n".format(spa.fsc.fsc_average(stats2.ncoeffs, stats2.fsc_model_half2))
    
    # for loggraph
    fsc_logfile = "{}_fsc.log".format(output_prefix)
    with open(fsc_logfile, "w") as ofs:
        columns = "1/resol^2 ncoef ln(Mn(|Fo|^2)) ln(Mn(|Fc|^2)) FSC(full,model) FSC_half FSC_full_sqrt FSC(half1,model) FSC(half2,model) Rcmplx(full,model)".split()
        
        ofs.write("$TABLE: Map-model FSC after refinement:\n")
        if len(maps) == 2:
            if cross_validation: fsc_cols = [5,6,7,8,9]
            else:                fsc_cols = [5,6,7]
        else: fsc_cols = [5]
        fsc_cols.append(10)
        if len(maps) == 2: ofs.write("$GRAPHS: FSC :A:1,5,6,7:\n")
        else:              ofs.write("$GRAPHS: FSC :A:1,5:\n")
        if cross_validation: ofs.write(": cross-validated FSC :A:1,8,9:\n".format(",".join(map(str,fsc_cols))))
        ofs.write(": Rcmplx :A:1,{}:\n".format(4+len(fsc_cols)))
        ofs.write(": ln(Mn(|F|^2)) :A:1,3,4:\n")
        ofs.write(": number of Fourier coeffs :A:1,2:\n")
        ofs.write("$$ {}$$\n".format(" ".join(columns[:4]+[columns[i-1] for i in fsc_cols])))
        ofs.write("$$\n")

        plot_columns = ["d_min", "ncoeffs", "power_FP", "power_FC", "fsc_model"]
        if len(maps) == 2:
            plot_columns.extend(["fsc_half", "fsc_full_sqrt"])
            if cross_validation:
                plot_columns.extend(["fsc_model_half1", "fsc_model_half2"])
        plot_columns.append("Rcmplx")
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

    return fscavg_text
# calc_fsc()

def main(args):
    if not args.model:
        raise SystemExit("Error: give --model.")

    if not (args.map or args.halfmaps):
        raise SystemExit("Error: give --map | --halfmaps.")

    if args.mask_for_fofc and not os.path.exists(args.mask_for_fofc):
        raise SystemExit("Error: --mask_for_fofc {} does not exist".format(args.mask_for_fofc))

    if args.mask_for_fofc and args.mask_radius_for_fofc:
        raise SystemExit("Error: you cannot specify both --mask_for_fofc and --mask_radius_for_fofc")

    if args.trim_fofc_mtz and not (args.mask_for_fofc or args.mask_radius_for_fofc):
        raise SystemExit("Error: --trim_fofc_mtz is specified but --mask_for_fofc is not given")

    if args.ligand: args.ligand = sum(args.ligand, [])

    st = utils.fileio.read_structure(args.model)
    try:
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                                       stop_for_unknowns=True, check_hydrogen=(args.hydrogen=="yes"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if args.mask_for_fofc and not args.no_check_mask_with_model:
        mask = utils.fileio.read_ccp4_map(args.mask_for_fofc)[0]
        if not utils.maps.test_mask_with_model(mask, st):
            raise SystemExit("\nError: Model is out of mask.\n"
                             "Please check your --model and --mask_for_fofc. You can disable this test with --no_check_mask_with_model.")
        
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

    chain_id_lens = [len(x) for x in utils.model.all_chain_ids(st)]
    keep_chain_ids = (chain_id_lens and max(chain_id_lens) == 1) # always kept unless one-letter chain IDs

    # FIXME if mtz is given and sfcalc() not ran?
    has_ncsc = "ncsc_file" in file_info
    if has_ncsc:
        args.keyword_file.append(file_info["ncsc_file"])

    if not args.no_trim:
        refmac_prefix = "{}_{}".format(args.shifted_model_prefix, args.output_prefix)
    else:
        refmac_prefix = args.output_prefix # XXX this should be different name (nomask etc?)

    # Weight auto scale
    if args.weight_matrix is None and args.weight_auto_scale is None:
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

    # Take care of TLS in
    if args.tlsin and not args.no_shift and "shifts" in file_info:
        logger.write("Shifting origin in tlsin")
        tlsgroups = utils.refmac.read_tls_file(args.tlsin)
        spa.shiftback.shift_back_tls(tlsgroups, -file_info["shifts"]) # tlsgroups is modified
        args.tlsin = "shifted.tls"
        utils.refmac.write_tls_file(tlsgroups, args.tlsin)
        
    # Run Refmac
    refmac = utils.refmac.Refmac(prefix=refmac_prefix, args=args, global_mode="spa",
                                 keep_chain_ids=keep_chain_ids)
    refmac.set_libin(args.ligand)
    try:
        refmac_summary = refmac.run_refmac()
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    json.dump(refmac_summary,
              open("{}_summary.json".format(refmac_prefix), "w"),
              indent=True)

    if args.halfmaps:
        maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]

    # Modify output
    st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(refmac_prefix+model_format)
    utils.model.adp_analysis(st)
    
    if not args.no_trim:
        st.cell = maps[0][0].unit_cell
        if not args.no_shift:
            spa.shiftback.shift_back_model(st, file_info["shifts"]) # st is modified
    
    if "refmac_fixes" in file_info:
        file_info["refmac_fixes"].modify_back(st)
    utils.fileio.write_model(st, prefix=args.output_prefix,
                             pdb=True, cif=True, cif_ref=cif_ref)

    # Take care of TLS out
    if not args.no_trim: # if no_trim, there is nothing to do.
        tlsout = refmac.tlsout()
        if os.path.exists(tlsout):
            if not args.no_shift:
                logger.write("Shifting origin in tlsout")
                tlsgroups = utils.refmac.read_tls_file(tlsout)
                spa.shiftback.shift_back_tls(tlsgroups, file_info["shifts"]) # tlsgroups is modified
                utils.refmac.write_tls_file(tlsgroups, args.output_prefix+".tls")
            else:
                logger.write("Copying tlsout")
                shutil.copyfile(refmac.tlsout(), args.output_prefix+".tls")

    # Expand sym here
    st_expanded = st.clone()
    if has_ncsc:
        utils.model.expand_ncs(st_expanded)
        utils.fileio.write_model(st_expanded, file_name=args.output_prefix+"_expanded"+model_format,
                                 cif_ref=cif_ref)

    if args.cross_validation and args.cross_validation_method == "shake":
        logger.write("Cross validation is requested.")
        refmac_prefix_shaken = refmac_prefix+"_shaken_refined"
        logger.write("Starting refinement using half map 1 (model is shaken first)")
        refmac_hm1 = refmac.copy(hklin=args.mtz_half[0],
                                 xyzin=refmac_prefix+model_format,
                                 prefix=refmac_prefix_shaken,
                                 shake=args.shake_radius,
                                 jellybody=False) # makes no sense to use jelly body after shaking
        if args.jellybody: logger.write("  Turning off jelly body")
        if "lab_f_half1" in file_info:
            refmac_hm1.lab_f = file_info["lab_f_half1"]
            refmac_hm1.lab_phi = file_info["lab_phi_half1"]
            # SIGMA?
            
        try:
            refmac_hm1.run_refmac()
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))
        
        # Modify output
        st_sr, cif_ref_sr = utils.fileio.read_structure_from_pdb_and_mmcif(refmac_prefix_shaken+model_format)
        if not args.no_trim:
            st_sr.cell = maps[0][0].unit_cell
            if not args.no_shift:
                spa.shiftback.shift_back_model(st_sr, file_info["shifts"])
            
        if "refmac_fixes" in file_info:
            file_info["refmac_fixes"].modify_back(st)

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

    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None
        
    # Calc FSC
    fscavg_text = calc_fsc(st_expanded, args.output_prefix, maps,
                           args.resolution, mask=mask, mask_radius=args.mask_radius if not args.no_mask else None,
                           b_before_mask=args.b_before_mask,
                           no_sharpen_before_mask=args.no_sharpen_before_mask,
                           make_hydrogen=args.hydrogen,
                           monlib=monlib, cross_validation=args.cross_validation,
                           blur=args.blur[0] if args.blur else None,
                           d_min_fsc=args.fsc_resolution,
                           cross_validation_method=args.cross_validation_method, st_sr=st_sr_expanded)

    # Calc Fo-Fc (and updated) maps
    logger.write("Starting Fo-Fc calculation..")
    if not args.halfmaps: logger.write(" with limited functionality because half maps were not provided")
    logger.write(" model: {}".format(args.output_prefix+model_format))

    # for Fo-Fc in case of helical reconstruction, expand model more
    # XXX should we do it for FSC calculation also? Probably we should not do sharpen-unsharpen procedure for FSC calc either.
    if args.twist is not None:
        logger.write("Generating all helical copies in the box")
        st_expanded = st.clone()
        utils.symmetry.update_ncs_from_args(args, st_expanded, map_and_start=maps[0], filter_model_helical_contacting=False)
        utils.model.expand_ncs(st_expanded)
        utils.fileio.write_model(st_expanded, file_name=args.output_prefix+"_expanded_all"+model_format,
                                 cif_ref=cif_ref)

    if args.mask_for_fofc:
        logger.write("  mask: {}".format(args.mask_for_fofc))
        mask = utils.fileio.read_ccp4_map(args.mask_for_fofc)[0]
    elif args.mask_radius_for_fofc:
        logger.write("  mask: using refined model with radius of {} A".format(args.mask_radius_for_fofc))
        mask = gemmi.FloatGrid(*maps[0][0].shape)
        mask.set_unit_cell(maps[0][0].unit_cell)
        mask.spacegroup = gemmi.SpaceGroup(1)
        mask.mask_points_in_constant_radius(st_expanded[0], args.mask_radius_for_fofc, 1.)
    else:
        logger.write("  mask: not used")
        mask = None
        
    hkldata, map_labs, stats_str = spa.fofc.calc_fofc(st_expanded, args.resolution, maps, mask=mask, monlib=monlib,
                                                      half1_only=(args.cross_validation and args.cross_validation_method == "throughout"),
                                                      sharpening_b=None if args.halfmaps else 0.) # assume already sharpened if fullmap is given
    spa.fofc.write_files(hkldata, map_labs, maps[0][1], stats_str,
                         mask=mask, output_prefix="diffmap",
                         trim_map=mask is not None, trim_mtz=args.trim_fofc_mtz)

    # Final summary
    if len(refmac_summary["cycles"]) > 1 and "actual_weight" in refmac_summary["cycles"][-2]:
        final_weight = refmac_summary["cycles"][-2]["actual_weight"]
    else:
        final_weight = "???"

    adpstats_txt = ""
    adp_stats = utils.model.adp_stats_per_chain(st[0])
    max_chain_len = max([len(x[0]) for x in adp_stats])
    max_num_len = max([len(str(x[1])) for x in adp_stats])
    for chain, natoms, qs in adp_stats:
        adpstats_txt += " Chain {0:{1}s}".format(chain, max_chain_len) if chain!="*" else " {0:{1}s}".format("All", max_chain_len+6)
        adpstats_txt += " ({0:{1}d} atoms) min={2:5.1f} median={3:5.1f} max={4:5.1f} A^2\n".format(natoms, max_num_len, qs[0],qs[2],qs[4])

    # Create Coot script
    with open("{}_coot.py".format(args.output_prefix), "w") as ofs:
        ofs.write('imol = read_pdb("{}.pdb")\n'.format(args.output_prefix)) # as Coot is not good at mmcif file..
        ofs.write('imol_fo = make_and_draw_map("diffmap.mtz", "FWT", "PHWT", "", 0, 0)\n')
        ofs.write('imol_fofc = make_and_draw_map("diffmap.mtz", "DELFWT", "PHDELWT", "", 0, 1)\n')
        if mask is not None:
            ofs.write('set_contour_level_absolute(imol_fo, 1.2)\n')
            ofs.write('set_contour_level_absolute(imol_fofc, 3.0)\n')
        
    logger.write("""
=============================================================================
* Final Summary *

Rmsd from ideal
  bond lengths: {rmsbond} A
  bond  angles: {rmsangle} deg

{fscavgs}
 Run loggraph {fsclog} to see plots

ADP statistics
{adpstats}

Weight used: {final_weight}
             If you want to change the weight, give larger (looser restraints)
             or smaller (tighter) value to --weight_auto_scale=.
             
Open refined model and diffmap.mtz with COOT:
coot --script {prefix}_coot.py
=============================================================================
""".format(rmsbond=refmac_summary["cycles"][-1].get("rms_bond", "???"),
           rmsangle=refmac_summary["cycles"][-1].get("rms_angle", "???"),
           fscavgs=fscavg_text.rstrip(),
           fsclog="{}_fsc.log".format(args.output_prefix),
           adpstats=adpstats_txt.rstrip(),
           final_weight=final_weight,
           prefix=args.output_prefix))
# main()
        
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)

