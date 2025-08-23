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
from servalcat import spa # this is not a right style
from servalcat.spa import shift_maps
from servalcat.refmac.refmac_wrapper import prepare_crd

def add_arguments(parser):
    parser.description = 'Run REFMAC5 for SPA'

    parser.add_argument('--exe', default="refmac5",
                        help='refmac5 binary (default: %(default)s)')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    # sfcalc options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--halfmaps", nargs=2, help="Input half map files")
    group.add_argument("--map", help="Use this only if you really do not have half maps.")
    parser.add_argument('--mask',
                        help='Mask file')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument('--mask_radius',
                        type=float, default=3,
                        help='')
    parser.add_argument('--mask_soft_edge',
                        type=float, default=0,
                        help='Add soft edge to model mask. Should use with --no_sharpen_before_mask?')
    parser.add_argument('--padding',
                        type=float, 
                        help='Default: 2*mask_radius')
    parser.add_argument('--no_mask', action='store_true')
    parser.add_argument('--invert_mask', action='store_true', help='not for refinement.')
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--resolution',
                        type=float,
                        help='')
    parser.add_argument('--no_trim',
                        action='store_true',
                        help='Keep original box (not recommended)')
    parser.add_argument('--blur',
                        type=float, default=0,
                        help='Sharpening or blurring B')
    utils.symmetry.add_symmetry_args(parser) # add --pg etc
    parser.add_argument('--contacting_only', action="store_true", help="Filter out non-contacting NCS")
    parser.add_argument('--ignore_symmetry', action='store_true',
                        help='Ignore symmetry information (MTRIX/_struct_ncs_oper) in the model file')
    parser.add_argument('--find_links', action='store_true', 
                        help='Automatically add links')
    parser.add_argument("--b_before_mask", type=float,
                        help="sharpening B value for sharpen-mask-unsharpen procedure. By default it is determined automatically.")
    parser.add_argument('--no_sharpen_before_mask', action='store_true',
                        help='By default half maps are sharpened before masking by std of signal and unsharpened after masking. This option disables it.')
    parser.add_argument('--no_fix_microheterogeneity', action='store_true', 
                        help='By default it will fix microheterogeneity for Refmac')
    parser.add_argument('--no_fix_resi9999', action='store_true', 
                        help='By default it will split chain if max residue number > 9999 which is not supported by Refmac')
    parser.add_argument('--no_check_ncs_overlaps', action='store_true', 
                        help='Disable model overlap (e.g. expanded model is used with --pg) test')
    parser.add_argument('--no_check_ncs_map', action='store_true', 
                        help='Disable map NCS consistency test')
    parser.add_argument('--no_check_mask_with_model', action='store_true', 
                        help='Disable mask test using model')
    parser.add_argument("--prepare_only", action='store_true',
                        help="Stop before refinement")
    parser.add_argument("--no_refmacat", action='store_true',
                        help="By default uses gemmi for crd/rst file preparation (do not use makecif)")
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
    parser.add_argument('--trim_fofc_mtz', action="store_true", help="maps mtz will have smaller cell (if --mask_for_fofc is given)")
    parser.add_argument("--fsc_resolution", type=float,
                        help="High resolution limit for FSC calculation. Default: Nyquist")

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_fsc(st, output_prefix, maps, d_min, mask, mask_radius, soft_edge, b_before_mask, no_sharpen_before_mask, make_hydrogen, monlib,
             blur=0, d_min_fsc=None, cross_validation=False, cross_validation_method=None, st_sr=None,
             source="electron"):
    # st_sr: shaken-and-refined st in case of cross_validation_method=="shake"
    if cross_validation:
        assert len(maps) == 2
        assert cross_validation_method in ("shake", "throughout")
    if cross_validation and cross_validation_method == "shake":
        assert st_sr is not None
    else:
        assert st_sr is None
    
    logger.writeln("Calculating map-model FSC..")
    ret = {"summary": {}}

    if d_min_fsc is None:
        d_min_fsc = utils.maps.nyquist_resolution(maps[0][0])
        logger.writeln("  --fsc_resolution is not specified. Using Nyquist resolution: {:.2f}".format(d_min_fsc))
    
    st = st.clone()
    if st_sr is not None: st_sr = st_sr.clone()

    if make_hydrogen == "all":
        utils.restraints.add_hydrogens(st, monlib)
        if st_sr is not None: utils.restraints.add_hydrogens(st_sr, monlib)

    if mask is not None or mask_radius is not None:
        if mask is None:
            assert maps[0][0].unit_cell == st.cell
            mask = utils.maps.mask_from_model(st, mask_radius, soft_edge=soft_edge, grid=maps[0][0])
        if no_sharpen_before_mask or len(maps) < 2:
            for ma in maps: ma[0].array[:] *= mask
        else:
            # It seems we need different B for different resolution limit
            if b_before_mask is None: b_before_mask = determine_b_before_mask(st, maps, maps[0][1], mask, d_min_fsc)
            maps = utils.maps.sharpen_mask_unsharpen(maps, mask, d_min_fsc, b=b_before_mask)
        
    hkldata = utils.maps.mask_and_fft_maps(maps, d_min_fsc)
    hkldata.df["FC"] = utils.model.calc_fc_fft(st, d_min_fsc - 1e-6, monlib=monlib, source=source,
                                               miller_array=hkldata.miller_array())
    # XXX didn't apply mask to FC!!
    labs_fc = ["FC"]

    if st_sr is not None:
        hkldata.df["FC_sr"] = utils.model.calc_fc_fft(st_sr, d_min_fsc - 1e-6, monlib=monlib, source=source,
                                                      miller_array=hkldata.miller_array())
        labs_fc.append("FC_sr")

    if blur != 0:
        logger.writeln(" Unblurring Fc with B={} for FSC calculation".format(blur))
        unblur = numpy.exp(blur/hkldata.d_spacings().to_numpy()**2/4.)
        for lab in labs_fc:
            hkldata.df[lab] *= unblur
    
    hkldata.setup_relion_binning("stat")
    stats = spa.fsc.calc_fsc_all(hkldata, labs_fc=labs_fc, lab_f="FP",
                                 labs_half=["F_map1", "F_map2"] if len(maps)==2 else None)

    hkldata2 = hkldata.copy(d_min=d_min) # for FSCaverage at resolution for refinement # XXX more efficient way
    hkldata2.setup_relion_binning("stat")
    stats2 = spa.fsc.calc_fsc_all(hkldata2, labs_fc=labs_fc, lab_f="FP",
                                  labs_half=["F_map1", "F_map2"] if len(maps)==2 else None)
    
    if "fsc_half" in stats:
        with numpy.errstate(invalid="ignore"): # XXX negative fsc results in nan!
            stats.loc[:,"fsc_full_sqrt"] = numpy.sqrt(2*stats.fsc_half/(1+stats.fsc_half))

    logger.writeln(stats.to_string()+"\n")

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
    ret["summary"]["d_min"] = d_min
    ret["summary"]["FSCaverage"] = spa.fsc.fsc_average(stats2.ncoeffs, stats2.fsc_model)
    if cross_validation:
        ret["summary"]["FSCaverage_half1"] = spa.fsc.fsc_average(stats2.ncoeffs, stats2.fsc_model_half1)
        ret["summary"]["FSCaverage_half2"] = spa.fsc.fsc_average(stats2.ncoeffs, stats2.fsc_model_half2)
    fscavg_text  = "Map-model FSCaverages (at {:.2f} A):\n".format(d_min)
    fscavg_text += " FSCaverage(full) = {: .4f}\n".format(ret["summary"]["FSCaverage"])
    if cross_validation:
        fscavg_text += "Cross-validated map-model FSCaverages:\n"
        fscavg_text += " FSCaverage(half1)= {: .4f}\n".format(ret["summary"]["FSCaverage_half1"])
        fscavg_text += " FSCaverage(half2)= {: .4f}\n".format(ret["summary"]["FSCaverage_half2"])
    
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
    
    logger.write(fscavg_text)
    logger.writeln("Run loggraph {} to see plots.".format(fsc_logfile))
    
    # write json
    with open("{}_fsc.json".format(output_prefix), "w") as f:
        json.dump(stats.to_dict("records"), f, indent=True)
    ret["binned"] = stats2.to_dict(orient="records")
    return fscavg_text, ret
# calc_fsc()

def calc_fofc(st, st_expanded, maps, monlib, model_format, args, diffmap_prefix="diffmap", source="electron"):
    logger.writeln("Starting Fo-Fc calculation..")
    if not args.halfmaps: logger.writeln(" with limited functionality because half maps were not provided")
    logger.writeln(" model: {}".format(args.output_prefix+model_format))
    
    # for Fo-Fc in case of helical reconstruction, expand model more
    # XXX should we do it for FSC calculation also? Probably we should not do sharpen-unsharpen procedure for FSC calc either.
    if args.twist is not None:
        logger.writeln("Generating all helical copies in the box")
        st_expanded = st.clone()
        utils.symmetry.update_ncs_from_args(args, st_expanded, map_and_start=maps[0], filter_contacting=False)
        utils.model.expand_ncs(st_expanded)
        utils.fileio.write_model(st_expanded, args.output_prefix+"_expanded_all", pdb=True, cif=True)

    if args.mask_for_fofc:
        logger.writeln("  mask: {}".format(args.mask_for_fofc))
        mask = utils.fileio.read_ccp4_map(args.mask_for_fofc)[0]
    elif args.mask_radius_for_fofc:
        logger.writeln("  mask: using refined model with radius of {} A".format(args.mask_radius_for_fofc))
        mask = utils.maps.mask_from_model(st_expanded, args.mask_radius_for_fofc, grid=maps[0][0]) # use soft edge?
    else:
        logger.writeln("  mask: not used")
        mask = None
        
    hkldata, map_labs, stats_str = spa.fofc.calc_fofc(st_expanded, args.resolution, maps, mask=mask, monlib=monlib,
                                                      half1_only=(args.cross_validation and args.cross_validation_method == "throughout"),
                                                      sharpening_b=None if args.halfmaps else 0., # assume already sharpened if fullmap is given
                                                      source=source)
    spa.fofc.write_files(hkldata, map_labs, maps[0][1], stats_str,
                         mask=mask, output_prefix=diffmap_prefix,
                         trim_map=mask is not None, trim_mtz=args.trim_fofc_mtz)
    
    # Create Coot script
    spa.fofc.write_coot_script("{}_coot.py".format(args.output_prefix),
                               model_file="{}.pdb".format(args.output_prefix), # as Coot is not good at mmcif file..
                               mtz_file="{}_maps.mtz".format(diffmap_prefix),
                               contour_fo=None if mask is None else 1.2,
                               contour_fofc=None if mask is None else 3.0,
                               ncs_ops=st.ncs)

    # Create ChimeraX script
    spa.fofc.write_chimerax_script(cxc_out="{}_chimerax.cxc".format(args.output_prefix),
                                   model_file="{}.mmcif".format(args.output_prefix), # ChimeraX handles mmcif just fine
                                   fo_mrc_file="{}_normalized_fo.mrc".format(diffmap_prefix),
                                   fofc_mrc_file="{}_normalized_fofc.mrc".format(diffmap_prefix))
# calc_fofc()

def write_final_summary(st, refmac_summary, fscavg_text, output_prefix, is_mask_given):
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

    if is_mask_given:
        map_peaks_str = """\
List Fo-Fc map peaks in the ASU:
servalcat util map_peaks --map diffmap_normalized_fofc.mrc --model {prefix}.pdb --abs_level 4.0 \
""".format(prefix=output_prefix)
    else:
        map_peaks_str = "WARNING: --mask_for_fofc was not given, so the Fo-Fc map was not normalized."

    logger.writeln("""
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
             
Open refined model and maps mtz with COOT:
coot --script {prefix}_coot.py

Open refined model, map and difference map with ChimeraX/ISOLDE:
chimerax {prefix}_chimerax.cxc

{map_peaks_msg}
=============================================================================
""".format(rmsbond=refmac_summary["cycles"][-1].get("rms_bond", "???"),
           rmsangle=refmac_summary["cycles"][-1].get("rms_angle", "???"),
           fscavgs=fscavg_text.rstrip(),
           fsclog="{}_fsc.log".format(output_prefix),
           adpstats=adpstats_txt.rstrip(),
           final_weight=final_weight,
           prefix=output_prefix,
           map_peaks_msg=map_peaks_str))
# write_final_summary()

def lab_f_suffix(blur):
    if blur is None or blur == 0.:
        return ""
    elif blur > 0:
        return "Blur_{:.2f}".format(blur)
    else:
        return "Sharp_{:.2f}".format(-blur)
# lab_f_suffix()

def write_map_mtz(hkldata, mtz_out, map_labs, sig_lab=None, blur=0):
    nblur = 2 if blur != 0 else 1
    mean_f = hkldata.df[map_labs].abs().mean().min()
    data_labs = map_labs + ([sig_lab] if sig_lab else [])

    if mean_f < 1:
        scale = 10. / mean_f
        logger.writeln("Mean(|F|)= {:.2e} may be too small for Refmac. Applying scale= {:.1f}".format(mean_f, scale))
        for lab in data_labs:
            hkldata.df[lab] *= scale

    mtz_labs = data_labs + []
    mtz_types = {}
    if sig_lab: mtz_types[sig_lab] = "Q"

    if blur != 0:
        temp = hkldata.debye_waller_factors(b_iso=blur)
        for lab in data_labs:
            data = hkldata.df[lab]
            newlab = lab + lab_f_suffix(blur)
            if numpy.iscomplexobj(data): data = numpy.abs(data)
            hkldata.df[newlab] = data * temp
            mtz_labs.append(newlab)
            mtz_types[newlab] = "F" if lab != sig_lab else "Q"

    hkldata.write_mtz(mtz_out, labs=mtz_labs, types=mtz_types,
                      phase_label_decorator=lambda x: "P"+x[1:])
# write_map_mtz()

def determine_b_before_mask(st, maps, grid_start, mask, resolution):
    logger.writeln("Determining b_before_mask..")
    # work in masked map for the speed
    new_cell, new_shape, starts, shifts = shift_maps.determine_shape_and_shift(mask=mask,
                                                                               grid_start=grid_start,
                                                                               padding=5,
                                                                               mask_cutoff=0.5,
                                                                               noncentered=True,
                                                                               noncubic=True,
                                                                               json_out=None)
    st = st.clone()
    st.cell = new_cell
    st.spacegroup_hm = "P 1"
    for cra in st[0].all():
        cra.atom.pos += shifts
        cra.atom.b_iso = 0
        cra.atom.aniso = gemmi.SMat33f(0,0,0,0,0,0)

    newmaps = []
    for i in range(len(maps)): # Update maps
        g = gemmi.FloatGrid(maps[i][0].array*mask,
                            maps[i][0].unit_cell, maps[i][0].spacegroup)

        suba = g.get_subarray(starts, new_shape)
        new_grid = gemmi.FloatGrid(suba, new_cell, st.find_spacegroup())
        newmaps.append([new_grid]+maps[i][1:])

    hkldata = utils.maps.mask_and_fft_maps(newmaps, resolution)
    hkldata.df["FC"] = utils.model.calc_fc_fft(st, resolution - 1e-6, source="electron",
                                               miller_array=hkldata.miller_array())
    k, b = hkldata.scale_k_and_b("FC", "FP")
    return -b
# determine_b_before_mask()

def process_input(st, maps, resolution, monlib, mask_in, args,
                  shifted_model_prefix="shifted",
                  output_masked_prefix="masked_fs",
                  output_mtz_prefix="starting_map",
                  use_gemmi_prep=False, no_refmac_fix=False,
                  use_refmac=True, find_links=False):
    ret = {} # instructions for refinement
    maps = utils.maps.copy_maps(maps) # not to modify maps
    
    grid_start = maps[0][1]
    unit_cell = maps[0][0].unit_cell
    spacegroup = gemmi.SpaceGroup(1)
    start_xyz = numpy.array(maps[0][0].get_position(*grid_start).tolist())
    A = unit_cell.orth.mat.array
    center = numpy.sum(A, axis=1) / 2 #+ start_xyz

    # Create mask
    if mask_in:
        logger.writeln("Input mask file: {}".format(mask_in))
        mask = utils.fileio.read_ccp4_map(mask_in)[0]
    else:
        mask = None
    
    st.cell = unit_cell
    st.spacegroup_hm = "P 1"
    if use_refmac:
        ret["model_format"] = ".mmcif" if st.input_format == gemmi.CoorFormat.Mmcif else ".pdb"
        max_seq_num = max([max(res.seqid.num for res in chain) for model in st for chain in model])
        if max_seq_num > 9999 and ret["model_format"] == ".pdb":
            logger.writeln("Max residue number ({}) exceeds 9999. Will use mmcif format".format(max_seq_num))
            ret["model_format"] = ".mmcif"

    if len(st.ncs) > 0 and args.ignore_symmetry:
        logger.writeln("Removing symmetry information from model.")
        st.ncs.clear()
    utils.symmetry.update_ncs_from_args(args, st, map_and_start=maps[0], filter_contacting=args.contacting_only)
    st_expanded = st.clone()
    if not all(op.given for op in st.ncs):
        # symmetry is not exact in helical reconstructions
        cc_cutoff = 0.9 if args.twist is None else 0.7
        if not args.no_check_ncs_overlaps and utils.model.check_symmetry_related_model_duplication(st):
            raise SystemExit("\nError: Too many symmetry-related contacts detected.\n"
                             "Please provide an asymmetric unit model along with symmetry operators.")
        if not args.no_check_ncs_map and utils.maps.check_symmetry_related_map_values(st, maps[0][0], cc_cutoff=cc_cutoff):
            raise SystemExit("\nError: Map correlation is too small. Please ensure your map follows the model's symmetry")
        args.keywords.extend(utils.symmetry.ncs_ops_for_refmac(st.ncs))
        utils.model.expand_ncs(st_expanded)
        logger.writeln(" Saving expanded model: input_model_expanded.*")
        utils.fileio.write_model(st_expanded, "input_model_expanded", pdb=True, cif=True)

    if mask is not None and not args.no_check_mask_with_model:
        if not utils.maps.test_mask_with_model(mask, st_expanded):
            raise SystemExit("\nError: Model is out of mask.\n"
                             "Please check your --model and --mask. You can disable this test with --no_check_mask_with_model.")
    if args.mask_for_fofc:
        masktmp = utils.fileio.read_ccp4_map(args.mask_for_fofc)[0]
        if masktmp.shape != maps[0][0].shape:
            raise SystemExit("\nError: mask from --mask_for_fofc has a different shape from input map(s)")
        if not args.no_check_mask_with_model and not utils.maps.test_mask_with_model(masktmp, st):
            raise SystemExit("\nError: Model is out of mask.\n"
                             "Please check your --model and --mask_for_fofc. You can disable this test with --no_check_mask_with_model.")
        del masktmp

    if mask is None and args.mask_radius:
        logger.writeln("Creating mask..")
        mask = utils.maps.mask_from_model(st_expanded, args.mask_radius, soft_edge=args.mask_soft_edge, grid=maps[0][0])
        #utils.maps.write_ccp4_map("mask_from_model.ccp4", mask)

    if use_refmac:
        logger.writeln(" Saving input model with unit cell information")
        utils.fileio.write_model(st, "starting_model", pdb=True, cif=True)
        ret["model_file"] = "starting_model" + ret["model_format"]

    if mask is not None:
        if args.invert_mask:
            logger.writeln("Inverting mask..")
            mask_max, mask_min = numpy.max(mask), numpy.min(mask)
            logger.writeln("  mask_max, mask_min= {}, {}".format(mask_max, mask_min))
            mask = mask_max + mask_min - mask
        
        # Mask maps
        if args.no_sharpen_before_mask or len(maps) < 2:
            logger.writeln("Applying mask..")
            for ma in maps: ma[0].array[:] *= mask
        else:
            logger.writeln("Sharpen-mask-unsharpen..")
            b_before_mask = args.b_before_mask
            if b_before_mask is None: b_before_mask = determine_b_before_mask(st, maps, grid_start, mask, resolution)                
            maps = utils.maps.sharpen_mask_unsharpen(maps, mask, resolution, b=b_before_mask)

        if not args.no_trim:
            logger.writeln(" Shifting maps and/or model..")
            if args.padding is None: args.padding = args.mask_radius * 2
            new_cell, new_shape, starts, shifts = shift_maps.determine_shape_and_shift(mask=mask,
                                                                                       grid_start=grid_start,
                                                                                       padding=args.padding,
                                                                                       mask_cutoff=0.5,
                                                                                       noncentered=True,
                                                                                       noncubic=True,
                                                                                       json_out=None)
            ret["shifts"] = shifts
            vol_mask = numpy.count_nonzero(mask.array>0.5)
            vol_map = new_shape[0] * new_shape[1] * new_shape[2]
            ret["vol_ratio"] = vol_mask / vol_map
            logger.writeln(" Vol_mask/Vol_map= {:.2e}".format(ret["vol_ratio"]))

            # Model may be built out of the box (with 'unit cell' translation symmetry)
            # It is only valid with original unit cell, but no longer valid with the new cell
            # It still would not work if model is built over multiple 'cells'.
            extra_shift = utils.model.translate_into_box(st,
                                                         origin=gemmi.Position(*start_xyz),
                                                         apply_shift=False)
            if numpy.linalg.norm(extra_shift) > 0:
                logger.writeln("Input model is out of the box. Required shift= {}".format(extra_shift))
                ret["shifts"] += gemmi.Position(*extra_shift)
                logger.writeln("Shift for model has been adjusted: {}".format(numpy.array(ret["shifts"].tolist())))
            
            st.cell = new_cell
            st.spacegroup_hm = "P 1"
            if use_refmac:
                logger.writeln(" Saving model in trimmed map..")
                utils.fileio.write_model(st, shifted_model_prefix, pdb=True, cif=True)
                ret["model_file"] = shifted_model_prefix + ret["model_format"]

            logger.writeln(" Trimming maps..")
            for i in range(len(maps)): # Update maps
                suba = maps[i][0].get_subarray(starts, new_shape)
                new_grid = gemmi.FloatGrid(suba, new_cell, spacegroup)
                maps[i][0] = new_grid

    st.setup_cell_images()
    utils.restraints.find_and_fix_links(st, monlib, add_found=find_links,
                                        # link via ncsc is not supported as of Refmac5.8.0411
                                        find_symmetry_related=not use_refmac)
    # workaround for Refmac
    # TODO need to check external restraints
    if use_refmac:
        if use_gemmi_prep:
            h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                        "yes":gemmi.HydrogenChange.NoChange,
                        "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
            topo, metal_kws = utils.restraints.prepare_topology(st, monlib, h_change=h_change, raise_error=False)
            args.keywords = metal_kws + args.keywords
        elif not no_refmac_fix:
            topo = gemmi.prepare_topology(st, monlib, warnings=logger.silent(), ignore_unknown_links=True)
        else:
            topo = None # not used
        if not no_refmac_fix:
            ret["refmac_fixes"] = utils.refmac.FixForRefmac()
            ret["refmac_fixes"].fix_before_topology(st, topo, 
                                                    fix_microheterogeneity=not args.no_fix_microheterogeneity and not use_gemmi_prep,
                                                    fix_resimax=not args.no_fix_resi9999,
                                                    fix_nonpolymer=False)
        chain_id_len_max = max([len(x) for x in utils.model.all_chain_ids(st)])
        if chain_id_len_max > 1 and ret["model_format"] == ".pdb":
            logger.writeln("Long chain ID (length: {}) detected. Will use mmcif format".format(chain_id_len_max))
            ret["model_format"] = ".mmcif"
        if not no_refmac_fix and ret["model_format"] == ".mmcif" and not use_gemmi_prep:
            ret["refmac_fixes"].fix_nonpolymer(st)

    if use_refmac and use_gemmi_prep:
        # TODO: make cispept, make link, remove unknown link id
        # TODO: cross validation?
        crdout = os.path.splitext(ret["model_file"])[0] + ".crd"
        ret["model_file"] = crdout
        ret["model_format"] = ".mmcif"
        args.keywords.append("make cr prepared")
        gemmi.setup_for_crd(st)
        doc = gemmi.prepare_refmac_crd(st, topo, monlib, h_change)
        doc.write_file(crdout, options=gemmi.cif.Style.NoBlankLines)
        logger.writeln("crd file written: {}".format(crdout))

    hkldata = utils.maps.mask_and_fft_maps(maps, resolution, None, with_000=False)
    hkldata.setup_relion_binning("ml")
    hkldata.copy_binning(src="ml", dst="stat") # todo test usual binning for ml
    if len(maps) == 2:
        map_labs = ["Fmap1", "Fmap2", "Fout"]
        ret["lab_f_half1"] = "Fmap1" + lab_f_suffix(args.blur)
        # TODO Add SIGF in case of half maps, when refmac is ready
        ret["lab_phi_half1"] = "Pmap1"
        ret["lab_f_half2"] = "Fmap2" + lab_f_suffix(args.blur)
        ret["lab_phi_half2"] = "Pmap2"
        utils.maps.calc_noise_var_from_halfmaps(hkldata)
        d_eff_full = hkldata.d_eff("ml", "FSCfull")
        logger.writeln("Effective resolution from FSCfull= {:.2f}".format(d_eff_full))
        ret["d_eff"] = d_eff_full
    else:
        map_labs = ["Fout"]
        sig_lab = None

    if use_refmac:
        if args.no_mask:
            logger.writeln("Saving unmasked maps as mtz file..")
            mtzout = output_mtz_prefix+".mtz"
        else:
            logger.writeln(" Saving masked maps as mtz file..")
            mtzout = output_masked_prefix+"_obs.mtz"

        hkldata.df.rename(columns=dict(F_map1="Fmap1", F_map2="Fmap2", FP="Fout"), inplace=True)
        if "shifts" in ret:
            for lab in map_labs: # apply phase shift
                logger.writeln("  applying phase shift for {} with translation {}".format(lab, -ret["shifts"]))
                hkldata.translate(lab, -ret["shifts"])

        write_map_mtz(hkldata, mtzout, map_labs=map_labs, blur=args.blur)
        ret["mtz_file"] = mtzout
        ret["lab_f"] = "Fout" + lab_f_suffix(args.blur)
        ret["lab_phi"] = "Pout"
    else:
        fac = hkldata.debye_waller_factors(b_iso=args.blur)
        if "shifts" in ret: fac *= hkldata.translation_factor(-ret["shifts"])
        for lab in ("F_map1", "F_map2", "FP"):
            if lab in hkldata.df: hkldata.df[lab] *= fac
    return hkldata, ret
# process_input()

def check_args(args):
    if not os.path.exists(args.model):
        raise SystemExit("Error: --model {} does not exist.".format(args.model))

    if args.cross_validation and not args.halfmaps:
        raise SystemExit("Error: half maps are needed when --cross_validation is given")
    
    if args.mask_for_fofc and not os.path.exists(args.mask_for_fofc):
        raise SystemExit("Error: --mask_for_fofc {} does not exist".format(args.mask_for_fofc))

    if args.mask_for_fofc and args.mask_radius_for_fofc:
        raise SystemExit("Error: you cannot specify both --mask_for_fofc and --mask_radius_for_fofc")

    if args.trim_fofc_mtz and not (args.mask_for_fofc or args.mask_radius_for_fofc):
        raise SystemExit("Error: --trim_fofc_mtz is specified but --mask_for_fofc is not given")

    if args.ligand: args.ligand = sum(args.ligand, [])

    if args.keywords:
        args.keywords = sum(args.keywords, [])
    else:
        args.keywords = []

    if args.keyword_file:
        args.keyword_file = sum(args.keyword_file, [])
        for f in args.keyword_file:
            if not os.path.exists(f):
                raise SystemExit(f"Error: keyword file was not found: {f}")
            logger.writeln("Keyword file: {}".format(f))
    else:
        args.keyword_file = []

    if (args.twist, args.rise).count(None) == 1:
        raise SystemExit("ERROR: give both helical parameters --twist and --rise")
    if args.twist is not None:
        logger.writeln("INFO: setting --contacting_only because helical symmetry is given")
        args.contacting_only = True
    if args.no_mask:
        args.mask_radius = None
        if not args.no_trim:
            logger.writeln("WARNING: setting --no_trim because --no_mask is given")
            args.no_trim = True
        if args.mask:
            logger.writeln("WARNING: Your --mask is ignored because --no_mask is given")
            args.mask = None

    #if args.mask_soft_edge > 0:
    #    logger.writeln("INFO: --mask_soft_edge={} is given. Turning off sharpen_before_mask.".format(args.mask_soft_edge))
    #    args.no_sharpen_before_mask = True

    if args.resolution is None and args.model and utils.fileio.splitext(args.model)[1].endswith("cif"):
        doc = gemmi.cif.read(args.model)
        if len(doc) != 1:
            raise SystemExit("cannot find resolution from cif. Give --resolution")
        block = doc.sole_block()
        reso_str = block.find_value("_em_3d_reconstruction.resolution")
        try:
            args.resolution = float(reso_str)
        except:
            raise SystemExit("ERROR: _em_3d_reconstruction.resolution is invalid. Give --resolution")
        logger.writeln("WARNING: --resolution not given. Using _em_3d_reconstruction.resolution = {}".format(reso_str))

    if args.resolution is None:
        raise SystemExit("ERROR: --resolution is needed.")
# check_args()

def main(args):
    check_args(args)
    use_gemmi_prep = False
    if not args.prepare_only:
        refmac_ver = utils.refmac.check_version(args.exe)
        if not refmac_ver:
            raise SystemExit("Error: Check Refmac installation or use --exe to give the location.")
        if not args.no_refmacat and refmac_ver >= (5, 8, 404):
            logger.writeln(" will use gemmi to prepare restraints")
            use_gemmi_prep = True
        else:
            logger.writeln(" will use makecif to prepare restraints")

    logger.writeln("Input model: {}".format(args.model))
    st = utils.fileio.read_structure(args.model)
    if len(st) > 1:
        logger.writeln(" Removing models 2-{}".format(len(st)))
        for i in reversed(range(1, len(st))):
            del st[i]

    try:
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                                       stop_for_unknowns=True)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
    try:
        utils.restraints.prepare_topology(st.clone(), monlib, h_change=gemmi.HydrogenChange.NoChange,
                                          check_hydrogen=(args.hydrogen=="yes"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if args.halfmaps:
        maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]
        
    utils.model.remove_charge([st])
    shifted_model_prefix = "shifted"
    _, file_info = process_input(st, maps, resolution=args.resolution - 1e-6, monlib=monlib,
                                 mask_in=args.mask, args=args,
                                 shifted_model_prefix=shifted_model_prefix,
                                 use_gemmi_prep=use_gemmi_prep,
                                 find_links=args.find_links)
    if args.prepare_only:
        logger.writeln("\n--prepare_only is given. Stopping.")
        return

    args.mtz = file_info["mtz_file"]
    if args.halfmaps: # FIXME if no_mask?
        args.mtz_half = [file_info["mtz_file"], file_info["mtz_file"]]
    args.lab_phi = file_info["lab_phi"]  #"Pout0"
    args.lab_f = file_info["lab_f"]
    args.lab_sigf = None
    args.model = file_info["model_file"] # refmac xyzin
    model_format = file_info["model_format"]

    if args.cross_validation and args.cross_validation_method == "throughout":
        args.lab_f = file_info["lab_f_half1"]
        args.lab_phi = file_info["lab_phi_half1"]
        # XXX args.lab_sigf?

    chain_id_lens = [len(x) for x in utils.model.all_chain_ids(st)]
    keep_chain_ids = (chain_id_lens and max(chain_id_lens) == 1) # always kept unless one-letter chain IDs

    # FIXME if mtz is given and sfcalc() not ran?

    if not args.no_trim:
        refmac_prefix = "{}_{}".format(shifted_model_prefix, args.output_prefix)
    else:
        refmac_prefix = args.output_prefix # XXX this should be different name (nomask etc?)

    # Weight auto scale
    if args.weight_auto_scale is None:
        reso = file_info["d_eff"] if "d_eff" in file_info else args.resolution
        if "vol_ratio" in file_info:
            if "d_eff" in file_info:
                rlmc = (-9.503541, 3.129882, 15.439744)
            else:
                rlmc = (-8.329418, 3.032409, 14.381907)
            logger.writeln("Estimating weight auto scale using resolution and volume ratio")
            ws = rlmc[0] + reso*rlmc[1] +file_info["vol_ratio"]*rlmc[2]
        else:
            if "d_eff" in file_info:
                rlmc = (-5.903807, 2.870723)
            else:
                rlmc = (-4.891140, 2.746791)
            logger.writeln("Estimating weight auto scale using resolution")
            ws =  rlmc[0] + args.resolution*rlmc[1]
        args.weight_auto_scale = max(0.2, min(18.0, ws))
        logger.writeln(" Will use weight auto {:.2f}".format(args.weight_auto_scale))
        
    # Run Refmac
    refmac = utils.refmac.Refmac(prefix=refmac_prefix, args=args, global_mode="spa",
                                 keep_chain_ids=keep_chain_ids)
    refmac.set_libin(args.ligand)
    try:
        refmac_summary = refmac.run_refmac()
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    # Modify output
    st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(refmac_prefix+model_format)
    utils.model.setup_entities(st, clear=True, overwrite_entity_type=True, force_subchain_names=True)

    if not args.no_trim:
        st.cell = maps[0][0].unit_cell
        st.setup_cell_images()
    if "refmac_fixes" in file_info:
        file_info["refmac_fixes"].modify_back(st)
    utils.model.adp_analysis(st)
    utils.fileio.write_model(st, prefix=args.output_prefix,
                             pdb=True, cif=True, cif_ref=cif_ref)

    # Take care of TLS out
    if not args.no_trim: # if no_trim, there is nothing to do.
        tlsout = refmac.tlsout()
        if os.path.exists(tlsout):
            logger.writeln("Copying tlsout")
            shutil.copyfile(refmac.tlsout(), args.output_prefix+".tls")

    # Expand sym here
    st_expanded = st.clone()
    if not all(op.given for op in st.ncs):
        utils.model.expand_ncs(st_expanded)
        utils.fileio.write_model(st_expanded, args.output_prefix+"_expanded", pdb=True, cif=True,
                                 cif_ref=cif_ref)

    if args.cross_validation and args.cross_validation_method == "shake":
        logger.writeln("Cross validation is requested.")
        refmac_prefix_shaken = refmac_prefix+"_shaken_refined"
        logger.writeln("Starting refinement using half map 1 (model is shaken first)")
        logger.writeln("In this refinement, hydrogen is removed regardless of --hydrogen option")
        if use_gemmi_prep:
            xyzin = refmac_prefix + ".crd"
            st_tmp = utils.fileio.read_structure(refmac_prefix+model_format)
            utils.model.setup_entities(st_tmp, clear=True, overwrite_entity_type=True, force_subchain_names=True)
            prepare_crd(st_tmp,
                        crdout=xyzin, ligand=[refmac_prefix+model_format],
                        make={"hydr":"n"},
                        fix_long_resnames=False) # we do not need output file - do we?
        else:
            xyzin = refmac_prefix + model_format
        refmac_hm1 = refmac.copy(hklin=args.mtz_half[0],
                                 xyzin=xyzin,
                                 prefix=refmac_prefix_shaken,
                                 shake=args.shake_radius,
                                 jellybody=False, # makes no sense to use jelly body after shaking
                                 hydrogen="no") # should not use hydrogen after shaking
        if args.jellybody: logger.writeln("  Turning off jelly body")
        if "lab_f_half1" in file_info:
            refmac_hm1.lab_f = file_info["lab_f_half1"]
            refmac_hm1.lab_phi = file_info["lab_phi_half1"]
            # SIGMA?
            
        try:
            refmac_hm1.run_refmac()
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))

        if args.hydrogen != "no": # does not work properly when 'yes' - we would need to keep hydrogen in input
            logger.writeln("Cross validation: 2nd run with hydrogen")
            if use_gemmi_prep:
                xyzin = refmac_prefix_shaken + ".crd"
                st_tmp = utils.fileio.read_structure(refmac_prefix_shaken+model_format)
                utils.model.setup_entities(st_tmp, clear=True, overwrite_entity_type=True, force_subchain_names=True)
                prepare_crd(st_tmp,
                            crdout=xyzin, ligand=[refmac_prefix+model_format],
                            make={"hydr":"a"},
                            fix_long_resnames=False) # we do not need output file - do we?
            else:
                xyzin = refmac_prefix_shaken + model_format
            refmac_prefix_shaken = refmac_prefix+"_shaken_refined2"
            refmac_hm1_2 = refmac_hm1.copy(xyzin=xyzin,
                                           prefix=refmac_prefix_shaken,
                                           shake=None,
                                           hydrogen="all")
            try:
                refmac_hm1_2.run_refmac()
            except RuntimeError as e:
                raise SystemExit("Error: {}".format(e))
        
        # Modify output
        st_sr, cif_ref_sr = utils.fileio.read_structure_from_pdb_and_mmcif(refmac_prefix_shaken+model_format)
        utils.model.setup_entities(st_sr, clear=True, overwrite_entity_type=True, force_subchain_names=True)
        if not args.no_trim:
            st_sr.cell = maps[0][0].unit_cell
            st_sr.setup_cell_images()
        if "refmac_fixes" in file_info:
            file_info["refmac_fixes"].modify_back(st_sr)

        utils.fileio.write_model(st_sr, prefix=args.output_prefix+"_shaken_refined",
                                 pdb=True, cif=True, cif_ref=cif_ref_sr)

        # Expand sym here
        st_sr_expanded = st_sr.clone()
        if not all(op.given for op in st_sr.ncs):
            utils.model.expand_ncs(st_sr_expanded)
            utils.fileio.write_model(st_sr_expanded, args.output_prefix+"_shaken_refined_expanded",
                                     pdb=True, cif=True, cif_ref=cif_ref_sr)
            if args.twist is not None: # as requested by a user
                st_sr_expanded_all = st_sr.clone()
                utils.symmetry.update_ncs_from_args(args, st_sr_expanded_all, map_and_start=maps[0], filter_contacting=False)
                utils.model.expand_ncs(st_sr_expanded_all)
                utils.fileio.write_model(st_sr_expanded_all, args.output_prefix+"_shaken_refined_expanded_all", pdb=True, cif=True,
                                         cif_ref=cif_ref)
    else:
        st_sr_expanded = None

    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None
        
    # Calc FSC
    fscavg_text = calc_fsc(st_expanded, args.output_prefix, maps,
                           args.resolution, mask=mask, mask_radius=args.mask_radius if not args.no_mask else None,
                           soft_edge=args.mask_soft_edge,
                           b_before_mask=args.b_before_mask,
                           no_sharpen_before_mask=args.no_sharpen_before_mask,
                           make_hydrogen=args.hydrogen,
                           monlib=monlib, cross_validation=args.cross_validation,
                           blur=args.blur,
                           d_min_fsc=args.fsc_resolution,
                           cross_validation_method=args.cross_validation_method, st_sr=st_sr_expanded)[0]

    # Calc Fo-Fc (and updated) maps
    calc_fofc(st, st_expanded, maps, monlib, model_format, args)
    
    # Final summary
    write_final_summary(st, refmac_summary, fscavg_text, args.output_prefix,
                        args.mask_for_fofc or args.mask_radius_for_fofc)
# main()
        
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)

