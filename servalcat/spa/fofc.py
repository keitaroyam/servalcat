# TODO shift map first using mask!
"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import time
from servalcat.utils import logger
from servalcat import utils
from servalcat.spa import shift_maps
import argparse

def add_arguments(parser):
    parser.description = 'Fo-Fc map calculation based on model and data errors'
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--halfmaps", nargs=2)
    group.add_argument("--map", help="Use only if you really do not have half maps.")
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument("-d", '--resolution', type=float, required=True)
    parser.add_argument('-m', '--mask', help="mask file")
    parser.add_argument('-r', '--mask_radius', type=float, help="mask radius (not used if --mask is given)")
    parser.add_argument('--no_check_mask_with_model', action='store_true', 
                        help='Disable mask test using model')
    parser.add_argument("-B", type=float, help="Estimated blurring")
    parser.add_argument("--half1_only", action='store_true', help="Only use half 1 for map calculation (use half 2 only for noise estimation)")
    parser.add_argument("--normalized_map", action='store_true',
                        help="Write normalized map in the masked region. Now this is on by default.")
    parser.add_argument("--no_fsc_weights", action='store_true',
                        help="Just for debugging purpose: turn off FSC-based weighting")
    parser.add_argument("--sharpening_b", type=float,
                        help="Use B value (negative value for sharpening) instead of standard deviation of the signal")
    parser.add_argument("--trim", action='store_true',
                        help="Write trimmed maps")
    parser.add_argument("--trim_mtz", action='store_true',
                        help="Write trimmed mtz")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument("--omit_proton", action='store_true',
                        #help="Omit hydrogen proton (leaving electrons) from model in map calculation")
                        help=argparse.SUPPRESS)
    parser.add_argument("--omit_h_electron", action='store_true',
                        #help="Omit hydrogen electrons (leaving protons) from model in map calculation")
                        help=argparse.SUPPRESS)
    parser.add_argument("-s", "--source", choices=["electron", "xray", "neutron", "custom"], default="electron")
    parser.add_argument('-o','--output_prefix', default="diffmap",
                        help='output file name prefix (default: %(default)s)')
    parser.add_argument('--keep_charges',  action='store_true',
                        help="Use scattering factor for charged atoms. Use it with care.")
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_D_and_S(hkldata, has_halfmaps=True, half1_only=False):#fo_asu, fc_asu, varn, bins, bin_idxes):
    bdf = hkldata.binned_df["ml"]
    bdf["D"] = 0.
    bdf["S"] = 0.
    stats_str = """$TABLE: Statistics :
$GRAPHS
: log(Mn(|F|^2)) and variances :A:1,6,7,8,13,14:
: FSC :A:1,9,10,11:
: weights :A:1,12,15,16:
: map weights :A:1,17:
$$
1/resol^2 bin n d_max d_min log(var(Fo)) log(var(Fc)) log(var(DFc)) FSC.model FSC.full sqrt(FSC.full) D log(var_U,T) log(var_noise) wFo wFc wFo.sharpen
$$
$$
"""
    tmpl = "{:.4f} {:3d} {:7d} {:7.3f} {:7.3f} {:.4e} {:.4e} {:4e} {: .4f}   {: .4f} {: .4f} {: .4e} {:.4e} {:.4e} {:.4f} {:.4f} {:.4e}\n"

    var_noise = None
    FP = hkldata.df.FP.to_numpy()
    if half1_only:
        FP = hkldata.df.F_map1.to_numpy()
        var_noise = hkldata.binned_df["ml"].var_noise * 2
    elif has_halfmaps:
        var_noise = hkldata.binned_df["ml"].var_noise
        
    for i_bin, idxes in hkldata.binned("ml"):
        bin_d_min = hkldata.binned_df["ml"].d_min[i_bin]
        bin_d_max = hkldata.binned_df["ml"].d_max[i_bin]
        Fo = FP[idxes]
        Fc = hkldata.df.FC.to_numpy()[idxes]
        fsc = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
        bdf.loc[i_bin, "D"] = numpy.sum(numpy.real(Fo * numpy.conj(Fc)))/numpy.sum(numpy.abs(Fc)**2)
        if has_halfmaps:
            varn = var_noise[i_bin]
            fsc_full = hkldata.binned_df["ml"].FSCfull[i_bin]
            S = max(0, numpy.average(numpy.abs(Fo-bdf.D[i_bin]*Fc)**2)-varn)
            bdf.loc[i_bin, "S"] = S
            w = S/(S+varn)
            if fsc_full < 0: # this should be fixed actually. needs smoothing to zero.
                w_sharpen = 0
            else:
                w_sharpen = w / numpy.sqrt(fsc_full) / numpy.std(Fo)
        else:
            varn = fsc_full = 0
            w = 1
            w_sharpen = 1

        with numpy.errstate(divide="ignore", invalid="ignore"):
            stats_str += tmpl.format(1/bin_d_min**2, i_bin, Fo.size, bin_d_max, bin_d_min,
                                     numpy.log(numpy.average(numpy.abs(Fo)**2)),
                                     numpy.log(numpy.average(numpy.abs(Fc)**2)),
                                     numpy.log(bdf.D[i_bin]**2*numpy.average(numpy.abs(Fc)**2)),
                                     fsc, fsc_full, numpy.sqrt(fsc_full), bdf.D[i_bin],
                                     numpy.log(bdf.S[i_bin]), numpy.log(varn),
                                     w, 1-w, w_sharpen)
    return stats_str
# calc_D_and_S()

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
#@profile
def calc_maps(hkldata, B=None, has_halfmaps=True, half1_only=False, no_fsc_weights=False, sharpening_b=None):
    has_fc = "FC" in hkldata.df

    if not has_fc:
        labs = ["FWT"]
        if B is not None: labs.append("FWT_b0")
    elif has_halfmaps:
        labs = ["Fupdate", "DELFWT", "FWT", "DELFWT_noscale", "Fupdate_noscale"]
        if B is not None: labs.extend(["Fupdate_b0", "DELFWT_b0", "FWT_b0"])
    else:
        labs = ["DELFWT"]
        
    tmp = {}
    for l in labs:
        tmp[l] = numpy.zeros(len(hkldata.df.index), numpy.complex128)

    logger.writeln("Calculating maps..")
    logger.write(" sharpening method: ")
    if sharpening_b is None:
        logger.writeln("1/sqrt(FSC * Mn(Fo)) for Fo and 1/sigma_U,T for Fo-Fc")
    else:
        logger.writeln("1/exp(-B*s^2/4) with B= {:.2f}".format(sharpening_b))

    time_t = time.time()

    if half1_only:
        FP = hkldata.df.F_map1.to_numpy()
    else:
        FP = hkldata.df.FP.to_numpy()

    s2 = 1./hkldata.d_spacings().to_numpy()**2

    fsc_became_negative = False
        
    for i_bin, idxes in hkldata.binned("ml"):
        if has_halfmaps:
            fsc = hkldata.binned_df["ml"].FSCfull[i_bin] # FSCfull
            if half1_only:
                varn = hkldata.binned_df["ml"].var_noise[i_bin] * 2
                fsc = fsc/(2-fsc) # to FSChalf
            else:
                varn = hkldata.binned_df["ml"].var_noise[i_bin]
        else:
            fsc, varn = 1., 0.
            
        w_nomodel = 1. if no_fsc_weights else fsc
        Fo = FP[idxes]
        sig_fo = numpy.std(Fo)
        s2_bin = s2[idxes]

        if has_fc:
            Fc = hkldata.df.FC.to_numpy()[idxes]
            D = hkldata.binned_df["ml"].D[i_bin]
            S = hkldata.binned_df["ml"].S[i_bin] # variance of unexplained signal
            w = 1. if no_fsc_weights or not has_halfmaps else S/(S+varn)
            delfwt = w * (Fo-D*Fc)
            fup = 2 * w * Fo + (1 - 2*w) * D*Fc # <F> + delfwt
            if has_halfmaps: # no point making this map when half maps not given
                tmp["DELFWT_noscale"][idxes] = delfwt
                tmp["Fupdate_noscale"][idxes] = fup

        if not fsc_became_negative and fsc <= 0:
            logger.writeln(" WARNING: cutting resolution at {:.2f} A because fsc < 0".format(hkldata.binned_df["ml"].d_max[i_bin]))
            fsc_became_negative = True
        if fsc_became_negative:
            continue

        if sharpening_b is None:
            k = sig_fo * numpy.sqrt(fsc)
            k_fofc = numpy.sqrt(S) if has_fc and S > 0 else 1. # to avoid zero-division. if S=0 then w=0.
        else:
            k = k_fofc = numpy.exp(-sharpening_b*s2_bin/4)
            
        lab_suf = "" if B is None else "_b0"
        if has_halfmaps:
            tmp["FWT"+lab_suf][idxes] = w_nomodel / k * Fo
            if has_fc:
                tmp["DELFWT"+lab_suf][idxes] = delfwt / k_fofc
                tmp["Fupdate"+lab_suf][idxes] = fup / k
        elif has_fc:
            tmp["DELFWT"+lab_suf][idxes] = delfwt

        if B is not None and has_halfmaps: # local B based map
            k_l = numpy.exp(-B*s2_bin/4.)
            k2_l = numpy.exp(-B*s2_bin/2.)
            fsc_l = k2_l*fsc/(1+(k2_l-1)*fsc)
            w_nomodel = 1. if no_fsc_weights else fsc_l
            tmp["FWT"][idxes] = Fo*w_nomodel/k/k_l
            if has_fc:
                S_l = S * k2_l
                w = 1. if no_fsc_weights or not has_halfmaps else S_l/(S_l+varn)            
                delfwt = (Fo-D*Fc)*w/k_fofc/k_l
                fup = (w*Fo+(1.-w)*D*Fc)/k/k_l
                logger.writeln("{:4d} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}".format(i_bin,
                                                                                      numpy.average(k),
                                                                                      numpy.average(sig_fo),
                                                                                      numpy.average(fsc_l),
                                                                                      numpy.average(k_l),
                                                                                      numpy.average(abs(fup)),
                                                                                      numpy.average(abs(delfwt))))
                tmp["DELFWT"][idxes] = delfwt
                tmp["Fupdate"][idxes] = fup

    for l in labs:
        hkldata.df[l] = tmp[l]

    logger.writeln(" finished in {:.3f} sec.".format(time.time()-time_t))
    return labs
# calc_maps()

def dump_to_mtz(hkldata, map_labs, mtz_out):
    extra_labs = list(filter(lambda x: x in hkldata.df, ["FP", "FC"]))
    map_labs = map_labs + extra_labs
    hkldata.write_mtz(mtz_out, map_labs)
# dump_to_mtz()

def calc_fofc(st, d_min, maps, mask=None, monlib=None, B=None, half1_only=False,
              no_fsc_weights=False, sharpening_b=None, omit_proton=False, omit_h_electron=False,
              source="electron"):
    if no_fsc_weights:
        logger.writeln("WARNING: --no_fsc_weights is requested.")
    if sharpening_b is not None:
        logger.writeln("WARNING: --sharpening_b={} is given".format(sharpening_b))
    
    hkldata = utils.maps.mask_and_fft_maps(maps, d_min, mask)
    hkldata.df["FC"] = utils.model.calc_fc_fft(st, d_min - 1e-6, monlib=monlib, source=source,
                                               miller_array=hkldata.miller_array())
    if mask is not None:
        fc_map = hkldata.fft_map("FC", grid_size=mask.shape)
        fc_map.array[:] *= mask
        hkldata.df["FC"] = gemmi.transform_map_to_f_phi(fc_map).get_value_by_hkl(hkldata.miller_array())
        
    hkldata.setup_relion_binning("ml")

    has_halfmaps = (len(maps) == 2)
    if has_halfmaps:
        utils.maps.calc_noise_var_from_halfmaps(hkldata)

    stats_str = calc_D_and_S(hkldata, has_halfmaps=has_halfmaps, half1_only=half1_only)

    if omit_proton or omit_h_electron:
        hkldata.df["FC"] = utils.model.calc_fc_fft(st, d_min - 1e-6, monlib=monlib, source=source,
                                                   omit_proton=omit_proton, omit_h_electron=omit_h_electron,
                                                   miller_array=hkldata.miller_array())
    
    map_labs = calc_maps(hkldata, B=B, has_halfmaps=has_halfmaps, half1_only=half1_only,
                         no_fsc_weights=no_fsc_weights, sharpening_b=sharpening_b)
    return hkldata, map_labs, stats_str
# calc_fofc()

def write_files(hkldata, map_labs, grid_start, stats_str,
                mask=None, output_prefix="diffmap", trim_map=False, trim_mtz=False,
                normalize_map=True, omit_h_electron=False):
    # this function may modify the overall scale of FWT/DELFWT.

    if mask is not None and (trim_map or trim_mtz):
        new_cell, new_shape, new_grid_start, shifts = shift_maps.determine_shape_and_shift(mask=gemmi.FloatGrid(mask.array,
                                                                                                                hkldata.cell,
                                                                                                                hkldata.sg),
                                                                                           grid_start=grid_start,
                                                                                           padding=5,
                                                                                           mask_cutoff=0.5,
                                                                                           noncentered=True,
                                                                                           noncubic=True,
                                                                                           json_out=None)
    else:
        new_cell, new_shape, new_grid_start, shifts = None, None, None, None

    if trim_map:
        grid_start_for_map = new_grid_start
        shape_for_map = new_shape
    else:
        grid_start_for_map = grid_start
        shape_for_map = None
        
    if normalize_map and mask is not None:
        cutoff = 0.5
        if "DELFWT" in hkldata.df:
            logger.writeln("Normalized Fo-Fc map requested.")
            delfwt_map = hkldata.fft_map("DELFWT", grid_size=mask.shape)
            masked = delfwt_map.array[mask.array>cutoff]
            logger.writeln("   Whole volume: {} voxels".format(delfwt_map.point_count))
            logger.writeln("  Masked volume: {} voxels (>{})".format(masked.size, cutoff))
            global_mean = numpy.average(delfwt_map)
            global_std = numpy.std(delfwt_map)
            logger.writeln("    Global mean: {:.3e}".format(global_mean))
            logger.writeln("     Global std: {:.3e}".format(global_std))
            masked_mean = numpy.average(masked)
            masked_std = numpy.std(masked)
            logger.writeln("    Masked mean: {:.3e}".format(masked_mean))
            logger.writeln("     Masked std: {:.3e}".format(masked_std))
            #logger.writeln(" If you want to scale manually: {}".format())
            scaled = (delfwt_map - masked_mean)/masked_std
            hkldata.df["DELFWT"] /= masked_std # it would work if masked_mean~0
            if omit_h_electron:
                scaled *= -1
                filename = "{}_normalized_fofc_flipsign.mrc".format(output_prefix)
            else:
                filename = "{}_normalized_fofc.mrc".format(output_prefix)
            logger.writeln("  Writing {}".format(filename))
            utils.maps.write_ccp4_map(filename, scaled, cell=hkldata.cell,
                                      grid_start=grid_start_for_map, grid_shape=shape_for_map)

        # Write Fo map as well
        if "FWT" in hkldata.df:
            fwt_map = hkldata.fft_map("FWT", grid_size=mask.shape)
            masked = fwt_map.array[mask.array>cutoff]
            masked_mean = numpy.average(masked)
            masked_std = numpy.std(masked)
            scaled = (fwt_map - masked_mean)/masked_std # does not make much sense for Fo map though
            hkldata.df["FWT"] /= masked_std # it would work if masked_mean~0
            filename = "{}_normalized_fo.mrc".format(output_prefix)
            logger.writeln("  Writing {}".format(filename))
            utils.maps.write_ccp4_map(filename, scaled, cell=hkldata.cell,
                                      grid_start=grid_start_for_map, grid_shape=shape_for_map)

    if trim_mtz and shifts is not None:
        hkldata2 = utils.hkl.HklData(new_cell, hkldata.sg, df=None)
        d_min = hkldata.d_min_max()[0]
        for lab in map_labs + ["FP", "FC"]:
            if lab not in hkldata.df: continue
            gr = hkldata.fft_map(lab, grid_size=mask.shape)
            gr = gemmi.FloatGrid(gr.get_subarray(new_grid_start, new_shape),
                                 new_cell, hkldata.sg)
            if hkldata2.df is None:
                ad = gemmi.transform_map_to_f_phi(gr).prepare_asu_data(dmin=d_min)
                hkldata2.merge_asu_data(ad, lab)
            else:
                hkldata2.df[lab] = gemmi.transform_map_to_f_phi(gr).get_value_by_hkl(hkldata2.miller_array())
            hkldata2.translate(lab, -shifts)
        hkldata = hkldata2
        
    dump_to_mtz(hkldata, map_labs, "{}_maps.mtz".format(output_prefix))
    if stats_str:
        with open("{}_Fstats.log".format(output_prefix), "w") as f:
            f.write(stats_str)
# write_files()

def write_coot_script(py_out, model_file, mtz_file, contour_fo=1.2, contour_fofc=3.0, ncs_ops=None):
    with open(py_out, "w") as ofs:
        ofs.write('imol = read_pdb("{}")\n'.format(model_file)) # TODO safer
        ofs.write('imol_fo = make_and_draw_map("{}", "FWT", "PHWT", "", 0, 0)\n'.format(mtz_file))
        ofs.write('imol_fofc = make_and_draw_map("{}", "DELFWT", "PHDELWT", "", 0, 1)\n'.format(mtz_file))
        if contour_fo is not None:
            ofs.write('set_contour_level_absolute(imol_fo, {:.1f})\n'.format(contour_fo))
        if contour_fofc is not None:
            ofs.write('set_contour_level_absolute(imol_fofc, {:.1f})\n'.format(contour_fofc))
        if ncs_ops is not None:
            for op in ncs_ops:
                if op.given: continue
                c, resid = utils.symmetry.find_center_of_origin(op.tr.mat, op.tr.vec)
                if resid.length() > 1e-6: continue # coot does not support translation..
                v = [y for x in op.tr.mat.tolist() for y in x] + c.tolist()
                ofs.write("add_molecular_symmetry(imol, {})\n".format(",".join(str(x) for x in v)))
# write_coot_script()

def write_chimerax_script(cxc_out, model_file, fo_mrc_file, fofc_mrc_file):
    with open(cxc_out, "w") as ofs:
        ofs.write('open {}\n'.format(model_file))
        ofs.write('open {}\n'.format(fo_mrc_file))
        ofs.write('open {}\n'.format(fofc_mrc_file))
        ofs.write('volume #3 level 4 level -4 color #00FF00 color #FF0000 squaremesh false cap false style mesh meshlighting false\n')
        ofs.write('isolde start\n')
        ofs.write('clipper associate #2 toModel #1\n')
        ofs.write('clipper associate #3 toModel #1\n')
# write_chimerax_script()

def main(args):
    if not args.halfmaps and not args.map:
        raise SystemExit("Error: give --halfmaps or --map")

    if not args.halfmaps and args.B is not None:
        raise SystemExit("Error: -B only works with half maps")

    if args.half1_only:
        if not args.halfmaps:
            raise SystemExit("--half1_only requires half maps")
        logger.error("--half1_only specified. Half map 2 is used only for noise estimation")

    if args.normalized_map:
        logger.writeln("DeprecationWarning: --normalized_map is now on by default. This option will be removed in the future.")
        
    if not args.halfmaps:
        logger.error("Warning: using --halfmaps is strongly recommended!")

    st = utils.fileio.read_structure(args.model)
    ccu = utils.model.CustomCoefUtil()
    if not args.keep_charges:
        utils.model.remove_charge([st])
    if args.source == "custom":
        ccu.read_from_cif(st, args.model)
        ccu.show_info()
        ccu.set_coeffs(st)
    else:
        utils.model.check_atomsf([st], args.source)
    ncs_org = gemmi.NcsOpList(st.ncs)
    utils.model.expand_ncs(st)

    if (args.omit_proton or args.omit_h_electron) and not st[0].has_hydrogen():
        raise SystemExit("ERROR! --omit_proton/--omit_h_electron requested, but no hydrogen atoms were found.")

    if args.halfmaps:
        maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
        has_halfmaps = True
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]
        has_halfmaps = False

    grid_start = maps[0][1]
    g = maps[0][0]
    st.spacegroup_hm = "P1"
    st.cell = g.unit_cell

    if st[0].has_hydrogen():
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib)
    else:
        monlib = None

    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
        if not args.no_check_mask_with_model:
            if not utils.maps.test_mask_with_model(mask, st):
                raise SystemExit("\nError: Model is out of mask.\n"
                                 "Please check your --model and --mask. You can disable this test with --no_check_mask_with_model.")
    elif args.mask_radius:
        mask = utils.maps.mask_from_model(st, args.mask_radius, grid=g)
        utils.maps.write_ccp4_map("mask_from_model.ccp4", mask)
    else:
        mask = None
        logger.writeln("Warning: Mask is needed for map normalization. Use --mask or --mask_radius if you want normalized map.")

    hkldata, map_labs, stats_str = calc_fofc(st, args.resolution, maps, mask=mask, monlib=monlib, B=args.B,
                                             half1_only=args.half1_only, no_fsc_weights=args.no_fsc_weights,
                                             sharpening_b=args.sharpening_b, omit_proton=args.omit_proton,
                                             omit_h_electron=args.omit_h_electron,
                                             source=args.source)
    write_files(hkldata, map_labs, grid_start, stats_str,
                mask=mask, output_prefix=args.output_prefix,
                trim_map=args.trim, trim_mtz=args.trim_mtz, omit_h_electron=args.omit_h_electron)

    py_out = "{}_coot.py".format(args.output_prefix)
    write_coot_script(py_out, model_file=args.model,
                      mtz_file=args.output_prefix+"_maps.mtz",
                      contour_fo=None if mask is None else 1.2,
                      contour_fofc=None if mask is None else 3.0,
                      ncs_ops=ncs_org)
    logger.writeln("\nOpen model and diffmap mtz with COOT:")
    logger.writeln("coot --script " + py_out)
    if mask is not None:
        logger.writeln("\nWant to list Fo-Fc map peaks? Try:")
        if args.omit_h_electron:
            logger.writeln("servalcat util map_peaks --map {}_normalized_fofc_flipsign.mrc --model {} --abs_level 4.0".format(args.output_prefix, args.model))
        else:
            logger.writeln("servalcat util map_peaks --map {}_normalized_fofc.mrc --model {} --abs_level 4.0".format(args.output_prefix, args.model))

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
