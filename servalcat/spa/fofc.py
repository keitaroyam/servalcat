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
    parser.add_argument("-B", type=float, help="Estimated blurring")
    parser.add_argument("--half1_only", action='store_true', help="Only use half 1 for map calculation (use half 2 only for noise estimation)")
    parser.add_argument("--normalized_map", action='store_true',
                        help="Write normalized map in the masked region")
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
                        help="Omit proton from model in map calculation")
    parser.add_argument("--omit_h_electron", action='store_true',
                        help="Omit hydrogen electrons from model in map calculation")
    parser.add_argument('-o','--output_prefix', default="diffmap",
                        help='output file name prefix')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_D_and_S(hkldata, has_halfmaps=True, half1_only=False):#fo_asu, fc_asu, varn, bins, bin_idxes):
    bdf = hkldata.binned_df
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
        var_noise = hkldata.binned_df.var_noise * 2
    elif has_halfmaps:
        var_noise = hkldata.binned_df.var_noise
        
    bin_limits = dict(hkldata.bin_and_limits())
    for i_bin, g in hkldata.binned():
        bin_d_max, bin_d_min = bin_limits[i_bin]
        Fo = FP[g.index]
        Fc = g.FC.to_numpy()
        fsc = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
        bdf.loc[i_bin, "D"] = numpy.sum(numpy.real(Fo * numpy.conj(Fc)))/numpy.sum(numpy.abs(Fc)**2)
        if has_halfmaps:
            varn = var_noise[i_bin]
            fsc_full = hkldata.binned_df.FSCfull[i_bin]
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
    if has_halfmaps:
        labs = ["Fupdate", "DELFWT", "FWT", "DELFWT_noscale", "Fupdate_noscale"]
        if B is not None: labs.extend(["Fupdate_b0", "DELFWT_b0", "FWT_b0"])
    else:
        labs = ["DELFWT"]
        
    tmp = {}
    for l in labs:
        tmp[l] = numpy.zeros(len(hkldata.df.index), numpy.complex128)

    logger.write("Calculating maps..")
    time_t = time.time()

    if half1_only:
        FP = hkldata.df.F_map1.to_numpy()
    else:
        FP = hkldata.df.FP.to_numpy()
        
    for i_bin, g in hkldata.binned():
        Fo = FP[g.index]
        Fc = g.FC.to_numpy()
        D = hkldata.binned_df.D[i_bin]
        if not has_halfmaps:
            delfwt = (Fo-D*Fc)
            tmp["DELFWT"][g.index] = delfwt
            continue

        S = hkldata.binned_df.S[i_bin] # variance of unexplained signal
        fsc = hkldata.binned_df.FSCfull[i_bin] # FSCfull
        if half1_only:
            varn = hkldata.binned_df.var_noise[i_bin] * 2
            fsc = fsc/(2-fsc) # to FSChalf
        else:
            varn = hkldata.binned_df.var_noise[i_bin]

        w = 1. if no_fsc_weights else S/(S+varn)
        w_nomodel = 1. if no_fsc_weights else fsc
        
        delfwt = w * (Fo-D*Fc)
        fup = w * Fo + (1.-w)*D*Fc

        tmp["DELFWT_noscale"][g.index] = delfwt
        tmp["Fupdate_noscale"][g.index] = fup

        sig_fo = numpy.std(Fo)
        if sharpening_b is None:
            if fsc < 0 or sig_fo < 1e-10: # FIXME probably we should compare sig_fo with mean(fo)
                logger.write("WARNING: skipping bin {} sig_fo={} fsc={}".format(i_bin, sig_fo, fsc))
                continue                
            k = sig_fo * numpy.sqrt(fsc)
        else:
            s2 = 1./hkldata.d_spacings()[g.index]**2
            k = numpy.exp(-sharpening_b*s2/4)
            
        #n_fofc = numpy.sqrt(var_cmpl(Fo-D*Fc))

        lab_suf = "" if B is None else "_b0"
        tmp["FWT"+lab_suf][g.index] = w_nomodel/k*Fo
        tmp["DELFWT"+lab_suf][g.index] = delfwt/k
        tmp["Fupdate"+lab_suf][g.index] = fup/k

        if B is not None: # local B based map
            s2 = 1./hkldata.d_spacings()[g.index]**2 # TODO fix duplicated calculation
            k_l = numpy.exp(-B*s2/4.)
            k2_l = numpy.exp(-B*s2/2.)
            fsc_l = k2_l*fsc/(1+(k2_l-1)*fsc)
            S_l = S * k2_l
            w = 1. if no_fsc_weights else S_l/(S_l+varn)            
            w_nomodel = 1. if no_fsc_weights else fsc_l
            
            delfwt = (Fo-D*Fc)*w/k/k_l
            fup = (w*Fo+(1.-w)*D*Fc)/k/k_l
            fwt = Fo*w_nomodel/k/k_l
            logger.write("{:4d} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}".format(i_bin,
                                                                           numpy.average(k),
                                                                           numpy.average(sig_fo),
                                                                           numpy.average(fsc_l),
                                                                           numpy.average(k_l),
                                                                           numpy.average(abs(fup)),
                                                                           numpy.average(abs(delfwt))))
            tmp["FWT"][g.index] = fwt
            tmp["DELFWT"][g.index] = delfwt
            tmp["Fupdate"][g.index] = fup

    for l in labs:
        hkldata.df[l] = tmp[l]

    logger.write(" finished in {:.3f} sec.".format(time.time()-time_t))
    return labs
# calc_maps()

def dump_to_mtz(hkldata, map_labs, mtz_out):
    map_labs = map_labs + ["FP", "FC"]
    data = numpy.empty((len(hkldata.df.index), len(map_labs)*2+3))
    data[:,:3] = hkldata.df[["H","K","L"]]
    for i, lab in enumerate(map_labs):
        data[:,3+i*2] = numpy.abs(hkldata.df[lab])
        data[:,3+i*2+1] = numpy.angle(hkldata.df[lab], deg=True)
        
    mtz = gemmi.Mtz()
    mtz.spacegroup = hkldata.sg
    mtz.cell = hkldata.cell
    mtz.add_dataset('HKL_base')
    for label in ['H', 'K', 'L']: mtz.add_column(label, 'H')

    for lab in map_labs:
        mtz.add_column(lab, "F")
        mtz.add_column(("PH"+lab).replace("FWT", "WT"), "P")
    
    mtz.set_data(data)
    mtz.write_to_file(mtz_out)
# dump_to_mtz()

def calc_fofc(st, d_min, maps, mask=None, monlib=None, B=None, half1_only=False,
              no_fsc_weights=False, sharpening_b=None, omit_proton=False, omit_h_electron=False):
    if no_fsc_weights:
        logger.write("WARNING: --no_fsc_weights is requested.")
    if sharpening_b is not None:
        logger.write("WARNING: --sharpening_b={} is given".format(sharpening_b))
    
    fc_asu = utils.model.calc_fc_fft(st, d_min, cutoff=1e-7, monlib=monlib, source="electron")
    hkldata = utils.maps.mask_and_fft_maps(maps, d_min, mask)
    hkldata.merge_asu_data(fc_asu, "FC")
    hkldata.setup_relion_binning()

    has_halfmaps = (len(maps) == 2)
    if has_halfmaps:
        utils.maps.calc_noise_var_from_halfmaps(hkldata)

    if B is not None:
        Bave = numpy.average(utils.model.all_B(st))
        logger.write("Using user-specified B: {}".format(B))
        logger.write("    Average B of model= {:.2f}".format(Bave))
        b_local = B - Bave
        logger.write("    Relative B for map= {:.2f}".format(b_local))
    else:
        b_local = None
        
    stats_str = calc_D_and_S(hkldata, has_halfmaps=has_halfmaps, half1_only=half1_only)

    if omit_proton or omit_h_electron:
        fc_asu_2 = utils.model.calc_fc_fft(st, d_min, cutoff=1e-7, monlib=monlib, source="electron",
                                           omit_proton=omit_proton, omit_h_electron=omit_h_electron)
        del hkldata.df["FC"]
        hkldata.merge_asu_data(fc_asu_2, "FC")
    
    map_labs = calc_maps(hkldata, B=b_local, has_halfmaps=has_halfmaps, half1_only=half1_only,
                         no_fsc_weights=no_fsc_weights, sharpening_b=sharpening_b)
    return hkldata, map_labs, stats_str
# calc_fofc()

def write_files(hkldata, map_labs, grid_start, stats_str,
                mask=None, output_prefix="diffmap", trim_map=False, trim_mtz=False,
                normalize_map=False, omit_h_electron=False):
    # this function may modify the overall scale of FWT/DELFWT.

    if mask is not None and (trim_map or trim_mtz):
        new_cell, new_shape, grid_start, shifts = shift_maps.determine_shape_and_shift(mask=gemmi.FloatGrid(mask,
                                                                                                            hkldata.cell,
                                                                                                            hkldata.sg),
                                                                                       grid_start=grid_start,
                                                                                       padding=5,
                                                                                       mask_cutoff=0.5,
                                                                                       noncentered=True,
                                                                                       noncubic=True,
                                                                                       json_out=None)
    else:
        new_cell, new_shape, shifts = None, None, None
        
    if normalize_map and mask is not None:
        logger.write("Normalized Fo-Fc map requested.")
        delfwt_map = hkldata.fft_map("DELFWT", grid_size=mask.shape)
        cutoff = 0.5
        masked = numpy.array(delfwt_map)[mask>cutoff]
        logger.write("   Whole volume: {} voxels".format(delfwt_map.point_count))
        logger.write("  Masked volume: {} voxels (>{})".format(masked.size, cutoff))
        global_mean = numpy.average(delfwt_map)
        global_std = numpy.std(delfwt_map)
        logger.write("    Global mean: {:.3e}".format(global_mean))
        logger.write("     Global std: {:.3e}".format(global_std))
        masked_mean = numpy.average(masked)
        masked_std = numpy.std(masked)
        logger.write("    Masked mean: {:.3e}".format(masked_mean))
        logger.write("     Masked std: {:.3e}".format(masked_std))
        #logger.write(" If you want to scale manually: {}".format())
        scaled = (delfwt_map - masked_mean)/masked_std
        hkldata.df["DELFWT"] /= masked_std # it would work if masked_mean~0
        if omit_h_electron:
            scaled *= -1
            filename = "{}_normalized_fofc_flipsign.mrc".format(output_prefix)
        else:
            filename = "{}_normalized_fofc.mrc".format(output_prefix)
        logger.write("  Writing {}".format(filename))
        utils.maps.write_ccp4_map(filename, scaled, cell=hkldata.cell,
                                  grid_start=grid_start, grid_shape=new_shape)

        # Write Fo map as well
        if "FWT" in hkldata.df:
            fwt_map = hkldata.fft_map("FWT", grid_size=mask.shape)
            masked = numpy.array(fwt_map)[mask>cutoff]
            masked_mean = numpy.average(masked)
            masked_std = numpy.std(masked)
            scaled = (fwt_map - masked_mean)/masked_std # does not make much sense for Fo map though
            hkldata.df["FWT"] /= masked_std # it would work if masked_mean~0
            filename = "{}_normalized_fo.mrc".format(output_prefix)
            logger.write("  Writing {}".format(filename))
            utils.maps.write_ccp4_map(filename, scaled, cell=hkldata.cell,
                                      grid_start=grid_start, grid_shape=new_shape)

    if trim_mtz and shifts is not None:
        hkldata2 = utils.hkl.HklData(new_cell, hkldata.sg, df=None)
        d_min = hkldata.d_min_max()[0]
        for lab in map_labs + ["FP", "FC"]:
            gr = hkldata.fft_map(lab, mask.shape)
            gr = gemmi.FloatGrid(gr.get_subarray(*(list(grid_start)+list(new_shape))),
                                 new_cell, hkldata.sg)
            ad = gemmi.transform_map_to_f_phi(gr).prepare_asu_data(dmin=d_min)
            hkldata2.merge_asu_data(ad, lab)
            hkldata2.translate(lab, -shifts)
        hkldata = hkldata2
        
    dump_to_mtz(hkldata, map_labs, "{}.mtz".format(output_prefix))
    open("{}_Fstats.log".format(output_prefix), "w").write(stats_str)
# write_files()

def main(args):
    if not args.halfmaps and not args.map:
        logger.error("Error: give --halfmaps or --map")
        return

    if not args.halfmaps and args.B is not None:
        logger.error("Error: -B only works for half maps")
        return

    if args.half1_only:
        if not args.halfmaps:
            logger.error("--half1_only requires half maps")
            return
        logger.error("--half1_only specified. Half map 2 is used only for noise estimation")

    if not args.halfmaps:
        logger.error("Warning: using --halfmaps is strongly recommended!")

    st = utils.fileio.read_structure(args.model)
    utils.model.expand_ncs(st)

    if (args.omit_proton or args.omit_h_electron) and st[0].count_hydrogen_sites() == 0:
        logger.error("ERROR! --omit_proton/--omit_h_electron requested, but no hydrogen atoms were found.")
        return

    if args.halfmaps:
        maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
        has_halfmaps = True
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]
        has_halfmaps = False

    grid_start = maps[0][1]
    g = maps[0][0]
    st.spacegroup_hm = "P1"
    st.cell = g.unit_cell

    if st[0].count_hydrogen_sites() > 0:
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib)
    else:
        monlib = None

    if args.mask:
        mask = numpy.array(utils.fileio.read_ccp4_map(args.mask)[0])
    elif args.mask_radius:
        mask = gemmi.FloatGrid(*g.shape)
        mask.set_unit_cell(g.unit_cell)
        mask.spacegroup = gemmi.SpaceGroup(1)
        mask.mask_points_in_constant_radius(st[0], args.mask_radius, 1.)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = mask
        ccp4.update_ccp4_header(2, True) # float, update stats
        ccp4.write_ccp4_map("mask_from_model.ccp4")
        mask = numpy.array(mask)
    else:
        mask = None
        if args.normalized_map:
            logger.error("Error: Provide --mask or --mask_radius if you want --normalized-map.")
            return

    hkldata, map_labs, stats_str = calc_fofc(st, args.resolution, maps, mask=mask, monlib=monlib, B=args.B,
                                             half1_only=args.half1_only, no_fsc_weights=args.no_fsc_weights,
                                             sharpening_b=args.sharpening_b, omit_proton=args.omit_proton,
                                             omit_h_electron=args.omit_h_electron)
    write_files(hkldata, map_labs, grid_start, stats_str,
                mask=mask, output_prefix=args.output_prefix,
                trim_map=args.trim, trim_mtz=args.trim_mtz, normalize_map=args.normalized_map, omit_h_electron=args.omit_h_electron)
    
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
