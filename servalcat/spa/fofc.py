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
import argparse

def add_arguments(parser):
    parser.description = 'Fo-Fc map calculation based on model and data errors'
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--halfmaps", nargs=2)
    group.add_argument("--map", help="Use only if you really do not have half maps.")
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file (PDB or mmCIF/PDBx')
    parser.add_argument("-d", '--resolution', type=float, required=True)
    parser.add_argument('-m', '--mask', help="mask file")
    parser.add_argument('-r', '--mask_radius', type=float, help="mask radius")
    parser.add_argument("-B", type=float, help="Estimated blurring.")
    parser.add_argument("--half1_only", action='store_true', help="Only use half 1 for map (use half 2 only for noise estimation)")
    parser.add_argument("--normalized_map", action='store_true',
                        help="Write normalized map in the masked region")
    parser.add_argument("--crop", action='store_true',
                        help="Write cropped maps")
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
$$
1/resol^2 bin n d_max d_min log(var(Fo)) log(var(Fc)) log(var(DFc)) FSC.model FSC.full sqrt(FSC.full) D log(var_U,T) log(var_noise) wFo wFc
$$
$$
"""
    tmpl = "{:.4f} {:3d} {:7d} {:7.3f} {:7.3f} {:.4e} {:.4e} {:4e} {: .4f}   {: .4f} {: .4f} {: .4e} {:.4e} {:.4e} {:.4f} {:.4f}\n"

    var_noise = None
    FP = hkldata.df.FP.to_numpy()
    if half1_only:
        FP = hkldata.df.F_map1.to_numpy()
        var_noise = hkldata.binned_df.var_noise * 2
    elif has_halfmaps:
        var_noise = hkldata.binned_df.var_noise
        
    FC = hkldata.df.FC.to_numpy()
    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        Fo = FP[sel]
        Fc = FC[sel]
        fsc = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
        bdf.loc[i_bin, "D"] = numpy.sum(numpy.real(Fo * numpy.conj(Fc)))/numpy.sum(numpy.abs(Fc)**2)
        if has_halfmaps:
            varn = var_noise[i_bin]
            fsc_full = hkldata.binned_df.FSCfull[i_bin]
            S = max(0, numpy.average(numpy.abs(Fo-bdf.D[i_bin]*Fc)**2)-varn)
            bdf.loc[i_bin, "S"] = S
            w = S/(S+varn)
        else:
            varn = fsc_full = 0
            w = 1

        with numpy.errstate(divide="ignore"):
            stats_str += tmpl.format(1/bin_d_min**2, i_bin, Fo.size, bin_d_max, bin_d_min,
                                     numpy.log(numpy.average(numpy.abs(Fo)**2)),
                                     numpy.log(numpy.average(numpy.abs(Fc)**2)),
                                     numpy.log(bdf.D[i_bin]**2*numpy.average(numpy.abs(Fc)**2)),
                                     fsc, fsc_full, numpy.sqrt(fsc_full), bdf.D[i_bin],
                                     numpy.log(bdf.S[i_bin]), numpy.log(varn),
                                     w, 1-w)
    return stats_str
# calc_D_and_S()

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
#@profile
def calc_maps(hkldata, B=None, has_halfmaps=True, half1_only=False):
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
        
    FC = hkldata.df.FC.to_numpy()
    
    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        Fo = FP[sel]
        Fc = FC[sel]
        D = hkldata.binned_df.D[i_bin]
        if not has_halfmaps:
            delfwt = (Fo-D*Fc)
            tmp["DELFWT"][sel] = delfwt
            continue

        S = hkldata.binned_df.S[i_bin]
        fsc = hkldata.binned_df.FSCfull[i_bin]
        if half1_only:
            varn = hkldata.binned_df.var_noise[i_bin] * 2
            fsc = fsc/(2-fsc) # to FSChalf
        else:
            varn = hkldata.binned_df.var_noise[i_bin]

        delfwt = (Fo-D*Fc)*S/(S+varn)
        fup = (Fo*S+varn*D*Fc)/(S+varn)

        tmp["DELFWT_noscale"][sel] = delfwt
        tmp["Fupdate_noscale"][sel] = fup

        sig_fo = numpy.std(Fo)
        n_fo = sig_fo * numpy.sqrt(fsc)
        if n_fo < 1e-10 or n_fo != n_fo:
            logger.write("WARNING: skipping bin {} sig_fo={} fsc={}".format(i_bin, sig_fo, fsc))
            continue
        #n_fofc = numpy.sqrt(var_cmpl(Fo-D*Fc))

        lab_suf = "" if B is None else "_b0"
        tmp["FWT"+lab_suf][sel] = numpy.sqrt(fsc)*Fo/sig_fo
        tmp["DELFWT"+lab_suf][sel] = delfwt/n_fo
        tmp["Fupdate"+lab_suf][sel] = fup/n_fo

        if B is not None:
            s2 = 1./hkldata.d_spacings()[sel]**2
            k = numpy.exp(-B*s2/4.)
            k2 = numpy.exp(-B*s2/2.)
            fsc_l = k2*fsc/(1+(k2-1)*fsc)
            #n_fo = sig_fo * numpy.sqrt(fsc_l) * k
            S_l = S * k2
            
            delfwt = (Fo-D*Fc)*S*k/(S_l+varn)/n_fo # S_l/k = S*k
            fup = (Fo*S_l+varn*D*Fc)/(S_l+varn)/n_fo/k
            fwt = Fo*fsc_l/n_fo
            logger.write("{:4d} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}".format(i_bin,
                                                                           numpy.average(n_fo),
                                                                           numpy.average(sig_fo),
                                                                           numpy.average(fsc_l),
                                                                           numpy.average(k),
                                                                           numpy.average(abs(fup)),
                                                                           numpy.average(abs(delfwt))))
            tmp["FWT"][sel] = fwt
            tmp["DELFWT"][sel] = delfwt
            tmp["Fupdate"][sel] = fup

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

def calc_fofc(st, d_min, maps, mask=None, monlib=None, B=None, half1_only=False, omit_proton=False, omit_h_electron=False):
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
    
    map_labs = calc_maps(hkldata, B=b_local, has_halfmaps=has_halfmaps, half1_only=half1_only)
    return hkldata, map_labs, stats_str
# calc_fofc()

def write_files(hkldata, map_labs, grid_start, stats_str,
                mask=None, output_prefix="diffmap", crop=False, normalize_map=False, omit_h_electron=False):
    dump_to_mtz(hkldata, map_labs, "{}.mtz".format(output_prefix))
    open("{}_Fstats.log".format(output_prefix), "w").write(stats_str)
    
    # DEBUG
    if mask is not None:
        utils.maps.write_ccp4_map("fc_map.mrc",  hkldata.fft_map("FC", grid_size=mask.shape), cell=hkldata.cell,
                                  mask_for_extent=mask if crop else None,
                                  grid_start=grid_start)
        hkldata.df["DFC"] = hkldata.df["FC"] * hkldata.binned_data_as_array("D")
        utils.maps.write_ccp4_map("dfc_map.mrc",  hkldata.fft_map("DFC", grid_size=mask.shape), cell=hkldata.cell,
                                  mask_for_extent=mask if crop else None,
                                  grid_start=grid_start)

    
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
        if omit_h_electron:
            scaled *= -1
            filename = "{}_normalized_fofc_flipsign.mrc".format(output_prefix)
        else:
            filename = "{}_normalized_fofc.mrc".format(output_prefix)
        logger.write("  Writing {}".format(filename))
        utils.maps.write_ccp4_map(filename, scaled, cell=hkldata.cell,
                                  mask_for_extent=mask if crop else None,
                                  grid_start=grid_start)

        # Write Fo map as well
        fwt_map = hkldata.fft_map("FWT", grid_size=mask.shape)
        masked = numpy.array(fwt_map)[mask>cutoff]
        masked_mean = numpy.average(masked)
        masked_std = numpy.std(masked)
        scaled = (fwt_map - masked_mean)/masked_std # does not make much sense for Fo map though
        filename = "{}_normalized_fo.mrc".format(output_prefix)
        logger.write("  Writing {}".format(filename))
        utils.maps.write_ccp4_map(filename, scaled, cell=hkldata.cell,
                                  mask_for_extent=mask if crop else None,
                                  grid_start=grid_start)

# write_files()

def main(args):
    if not args.halfmaps and not args.map:
        logger.write("Error: give --halfmaps or --map")
        return

    if not args.halfmaps and args.B is not None:
        logger.write("Error: -B only works for half maps")
        return

    if args.half1_only:
        if not args.halfmaps:
            logger.write("--half1_only requires half maps")
            return
        logger.write("--half1_only specified. Half map 2 is used only for noise estimation")

    if not args.halfmaps:
        logger.write("Warning: using --halfmaps is strongly recommended!")

    st = utils.fileio.read_structure(args.model)
    utils.model.expand_ncs(st)

    if (args.omit_proton or args.omit_h_electron) and st[0].count_hydrogen_sites() == 0:
        logger.write("ERROR! --omit_proton/--omit_h_electron requested, but no hydrogen atoms were found.")
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
        monlib = utils.restraints.load_monomer_library(st[0].get_all_residue_names(),
                                                       monomer_dir=args.monlib)
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
            logger.write("Error: Provide --mask or --mask_radius if you want --normalized-map.")
            return

    hkldata, map_labs, stats_str = calc_fofc(st, args.resolution, maps, mask=mask, monlib=monlib, B=args.B,
                                             half1_only=args.half1_only, omit_proton=args.omit_proton,
                                             omit_h_electron=args.omit_h_electron)
    write_files(hkldata, map_labs, grid_start, stats_str,
                mask=mask, output_prefix=args.output_prefix,
                crop=args.crop, normalize_map=args.normalized_map, omit_h_electron=args.omit_h_electron)
    
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
