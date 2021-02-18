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

var_cmpl = lambda x: numpy.var(x.real)+numpy.var(x.imag)
var_cmpl = lambda x: numpy.var(x)

def add_arguments(parser):
    parser.description = 'Fo-Fc map calculation based on model and data errors'
    parser.add_argument("--halfmaps", required=True, nargs=2)
    #parser.add_argument('--mapref', help='Reference map file')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file (PDB or mmCIF/PDBx')
    parser.add_argument("-d", '--resolution', type=float, required=True)
    parser.add_argument('-m', '--mask', help="mask file")
    parser.add_argument('-r', '--mask_radius', type=float, help="mask radius")
    parser.add_argument("-B", type=float, help="Estimated blurring.")
    parser.add_argument("--normalized_map", action='store_true',
                        help="Write normalized map in the masked region")
    parser.add_argument("--crop", action='store_true',
                        help="Write cropped maps")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument("--omit_proton", action='store_true',
                        help="Omit proton from model in map calculation")
    parser.add_argument('--output_prefix', default="diffmap",
                        help='output file name prefix')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def read_maps(halfmaps, d_min, mask=None):
    assert len(halfmaps) == 2
    map1 = utils.fileio.read_ccp4_map(halfmaps[0])
    map2 = utils.fileio.read_ccp4_map(halfmaps[1])
    
    if mask is not None:
        map1.grid = gemmi.FloatGrid(numpy.array(map1.grid)*mask,
                                    map1.grid.unit_cell, map1.grid.spacegroup)
        map2.grid = gemmi.FloatGrid(numpy.array(map2.grid)*mask,
                                    map2.grid.unit_cell, map2.grid.spacegroup)

    asu1 = gemmi.transform_map_to_f_phi(map1.grid).prepare_asu_data(dmin=d_min)
    asu2 = gemmi.transform_map_to_f_phi(map2.grid).prepare_asu_data(dmin=d_min)

    return asu1, asu2
# read_maps()
    
def calc_noise_var(asu1, asu2, fc_asu):
    df = utils.hkl.df_from_asu_data(asu1, "F_map1")
    hkldata = utils.hkl.HklData(asu1.unit_cell, asu1.spacegroup, False, df)
    hkldata.merge_asu_data(asu2, "F_map2")
    hkldata.merge_asu_data(fc_asu, "FC")
    hkldata.df["FP"] = (hkldata.df.F_map1 + hkldata.df.F_map2)/2.
    hkldata.setup_relion_binning()
    # Scale
    #iniscale = utils.scaling.InitialScaler(fo_asu, fc_asu, aniso=True)
    #iniscale.run()
    #scale_for_fo = iniscale.get_scales()
    #fo_asu.value_array[:] *= scale_for_fo
    #asu1.value_array[:] *= scale_for_fo
    #asu2.value_array[:] *= scale_for_fo

    s_array = 1./hkldata.d_spacings()
    hkldata.binned_df["var_noise"] = 0.
    hkldata.binned_df["FSCfull"] = 0.

    logger.write("Bin Ncoeffs d_max   d_min   FSChalf var.noise   scale")
    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        # scale
        #fo = numpy.array(hkldata.df.FP[sel])
        #fc = numpy.array(hkldata.df.FC[sel])
        fo = hkldata.df.FP[sel]
        fc = hkldata.df.FC[sel]
        scale = 1. #numpy.sqrt(var_cmpl(fc)/var_cmpl(fo))
        #hkldata.df.loc[sel, "FP"] *= scale
        #hkldata.df.loc[sel, "F_map1"] *= scale
        #hkldata.df.loc[sel, "F_map2"] *= scale
        
        sel1 = numpy.array(hkldata.df.F_map1[sel])
        sel2 = numpy.array(hkldata.df.F_map2[sel])

        fsc = numpy.real(numpy.corrcoef(sel1, sel2)[1,0])
        varn = var_cmpl(sel1-sel2)/4
        logger.write("{:3d} {:7d} {:7.3f} {:7.3f} {:.4f} {:e} {}".format(i_bin, sel1.size, bin_d_max, bin_d_min,
                                                                         fsc, varn, scale))
        hkldata.binned_df.loc[i_bin, "var_noise"] = varn
        hkldata.binned_df.loc[i_bin, "FSCfull"] = 2*fsc/(1+fsc)

    return hkldata
# calc_noise_var()

def calc_D_and_S(hkldata, output_prefix):#fo_asu, fc_asu, varn, bins, bin_idxes):
    bdf = hkldata.binned_df
    bdf["D"] = 0.
    bdf["S"] = 0.
    ofs = open("{}_Fstats.dat".format(output_prefix), "w")
    ofs.write("bin       n   d_max   d_min         Fo         Fc FSC.model FSC.full      D          S          N\n")
    tmpl = "{:3d} {:7d} {:7.3f} {:7.3f} {:.4e} {:.4e} {: .4f}   {: .4f} {: .4e} {:.4e} {:.4e}\n"
    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        Fo = numpy.array(hkldata.df.FP[sel])
        Fc = numpy.array(hkldata.df.FC[sel])
        varn = hkldata.binned_df.var_noise[i_bin]
        fsc = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
        fsc_full = hkldata.binned_df.FSCfull
        bdf.loc[i_bin, "D"] = numpy.sum(numpy.real(Fo * numpy.conj(Fc)))/numpy.sum(numpy.abs(Fc)**2)
        bdf.loc[i_bin, "S"] = max(0, numpy.average(numpy.abs(Fo-bdf.D[i_bin]*Fc)**2)-varn)
        ofs.write(tmpl.format(i_bin, Fo.size, bin_d_max, bin_d_min,
                              numpy.average(numpy.abs(Fo)),
                              numpy.average(numpy.abs(Fc)),
                              fsc, fsc_full[i_bin], bdf.D[i_bin], bdf.S[i_bin], varn))
# calc_D_and_S()

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
#@profile
def calc_maps(hkldata, B=None):#fo_asu, fc_asu, D, S, varn, bins, bin_idxes):
    labs = ["DELFWT", "FWT", "DELFWT_noscale", "FWT_noscale"]
    if B is not None: labs.extend(["DELFWT_b0", "FWT_b0"])
    tmp = {}
    for l in labs:
        tmp[l] = numpy.zeros(len(hkldata.df.index), numpy.complex128)

    logger.write("Calculating maps..")
    time_t = time.time()

    FP = numpy.array(hkldata.df.FP)
    FC = numpy.array(hkldata.df.FC)
    
    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        Fo = FP[sel]
        Fc = FC[sel]
        D = hkldata.binned_df.D[i_bin]
        S = hkldata.binned_df.S[i_bin]
        FSCfull = hkldata.binned_df.FSCfull[i_bin]
        varn = hkldata.binned_df.var_noise[i_bin]
        FSCfull = max(1e-6, FSCfull)

        delfwt = (Fo-D*Fc)*S/(S+varn)
        fwt = (Fo*S+varn*D*Fc)/(S+varn)

        tmp["DELFWT_noscale"][sel] = delfwt
        tmp["FWT_noscale"][sel] = fwt

        #sig_fo = numpy.sqrt(var_cmpl(Fo))
        sig_fo = numpy.std(Fo)
        n_fo = sig_fo * numpy.sqrt(FSCfull)
        #n_fofc = numpy.sqrt(var_cmpl(Fo-D*Fc))

        lab_suf = "" if B is None else "_b0"
        tmp["DELFWT"+lab_suf][sel] = delfwt/n_fo
        tmp["FWT"+lab_suf][sel] = fwt/n_fo

        if B is not None:
            s2 = 1./hkldata.d_spacings()[sel]**2
            k = numpy.exp(-B*s2/4.)
            k2 = numpy.exp(-B*s2/2.)
            fsc_l = k2*FSCfull/(1+(k2-1)*FSCfull)
            #n_fo = sig_fo * numpy.sqrt(fsc_l) * k
            S_l = S * k2
            
            delfwt = (Fo-D*Fc)*S*k/(S_l+varn)/n_fo # S_l/k = S*k
            fwt = (Fo*S_l+varn*D*Fc)/(S_l+varn)/n_fo/k
            logger.write("{:4d} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}".format(i_bin,
                                                                           numpy.average(n_fo),
                                                                           numpy.average(sig_fo),
                                                                           numpy.average(fsc_l),
                                                                           numpy.average(k),
                                                                           numpy.average(abs(fwt)),
                                                                           numpy.average(abs(delfwt))))
            tmp["DELFWT"][sel] = delfwt
            tmp["FWT"][sel] = fwt

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
        
def main(args):
    st = gemmi.read_structure(args.model)
    st.expand_ncs(gemmi.HowToNameCopiedChain.Short)

    g = utils.fileio.read_ccp4_map(args.halfmaps[0]).grid
    st.spacegroup_hm = "P1"
    st.cell = g.unit_cell

    if args.omit_proton and st[0].count_hydrogen_sites() == 0:
        logger.write("ERROR! --omit_proton requested, but no hydrogen atoms were found.")
        return
    
    if st[0].count_hydrogen_sites() > 0:
        monlib = utils.model.load_monomer_library(st[0].get_all_residue_names(),
                                                  monomer_dir=args.monlib)
    else:
        monlib = None

    if args.mask:
        mask = numpy.array(utils.fileio.read_ccp4_map(args.mask).grid)
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
            logger.write("WARNING: Mask is not available. --normalized-map will have no effect.")
        
    fc_asu = utils.model.calc_fc_fft(st, args.resolution, r_cut=1e-7, monlib=monlib, source="electron")
    asu1, asu2 = read_maps(args.halfmaps, args.resolution, mask)
    hkldata = calc_noise_var(asu1, asu2, fc_asu)
    #dump_to_mtz(fo_asu, "Fo.mtz")

    if args.B is not None:
        Bave = numpy.average(utils.model.all_B(st))
        logger.write("Using user-specified B: {}".format(args.B))
        logger.write("    Average B of model= {:.2f}".format(Bave))
        B = args.B - Bave
        logger.write("    Relative B for map= {:.2f}".format(B))
    else:
        B = None
        
    calc_D_and_S(hkldata, args.output_prefix)

    if args.omit_proton:
        fc_asu_2 = utils.model.calc_fc_fft(st, args.resolution, r_cut=1e-7, monlib=monlib, source="electron",
                                           omit_proton=True)
        del hkldata.df["FC"]
        hkldata.merge_asu_data(fc_asu_2, "FC")
    
    map_labs = calc_maps(hkldata, B=B)
    dump_to_mtz(hkldata, map_labs, "{}.mtz".format(args.output_prefix))

    if args.normalized_map and mask is not None:
        logger.write("Normalized Fo-Fc map requested.")
        delfwt_map = hkldata.fft_map("DELFWT", grid_size=mask.shape)
        cutoff = 1. - 1e-6
        masked = numpy.array(delfwt_map)[mask>cutoff]
        logger.write("   Whole volume: {} voxels".format(delfwt_map.point_count))
        logger.write("  Masked volume: {} voxels (>{})".format(masked.size, cutoff))
        global_mean = numpy.average(delfwt_map)
        global_std = numpy.std(delfwt_map)
        logger.write("    Global mean: {}".format(global_mean))
        logger.write("     Global std: {}".format(global_std))
        masked_mean = numpy.average(masked)
        masked_std = numpy.std(masked)
        logger.write("    Masked mean: {}".format(masked_mean))
        logger.write("     Masked std: {}".format(masked_std))
        #logger.write(" If you want to scale manually: {}".format())
        scaled = (delfwt_map - masked_mean)/masked_std
        filename = "{}_normalized_fofc.mrc".format(args.output_prefix)
        logger.write("  Writing {}".format(filename))
        utils.maps.write_ccp4_map(filename, scaled, cell=st.cell,
                                  mask_for_extent=mask if args.crop else None)

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
