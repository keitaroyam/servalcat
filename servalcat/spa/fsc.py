"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import gemmi
import numpy
import pandas
from servalcat.utils import logger
from servalcat import spa
from servalcat import utils

def add_arguments(parser):
    parser.description = 'FSC calculation'

    parser.add_argument('--model', nargs="*", action="append",
                        help="")
    parser.add_argument('--map',
                        help='Input map file(s)')
    parser.add_argument('--mtz',
                        help='Input mtz file.')
    parser.add_argument('--labin', nargs=2,
                        help='label (F and PHI) for mtz')
    parser.add_argument("--halfmaps",  nargs=2)
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--mask', help='Mask file')
    utils.symmetry.add_symmetry_args(parser) # add --pg etc
    parser.add_argument('-d', '--resolution',
                        type=float,
                        help='Default: Nyquist')
    parser.add_argument('--random_seed', type=float, default=1234,
                        help="random seed for phase randomized FSC")
    parser.add_argument('-o', '--fsc_out',
                        default="fsc.dat",
                        help='')
    parser.add_argument('--csv', action='store_true',
                        help="Write csv file")

# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def write_loggraph(stats, labs_fc, log_out):
    model_labs1 = [l for l in stats if any(l.startswith("fsc_"+fc) for fc in labs_fc)]
    model_labs2 = [l for l in stats if any(l.startswith(("cc_"+fc, "mcos_"+fc)) for fc in labs_fc)]
    power_labs = [l for l in stats if l.startswith("power_")]
    half_labs1 = ["fsc_half_unmasked", "fsc_half_masked", "fsc_half_masked_rand", "fsc_half_masked_corrected"]
    half_labs2 = ["cc_half", "mcos_half"]
    if not all(l in stats for l in half_labs1):
        if "fsc_half" in stats:
            half_labs1 = ["fsc_half"]
        else:
            half_labs1 = []

    s2lab = "1/resol^2"
    stats2 = stats.copy()
    stats2.insert(0, s2lab, 1./stats["d_min"]**2)
    stats2.insert(1, "bin", stats.index)
    for l in power_labs: stats2[l] = numpy.log(stats2[l])
    title_labs = []
    if half_labs1:
        title_labs.append(("Phase randomized FSC" if len(half_labs1) > 1 else "Half map FSC",
                           [s2lab] + half_labs1))
    if half_labs2:
        title_labs.append(("Half map amplitude CC and Mean(cos(dphi))",
                           [s2lab] + half_labs2))
    if model_labs1:
        title_labs.append(("Map-model FSC", [s2lab] + model_labs1))
    if model_labs2:
        title_labs.append(("Map-model amplitude CC and Mean(cos(dphi))", [s2lab] + model_labs2))
    if power_labs:
        title_labs.append(("log(Power)", [s2lab] + power_labs))

    title_labs.append(("number of Fourier coefficients", [s2lab, "ncoeffs"]))
    with open(log_out, "w") as ofs:
        ofs.write(utils.make_loggraph_str(stats2, main_title="FSC", title_labs=title_labs))
# write_loggraph()

def fsc_average(n, fsc):
    return numpy.nansum(n * fsc) / numpy.nansum(n)
# fsc_average()

def randomized_f(f):
    phase = numpy.random.uniform(0, 2, size=len(f)) * numpy.pi
    rf = numpy.abs(f) * (numpy.cos(phase) + 1j*numpy.sin(phase))
    return rf
# randomized_f()

def calc_fsc(hkldata, labs=None, fs=None):
    if labs is not None:
        assert len(labs) == 2
        fs = [hkldata.df[l].to_numpy() for l in labs]
    else:
        assert fs is not None and len(fs) == 2
    ret = []
    for i_bin, idxes in hkldata.binned():
        F1, F2 = fs[0][idxes], fs[1][idxes]
        fsc = numpy.real(numpy.corrcoef(F1, F2)[1,0])
        ret.append(fsc)
    return ret
# calc_fsc()

def calc_phase_randomized_fsc(hkldata, mask, labs_half, labs_half_masked, randomize_fsc_at=0.8):
    stats = hkldata.binned_df[["d_min", "d_max"]].copy()
    stats["fsc_half_unmasked"] = calc_fsc(hkldata, labs=labs_half)
    stats["fsc_half_masked"] = calc_fsc(hkldata, labs=labs_half_masked)
    stats["ncoeffs"] = 0

    # Randomize F
    f_rands = [numpy.copy(hkldata.df[labs_half[i]]) for i in range(2)]
    rand_start_bin = None
    for i_bin, idxes in hkldata.binned():
        stats.loc[i_bin, "ncoeffs"] = len(idxes)
        fsc_half = stats["fsc_half_unmasked"][i_bin]
        if rand_start_bin is None and fsc_half < randomize_fsc_at:
            rand_start_bin = i_bin
            logger.writeln(" randomize phase beyond {:.2f} A (bin {})".format(stats["d_max"][i_bin], i_bin))

        if rand_start_bin is not None:
            for i in range(2):
                f_rands[i][idxes] = randomized_f(hkldata.df[labs_half[i]].to_numpy()[idxes])

    # Multiply mask
    for i in range(2):
        g = hkldata.fft_map(data=f_rands[i], grid_size=mask.shape)
        g.array[:] *= mask
        fg = gemmi.transform_map_to_f_phi(g)
        f_rands[i] = fg.get_value_by_hkl(hkldata.miller_array())

    # Calc randomized fsc
    stats["fsc_half_masked_rand"] = calc_fsc(hkldata, fs=f_rands)

    # Calc corrected fsc
    stats["fsc_half_masked_corrected"] = 0.
    for i_bin in stats.index:
        if i_bin < rand_start_bin + 2: # RELION way # FIXME rand_start_bin can be None
            stats.loc[i_bin, "fsc_half_masked_corrected"] = stats["fsc_half_masked"][i_bin]
        else:
            fscn = stats["fsc_half_masked_rand"][i_bin]
            fsct = stats["fsc_half_masked"][i_bin]
            stats.loc[i_bin, "fsc_half_masked_corrected"] = (fsct - fscn) / (1. - fscn)
            
    global_res = 999.
    for i_bin in stats.index:
        if stats["fsc_half_masked_corrected"][i_bin] < 0.143:
            break
        global_res = 1./(0.5*(1./stats["d_min"][i_bin]+1./stats["d_max"][i_bin])) # definition is slightly different from RELION

    logger.writeln("resolution from mask corrected FSC = {:.2f} A".format(global_res))

    return stats, global_res
# calc_maskphase_randomized_fsc()

def calc_fsc_all(hkldata, labs_fc, lab_f, labs_half=None,
                 labs_half_nomask=None, mask=None): # TODO name changed
    if labs_half: assert len(labs_half) == 2
    if labs_half_nomask: assert len(labs_half_nomask) == 2 # only used when mask is not None

    if mask is not None and labs_half_nomask:
        stats, global_res = calc_phase_randomized_fsc(hkldata, mask,
                                                      labs_half=labs_half_nomask,
                                                      labs_half_masked=labs_half)
        half_fsc_done = True
    else:
        stats = hkldata.binned_df[["d_min", "d_max"]].copy()
        half_fsc_done = False
        
    stats["ncoeffs"] = 0
    stats["power_{}".format(lab_f)] = 0.
    for lab in labs_fc:
        stats["power_{}".format(lab)] = 0.
        stats["fsc_{}_full".format(lab)] = 0.
        stats["Rcmplx_{}_full".format(lab)] = 0.
        stats["cc_{}_full".format(lab)] = 0.
        stats["mcos_{}_full".format(lab)] = 0.
    if labs_half:
        if not half_fsc_done: stats["fsc_half"] = 0.
        stats["cc_half"] = 0.
        stats["mcos_half"] = 0.
        for lab in labs_fc:
            stats["fsc_{}_half1".format(lab)] = 0.
            stats["fsc_{}_half2".format(lab)] = 0.

    for i_bin, idxes in hkldata.binned():
        stats.loc[i_bin, "ncoeffs"] = len(idxes)
        Fo = hkldata.df[lab_f].to_numpy()[idxes]
        stats.loc[i_bin, "power_{}".format(lab_f)] = numpy.average(numpy.abs(Fo)**2)
        if labs_half:
            F1, F2 = hkldata.df[labs_half[0]].to_numpy()[idxes], hkldata.df[labs_half[1]].to_numpy()[idxes]
            if not half_fsc_done: stats.loc[i_bin, "fsc_half"] = numpy.real(numpy.corrcoef(F1, F2)[1,0])
            cc_half = numpy.corrcoef(numpy.abs(F1), numpy.abs(F2))[1,0]
            mcos_half = numpy.mean(numpy.cos(numpy.angle(F1) - numpy.angle(F2))) # f1*f2.conj()/abs(f1)/abs(f2) is much faster, but in case zero..
            stats.loc[i_bin, "cc_half"] = cc_half
            stats.loc[i_bin, "mcos_half"] = mcos_half
        else:
            F1, F2 = None, None

        for labfc in labs_fc:
            Fc = hkldata.df[labfc].to_numpy()[idxes]
            fsc_model = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
            cc_model = numpy.corrcoef(numpy.abs(Fo), numpy.abs(Fc))[1,0]
            mcos_model = numpy.mean(numpy.cos(numpy.angle(Fo) - numpy.angle(Fc)))
            D = numpy.sum(numpy.real(Fo * numpy.conj(Fc)))/numpy.sum(numpy.abs(Fc)**2)
            rcmplx_model = numpy.sum(numpy.abs(Fo-D*Fc))/numpy.sum(numpy.abs(Fo))
            stats.loc[i_bin, "fsc_{}_full".format(labfc)] = fsc_model
            stats.loc[i_bin, "cc_{}_full".format(labfc)] = cc_model
            stats.loc[i_bin, "mcos_{}_full".format(labfc)] = mcos_model
            stats.loc[i_bin, "Rcmplx_{}_full".format(labfc)] = rcmplx_model
            stats.loc[i_bin, "power_{}".format(labfc)] = numpy.average(numpy.abs(Fc)**2)
            if labs_half:
                stats.loc[i_bin, "fsc_{}_half1".format(labfc)] = numpy.real(numpy.corrcoef(F1, Fc)[1,0])
                stats.loc[i_bin, "fsc_{}_half2".format(labfc)] = numpy.real(numpy.corrcoef(F2, Fc)[1,0])
    return stats
# calc_fsc_all()

def main(args):
    numpy.random.seed(args.random_seed)
    if args.model:
        args.model = sum(args.model, [])
    else:
        args.model = []
        
    if args.mask:
        logger.writeln("Input mask file: {}".format(args.mask))
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None
        
    if args.halfmaps:
        maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
        unit_cell = maps[0][0].unit_cell
    elif args.map:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]
        unit_cell = maps[0][0].unit_cell
    elif args.mtz:
        mtz = utils.fileio.read_mmhkl(hklin)
        if mask is not None and mask.unit_cell != mtz.cell:
            raise SystemExit("Error: Inconsistent unit cell between mtz and mask")
        gr = mtz.transform_f_phi_to_map(f=args.labin[0],
                                        phi=args.labin[1],
                                        exact_size=mask.shape if mask is not None else (0,0,0),
                                        sample_rate=3 if mask is None else 0)
        maps = [[gr, [0,0,0]]]
        unit_cell = mtz.cell # TODO check cell of given label
        d_min = numpy.min(mtz.make_d_array()[~numpy.isnan(mtz.column_with_label(args.labin[0]).array)])
        if args.resolution is None:
            args.resolution = d_min
        elif args.resolution < d_min:
            raise SystemExit("Error: --resolution ({}) is higher than actual resolution in mtz ({:.2f}).".format(args.resolution, d_min))
    else:
        raise SystemExit("Error: No input map/mtz found.")

    if args.resolution is None:
        args.resolution = utils.maps.nyquist_resolution(maps[0][0])
        logger.writeln("WARNING: --resolution is not specified. Using Nyquist resolution: {:.2f}".format(args.resolution))
        
    sts = []
    for xyzin in args.model:
        st = utils.fileio.read_structure(xyzin)
        st.cell = unit_cell
        st.spacegroup_hm = "P1"
        utils.symmetry.update_ncs_from_args(args, st, map_and_start=maps[0])
        if len(st.ncs) > 0:
            utils.model.expand_ncs(st)
            
        sts.append(st)

    hkldata = None
    for i, m in enumerate(maps):
        if len(maps) == 2:
            lab = "F_map{}".format(i+1)
        else:
            lab = "FP"
        for j in range(2):
            if j == 1 and mask is None: break
            lab_suffix = ["_nomask", "_mask"][j]
            g = m[0]
            if j == 1:
                g.array[:] *= mask # modifies original data
            f_grid = gemmi.transform_map_to_f_phi(g)

            if hkldata is None:
                asudata = f_grid.prepare_asu_data(dmin=args.resolution, with_000=True)
                hkldata = utils.hkl.hkldata_from_asu_data(asudata, lab + lab_suffix)
            else:
                hkldata.df[lab + lab_suffix] = f_grid.get_value_by_hkl(hkldata.miller_array())

    if len(maps) == 2:
        hkldata.df["FP_nomask"] = (hkldata.df.F_map1_nomask + hkldata.df.F_map2_nomask) * 0.5
        if mask is not None:
            hkldata.df["FP_mask"] = (hkldata.df.F_map1_mask + hkldata.df.F_map2_mask) * 0.5

    if len(maps) == 2:
        labs_half = ["F_map1_nomask", "F_map2_nomask"]
        if mask is not None:
            labs_half_masked = ["F_map1_mask", "F_map2_mask"]
        else:
            labs_half_masked = []
    else:
        labs_half, labs_half_masked = [], []
    lab_f = "FP_nomask" if mask is None else "FP_mask"
    labs_fc = []
    for i, st in enumerate(sts): 
        labs_fc.append("FC_{}".format(i) if len(sts)>1 else "FC")
        hkldata.df[labs_fc[-1]] = utils.model.calc_fc_fft(st, args.resolution - 1e-6, source="electron",
                                                          miller_array=hkldata.miller_array())
        if mask is not None: # TODO F000
            g = hkldata.fft_map(labs_fc[-1], grid_size=mask.shape)
            g.array[:] *= mask
            fg = gemmi.transform_map_to_f_phi(g)
            hkldata.df[labs_fc[-1]] = fg.get_value_by_hkl(hkldata.miller_array())

    hkldata.setup_relion_binning()
    stats = calc_fsc_all(hkldata, labs_fc=labs_fc, lab_f=lab_f,
                         labs_half=labs_half_masked if mask is not None else labs_half,
                         labs_half_nomask=labs_half, mask=mask)
    with open(args.fsc_out, "w") as ofs:
        if args.mask:
            ofs.write("# Mask= {}\n".format(args.mask))
        for lab, xyzin in zip(labs_fc, args.model):
            ofs.write("# {} from {}\n".format(lab, xyzin))

        ofs.write(stats.to_string(index=False, index_names=False)+"\n")
        for k in stats:
            if k.startswith("fsc_FC_"):
                logger.writeln("# FSCaverage of {} = {:.4f}".format(k, fsc_average(stats.ncoeffs, stats[k])), fs=ofs)
            if k.startswith("Rcmplx_FC_"):
                logger.writeln("# Average of {} = {:.4f}".format(k, fsc_average(stats.ncoeffs, stats[k])), fs=ofs)

    logger.writeln("Data file: {}".format(args.fsc_out))

    if args.csv:
        csv_out = os.path.splitext(args.fsc_out)[0] + ".csv"
        stats.to_csv(csv_out)
        logger.writeln("CSV file: {}".format(csv_out))

    log_out = os.path.splitext(args.fsc_out)[0] + ".log"
    write_loggraph(stats, labs_fc, log_out)
    logger.writeln("Run loggraph {} to see plots.".format(log_out))
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
