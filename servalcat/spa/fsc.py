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
from servalcat.spa.run_refmac import determine_b_before_mask
from servalcat import utils

def add_arguments(parser):
    parser.description = 'FSC calculation'

    parser.add_argument('--model',
                        help="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--map',
                        help='Input map file(s)')
    group.add_argument("--halfmaps",  nargs=2)
    group.add_argument('--mtz',
                        help='Input mtz file.')
    parser.add_argument('--labin', nargs=2,
                        help='label (F and PHI) for mtz')
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--mask', help='Mask file')
    parser.add_argument('--mask_radius',
                        type=float, default=3,
                        help='calculate mask from model if provided')
    parser.add_argument('--mask_soft_edge',
                        type=float, default=0,
                        help='Add soft edge to model mask.')
    parser.add_argument('--mask_model', action='store_true',
                        help='Apply mask to model density')
    parser.add_argument("--b_before_mask", type=float,
                        help="when model-based mask is used: sharpening B value for sharpen-mask-unsharpen procedure. By default it is determined automatically.")
    parser.add_argument('--no_sharpen_before_mask', action='store_true',
                        help='when model-based mask is used: by default half maps are sharpened before masking by std of signal and unsharpened after masking. This option disables it.')
    utils.symmetry.add_symmetry_args(parser) # add --pg etc
    parser.add_argument('-d', '--resolution',
                        type=float,
                        help='Default: Nyquist')
    parser.add_argument('--random_seed', type=float, default=1234,
                        help="random seed for phase randomized FSC")
    parser.add_argument("-s", "--source", choices=["electron", "xray", "neutron", "custom"], default="electron")
    parser.add_argument('-o', '--fsc_out',
                        default="fsc.dat",
                        help='')
    parser.add_argument('--csv', action='store_true',
                        help="Write csv file")
    parser.add_argument('--keep_charges',  action='store_true',
                        help="Use scattering factor for charged atoms. Use it with care.")

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
    half_labs1 = [l for l in ("fsc_half_unmasked", "fsc_half_masked", "fsc_half_masked_rand", "fsc_half_masked_corrected") if l in stats]
    half_labs2 = [l for l in ("cc_half", "mcos_half") if l in stats]
    if not half_labs1 and "fsc_half" in stats:
        half_labs1 = ["fsc_half"]

    stats2 = stats.copy()
    stats2.insert(0, "bin", stats.index)
    for l in power_labs: stats2[l] = numpy.log(stats2[l])
    title_labs = []
    if half_labs1:
        title_labs.append(("Phase randomized FSC" if len(half_labs1) > 1 else "Half map FSC",
                           half_labs1))
    if half_labs2:
        title_labs.append(("Half map amplitude CC and Mean(cos(dphi))",
                           half_labs2))
    if model_labs1:
        title_labs.append(("Map-model FSC", model_labs1))
    if model_labs2:
        title_labs.append(("Map-model amplitude CC and Mean(cos(dphi))", model_labs2))
    if power_labs:
        title_labs.append(("log(Power)", power_labs))

    title_labs.append(("number of Fourier coefficients", ["ncoeffs"]))
    with open(log_out, "w") as ofs:
        ofs.write(utils.make_loggraph_str(stats2, main_title="FSC", title_labs=title_labs,
                                          s2=1./stats2["d_min"]**2))
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
    for i_bin, idxes in hkldata.binned("stat"):
        F1, F2 = fs[0][idxes], fs[1][idxes]
        fsc = numpy.real(numpy.corrcoef(F1, F2)[1,0])
        ret.append(fsc)
    return ret
# calc_fsc()

def calc_phase_randomized_fsc(hkldata, mask, labs_half, labs_half_masked, randomize_fsc_at=0.8):
    stats = hkldata.binned_df["stat"][["d_min", "d_max"]].copy()
    stats["fsc_half_unmasked"] = calc_fsc(hkldata, labs=labs_half)
    stats["fsc_half_masked"] = calc_fsc(hkldata, labs=labs_half_masked)
    stats["ncoeffs"] = 0

    # Randomize F
    f_rands = [numpy.copy(hkldata.df[labs_half[i]]) for i in range(2)]
    rand_start_bin = None
    for i_bin, idxes in hkldata.binned("stat"):
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
        stats = hkldata.binned_df["stat"][["d_min", "d_max"]].copy()
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

    for i_bin, idxes in hkldata.binned("stat"):
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
    if args.b_before_mask is not None and args.model is None:
        raise SystemExit("--b_before_mask can be only used with --model.")
    
    numpy.random.seed(args.random_seed)
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
        
    if args.model:
        st = utils.fileio.read_structure(args.model)
        st.cell = unit_cell
        st.spacegroup_hm = "P1"
        ccu = utils.model.CustomCoefUtil()
        if not args.keep_charges:
            utils.model.remove_charge([st])
        if args.source == "custom":
            ccu.read_from_cif(st, args.model)
            ccu.show_info()
            ccu.set_coeffs(st)
        utils.symmetry.update_ncs_from_args(args, st, map_and_start=maps[0])
        st_expanded = st.clone()
        if len(st.ncs) > 0:
            utils.model.expand_ncs(st_expanded)
        if mask is None and args.mask_radius > 0:
            # XXX if helical..
            if args.twist is not None:
                logger.writeln("Generating all helical copies in the box")
                st_for_mask = st.clone()
                utils.symmetry.update_ncs_from_args(args, st_for_mask, map_and_start=maps[0], filter_contacting=True)
                utils.model.expand_ncs(st_for_mask)
            else:
                st_for_mask = st_expanded
            mask = utils.maps.mask_from_model(st_for_mask, args.mask_radius, soft_edge=args.mask_soft_edge, grid=maps[0][0])
            #utils.maps.write_ccp4_map("mask_from_model.ccp4", mask)
            if not args.no_sharpen_before_mask and args.b_before_mask is None:
                args.b_before_mask = determine_b_before_mask(st_for_mask, maps, maps[0][1], mask, args.resolution)
    else:
        st_expanded = None

    hkldata = None
    for j in range(2):
        if j == 1:
            if mask is None: break
            if args.b_before_mask is None:
                # modifies original data
                for ma in maps: ma[0].array[:] *= mask
            else:
                maps = utils.maps.sharpen_mask_unsharpen(maps, mask, args.resolution, b=args.b_before_mask)
        lab_suffix = ["_nomask", "_mask"][j]
        for i, m in enumerate(maps):
            if len(maps) == 2:
                lab = "F_map{}".format(i+1)
            else:
                lab = "FP"
            f_grid = gemmi.transform_map_to_f_phi(m[0])
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
    if st_expanded is not None:
        labs_fc.append("FC")
        hkldata.df[labs_fc[-1]] = utils.model.calc_fc_fft(st_expanded, args.resolution - 1e-6, source=args.source,
                                                          miller_array=hkldata.miller_array())
        if args.mask_model and mask is not None:
            if args.b_before_mask is None:
                normalizer = 1.
            else:
                normalizer = hkldata.debye_waller_factors(b_iso=args.b_before_mask)
            g = hkldata.fft_map(data=hkldata.df[labs_fc[-1]] / normalizer, grid_size=mask.shape)
            g.array[:] *= mask
            fg = gemmi.transform_map_to_f_phi(g)
            hkldata.df[labs_fc[-1]] = fg.get_value_by_hkl(hkldata.miller_array()) * normalizer

    hkldata.setup_relion_binning("stat")
    stats = calc_fsc_all(hkldata, labs_fc=labs_fc, lab_f=lab_f,
                         labs_half=labs_half_masked if mask is not None else labs_half,
                         labs_half_nomask=labs_half, mask=mask)
    with open(args.fsc_out, "w") as ofs:
        if args.mask:
            ofs.write("# Mask= {}\n".format(args.mask))
        if args.model is not None:
            ofs.write("# {} from {}\n".format(labs_fc[0], args.model))

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
