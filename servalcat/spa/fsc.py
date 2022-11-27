"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import pandas
from servalcat.utils import logger
from servalcat import spa
from servalcat import utils

def add_arguments(parser):
    parser.description = 'FSC calculation'

    parser.add_argument('--model', nargs="+", action="append",
                        required=True, 
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
    parser.add_argument('-d', '--resolution',
                        type=float,
                        help='Default: Nyquist')
    parser.add_argument('-o', '--fsc_out',
                        default="fsc.dat", # TODO csv
                        help='')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def fsc_average(n, fsc):
    sel = fsc == fsc # filter nan
    n, fsc = n[sel], fsc[sel]
    return sum(n*fsc)/sum(n)
# fsc_average()

def calc_fsc(hkldata, labs_fc, lab_f, labs_half=None):
    if labs_half is not None: assert len(labs_half) == 2
    
    stats = hkldata.binned_df[["d_min", "d_max"]].copy()
    stats["ncoeffs"] = 0
    stats["power_{}".format(lab_f)] = 0.
    for lab in labs_fc: stats["power_{}".format(lab)] = 0.
    for lab in labs_fc: stats["fsc_{}_full".format(lab)] = 0.
    for lab in labs_fc: stats["Rcmplx_{}_full".format(lab)] = 0.
    if labs_half is not None:
        stats["fsc_half"] = 0.
        for lab in labs_fc:
            stats["fsc_{}_half1".format(lab)] = 0.
            stats["fsc_{}_half2".format(lab)] = 0.

    for i_bin, idxes in hkldata.binned():
        stats.loc[i_bin, "ncoeffs"] = len(idxes)
        Fo = hkldata.df[lab_f].to_numpy()[idxes]
        stats.loc[i_bin, "power_{}".format(lab_f)] = numpy.average(numpy.abs(Fo)**2)
        if labs_half is not None:
            F1, F2 = hkldata.df[labs_half[0]].to_numpy()[idxes], hkldata.df[labs_half[1]].to_numpy()[idxes]
            stats.loc[i_bin, "fsc_half"] = numpy.real(numpy.corrcoef(F1, F2)[1,0])
        else:
            F1, F2 = None, None

        for labfc in labs_fc:
            Fc = hkldata.df[labfc].to_numpy()[idxes]
            fsc_model = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
            D = numpy.sum(numpy.real(Fo * numpy.conj(Fc)))/numpy.sum(numpy.abs(Fc)**2)
            rcmplx_model = numpy.sum(numpy.abs(Fo-D*Fc))/numpy.sum(numpy.abs(Fo))
            stats.loc[i_bin, "fsc_{}_full".format(labfc)] = fsc_model
            stats.loc[i_bin, "Rcmplx_{}_full".format(labfc)] = rcmplx_model
            stats.loc[i_bin, "power_{}".format(labfc)] = numpy.average(numpy.abs(Fc)**2)
            if labs_half is not None:
                stats.loc[i_bin, "fsc_{}_half1".format(labfc)] = numpy.real(numpy.corrcoef(F1, Fc)[1,0])
                stats.loc[i_bin, "fsc_{}_half2".format(labfc)] = numpy.real(numpy.corrcoef(F2, Fc)[1,0])
    return stats
# calc_fsc()

def main(args):
    args.model = sum(args.model, [])
    
    if args.mask:
        logger.write("Input mask file: {}".format(args.mask))
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
        mtz = gemmi.read_mtz_file(args.mtz)
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
        logger.write("WARNING: --resolution is not specified. Using Nyquist resolution: {:.2f}".format(args.resolution))
        
    sts = []
    for xyzin in args.model:
        st = utils.fileio.read_structure(xyzin)
        st.cell = unit_cell
        st.spacegroup_hm = "P1"
        if len(st.ncs) > 0:
            utils.model.expand_ncs(st)
            
        sts.append(st)
    
    if mask is not None:
        logger.write("Applying mask..")
        for ma in maps: ma[0].array[:] *= mask

    hkldata = utils.maps.mask_and_fft_maps(maps, args.resolution)

    labs_fc = []
    for i, st in enumerate(sts): 
        labs_fc.append("FC_{}".format(i) if len(sts)>1 else "FC")
        hkldata.df[labs_fc[-1]] = utils.model.calc_fc_fft(st, args.resolution - 1e-6, source="electron",
                                                          miller_array=hkldata.miller_array())

    hkldata.setup_relion_binning()
    stats = calc_fsc(hkldata, labs_fc=labs_fc, lab_f="FP", labs_half=["F_map1","F_map2"] if len(maps)==2 else None)
    with open(args.fsc_out, "w") as ofs:
        if args.mask:
            ofs.write("# Mask= {}\n".format(args.mask))
        for lab, xyzin in zip(labs_fc, args.model):
            ofs.write("# {} from {}\n".format(lab, xyzin))

        ofs.write(stats.to_string(index=False, index_names=False)+"\n")
        for k in stats:
            if k.startswith("fsc_FC_"):
                logger.write("# FSCaverage of {} = {:.4f}".format(k, fsc_average(stats.ncoeffs, stats[k])), fs=ofs)
            if k.startswith("Rcmplx_FC_"):
                logger.write("# Average of {} = {:.4f}".format(k, fsc_average(stats.ncoeffs, stats[k])), fs=ofs)

    logger.write("See {}".format(args.fsc_out))
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
