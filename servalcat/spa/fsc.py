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
from servalcat import utils

def add_arguments(parser):
    parser.description = 'FSC calculation'

    parser.add_argument('--model',
                        required=True,
                        help="")
    parser.add_argument('--map',
                        help='Input map file(s)')
    parser.add_argument("--halfmaps",  nargs=2)
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--mask', help='Mask file')
    parser.add_argument('-r', '--mask_radius',
                        type=float,
                        help='')
    parser.add_argument('-d', '--resolution',
                        type=float,
                        required=True,
                        help='')
    parser.add_argument('--fsc_out',
                        default="fsc.dat",
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
    
    stats_columns = ["d_max", "d_min", "ncoeffs", "power_{}".format(lab_f)]
    for lab in labs_fc: stats_columns.append("power_{}".format(lab))
    for lab in labs_fc: stats_columns.append("fsc_{}_full".format(lab))
    if labs_half is not None:
        stats_columns.append("fsc_half")
        for lab in labs_fc: stats_columns.extend(["fsc_{}_half1".format(lab),
                                                  "fsc_{}_half2".format(lab)])

    stats = pandas.DataFrame(index=[x[0] for x in hkldata.bin_and_limits()],
                             columns=stats_columns, dtype=numpy.float)
    stats.ncoeffs = 0 # to int

    bin_limits = dict(hkldata.bin_and_limits())
    for i_bin, g in hkldata.binned():
        bin_d_max, bin_d_min = bin_limits[i_bin]
        stats.loc[i_bin, "d_min"] = bin_d_min
        stats.loc[i_bin, "d_max"] = bin_d_max
        stats.loc[i_bin, "ncoeffs"] = len(g.index)
        Fo = g[lab_f].to_numpy()
        stats.loc[i_bin, "power_{}".format(lab_f)] = numpy.average(numpy.abs(Fo)**2)
        if labs_half is not None:
            F1, F2 = g[labs_half[0]].to_numpy(), g[labs_half[1]].to_numpy()
            stats.loc[i_bin, "fsc_half"] = numpy.real(numpy.corrcoef(F1, F2)[1,0])
        else:
            F1, F2 = None, None

        for labfc in labs_fc:
            Fc = g[labfc].to_numpy()
            fsc_model = numpy.real(numpy.corrcoef(Fo, Fc)[1,0])
            stats.loc[i_bin, "fsc_{}_full".format(labfc)] = fsc_model
            stats.loc[i_bin, "power_{}".format(labfc)] = numpy.average(numpy.abs(Fc)**2)
            if labs_half is not None:
                stats.loc[i_bin, "fsc_{}_half1".format(labfc)] = numpy.real(numpy.corrcoef(F1, Fc)[1,0])
                stats.loc[i_bin, "fsc_{}_half2".format(labfc)] = numpy.real(numpy.corrcoef(F2, Fc)[1,0])
    return stats
# calc_fsc()

def main(args):
    st = utils.fileio.read_structure(args.model)
    if args.halfmaps:
        maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
        assert maps[0][0].shape == maps[1][0].shape
        assert maps[0][0].unit_cell == maps[1][0].unit_cell
        assert maps[0][1] == maps[1][1]
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]

    st.cell = maps[0][0].unit_cell
    st.spacegroup_hm = "P1"

    if len(st.ncs) > 0:
        utils.model.expand_ncs(st)
    
    if args.mask:
        logger.write("Input mask file: {}".format(args.mask))
        mask = numpy.array(utils.fileio.read_ccp4_map(args.mask)[0])
    elif args.mask_radius is not None:
        mask = gemmi.FloatGrid(*ref_grid.shape)
        mask.set_unit_cell(st.cell)
        mask.spacegroup = st.find_spacegroup()
        mask.mask_points_in_constant_radius(st[0], args.mask_radius, 1.)    
    else:
        mask = None
    
    fc = utils.model.calc_fc_fft(st, args.resolution, source="electron")
    hkldata = utils.maps.mask_and_fft_maps(maps, args.resolution, mask)
    hkldata.merge_asu_data(fc, "FC")
    hkldata.setup_relion_binning()
    stats = calc_fsc(hkldata, labs_fc=["FC"], lab_f="FP", labs_half=["F_map1","F_map2"] if len(maps)==2 else None)
    with open(args.fsc_out, "w") as ofs:
        if args.mask:
            ofs.write("# Mask= {}\n".format(args.mask))
        elif args.mask_radius:
            ofs.write("# Mask_radius= {}\n".format(args.mask_radius))

        ofs.write(stats.to_string(index=False, index_names=False)+"\n")
        for k in stats:
            if k.startswith("fsc_FC_"):
                logger.write("# FSCaverage of {} = {:.4f}".format(k, fsc_average(stats.ncoeffs, stats[k])), fs=ofs)

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
