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
from servalcat import spa

def add_arguments(parser):
    parser.description = 'Find translation of the model in the map'
    parser.add_argument('--model',
                        required=True,
                        help="")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--halfmaps", nargs=2, help="Input half map files")
    group.add_argument("--map", help="Use this only if you really do not have half maps.")
    parser.add_argument('--mask',
                        help='Mask file')
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('-d', '--resolution',
                        type=float,
                        required=True,
                        help='')
    parser.add_argument('--no_interpolation', action="store_true",
                        help="No interpolation in peak finding of translation function")
    parser.add_argument('-o', '--output_prefix', default="translated")

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_fsc(hkldata, lab1, lab2):
    stats = hkldata.binned_df["ml"][["d_min", "d_max"]].copy()
    stats["ncoeffs"] = 0
    stats["fsc"] = 0.
    for i_bin, idxes in hkldata.binned("ml"):
        stats.loc[i_bin, "ncoeffs"] = len(idxes)
        stats.loc[i_bin, "fsc"] = numpy.real(numpy.corrcoef(hkldata.df[lab1].to_numpy()[idxes],
                                                            hkldata.df[lab2].to_numpy()[idxes])[1,0])

    sum_n = sum(stats.ncoeffs)
    fscavg = sum(stats.ncoeffs*stats.fsc)/sum_n
    return stats, fscavg
# calc_fsc()

def find_peak(tf_map, ini_pos):
    logger.writeln("Finding peak using interpolation..")

    x = tf_map.unit_cell.fractionalize(ini_pos)
    logger.writeln("       x0: [{}, {}, {}]".format(*x.tolist()))
    logger.writeln("       f0: {}".format(-tf_map.interpolate_value(x, order=3)))
    
    res = scipy.optimize.minimize(fun=lambda x:-tf_map.interpolate_value(gemmi.Fractional(*x), order=3),
                                  x0=x.tolist(),
                                  jac=lambda x:-numpy.array(tf_map.tricubic_interpolation_der(gemmi.Fractional(*x))[1:]))
    logger.writeln(str(res))
    final_pos = tf_map.unit_cell.orthogonalize(gemmi.Fractional(*res.x))
    logger.writeln(" Move from initial: [{:.3f}, {:.3f}, {:.3f}] A".format(*(final_pos-ini_pos).tolist()))
    return final_pos
# find_peak()

def main(args):
    if args.halfmaps:
        maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
        assert maps[0][0].shape == maps[1][0].shape
        assert maps[0][0].unit_cell == maps[1][0].unit_cell
        assert maps[0][1] == maps[1][1]
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]

    model_format = utils.fileio.check_model_format(args.model)
    st = utils.fileio.read_structure(args.model)
    st.cell = maps[0][0].unit_cell
    st.spacegroup_hm = "P1"

    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None

    hkldata = utils.maps.mask_and_fft_maps(maps, args.resolution, mask=None)
    hkldata.df["FC"] = utils.model.calc_fc_fft(st, args.resolution - 1e-6, source="electron",
                                               miller_array=hkldata.miller_array())
    hkldata.setup_relion_binning("ml")

    stats, fscavg = calc_fsc(hkldata, "FP", "FC")
    logger.writeln(stats.to_string())
    logger.writeln("FSCaverage before translation = {:.4f}".format(fscavg))

    hkldata.df["TF"] = hkldata.df.FP.to_numpy() * numpy.conj(hkldata.df.FC.to_numpy())

    tf_map = hkldata.fft_map("TF")
    max_idx = numpy.unravel_index(numpy.argmax(tf_map), tf_map.shape)
    shift = tf_map.get_position(*max_idx)

    if not args.no_interpolation:
        shift = utils.maps.optimize_peak(tf_map, shift)

    logger.writeln("shift= {:.4f}, {:.4f}, {:.4f} ".format(*shift))

    # phase shift for translation
    hkldata.df.FC *= numpy.exp(2.j*numpy.pi*numpy.dot(hkldata.miller_array(),
                                                      hkldata.cell.fractionalize(shift).tolist()))
    stats, fscavg = calc_fsc(hkldata, "FP", "FC")
    logger.writeln(stats.to_string())
    logger.writeln("FSCaverage after translation = {:.4f}".format(fscavg))

    tr = gemmi.Transform(gemmi.Mat33(), shift)
    st[0].transform_pos_and_adp(tr)
    utils.model.translate_into_box(st)
    utils.fileio.write_model(st, file_name=args.output_prefix+model_format)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
