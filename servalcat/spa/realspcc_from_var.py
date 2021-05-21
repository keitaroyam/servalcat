"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import scipy.integrate
import os
import argparse
from servalcat.utils import logger
from servalcat import utils
from servalcat import spa

def add_arguments(parser):
    parser.description = 'Calculate real space correlation radius from variances in reciprocal space'
    parser.add_argument("--halfmaps", nargs=2)
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument("-d", '--resolution', type=float, required=True)
    parser.add_argument('-m', '--mask', help="mask file")
    parser.add_argument('-w', '--weight', help="weight")
    parser.add_argument('-f', help="noise, signal, total")
    parser.add_argument('--sharpen_signal', action="store_true", help="")
    parser.add_argument('--x_max', type=float, default=20)
    #parser.add_argument("-B", type=float, help="Estimated blurring.")
    parser.add_argument('-o','--output_prefix', default="cc",
                        help='output file name prefix')
# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

#def f_noise(x, s_list, w_list):
#    return lambda s: s**2 * numpy.interp(s, s_list, w_list) * numpy.sinc(2*numpy.abs(s)*numpy.abs(x))

def f_noise(s, x, s_list, w_list):
    return s**2 * numpy.interp(s, s_list, w_list) * numpy.sinc(2*numpy.abs(s)*numpy.abs(x))

def calc_cc_from_var(hkldata, smax, x_max=20, x_step=0.1, kind="noise", sharpen_signal=False, weight=None):
    assert 1./hkldata.bin_and_limits()[-1][2] <= smax
    bin_s = numpy.array([(1/dmax+1/dmin)/2 for _,dmax,dmin in hkldata.bin_and_limits()])
    bin_start = hkldata.bin_and_limits()[0][0] # grr

    logger.write("kind= {} sharpen_signal={}, weight={}".format(kind, sharpen_signal, weight))

    wsq = None
    if kind == "noise":
        wsq = hkldata.binned_df.var_noise[bin_start:].to_numpy(copy=True)
    elif kind == "signal":
        wsq = hkldata.binned_df.var_signal[bin_start:].to_numpy(copy=True)
    elif kind == "total":
        wsq = hkldata.binned_df.var_signal[bin_start:].to_numpy() + hkldata.binned_df.var_noise[bin_start:].to_numpy()
    else:
        raise RuntimeError("unknown kind")

    if sharpen_signal:
        wsq /= hkldata.binned_df.var_signal[bin_start:].to_numpy()
    
    if weight is None:
        wsq *= 1.
    elif weight == "fscfull":
        wsq *= hkldata.binned_df.FSCfull[bin_start:].to_numpy()**2
    else:
        raise RuntimeError("unknown weight")
        
    cov_xx = scipy.integrate.quad(f_noise, 0, smax, args=(0, bin_s, wsq))[0]
    x_all = numpy.arange(0, x_max, x_step)
    cc_all = []
    for x in x_all:
        cov_xy = scipy.integrate.quad(f_noise, 0, smax, args=(x, bin_s, wsq))[0]
        cc_all.append(cov_xy/cov_xx)

    cc_all = numpy.array(cc_all)
    return x_all, cc_all
# calc_cc_from_var()

def main(args):
    maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
    if args.mask:
        mask = numpy.array(utils.fileio.read_ccp4_map(args.mask)[0])
    else:
        mask = None

    hkldata = utils.maps.mask_and_fft_maps(maps, args.resolution, mask)
    hkldata.setup_relion_binning()
    utils.maps.calc_noise_var_from_halfmaps(hkldata)

    smax = 1. / args.resolution
    x_all, cc_all = calc_cc_from_var(hkldata, smax, x_max=args.x_max, kind=args.f,
                                     sharpen_signal=args.sharpen_signal, weight=args.weight)

    ofs = open("{}.dat".format(args.output_prefix), "w")
    ofs.write("# smax= {}\n".format(smax))
    ofs.write("# halfmaps= {}\n".format(*args.halfmaps))
    ofs.write("# mask= {}\n".format(args.mask))
    ofs.write("# weight= {}\n".format(args.weight))
    ofs.write("# f= {}\n".format(args.f))
    ofs.write("x cc dmin weight f sharpen\n")
    for x, cc in zip(x_all, cc_all):
        ofs.write("{:.2f} {:.4f} {:.2f} {} {} {}\n".format(x, cc, args.resolution, args.weight, args.f,
                                                           "TRUE" if args.sharpen_signal else "FALSE"))
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)

