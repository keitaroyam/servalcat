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
    parser.add_argument('-f', choices=["noise", "signal", "total"], required=True)
    parser.add_argument('--sharpen_signal', action="store_true", help="")
    parser.add_argument('--x_max', type=float, default=20)
    parser.add_argument("-B", type=float, help="Sharpening (negative)/blurring (positive) B value")
    parser.add_argument('--find_B_at', type=float, help="")
    parser.add_argument('-o','--output_prefix', default="cc",
                        help='output file name prefix')
# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def integ_var(x, s_list, w_list, B):
    tinv = numpy.exp(-B*s_list**2/2) if B is not None else 1.
    return scipy.integrate.simps(s_list**2 * tinv * w_list * numpy.sinc(2*numpy.abs(s_list)*numpy.abs(x)),
                                 s_list)

def calc_var(hkldata, kind="noise", sharpen_signal=False, weight=None):
    wsq = None
    if kind == "noise":
        wsq = hkldata.binned_df["ml"].var_noise.to_numpy(copy=True)
    elif kind == "signal":
        wsq = hkldata.binned_df["ml"].var_signal.to_numpy(copy=True)
    elif kind == "total":
        wsq = hkldata.binned_df["ml"].var_signal.to_numpy() + hkldata.binned_df["ml"].var_noise.to_numpy()
    else:
        raise RuntimeError("unknown kind")

    if sharpen_signal:
        wsq /= hkldata.binned_df["ml"].var_signal.to_numpy()
    
    if weight is None:
        wsq *= 1.
    elif weight == "fscfull":
        wsq *= hkldata.binned_df["ml"].FSCfull.to_numpy()**2
    else:
        raise RuntimeError("unknown weight")
        
    return wsq
# calc_var()

def find_b(hkldata, smax, x, kind="noise", sharpen_signal=False, weight=None): # XXX unfinished
    bin_s = 0.5*(1./hkldata.binned_df["ml"][["d_min", "d_max"]]).sum(axis=1).to_numpy() # 0.5 * (1/d_max + 1/d_min)
    logger.writeln("kind= {} sharpen_signal={}, weight={}".format(kind, sharpen_signal, weight))
    wsq = calc_var(hkldata, kind, sharpen_signal, weight)
    for B in -numpy.arange(0,100,5):
        cov_xx = integ_var(0., bin_s, wsq, B)
        cov_xy = integ_var(x, bin_s, wsq, B)
        print(B, cov_xy/cov_xx)
# find_b()        

def calc_cc_from_var(hkldata, x_list, kind="noise", sharpen_signal=False, weight=None, B=None):
    bin_s = 0.5*(1./hkldata.binned_df["ml"][["d_min", "d_max"]]).sum(axis=1).to_numpy() # 0.5 * (1/d_max + 1/d_min)
    logger.writeln("kind= {} sharpen_signal={}, weight={}".format(kind, sharpen_signal, weight))
    wsq = calc_var(hkldata, kind, sharpen_signal, weight)
    cov_xx = integ_var(0., bin_s, wsq, B)
    cov_xy = integ_var(x_list[:,None], bin_s, wsq, B)
    cc_all = cov_xy / cov_xx
    return cc_all
# calc_cc_from_var()

def main(args):
    maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None

    hkldata = utils.maps.mask_and_fft_maps(maps, args.resolution, mask)
    hkldata.setup_relion_binning("ml")
    utils.maps.calc_noise_var_from_halfmaps(hkldata)

    smax = 1. / args.resolution
    if args.find_B_at is not None:
        find_b(hkldata, smax, args.find_B_at,
               kind=args.f, sharpen_signal=args.sharpen_signal, weight=args.weight)
    else:
        x_all = numpy.arange(0, args.x_max, 0.1)
        cc_all = calc_cc_from_var(hkldata, x_list=x_all, kind=args.f,
                                  sharpen_signal=args.sharpen_signal, weight=args.weight,
                                  B=args.B)

        ofs = open("{}.dat".format(args.output_prefix), "w")
        ofs.write("# smax= {}\n".format(smax))
        ofs.write("# halfmaps= {}\n".format(*args.halfmaps))
        ofs.write("# mask= {}\n".format(args.mask))
        ofs.write("# weight= {}\n".format(args.weight))
        ofs.write("# f= {}\n".format(args.f))
        ofs.write("x cc dmin weight f sharpen b\n")
        for x, cc in zip(x_all, cc_all):
            ofs.write("{:.2f} {:.4f} {:.2f} {} {} {} {}\n".format(x, cc, args.resolution, args.weight, args.f,
                                                                  "TRUE" if args.sharpen_signal else "FALSE",
                                                                  args.B))
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)

