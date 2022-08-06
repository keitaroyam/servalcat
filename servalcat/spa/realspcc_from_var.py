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

def f_noise(s, x, s_list, w_list, B):
    tinv = numpy.exp(-B*s**2/2) if B is not None else 1.
    return s**2 * tinv * numpy.interp(s, s_list, w_list) * numpy.sinc(2*numpy.abs(s)*numpy.abs(x))

def calc_var(hkldata, kind="noise", sharpen_signal=False, weight=None):
    wsq = None
    if kind == "noise":
        wsq = hkldata.binned_df.var_noise.to_numpy(copy=True)
    elif kind == "signal":
        wsq = hkldata.binned_df.var_signal.to_numpy(copy=True)
    elif kind == "total":
        wsq = hkldata.binned_df.var_signal.to_numpy() + hkldata.binned_df.var_noise.to_numpy()
    else:
        raise RuntimeError("unknown kind")

    if sharpen_signal:
        wsq /= hkldata.binned_df.var_signal.to_numpy()
    
    if weight is None:
        wsq *= 1.
    elif weight == "fscfull":
        wsq *= hkldata.binned_df.FSCfull.to_numpy()**2
    else:
        raise RuntimeError("unknown weight")
        
    return wsq
# calc_var()

def find_b(hkldata, smax, x, kind="noise", sharpen_signal=False, weight=None): # XXX unfinished
    bin_s = 0.5*(1./hkldata.binned_df[["d_min", "d_max"]]).sum(axis=1).to_numpy() # 0.5 * (1/d_max + 1/d_min)
    logger.write("kind= {} sharpen_signal={}, weight={}".format(kind, sharpen_signal, weight))
    wsq = calc_var(hkldata, kind, sharpen_signal, weight)
    for B in -numpy.arange(0,100,5):
        cov_xx = scipy.integrate.quad(f_noise, 0, smax, args=(0, bin_s, wsq, B))[0]
        cov_xy = scipy.integrate.quad(f_noise, 0, smax, args=(x, bin_s, wsq, B))[0]
        print(B, cov_xy/cov_xx)
# find_b()        

def calc_cc_from_var(hkldata, smax, x_max=20, x_step=0.1, kind="noise", sharpen_signal=False, weight=None, B=None):
    assert 1./hkldata.binned_df.d_min.iloc[-1] <= smax
    bin_s = 0.5*(1./hkldata.binned_df[["d_min", "d_max"]]).sum(axis=1).to_numpy() # 0.5 * (1/d_max + 1/d_min)

    logger.write("kind= {} sharpen_signal={}, weight={}".format(kind, sharpen_signal, weight))
    wsq = calc_var(hkldata, kind, sharpen_signal, weight)
    cov_xx = scipy.integrate.quad(f_noise, 0, smax, args=(0, bin_s, wsq, B))[0]
    x_all = numpy.arange(0, x_max, x_step)
    cc_all = []
    for x in x_all:
        cov_xy = scipy.integrate.quad(f_noise, 0, smax, args=(x, bin_s, wsq, B))[0]
        cc_all.append(cov_xy/cov_xx)

    cc_all = numpy.array(cc_all)
    return x_all, cc_all
# calc_cc_from_var()

def main(args):
    maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None

    hkldata = utils.maps.mask_and_fft_maps(maps, args.resolution, mask)
    hkldata.setup_relion_binning()
    utils.maps.calc_noise_var_from_halfmaps(hkldata)

    smax = 1. / args.resolution
    if args.find_B_at is not None:
        find_b(hkldata, smax, args.find_B_at,
               kind=args.f, sharpen_signal=args.sharpen_signal, weight=args.weight)
    else:
        x_all, cc_all = calc_cc_from_var(hkldata, smax, x_max=args.x_max, kind=args.f,
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

