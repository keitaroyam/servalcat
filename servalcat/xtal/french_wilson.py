"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import gemmi
import numpy
import pandas
import time
import os
import scipy.special
import scipy.optimize
from servalcat.utils import logger
from servalcat.xtal.sigmaa import process_input
from servalcat import utils
from servalcat import ext

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)

def add_arguments(parser):
    parser.description = 'Convert intensity to amplitude'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin', 
                        help='MTZ column for I,SIGI')
    parser.add_argument('--d_min', type=float)
    parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int,
                        help="Number of bins (default: auto)")
    parser.add_argument('-o','--output_prefix',
                        help='output file name prefix')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def determine_Sigma_and_aniso(hkldata):
    # initial estimate
    hkldata.binned_df["S"] = 1.
    I_over_eps = hkldata.df.I.to_numpy() / hkldata.df.epsilon.to_numpy()
    x = []
    for i_bin, idxes in hkldata.binned():
        #S = max(numpy.nanmean(I_over_eps[idxes]), 1e-3)
        # var(I) = var_signal + var_noise, so this overestimates S,
        # but it should be better than having negative values (in noisy shell)
        S = numpy.nanstd(I_over_eps[idxes])
        hkldata.binned_df.loc[i_bin, "S"] = S
        x.append(S)
    logger.writeln("Initial estimates:")
    logger.writeln(hkldata.binned_df.to_string())

    B = gemmi.SMat33d(0,0,0,0,0,0)
    SMattolist = lambda B: [B.u11, B.u22, B.u33, B.u12, B.u13, B.u23]
    adpdirs = utils.model.adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=True)
    logger.writeln("ADP free parameters = {}".format(adpdirs.shape[0]))
    ssqmat = hkldata.ssq_mat()
    cycle_data = [[0] + SMattolist(B) + list(hkldata.binned_df.S)]
    for icyc in range(100):
        #logger.writeln("Refine B")
        B_converged = False
        t0 = time.time()
        args=(ssqmat, hkldata, adpdirs)
        for j in range(10):
            x = numpy.dot(SMattolist(B), numpy.linalg.pinv(adpdirs))
            f0 = ll_all_B(x, *args)
            shift = ll_shift_B(x, *args)
            for i in range(3):
                ss = shift / 2**i
                f1 = ll_all_B(x + ss, *args)
                #logger.writeln("f0 = {:.3e} shift = {} df = {:.3e}".format(f0, ss, f1 - f0))
                if f1 < f0:
                    B = gemmi.SMat33d(*numpy.dot(x+ss, adpdirs))
                    if numpy.max(numpy.abs(ss)) < 1e-4: B_converged = True
                    break
            else:
                B_converged = True
            if B_converged: break

        #logger.writeln("time= {}".format(time.time() - t0))
        #logger.writeln("B_aniso= {}".format(B))
        #logger.writeln("Refine S")
        S_converged = [False for _ in hkldata.binned()]
        k_ani = hkldata.debye_waller_factors(b_cart=B)
        for i, (i_bin, idxes) in enumerate(hkldata.binned()):
            #logger.writeln("Bin {}".format(i_bin))
            for j in range(10):
                S = hkldata.binned_df.loc[i_bin, "S"]
                f0 = numpy.nansum(ext.ll_int(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes],
                                             S * hkldata.df.epsilon.to_numpy()[idxes],
                                             0, hkldata.df.centric.to_numpy()[idxes]+1))
                shift = numpy.exp(ll_shift_bin_S(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes],
                                                 S, hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes]))
                for k in range(3):
                    ss = shift**(1. / 2**k)
                    f1 = numpy.nansum(ext.ll_int(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes],
                                                 S * ss * hkldata.df.epsilon.to_numpy()[idxes],
                                                 0, hkldata.df.centric.to_numpy()[idxes]+1))
                    #logger.writeln("bin {:3d} f0 = {:.3e} shift = {:.3e} df = {:.3e}".format(i_bin, f0, ss, f1 - f0))
                    if f1 < f0:
                        hkldata.binned_df.loc[i_bin, "S"] = S * ss
                        if ss > 0.9999: S_converged[i] = True
                        break
                else:
                    S_converged[i] = True
                if S_converged[i]: break

        #logger.writeln("Refined estimates in cycle {}:".format(icyc))
        #logger.writeln(hkldata.binned_df.to_string())
        #logger.writeln("B_aniso= {}".format(B))
        cycle_data.append([icyc] + SMattolist(B) + list(hkldata.binned_df.S))
        if B_converged and all(S_converged):
            logger.writeln("Converged in cycle {}".format(icyc))
            logger.writeln("Refined estimates:")
            logger.writeln(hkldata.binned_df.to_string())
            logger.writeln("B_aniso= {}".format(B))
            break

    with open("fw_cycles.dat", "w") as ofs:
        ofs.write("cycle B11 B22 B33 B12 B13 B23 " + " ".join("S{}".format(i) for i in hkldata.binned_df.index) + "\n")
        for data in cycle_data:
            ofs.write("{:2d} ".format(data[0]+1))
            ofs.write(" ".join("{:.4e}".format(x) for x in data[1:]))
            ofs.write("\n")
        
    return B

def ll_all_B(x, ssqmat, hkldata, adpdirs):
    B = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k_ani = hkldata.debye_waller_factors(b_cart=B)
    ret = 0.
    for i_bin, idxes in hkldata.binned():
        ret += numpy.nansum(ext.ll_int(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes],
                                       hkldata.binned_df.S[i_bin] * hkldata.df.epsilon.to_numpy()[idxes],
                                       0, hkldata.df.centric.to_numpy()[idxes]+1))
    return ret

def ll_shift_bin_S(Io, sigIo, k_ani, S, c, eps, exp_trans=True):
    tmp = ext.ll_int_fw_der1_S(Io, sigIo, k_ani, S, c, eps)
    g = numpy.nansum(tmp)
    H = numpy.nansum(tmp**2)
    if exp_trans:
        return -g / (H * S + g)
    else:
        return -g / H

def ll_shift_B(x, ssqmat, hkldata, adpdirs):
    b_aniso = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    Io = hkldata.df.I.to_numpy()
    sigIo = hkldata.df.SIGI.to_numpy()
    c = hkldata.df.centric.to_numpy() + 1
    epsilon = hkldata.df.epsilon.to_numpy()
    r = numpy.empty(len(Io)) * numpy.nan
    for i_bin, idxes in hkldata.binned():
        r[idxes] = ext.ll_int_fw_der1_ani(Io[idxes], sigIo[idxes],
                                          k_ani[idxes], hkldata.binned_df.S[i_bin],
                                          c[idxes], epsilon[idxes])
    g = -numpy.nansum(ssqmat * r, axis=1)
    H = numpy.nansum(numpy.matmul(ssqmat[None,:].T, ssqmat.T[:,None]) * (r**2)[:,None,None], axis=0)
    g, H = numpy.dot(g, adpdirs.T), numpy.dot(adpdirs, numpy.dot(H, adpdirs.T))
    return -numpy.dot(g, numpy.linalg.pinv(H))

def french_wilson(hkldata, B_aniso, labout=None):
    if labout is None: labout = ["F", "SIGF"]
    hkldata.df[labout[0]] = numpy.nan
    hkldata.df[labout[1]] = numpy.nan
    hkldata.df["to1"] = numpy.nan
    k_ani = hkldata.debye_waller_factors(b_cart=B_aniso)
    
    for i_bin, idxes in hkldata.binned():
        S = hkldata.binned_df.S[i_bin]
        c = hkldata.df.centric.to_numpy()[idxes] + 1 # 1 for acentric, 2 for centric
        Io = hkldata.df.I.to_numpy()[idxes]
        sigo = hkldata.df.SIGI.to_numpy()[idxes]
        eps = hkldata.df.epsilon.to_numpy()[idxes]
        to = Io / sigo - sigo / c / k_ani[idxes]**2 / S / eps
        k_num = numpy.where(c == 1,  0.5, 0.)
        F = numpy.sqrt(sigo) * ext.integ_J_ratio(k_num, k_num - 0.5, False, to, 0., 1., c)
        Fsq = sigo * ext.integ_J_ratio(k_num + 0.5, k_num - 0.5, False, to, 0., 1., c)
        varF = Fsq - F**2
        hkldata.df.loc[idxes, labout[0]] = F
        hkldata.df.loc[idxes, labout[1]] = numpy.sqrt(varF)
        hkldata.df.loc[idxes, "to1"] = to

def main(args):
    if not args.output_prefix:
        args.output_prefix = utils.fileio.splitext(os.path.basename(args.hklin))[0] + "_fw"
    if not args.labin:
        mtz = utils.fileio.read_mmhkl(args.hklin)
        dlabs = utils.hkl.mtz_find_data_columns(mtz)
        if dlabs["J"]:
            labin = dlabs["J"][0]
        else:
            raise SystemExit("Intensity not found from mtz")
        flabs = utils.hkl.mtz_find_free_columns(mtz)
        if flabs:
            labin += [flabs[0]]
        logger.writeln("MTZ columns automatically selected: {}".format(labin))
    else:
        labin = args.labin.split(",")
        
    hkldata, _, _, _ = process_input(hklin=args.hklin,
                                     labin=labin,
                                     n_bins=args.nbins,
                                     free=None,
                                     xyzins=[],
                                     source=None,
                                     d_min=args.d_min,
                                     n_per_bin=500,
                                     max_bins=30)
    
    B_aniso = determine_Sigma_and_aniso(hkldata)
    french_wilson(hkldata, B_aniso)
    mtz_out = args.output_prefix+".mtz"
    lab_out = ["F", "SIGF", "I", "SIGI"]
    labo_types = {"F":"F", "SIGF":"Q", "I":"J", "SIGI":"Q"}
    if len(labin) == 3:
        lab_out.append("FREE")
        labo_types[lab_out[-1]] = "I"
    hkldata.write_mtz(mtz_out, lab_out, types=labo_types)
    return B_aniso, hkldata
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
