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
import scipy.special
import scipy.optimize
from servalcat.utils import logger
from servalcat import utils
from . import fw_helper as ext

import line_profiler
profile = line_profiler.LineProfiler()
import atexit
atexit.register(profile.print_stats)

def add_arguments(parser):
    parser.description = 'Convert intensity to amplitude'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin', required=True,
                        help='MTZ column for I,SIGI')
    parser.add_argument('--d_min', type=float)
    parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int, default=20,
                        help="Number of bins (default: %(default)d)")
    parser.add_argument('-o','--output_prefix', default="servalcat_fw",
                        help='output file name prefix (default: %(default)s)')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

# TODO this function can be generalised and merged with sigmaa.process_input
def process_input(hklin, labin, n_bins, d_max=None, d_min=None):
    assert len(labin) == 2
    
    mtz = gemmi.read_mtz_file(hklin)
    logger.write("Input mtz: {}".format(hklin))
    logger.write("    Unit cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*mtz.cell.parameters))
    logger.write("  Space group: {}".format(mtz.spacegroup.hm))
    logger.write("")
    
    hkldata = utils.hkl.hkldata_from_mtz(mtz, labin, newlabels=["I","SIGI"])
    hkldata.df = hkldata.df.astype({name: 'float64' for name in ["I","SIGI"]})

    # TODO perhaps we should switch type to float64
    if (d_min, d_max).count(None) != 2:
        hkldata = hkldata.copy(d_min=d_min, d_max=d_max)
    d_min, d_max = hkldata.d_min_max()
    
    hkldata.complete()
    hkldata.sort_by_resolution()
    hkldata.calc_epsilon()
    hkldata.calc_centric()
    hkldata.setup_binning(n_bins=n_bins)
    logger.write("Data completeness: {:.2%}".format(hkldata.completeness()))

    # Create a centric selection table for faster look up
    centric_and_selections = {}
    for i_bin, idxes in hkldata.binned():
        centric_and_selections[i_bin] = []
        for c, g2 in hkldata.df.loc[idxes].groupby("centric", sort=False):
            valid_sel = numpy.isfinite(g2.I) & (g2.SIGI > 0)
            vidxes = g2.index[valid_sel]
            nidxes = g2.index[~valid_sel] # missing reflections
            centric_and_selections[i_bin].append((c, vidxes, nidxes))
    
    return hkldata, centric_and_selections
# process_input()

def determine_Sigma_and_aniso(hkldata, centric_and_selections):
    # initial estimate
    hkldata.binned_df["S"] = 1.
    I_over_eps = hkldata.df.I.to_numpy() / hkldata.df.epsilon.to_numpy()
    x = []
    for i_bin, idxes in hkldata.binned():
        S = max(numpy.nanmean(I_over_eps[idxes]), 1e-3)
        hkldata.binned_df.loc[i_bin, "S"] = S
        x.append(S)
    x.extend([0. for _ in range(5)])
    x = numpy.array(x)
    logger.write("Initial estimates:")
    logger.write(hkldata.binned_df.to_string())

    if 0:
        e = 1.e-4
        args = (hkldata.s_array(), hkldata, centric_and_selections)
        f0 = ll_all(x, *args)
        ad = ll_1der_all(x, *args)
        print("ana=", ad)
        print("f0=", f0)
        print("x=", x)
        for i in range(len(x)):
            x2 = [_ for _ in x]
            x2[i] += e
            f1 = ll_all(x2, *args)
            nd = (f1 - f0) / e
            print(i, nd, ad[i] / nd)
        quit()

    if 0:
        ret = scipy.optimize.minimize(fun=ll_all,
                                      jac=ll_1der_all,
                                      x0=x,
                                      args=(hkldata.s_array(), hkldata, centric_and_selections))
        
        print(ret)
    svecs = hkldata.s_array()
    B = gemmi.SMat33d(0,0,0,0,0,0)

    for icyc in range(5):
        logger.write("Refine B")
        t0 = time.time()
        ret = scipy.optimize.minimize(fun=ll_all_B,
                                      jac=ll_1der_all_B,
                                      x0=[B.u11, B.u22, B.u12, B.u13, B.u23],
                                      args=(svecs, hkldata, centric_and_selections))
        logger.write(str(ret))
        logger.write("time= {}".format(time.time() - t0))
        B = gemmi.SMat33d(ret.x[0], ret.x[1], -ret.x[0] - ret.x[1],
                          ret.x[2], ret.x[3], ret.x[4])
        logger.write("B_aniso= {}".format(B))
        logger.write("Refine S")
        for i, (i_bin, idxes) in enumerate(hkldata.binned()):
            S = hkldata.binned_df.loc[i_bin, "S"]
            ret = scipy.optimize.minimize(fun=ll_bin,
                                          jac=ll_1der_bin_S,
                                          hess=ll_2der_bin_S,
                                          method="Newton-CG",
                                          x0=[S],
                                          args=(B, i_bin, svecs, hkldata, centric_and_selections))
            logger.write("Bin {}".format(i_bin))
            logger.write(str(ret))
            hkldata.binned_df.loc[i_bin, "S"] = ret.x
        logger.write("Refined estimates in cycle {}:".format(icyc))
        logger.write(hkldata.binned_df.to_string())
        logger.write("B_aniso= {}".format(B))
    
    return B

# For the calculation of J = \int_0^\infty x^{4z-1} e^{-(x^4 - 2t0 x^2)/2} dx
def f1_orig2_value(x, z, t0, where=True):
    return (x**4 - 2 * t0 * x**2) / 2.0 - (4 * z - 1) * numpy.log(x, where=where)

def f1_orig2_1der(x, z, t0):
    return 2.0*(x**3 - t0 * x) - (4 * z - 1) / x

def f1_orig2_2der(x, z, t0):
    return 6.0 * x**2 - 2.0 * t0 + (4 * z - 1) / x**2

def calc_f1_orig2_x0(z, t0):
    return numpy.sqrt((t0 + numpy.sqrt(t0**2 + 8 * z - 2)) / 2.0)

# with variable transformation exp(x-exp(-x))
def f1_exp2_value(x, z, t0):
    ex = numpy.exp(-x)
    exx = x - ex
    ex1 = numpy.exp(2.0 * exx)
    return (ex1 * ex1 - 2.0 * t0 * ex1) / 2.0 - 4 * z * exx - numpy.log(1.0 + ex)

def f1_exp2_1der(x, z, t0):
    ex = numpy.exp(-x)
    exx = x - ex
    ex1 = numpy.exp(2.0 * exx)
    ret = (1 + ex) * (2 * ex1 * ex1 - 2.0 * t0 * ex1 - 4 * z) + 1 - 1 / (1 + ex)
    return ret

def f1_exp2_2der(x, z, t0):
    ex = numpy.exp(-x)
    exx = x - ex
    ex1 = numpy.exp(2.0 * exx)
    return -ex * (2.0 * ex1 * ex1 - 2.0 * t0 * ex1 - 4 * z) + (1 + ex)**2 * (8.0 * ex1 * ex1 - 4.0 * t0 * ex1) - ex / (1 + ex)**2

def calc_f1_exp2_x0(z, t0):
    v = (t0 + numpy.sqrt(t0**2 + 8 * z)) * 0.5
    a = 0.5 * numpy.log(v)
    exp_a = v**(-0.5)
    return scipy.special.lambertw(exp_a).real + a

@profile
def J_ratio_1(k_num, k_den, to1, h, py=False): # case 1
    N = 100 # we should use N1 and N2 optimized
    z = 0.5 * (k_den + 1)
    deltaz = 0.5 * (k_num - k_den)
    root = calc_f1_exp2_x0(z, to1)
    f1val = f1_exp2_value(root, z, to1)
    f2der = f1_exp2_2der(root, z, to1)
    if py:
        xx = h * numpy.sqrt(2 / f2der) * numpy.arange(-N, N+1)[:,None] + root
        ff = f1_exp2_value(xx, z, to1) - f1val
        laplace_correct = numpy.sum(numpy.exp(-ff), axis=0)
    
        # for numerator
        g = numpy.exp(4 * deltaz * (xx - numpy.exp(-xx)))
        laplace_correct_num = numpy.sum(numpy.exp(-ff) * g, axis=0)
        return laplace_correct_num / laplace_correct
    else:
        return ext.integ_J_ratio_1(delta=h * numpy.sqrt(2 / f2der),
                                   root=root, to1=to1, f1val=f1val, z=z, deltaz=deltaz)

@profile
def J_1(k, to1, h, log=False, py=False): # case 1
    N = 100 # we should use N1 and N2 optimized
    z = 0.5 * (k + 1)
    root = calc_f1_exp2_x0(z, to1)
    f1val = f1_exp2_value(root, z, to1)
    f2der = f1_exp2_2der(root, z, to1)
    if py:
        xx = h * numpy.sqrt(2 / f2der) * numpy.arange(-N, N+1)[:,None] + root
        ff = f1_exp2_value(xx, z, to1) - f1val
        laplace_correct = numpy.sum(numpy.exp(-ff), axis=0) * h
    else:
        laplace_correct = ext.integ_J_1(delta=h * numpy.sqrt(2 / f2der),
                                        root=root, to1=to1, f1val=f1val, z=z) * h
        
    expon = -f1val + 0.5 * (numpy.log(2.0) - numpy.log(f2der))
    if log:
        return expon + numpy.log(laplace_correct)
    else:
        return numpy.exp(expon) * laplace_correct

@profile
def J_ratio_2(k_num, k_den, to1, h, py=False): # case 2
    N = 100 # we should use N1 and N2 optimized
    z = 0.5 * (k_den + 1)
    deltaz = 0.5 * (k_num - k_den)
    root = calc_f1_orig2_x0(z, to1)
    f1val = f1_orig2_value(root, z, to1)
    f2der = f1_orig2_2der(root, z, to1)
    if py:
        xx = h * numpy.sqrt(2 / f2der) * numpy.arange(-N, N+1)[:,None] + root
        sel = xx > 0
        ff = f1_orig2_value(xx, z, to1, where=sel) - f1val
        laplace_correct = numpy.sum(numpy.exp(-ff, where=sel),  where=sel, axis=0)
    
        # for numerator
        g = numpy.exp(4 * deltaz * numpy.log(xx, where=sel), where=sel)
        laplace_correct_num = numpy.sum(numpy.exp(-ff, where=sel) * g, where=sel, axis=0)
        return laplace_correct_num / laplace_correct
    else:
        return ext.integ_J_ratio_2(delta=h * numpy.sqrt(2 / f2der),
                                   root=root, to1=to1, f1val=f1val, z=z, deltaz=deltaz)


@profile
def J_2(k, to1, h, log=False, py=False): # case 2
    N = 100 # we should use N1 and N2 optimized
    z = 0.5 * (k + 1)
    root = calc_f1_orig2_x0(z, to1)
    f1val = f1_orig2_value(root, z, to1)
    f2der = f1_orig2_2der(root, z, to1)
    if py:
        xx = h * numpy.sqrt(2 / f2der) * numpy.arange(-N, N+1)[:,None] + root
        sel = xx > 0
        ff = f1_orig2_value(xx, z, to1, where=sel) - f1val
        laplace_correct = numpy.sum(numpy.exp(-ff, where=sel), where=sel, axis=0) * h # need where to avoid exp(-ff)
    else:
        laplace_correct = ext.integ_J_2(delta=h * numpy.sqrt(2 / f2der),
                                        root=root, to1=to1, f1val=f1val, z=z) * h
        
    expon = -f1val + 0.5 * (numpy.log(2.0) - numpy.log(f2der))
    if log:
        return expon + numpy.log(laplace_correct)
    else:
        return numpy.exp(expon) * laplace_correct

def J_conditions(k_den, to1, case1_lim=10):
    d = to1 + numpy.sqrt(to1**2 + 4 * (k_den + 1) - 2)
    idxes = numpy.digitize(d, [case1_lim, numpy.inf])
    return idxes

@profile
def J_ratio(k_num, k_den, to1, h=0.5, case1_lim=10):
    idxes = J_conditions(k_den, to1, case1_lim)
    ret = numpy.zeros(to1.shape)
    sel0 = idxes==0
    sel1 = idxes==1
    ret[sel0] = J_ratio_1(k_num, k_den, to1[sel0], h)
    ret[sel1] = J_ratio_2(k_num, k_den, to1[sel1], h)
    return ret

@profile
def J(k, to1, h=0.5, case1_lim=10, log=False):
    idxes = J_conditions(k, to1, case1_lim)
    ret = numpy.zeros(to1.shape)
    sel0 = idxes==0
    sel1 = idxes==1
    ret[sel0] = J_1(k, to1[sel0], h, log)
    ret[sel1] = J_2(k, to1[sel1], h, log)
    return ret
# J()

def ll_acentric(S, Io, sigIo, eps, h=0.5): # S includes k^2
    to1 = numpy.asarray(Io / sigIo - sigIo / S / eps)
    return -J(0., to1, h, log=True) + numpy.log(S)

@profile
def ll_1der_acentric(S, svecs, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - sigIo / S / eps / k2)
    tmp = -J_ratio(1., 0., to1, h) * sigIo / eps / S / k2
    tmp2 = -tmp - 1
    ret = numpy.zeros(1+5) # dS, dBij
    #return -J_ratio(1., 0., to1, h) * sigIo / eps / S**2 + 1. / S
    ret[0] = numpy.sum(tmp / S + 1./S) # S
    ret[1] = 0.5 * numpy.sum((svecs[:,0]**2 - svecs[:,2]**2) * tmp2) # B11
    ret[2] = 0.5 * numpy.sum((svecs[:,1]**2 - svecs[:,2]**2) * tmp2) # B22
    ret[3] = numpy.sum(svecs[:,0] * svecs[:,1] * tmp2) # B12
    ret[4] = numpy.sum(svecs[:,0] * svecs[:,2] * tmp2) # B13
    ret[5] = numpy.sum(svecs[:,1] * svecs[:,2] * tmp2) # B23
    return ret

def ll_2der_acentric(x, Io, sigIo, eps, h=0.5):
    S = x[0]
    to1 = numpy.asarray(Io / sigIo - sigIo / S / eps)
    J_2_0 = J_ratio(2., 0., to1, h)
    J_1_0 = J_ratio(1., 0., to1, h)
    return (J_2_0 - J_1_0**2) * (sigIo / eps / S**2)**2 + 2. * J_1_0 * sigIo / eps / S**3 - 1. / S**2

def ll_centric(S, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / eps)
    return -J(-0.5, to1, h, log=True) + 0.5 * numpy.log(S)

@profile
def ll_1der_centric(S, svecs, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / eps / k2)
    tmp = -J_ratio(0.5, -0.5, to1, h) * 0.5 * sigIo / eps / S / k2
    tmp2 = -tmp - 0.5
    ret = numpy.zeros(1+5) # dS, dBij
    #return -J_ratio(0.5, -0.5, to1, h) * 0.5 * sigIo / eps / S**2 + 0.5 / S
    ret[0] = numpy.sum(tmp / S + 0.5 / S) # S
    ret[1] = 0.5 * numpy.sum((svecs[:,0]**2 - svecs[:,2]**2) * tmp2) # B11
    ret[2] = 0.5 * numpy.sum((svecs[:,1]**2 - svecs[:,2]**2) * tmp2) # B22
    ret[3] = numpy.sum(svecs[:,0] * svecs[:,1] * tmp2) # B12
    ret[4] = numpy.sum(svecs[:,0] * svecs[:,2] * tmp2) # B13
    ret[5] = numpy.sum(svecs[:,1] * svecs[:,2] * tmp2) # B23
    return ret

def ll_2der_centric(x, Io, sigIo, eps, h=0.5):
    S = x[0]
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / eps)
    J_15_05 = J_ratio(1.5, -0.5, to1, h)
    J_05_05 = J_ratio(0.5, -0.5, to1, h)
    return (-J_15_05 + J_05_05**2) * (0.5 * sigIo / eps / S**2)**2 + J_05_05 * sigIo / eps / S**3 - 0.5 / S**2

def ll_all(x, svecs, hkldata, centric_and_selections):
    ll = (ll_acentric, ll_centric)
    n_bins = len(hkldata.binned())
    S = x[:n_bins]
    B = gemmi.SMat33d(x[n_bins], x[n_bins+1], -x[n_bins]-x[n_bins+1],
                      x[n_bins+2], x[n_bins+3], x[n_bins+4])
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    ret = 0.
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            ret += numpy.sum(ll[c](S[i] * k2[cidxes], Io, sigo, eps))
    return ret

def ll_bin(x, B, i_bin, svecs, hkldata, centric_and_selections):
    S = x[0]
    ll = (ll_acentric, ll_centric)
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    ret = 0.
    for c, cidxes, nidxes in centric_and_selections[i_bin]:
        Io = hkldata.df.I.to_numpy()[cidxes]
        sigo = hkldata.df.SIGI.to_numpy()[cidxes]
        eps = hkldata.df.epsilon.to_numpy()[cidxes]
        ret += numpy.sum(ll[c](S * k2[cidxes], Io, sigo, eps))
    return ret
    
@profile
def ll_all_B(x, svecs, hkldata, centric_and_selections):
    ll = (ll_acentric, ll_centric)
    n_bins = 0
    B = gemmi.SMat33d(x[n_bins], x[n_bins+1], -x[n_bins]-x[n_bins+1],
                      x[n_bins+2], x[n_bins+3], x[n_bins+4])
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    ret = 0.
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            ret += numpy.sum(ll[c](hkldata.binned_df.S[i_bin] * k2[cidxes], Io, sigo, eps))
    return ret

def ll_1der_all(x, svecs, hkldata, centric_and_selections):
    ll_1der = (ll_1der_acentric, ll_1der_centric)
    n_bins = len(hkldata.binned())
    S = x[:n_bins]
    B = gemmi.SMat33d(x[n_bins], x[n_bins+1], -x[n_bins]-x[n_bins+1],
                      x[n_bins+2], x[n_bins+3], x[n_bins+4])
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #svecs = hkldata.s_array()
    ret = numpy.zeros_like(x)
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            tmp = ll_1der[c](S[i], svecs[cidxes], k2[cidxes], Io, sigo, eps)
            ret[i] += tmp[0]
            for j in range(5):
                ret[n_bins+j] += tmp[1+j]
    return ret

def ll_1der_bin_S(x, B, i_bin, svecs, hkldata, centric_and_selections):
    S = x[0]
    ll_1der = (ll_1der_acentric, ll_1der_centric)
    n_bins = len(hkldata.binned())
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #svecs = hkldata.s_array()
    #ret = numpy.zeros_like(x)
    ret = 0.
    for c, cidxes, nidxes in centric_and_selections[i_bin]:
        Io = hkldata.df.I.to_numpy()[cidxes]
        sigo = hkldata.df.SIGI.to_numpy()[cidxes]
        eps = hkldata.df.epsilon.to_numpy()[cidxes]
        tmp = ll_1der[c](S, svecs[cidxes], k2[cidxes], Io, sigo, eps)
        ret += tmp[0]
    return [ret]

@profile
def ll_1der_all_B(x, svecs, hkldata, centric_and_selections):
    ll_1der = (ll_1der_acentric, ll_1der_centric)
    n_bins = 0
    B = gemmi.SMat33d(x[n_bins], x[n_bins+1], -x[n_bins]-x[n_bins+1],
                      x[n_bins+2], x[n_bins+3], x[n_bins+4])
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #svecs = hkldata.s_array()
    ret = numpy.zeros_like(x)
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            tmp = ll_1der[c](hkldata.binned_df.S[i_bin], svecs[cidxes], k2[cidxes], Io, sigo, eps)
            for j in range(5):
                ret[n_bins+j] += tmp[1+j]
    return ret

def ll_2der_bin_S(x, B, i_bin, svecs, hkldata, centric_and_selections):
    S = x[0]
    ll_2der = (ll_2der_acentric, ll_2der_centric)
    n_bins = len(hkldata.binned())
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #ret = numpy.zeros_like(x)
    ret = 0.
    for c, cidxes, nidxes in centric_and_selections[i_bin]:
        Io = hkldata.df.I.to_numpy()[cidxes]
        sigo = hkldata.df.SIGI.to_numpy()[cidxes]
        eps = hkldata.df.epsilon.to_numpy()[cidxes]
        #print(S, k2.shape, k2[cidxes].shape)
        ret += numpy.sum(ll_2der[c]([S*k2[cidxes]], Io, sigo, eps))
    return [ret]

def french_wilson(hkldata, centric_and_selections, B_aniso):
    hkldata.df["F"] = numpy.nan
    hkldata.df["SIGF"] = numpy.nan
    hkldata.df["to1"] = numpy.nan
    k2 = hkldata.debye_waller_factors(b_cart=B_aniso)**2
    
    for i_bin, idxes in hkldata.binned():
        S = hkldata.binned_df.S[i_bin]
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            epsS = hkldata.df.epsilon.to_numpy()[cidxes] * S * k2[cidxes]
            
            if c == 0: # acentric
                to1 = Io / sigo - sigo / epsS
                F = numpy.sqrt(sigo) * J_ratio(0.5, 0., to1)
                Fsq = sigo * J_ratio(1., 0., to1)
            else: # centric
                to1 = Io / sigo - 0.5 * sigo / epsS
                F = numpy.sqrt(sigo) * J_ratio(0., -0.5, to1)
                Fsq = sigo * J_ratio(0.5, -0.5, to1)

            print("bin=",i_bin, "cen=", c, "min_to1=", numpy.min(to1))
            varF = Fsq - F**2
            hkldata.df.loc[cidxes, "F"] = F
            hkldata.df.loc[cidxes, "SIGF"] = numpy.sqrt(varF)
            hkldata.df.loc[cidxes, "to1"] = to1

def main(args):
    if args.nbins < 1:
        raise SystemExit("--nbins must be > 0")

    hkldata, centric_and_selections = process_input(hklin=args.hklin,
                                                    labin=args.labin.split(","),
                                                    n_bins=args.nbins,
                                                    d_min=args.d_min,
                                                    d_max=args.d_max)
    print(hkldata.df)
    B_aniso = determine_Sigma_and_aniso(hkldata, centric_and_selections)
    french_wilson(hkldata, centric_and_selections, B_aniso)
    print(hkldata.df)

    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=["F","SIGF","I","SIGI","d","bin","centric","to1"],
                      types={"F":"F", "SIGF":"Q"})
    logger.write("output mtz: {}".format(mtz_out))
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
