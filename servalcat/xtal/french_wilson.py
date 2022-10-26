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

USE_FISHER = True

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def adp_constraints(ops, cell, tr0=True):
    # think about f = (b-Rb)^T (b-Rb) = b^T b - b^TRb -b^TR^Tb + b^TR^TRb
    # d^2f/db^2 = 2I - 2(R+R^T) + (R^TR + RR^T)
    # eigenvectors of this second derivative matrix corresponding to 0-valeud eigenvalues are directions to refine
    def get_6x6(m):
        r = numpy.zeros((6,6))
        for k, (i, j) in enumerate(((0,0), (1,1), (2,2), (0,1), (0,2), (1,2))):
            r[k,:] = (m[i][0] * m[j][0],
                      m[i][1] * m[j][1],
                      m[i][2] * m[j][2],
                      m[i][0] * m[j][1] + m[i][1] * m[j][0],
                      m[i][0] * m[j][2] + m[i][2] * m[j][0],
                      m[i][1] * m[j][2] + m[i][2] * m[j][1])
        return r

    Ainv = cell.frac.mat
    x = numpy.zeros((6,6))
    if tr0:
        x[:3,:3] += numpy.ones((3,3)) * 2
    for op in ops:
        m = gemmi.Mat33(op.rot).multiply(Ainv)
        r1 = get_6x6(Ainv.tolist())
        r2 = get_6x6(m.tolist()) / op.DEN**2
        r1r2 = numpy.dot(r1.T, r2)
        x += 2 * numpy.dot(r1.T, r1) - 2 * (r1r2 + r1r2.T) + 2 * numpy.dot(r2.T, r2)

    evals, evecs = numpy.linalg.eig(x)
    ret = []
    print(evals)
    for i in range(6):
        if numpy.isclose(evals[i], 0):
            ret.append(evecs[:, i])

    return numpy.vstack(ret)
# adp_constraints()

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
        #S = max(numpy.nanmean(I_over_eps[idxes]), 1e-3)
        # var(I) = var_signal + var_noise, so this overestimates S,
        # but it should be better than having negative values (in noisy shell)
        S = numpy.nanstd(I_over_eps[idxes])
        hkldata.binned_df.loc[i_bin, "S"] = S
        x.append(S)
    logger.write("Initial estimates:")
    logger.write(hkldata.binned_df.to_string())

    B = gemmi.SMat33d(0,0,0,0,0,0)
    SMattolist = lambda B: [B.u11, B.u22, B.u33, B.u12, B.u13, B.u23]
    adpdirs = adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=True)
    logger.write("ADP free parameters = {}".format(adpdirs.shape[0]))
    if 0:
        B = gemmi.SMat33d(1,2,3,4,5,6)
        x = numpy.hstack([x, numpy.dot(SMattolist(B), numpy.linalg.pinv(adpdirs))])
        e = 1.e-3
        args = (hkldata.s_array(), hkldata, centric_and_selections, adpdirs)
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
        x = [100.]
        e = 1.e-4
        args = (hkldata.s_array(), hkldata, centric_and_selections)
        for i, (i_bin, idxes) in enumerate(hkldata.binned()):
            f0 = ll_bin(x, B, i_bin, *args)
            ad = ll_1der_bin_S(x, B, i_bin, *args)[0]
            f1 = ll_bin([x[0]+e], B, i_bin, *args)
            nd = (f1 - f0) / e
            print(i_bin, nd, ad / nd)
        quit()

    if 0:
        B = gemmi.SMat33d(1,2,3,4,5,6)
        #adpdirs = numpy.identity(6)
        x = numpy.dot(SMattolist(B), numpy.linalg.pinv(adpdirs))
        e = 1.e-4
        args = (hkldata.s_array(), hkldata, centric_and_selections, adpdirs)
        f0 = ll_1der_all_B(x, *args)
        ad = ll_2der_all_B(x, *args)
        print("ana2=", ad)
        print("ana1=", f0)
        print("x=", x)
        for i in range(len(x)):
            x2 = [_ for _ in x]
            x2[i] += e
            f1 = ll_1der_all_B(x2, *args)
            nd = (f1 - f0) / e
            print(i, nd, ad[i] / nd)
        
        quit()

    if 0:
        B = gemmi.SMat33d(1,2,3,4,5,6)
        x = [100.]
        e = 1.e-4
        args = (hkldata.s_array(), hkldata, centric_and_selections)
        for i, (i_bin, idxes) in enumerate(hkldata.binned()):
            f0 = ll_1der_bin_S(x, B, i_bin, *args)[0]
            ad = ll_2der_bin_S(x, B, i_bin, *args)[0]
            print("ana2=", ad)
            print("ana1=", f0)
            print("x=", x)
            f1 = ll_1der_bin_S([x[0]+e], B, i_bin, *args)[0]
            nd = (f1 - f0) / e
            print(i_bin, nd, ad / nd)
        quit()
        
    if 0:
        ret = scipy.optimize.minimize(fun=ll_all,
                                      jac=ll_1der_all,
                                      x0=x,
                                      args=(hkldata.s_array(), hkldata, centric_and_selections, adpdirs))
        
        print(ret)

    svecs = hkldata.s_array()
    cycle_data = [[0] + SMattolist(B) + list(hkldata.binned_df.S)]
    for icyc in range(100):
        logger.write("Refine B")
        B_converged = False
        t0 = time.time()
        if 0:
            x = numpy.dot(SMattolist(B), numpy.linalg.pinv(adpdirs))
            ret = scipy.optimize.minimize(fun=ll_all_B,
                                          jac=ll_1der_all_B,
                                          hess=ll_2der_all_B,
                                          method="Newton-CG",
                                          x0=x,
                                          args=(svecs, hkldata, centric_and_selections, adpdirs))
            B = gemmi.SMat33d(*numpy.dot(ret.x, adpdirs))
            logger.write(str(ret))
        else:
            args=(svecs, hkldata, centric_and_selections, adpdirs)
            for j in range(10):
                x = numpy.dot(SMattolist(B), numpy.linalg.pinv(adpdirs))
                f0 = ll_all_B(x, *args)
                shift = ll_shift_B(x, *args)
                for i in range(3):
                    ss = shift / 2**i
                    f1 = ll_all_B(x + ss, *args)
                    logger.write("f0 = {:.3e} shift = {} f1 = {:.3e} dec? {}".format(f0, ss, f1, f1 < f0))
                    if f1 < f0:
                        B = gemmi.SMat33d(*numpy.dot(x+ss, adpdirs))
                        if numpy.max(numpy.abs(ss)) < 1e-4: B_converged = True
                        break
                else:
                    B_converged = True
                if B_converged: break

        logger.write("time= {}".format(time.time() - t0))
        logger.write("B_aniso= {}".format(B))
        logger.write("Refine S")
        S_converged = [False for _ in hkldata.binned()]
        for i, (i_bin, idxes) in enumerate(hkldata.binned()):
            #logger.write("Bin {}".format(i_bin))
            if 0:
                S = hkldata.binned_df.loc[i_bin, "S"]
                ret = scipy.optimize.minimize(fun=ll_bin,
                                              jac=ll_1der_bin_S,
                                              hess=ll_2der_bin_S,
                                              method="Newton-CG",
                                              x0=[S],
                                              args=(B, i_bin, svecs, hkldata, centric_and_selections))
                logger.write(str(ret))
                hkldata.binned_df.loc[i_bin, "S"] = ret.x
            else:
                args=(B, i_bin, svecs, hkldata, centric_and_selections)
                for j in range(10):
                    S = hkldata.binned_df.loc[i_bin, "S"]
                    f0 = ll_bin([S], *args)
                    shift = numpy.exp(ll_shift_bin_S(S, *args))
                    if 0:#i_bin == 16:
                        with open("debug.txt", "w") as ofs:
                            for i in range(0,31):
                                ss = shift**(1. / 2**i)
                                print(ss, ll_bin([S*ss], *args), file=ofs)
                        quit()
                    for k in range(3):
                        ss = shift**(1. / 2**k)
                        f1 = ll_bin([S*ss], *args)
                        logger.write("bin {:3d} f0 = {:.3e} shift = {:.3e} f1 = {:.3e} dec? {}".format(i_bin, f0, ss, f1, f1 < f0))
                        if f1 < f0:
                            hkldata.binned_df.loc[i_bin, "S"] = S * ss
                            if ss > 0.9999: S_converged[i] = True
                            break
                    else:
                        S_converged[i] = True
                    if S_converged[i]: break

        logger.write("Refined estimates in cycle {}:".format(icyc))
        logger.write(hkldata.binned_df.to_string())
        logger.write("B_aniso= {}".format(B))
        cycle_data.append([icyc] + SMattolist(B) + list(hkldata.binned_df.S))
        print(B_converged, S_converged)
        if B_converged and all(S_converged):
            logger.write("Converged. Finished in cycle {}".format(icyc))
            break

    with open("fw_cycles.dat", "w") as ofs:
        ofs.write("cycle B11 B22 B33 B12 B13 B23 " + " ".join("S{}".format(i) for i in hkldata.binned_df.index) + "\n")
        for data in cycle_data:
            ofs.write("{:2d} ".format(data[0]+1))
            ofs.write(" ".join("{:.4e}".format(x) for x in data[1:]))
            ofs.write("\n")
        
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
    # Want to solve x - exp(-x) = 0.5 * log(v) = A
    # solution: x = W(exp(-A)) + A
    tmp = numpy.sqrt(t0**2 + 8 * z)
    v = 0.5 * numpy.where(t0 > 0, t0 + tmp,
                          8 * z / (tmp - t0)) # to avoid precision loss
    if numpy.any(v == 0):
        sel = v == 0
        print("ERROR: v=0, t0=", t0[sel], "z=", z)
    sel = v > 0
    a = 0.5 * numpy.log(v, where=sel)
    exp_a = numpy.power(v, -0.5, where=sel)
    return numpy.where(sel, scipy.special.lambertw(exp_a).real + a, -numpy.inf)

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
    ret = numpy.zeros(1+6) # dS, dBij
    #return -J_ratio(1., 0., to1, h) * sigIo / eps / S**2 + 1. / S
    ret[0] = numpy.sum(tmp / S + 1. / S) # S
    ret[1] = 0.5 * numpy.sum((svecs[:,0]**2) * tmp2) # B11
    ret[2] = 0.5 * numpy.sum((svecs[:,1]**2) * tmp2) # B22
    ret[3] = 0.5 * numpy.sum((svecs[:,2]**2) * tmp2) # B33
    ret[4] = numpy.sum(svecs[:,0] * svecs[:,1] * tmp2) # B12
    ret[5] = numpy.sum(svecs[:,0] * svecs[:,2] * tmp2) # B13
    ret[6] = numpy.sum(svecs[:,1] * svecs[:,2] * tmp2) # B23
    return ret

def ll_2der_S_acentric(x, k2, Io, sigIo, eps, h=0.5):
    S = x[0]
    to1 = numpy.asarray(Io / sigIo - sigIo / S / k2 / eps)
    J_1_0 = J_ratio(1., 0., to1, h)
    if USE_FISHER:
        return (J_1_0 * sigIo / eps / k2 / S**2 - 1. / S)**2
    else:
        J_2_0 = J_ratio(2., 0., to1, h)
        return -(J_2_0 - J_1_0**2) * (sigIo / eps / k2 / S**2)**2 - 1. / S**2 + 2. * J_1_0 * sigIo / eps / k2 / S**3

def ll_ders_S_acentric(S, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - sigIo / S / eps / k2)
    tmp = -J_ratio(1., 0., to1, h) * sigIo / eps / S**2 / k2 + 1. / S
    g = numpy.sum(tmp)
    H = numpy.sum(tmp**2)
    return g, H

@profile
def ll_2der_B_acentric(S, svecs, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - sigIo / S / k2 / eps)
    J_1_0 = J_ratio(1., 0., to1, h)
    if USE_FISHER:
        tmp = (J_1_0 * sigIo / eps / k2 / S - 1.)**2
    else:
        J_2_0 = J_ratio(2., 0., to1, h)
        tmp = -(J_2_0 - J_1_0**2) * (sigIo / eps / S / k2)**2 + J_1_0 * sigIo / eps / S / k2
    ret = numpy.zeros((6,6))
    tmp2 = (0.5 * svecs[:,0]**2, 0.5 * svecs[:,1]**2, 0.5 * svecs[:,2]**2,
           svecs[:,0] * svecs[:,1], svecs[:,0] * svecs[:,2], svecs[:,1] * svecs[:,2])
    for i, j in ((0,0), (1,1), (2,2), (0,1), (0,2), (1,2)):
        ret[i,j] = numpy.sum(tmp2[i] * tmp2[j] * tmp)
        if i != j: ret[j,i] = ret[i,j]

    return ret

def ll_ders_B_acentric(S, svecs, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - sigIo / S / k2 / eps)
    g = numpy.zeros(6)
    H = numpy.zeros((6, 6))
    tmp = J_ratio(1., 0., to1, h) * sigIo / eps / k2 / S - 1.
    tmpsqr = tmp**2
    tmp2 = (0.5 * svecs[:,0]**2, 0.5 * svecs[:,1]**2, 0.5 * svecs[:,2]**2,
           svecs[:,0] * svecs[:,1], svecs[:,0] * svecs[:,2], svecs[:,1] * svecs[:,2])
    for k, (i, j) in enumerate(((0,0), (1,1), (2,2), (0,1), (0,2), (1,2))):
        H[i,j] = numpy.sum(tmp2[i] * tmp2[j] * tmpsqr)
        if i != j: H[j,i] = H[i,j]
        g[k] = numpy.sum(tmp2[k] * tmp)

    return g, H

def ll_centric(S, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / eps)
    return -J(-0.5, to1, h, log=True) + 0.5 * numpy.log(S)

@profile
def ll_1der_centric(S, svecs, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / eps / k2)
    tmp = -J_ratio(0.5, -0.5, to1, h) * 0.5 * sigIo / eps / S / k2
    tmp2 = -tmp - 0.5
    ret = numpy.zeros(1+6) # dS, dBij
    #return -J_ratio(0.5, -0.5, to1, h) * 0.5 * sigIo / eps / S**2 + 0.5 / S
    ret[0] = numpy.sum(tmp / S + 0.5 / S) # S
    ret[1] = 0.5 * numpy.sum((svecs[:,0]**2) * tmp2) # B11
    ret[2] = 0.5 * numpy.sum((svecs[:,1]**2) * tmp2) # B22
    ret[3] = 0.5 * numpy.sum((svecs[:,2]**2) * tmp2) # B33
    ret[4] = numpy.sum(svecs[:,0] * svecs[:,1] * tmp2) # B12
    ret[5] = numpy.sum(svecs[:,0] * svecs[:,2] * tmp2) # B13
    ret[6] = numpy.sum(svecs[:,1] * svecs[:,2] * tmp2) # B23
    return ret

def ll_2der_S_centric(x, k2, Io, sigIo, eps, h=0.5):
    S = x[0]
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / k2 / eps)
    J_05_05 = J_ratio(0.5, -0.5, to1, h)
    if USE_FISHER:
        return (J_05_05 * sigIo / eps / k2 / S**2 * 0.5 - 0.5 / S)**2
    else:
        J_15_05 = J_ratio(1.5, -0.5, to1, h)
        return (-J_15_05 + J_05_05**2) * (0.5 * sigIo / eps / k2 / S**2)**2 - 0.5 / S**2 + J_05_05 * sigIo / eps / k2 / S**3 

def ll_ders_S_centric(S, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / eps / k2)
    tmp = -J_ratio(0.5, -0.5, to1, h) * 0.5 * sigIo / eps / S**2 / k2 + 0.5 / S
    g = numpy.sum(tmp)
    H = numpy.sum(tmp**2)
    return g, H

def ll_ders_B_centric(S, svecs, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / eps / k2)
    g = numpy.zeros(6)
    H = numpy.zeros((6, 6))
    tmp = J_ratio(0.5, -0.5, to1, h) * 0.5 * sigIo / eps / S / k2 - 0.5
    tmpsqr = tmp**2
    tmp2 = (0.5 * svecs[:,0]**2, 0.5 * svecs[:,1]**2, 0.5 * svecs[:,2]**2,
           svecs[:,0] * svecs[:,1], svecs[:,0] * svecs[:,2], svecs[:,1] * svecs[:,2])
    for k, (i, j) in enumerate(((0,0), (1,1), (2,2), (0,1), (0,2), (1,2))):
        H[i,j] = numpy.sum(tmp2[i] * tmp2[j] * tmpsqr)
        if i != j: H[j,i] = H[i,j]
        g[k] = numpy.sum(tmp2[k] * tmp)

    return g, H
    
@profile
def ll_2der_B_centric(S, svecs, k2, Io, sigIo, eps, h=0.5):
    to1 = numpy.asarray(Io / sigIo - 0.5 * sigIo / S / k2 / eps)
    J_05_05 = J_ratio(0.5, -0.5, to1, h)
    if USE_FISHER:
        tmp = (J_05_05 * sigIo / eps / k2 / S * 0.5 - 0.5)**2
    else:
        J_15_05 = J_ratio(1.5, -0.5, to1, h)
        tmp = (-J_15_05 + J_05_05**2) * (0.5 * sigIo / eps / S / k2)**2 + J_05_05 * sigIo / eps / S / k2 * 0.5
    ret = numpy.zeros((6,6))
    tmp2 = (0.5 * svecs[:,0]**2, 0.5 * svecs[:,1]**2, 0.5 * svecs[:,2]**2,
           svecs[:,0] * svecs[:,1], svecs[:,0] * svecs[:,2], svecs[:,1] * svecs[:,2])
    for i, j in ((0,0), (1,1), (2,2), (0,1), (0,2), (1,2)):
        ret[i,j] = numpy.sum(tmp2[i] * tmp2[j] * tmp)
        if i != j: ret[j,i] = ret[i,j]

    return ret

def ll_all(x, svecs, hkldata, centric_and_selections, adpdirs):
    ll = (ll_acentric, ll_centric)
    n_bins = len(hkldata.binned())
    S = x[:n_bins]
    B = gemmi.SMat33d(*numpy.dot(x[n_bins:], adpdirs))
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
def ll_all_B(x, svecs, hkldata, centric_and_selections, adpdirs):
    ll = (ll_acentric, ll_centric)
    B = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    ret = 0.
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            ret += numpy.sum(ll[c](hkldata.binned_df.S[i_bin] * k2[cidxes], Io, sigo, eps))
    return ret

def ll_1der_all(x, svecs, hkldata, centric_and_selections, adpdirs):
    ll_1der = (ll_1der_acentric, ll_1der_centric)
    n_bins = len(hkldata.binned())
    S = x[:n_bins]
    B = gemmi.SMat33d(*numpy.dot(x[n_bins:], adpdirs))
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
            ret[n_bins:] += numpy.dot(tmp[1:], adpdirs.T)
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
def ll_1der_all_B(x, svecs, hkldata, centric_and_selections, adpdirs):
    ll_1der = (ll_1der_acentric, ll_1der_centric)
    B = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #print("x=", x)
    #svecs = hkldata.s_array()
    ret = numpy.zeros_like(x)
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            tmp = ll_1der[c](hkldata.binned_df.S[i_bin], svecs[cidxes], k2[cidxes], Io, sigo, eps)
            #print(numpy.dot(tmp[1:].reshape(1,6), adpdirs.T))
            ret += numpy.dot(tmp[1:], adpdirs.T)
    return ret

def ll_2der_bin_S(x, B, i_bin, svecs, hkldata, centric_and_selections):
    S = x[0]
    ll_2der = (ll_2der_S_acentric, ll_2der_S_centric)
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #ret = numpy.zeros_like(x)
    ret = 0.
    for c, cidxes, nidxes in centric_and_selections[i_bin]:
        Io = hkldata.df.I.to_numpy()[cidxes]
        sigo = hkldata.df.SIGI.to_numpy()[cidxes]
        eps = hkldata.df.epsilon.to_numpy()[cidxes]
        #print(S, k2.shape, k2[cidxes].shape)
        ret += numpy.sum(ll_2der[c]([S], k2[cidxes], Io, sigo, eps))
    return [ret]

@profile
def ll_2der_all_B(x, svecs, hkldata, centric_and_selections, adpdirs):
    ll_2der = (ll_2der_B_acentric, ll_2der_B_centric)
    B = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #print("x=", x)
    #svecs = hkldata.s_array()
    ret = numpy.zeros((len(x), len(x)))
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            tmp = ll_2der[c](hkldata.binned_df.S[i_bin], svecs[cidxes], k2[cidxes], Io, sigo, eps)
            #print(numpy.dot(tmp[1:].reshape(1,6), adpdirs.T))
            ret += numpy.dot(adpdirs, numpy.dot(tmp, adpdirs.T))
    return ret

def ll_shift_bin_S(S, B, i_bin, svecs, hkldata, centric_and_selections, exp_trans=True):
    ll_ders = (ll_ders_S_acentric, ll_ders_S_centric)
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    g = 0.
    H = 0.
    for c, cidxes, nidxes in centric_and_selections[i_bin]:
        Io = hkldata.df.I.to_numpy()[cidxes]
        sigo = hkldata.df.SIGI.to_numpy()[cidxes]
        eps = hkldata.df.epsilon.to_numpy()[cidxes]
        #print(S, k2.shape, k2[cidxes].shape)
        g_tmp, H_tmp = ll_ders[c](S, k2[cidxes], Io, sigo, eps)
        g += g_tmp
        H += H_tmp
    if exp_trans:
        return -g / (H * S + g)
    else:
        return -g / H

def ll_shift_B(x, svecs, hkldata, centric_and_selections, adpdirs):
    ll_ders = (ll_ders_B_acentric, ll_ders_B_centric)
    B = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k2 = hkldata.debye_waller_factors(b_cart=B)**2
    #g, H = numpy.zeros(len(x)), numpy.zeros((len(x), len(x)))
    g, H = numpy.zeros(6), numpy.zeros((6,6))
    for i, (i_bin, idxes) in enumerate(hkldata.binned()):
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            eps = hkldata.df.epsilon.to_numpy()[cidxes]
            g_tmp, H_tmp = ll_ders[c](hkldata.binned_df.S[i_bin], svecs[cidxes], k2[cidxes], Io, sigo, eps)
            g += g_tmp
            H += H_tmp
    g = numpy.dot(g, adpdirs.T)
    H = numpy.dot(adpdirs, numpy.dot(H, adpdirs.T))
    return -numpy.dot(g, numpy.linalg.pinv(H))

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
