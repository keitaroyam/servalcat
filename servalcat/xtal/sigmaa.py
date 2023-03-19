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
import itertools
import scipy.special
import scipy.optimize
from servalcat.utils import logger
from servalcat import utils
from servalcat import ext

"""
DFc = sum_j D_j F_c,j
The last Fc,n is bulk solvent contribution.
"""

def add_arguments(parser):
    parser.description = 'Sigma-A parameter estimation for crystallographic data'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin', required=True,
                        help='MTZ column for F,SIGF,FREE')
    parser.add_argument('--free', type=int, default=0,
                        help='flag number for test set')
    parser.add_argument('--model', required=True, nargs="+", action="append",
                        help='Input atomic model file(s)')
    parser.add_argument("-d", '--d_min', type=float)
    #parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int, default=20,
                        help="Number of bins (default: %(default)d)")
    parser.add_argument('-s', '--source', choices=["electron", "xray", "neutron"], default="xray",
                        help="Scattering factor choice (default: %(default)s)")
    parser.add_argument('--D_as_exp',  action='store_true',
                        help="estimate D through exp(x) as a positivity constraint")
    parser.add_argument('--S_as_exp',  action='store_true',
                        help="estimate variance of unexplained signal through exp(x) as a positivity constraint")
    parser.add_argument('--no_solvent',  action='store_true',
                        help="Do not consider bulk solvent contribution")
    parser.add_argument('--use_cc',  action='store_true',
                        help="Use CC(|F1|,|F2|) to CC(F1,F2) conversion to derive D and S")
    parser.add_argument('--use', choices=["all", "work", "test"], default="all",
                        help="Which reflections to be used for the parameter estimate.")
    parser.add_argument('-o','--output_prefix', default="sigmaa",
                        help='output file name prefix (default: %(default)s)')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_DFc(Ds, Fcs):
    DFc = sum(Ds[i] * Fcs[i] for i in range(len(Ds)))
    return DFc
# calc_DFc()

def calc_abs_DFc(Ds, Fcs):
    DFc = sum(Ds[i] * Fcs[i] for i in range(len(Ds)))
    return numpy.abs(DFc)
# calc_abs_DFc()

def deriv_DFc2_and_DFc_dDj(Ds, Fcs):
    """
    [(d/dDj |sum(Dk * Fc,k)|^2,
      d/dDj |sum(Dk * Fc,k)|), ....] for j = 0 .. N-1
    """
    
    DFc = sum(Ds[i] * Fcs[i] for i in range(len(Ds)))
    DFc_abs = numpy.abs(DFc)
    
    ret = []
    for j in range(len(Ds)):
        rsq = numpy.real(Fcs[j] * DFc.conj())
        ret.append((2 * rsq,
                    rsq / DFc_abs))
    return DFc_abs, ret
# deriv_DFc2_and_DFc_dDj()

def mlf_acentric(Fo, varFo, Fcs, Ds, S, epsilon):
    # https://doi.org/10.1107/S0907444911001314
    # eqn (4)
    Sigma = 2 * varFo + epsilon * S
    DFc = calc_abs_DFc(Ds, Fcs)
    ret = numpy.log(2) + numpy.log(Fo) - numpy.log(Sigma)
    ret += -(Fo**2 + DFc**2)/Sigma
    ret += gemmi.log_bessel_i0(2*Fo*DFc/Sigma)
    return -ret
# mlf_acentric()

def deriv_mlf_wrt_D_S_acentric(Fo, varFo, Fcs, Ds, S, epsilon):
    deriv = numpy.zeros(1+len(Ds))
    Sigma = 2 * varFo + epsilon * S
    Fo2 = Fo**2
    DFc, tmp = deriv_DFc2_and_DFc_dDj(Ds, Fcs)
    i1_i0_x = gemmi.bessel_i1_over_i0(2*Fo*DFc/Sigma) # m
    for i, (sqder, der) in enumerate(tmp):
        deriv[i] = -numpy.nansum(-sqder / Sigma + i1_i0_x * 2 * Fo * der / Sigma)
    
    deriv[-1] = -numpy.nansum((-1/Sigma + (Fo2 + DFc**2 - i1_i0_x * 2 * Fo * DFc) / Sigma**2) * epsilon)
    return deriv
# deriv_mlf_wrt_D_S_acentric()

def mlf_centric(Fo, varFo, Fcs, Ds, S, epsilon):
    # https://doi.org/10.1107/S0907444911001314
    # eqn (4)
    Sigma = varFo + epsilon * S
    DFc = calc_abs_DFc(Ds, Fcs)
    ret = 0.5 * (numpy.log(2 / numpy.pi) - numpy.log(Sigma))
    ret += -0.5 * (Fo**2 + DFc**2) / Sigma
    ret += gemmi.log_cosh(Fo * DFc / Sigma)
    return -ret
# mlf_centric()

def deriv_mlf_wrt_D_S_centric(Fo, varFo, Fcs, Ds, S, epsilon):
    deriv = numpy.zeros(1+len(Ds))
    Sigma = varFo + epsilon * S
    Fo2 = Fo**2
    DFc, tmp = deriv_DFc2_and_DFc_dDj(Ds, Fcs)
    tanh_x = numpy.tanh(Fo*DFc/Sigma)
    for i, (sqder, der) in enumerate(tmp):
        deriv[i] = -numpy.nansum(-0.5 * sqder / Sigma + tanh_x * Fo * der / Sigma)
    deriv[-1] = -numpy.nansum((-0.5 / Sigma + (0.5*(Fo2+DFc**2) - tanh_x * Fo*DFc)/Sigma**2)*epsilon)
    return deriv
# deriv_mlf_wrt_D_S_centric()

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
#@profile
def mlf(df, fc_labs, Ds, S, centric_sel, use):
    ret = 0.
    func = (mlf_acentric, mlf_centric)
    for c, work, test in centric_sel:
        if use == "all":
            cidxes = numpy.concatenate([work, test])
        else:
            cidxes = work if use == "work" else test
        Fcs = [df[lab].to_numpy()[cidxes] for lab in fc_labs]
        ret += numpy.nansum(func[c](df.FP.to_numpy()[cidxes], df.SIGFP.to_numpy()[cidxes]**2, Fcs, Ds, S, df.epsilon.to_numpy()[cidxes]))
    return ret
# mlf()

#@profile
def deriv_mlf_wrt_D_S(df, fc_labs, Ds, S, centric_sel, use):
    ret = []
    func = (deriv_mlf_wrt_D_S_acentric, deriv_mlf_wrt_D_S_centric)
    for c, work, test in centric_sel:
        if use == "all":
            cidxes = numpy.concatenate([work, test])
        else:
            cidxes = work if use == "work" else test
        Fcs = [df[lab].to_numpy()[cidxes] for lab in fc_labs]
        ret.append((func[c](df.FP.to_numpy()[cidxes], df.SIGFP.to_numpy()[cidxes]**2, Fcs, Ds, S, df.epsilon.to_numpy()[cidxes])))
    return sum(ret)
# deriv_mlf_wrt_D_S()

def determine_mlf_params_from_cc(hkldata, fc_labs, D_labs, centric_and_selections, use="all"):
    # theorhetical values
    cc_a = lambda cc: (numpy.pi/4*(1-cc**2)**2 * scipy.special.hyp2f1(3/2, 3/2, 1, cc**2) - numpy.pi/4) / (1-numpy.pi/4)
    cc_c = lambda cc: 2/(numpy.pi-2) * (cc**2*numpy.sqrt(1-cc**2) + cc * numpy.arctan(cc/numpy.sqrt(1-cc**2)) + (1-cc**2)**(3/2)-1)
    table_fsc = numpy.arange(0, 1, 1e-3)
    table_cc = [cc_a(table_fsc), cc_c(table_fsc)]

    for lab in D_labs: hkldata.binned_df[lab] = 1.
    hkldata.binned_df["S"] = 1.
    
    stats = hkldata.binned_df[["d_max", "d_min"]].copy()
    for i, labi in enumerate(fc_labs):
        stats["CC(FP,{})".format(labi)] = numpy.nan
    for i, labi in enumerate(fc_labs):
        for j in range(i+1, len(fc_labs)):
            labj = fc_labs[j]
            stats["CC({},{})".format(labi, labj)] = numpy.nan

    # sqrt of eps * c; c = 1 for acentrics and 2 for centrics
    inv_sqrt_c_eps = 1. / numpy.sqrt(hkldata.df.epsilon.to_numpy() * (hkldata.df.centric.to_numpy() + 1))
    for i_bin, _ in hkldata.binned():
        # assume they are all acentrics.. only correct by c
        if use == "all":
            cidxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin] for i in (1,2)])
        else:
            i = 1 if use == "work" else 2
            cidxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin]])
        valid_sel = numpy.isfinite(hkldata.df.FP.to_numpy()[cidxes])
        cidxes = cidxes[valid_sel]
        factor = inv_sqrt_c_eps[cidxes]
        Fo = hkldata.df.FP.to_numpy()[cidxes] * factor
        mean_Fo2 = numpy.mean(Fo**2)
        SigFo = hkldata.df.SIGFP.to_numpy()[cidxes]
        Fcs = [hkldata.df[lab].to_numpy()[cidxes] * factor for lab in fc_labs]
        mean_Fk2 = numpy.array([numpy.mean(numpy.abs(fk)**2) for fk in Fcs])
        
        # estimate D
        cc_fo_fj = [numpy.corrcoef(numpy.abs(fj), Fo)[1,0] for fj in Fcs]
        for i in range(len(fc_labs)): stats.loc[i_bin, "CC(FP,{})".format(fc_labs[i])] = cc_fo_fj[i]
        mat = [[numpy.sqrt(numpy.mean(numpy.abs(fk)**2)/mean_Fo2) * numpy.real(numpy.corrcoef(fk, fj)[1,0])
                for fk in Fcs]
               for fj in Fcs]
        A = [[numpy.sqrt(numpy.mean(numpy.abs(fk)**2) * numpy.mean(numpy.abs(fj)**2))/mean_Fo2 * numpy.real(numpy.corrcoef(fk, fj)[1,0])
                for fk in Fcs]
               for fj in Fcs]
        A = numpy.array([[numpy.real(numpy.corrcoef(fk, fj)[1,0]) for fk in Fcs] for fj in Fcs])
        v = numpy.interp(cc_fo_fj, table_cc[0], table_fsc)

        for i in range(len(fc_labs)):
            labi = fc_labs[i]
            for j in range(i+1, len(fc_labs)):
                labj = fc_labs[j]
                stats.loc[i_bin, "CC({},{})".format(labi, labj)] = numpy.real(numpy.corrcoef(Fcs[i], Fcs[j])[1,0])

        # test all signs, fixing first Fc positive.
        cc_max = -2
        for v_test in itertools.product(*((x, -x) for x in v[1:])):
            v_test = numpy.array((v[0],)+v_test)
            Dj_test = numpy.dot(numpy.linalg.pinv(A), v_test) * numpy.sqrt(mean_Fo2 / mean_Fk2)
            DFc_test = calc_abs_DFc(Dj_test, Fcs)
            cc_test = numpy.corrcoef(Fo, numpy.abs(DFc_test))[1,0]
            if cc_test > cc_max:
                cc_max = cc_test
                v_max = v_test
                DFc = DFc_test
                Dj = Dj_test

        for lab, D in zip(D_labs, Dj):
            hkldata.binned_df.loc[i_bin, lab] = D

        # estimate S
        mean_DFc2 = numpy.mean(DFc**2)
        est_fsc_fo_fc = numpy.interp(numpy.corrcoef(Fo, DFc)[1,0], table_cc[0], table_fsc)
        S = mean_Fo2 - 2 * numpy.sqrt(mean_Fo2 * mean_DFc2) * est_fsc_fo_fc + mean_DFc2 - numpy.mean(SigFo**2)
        hkldata.binned_df.loc[i_bin, "S"] = S

    logger.writeln("\nCC:")
    logger.writeln(stats.to_string())
    logger.writeln("\nEstimates:")
    logger.writeln(hkldata.binned_df.to_string())
# determine_mlf_params_from_cc()
    
def determine_mlf_params(hkldata, fc_labs, D_labs, centric_and_selections, D_as_exp=False, S_as_exp=False, use="all"):
    assert use in ("all", "work", "test")
    if D_as_exp:
        transD = numpy.exp # D = transD(x)
        transD_deriv = numpy.exp # dD/dx
        transD_inv = numpy.log # x = transD_inv(D)
    else:
        transD = lambda x: x
        transD_deriv = lambda x: 1
        transD_inv = lambda x: x

    if S_as_exp:
        transS = numpy.exp
        transS_deriv = numpy.exp
        transS_inv = numpy.log
    else:
        transS = lambda x: x
        transS_deriv = lambda x: 1
        transS_inv = lambda x: x
    
    # Initial values
    for lab in D_labs: hkldata.binned_df[lab] = 1.
    hkldata.binned_df["S"] = 10000.
    for i_bin, _ in hkldata.binned():
        if use == "all":
            idxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin] for i in (1,2)])
        else:
            i = 1 if use == "work" else 2
            idxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin]])
        valid_sel = numpy.isfinite(hkldata.df.FP[idxes]) # as there is no nan-safe numpy.corrcoef
        idxes = idxes[valid_sel]
        FC = numpy.abs(hkldata.df.FC.to_numpy()[idxes])
        FP = hkldata.df.FP.to_numpy()[idxes]
        D = numpy.corrcoef(FP, FC)[1,0]
        hkldata.binned_df.loc[i_bin, D_labs[0]] = D
        hkldata.binned_df.loc[i_bin, "S"] = numpy.var(FP - D * FC)

    for D_lab in D_labs:
        if hkldata.binned_df[D_lab].min() <= 0:
            min_D = hkldata.binned_df[D_lab][hkldata.binned_df[D_lab] > 0].min() * 0.1
            logger.writeln("WARNING: negative {} is detected from initial estimates. Replacing it using minimum positive value {:.2e}".format(D_lab, min_D))
            hkldata.binned_df[D_lab].where(hkldata.binned_df[D_lab] > 0, min_D, inplace=True) # arbitrary
        
    logger.writeln("Initial estimates:")
    logger.writeln(hkldata.binned_df.to_string())

    for i_bin, idxes in hkldata.binned():
        x0 = [transD_inv(hkldata.binned_df[lab][i_bin]) for lab in D_labs] + [transS_inv(hkldata.binned_df.S[i_bin])]
        def target(x):
            return mlf(hkldata.df, fc_labs, transD(x[:-1]), transS(x[-1]), centric_and_selections[i_bin], use)
        def grad(x):
            g = deriv_mlf_wrt_D_S(hkldata.df, fc_labs, transD(x[:-1]), transS(x[-1]), centric_and_selections[i_bin], use)
            g[:-1] *= transD_deriv(x[:-1])
            g[-1] *= transS_deriv(x[-1])
            return g

        # test derivative
        if 0:
            gana = grad(x0)
            e = 1e-4
            for i in range(len(x0)):
                tmp = x0.copy()
                f0 = target(tmp)
                tmp[i] += e
                fe = target(tmp)
                gnum = (fe-f0)/e
                print("DERIV:", i, gnum, gana[i], gana[i]/gnum)

        #print("Bin", i_bin)
        res = scipy.optimize.minimize(fun=target, x0=x0, jac=grad)
        #print(res)
        
        for i, lab in enumerate(D_labs):
            hkldata.binned_df.loc[i_bin, lab] = transD(res.x[i])
        hkldata.binned_df.loc[i_bin, "S"] = transS(res.x[-1])

    logger.writeln("Refined estimates:")
    logger.writeln(hkldata.binned_df.to_string())
    return D_labs
# determine_mlf_params()

def determine_mli_params(hkldata, fc_labs, D_labs, b_aniso, centric_and_selections, D_as_exp=False, S_as_exp=False, use="all"):
    assert use == "all" # for now
    # TODO sort out redundant code
    assert use in ("all", "work", "test")
    if D_as_exp:
        transD = numpy.exp # D = transD(x)
        transD_deriv = numpy.exp # dD/dx
        transD_inv = numpy.log # x = transD_inv(D)
    else:
        transD = lambda x: x
        transD_deriv = lambda x: 1
        transD_inv = lambda x: x

    if S_as_exp:
        transS = numpy.exp
        transS_deriv = numpy.exp
        transS_inv = numpy.log
    else:
        transS = lambda x: x
        transS_deriv = lambda x: 1
        transS_inv = lambda x: x
    
    # Initial values
    for lab in D_labs: hkldata.binned_df[lab] = 1.
    hkldata.binned_df["S"] = 10000.
    for i_bin, _ in hkldata.binned():
        if use == "all":
            idxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin] for i in (1,2)])
        else:
            i = 1 if use == "work" else 2
            idxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin]])
        valid_sel = numpy.isfinite(hkldata.df.FP[idxes]) # as there is no nan-safe numpy.corrcoef
        idxes = idxes[valid_sel]
        FC = numpy.abs(hkldata.df.FC.to_numpy()[idxes])
        FP = hkldata.df.FP.to_numpy()[idxes]
        D = numpy.corrcoef(FP, FC)[1,0]
        hkldata.binned_df.loc[i_bin, D_labs[0]] = D
        hkldata.binned_df.loc[i_bin, "S"] = numpy.var(FP - D * FC)

    for D_lab in D_labs:
        if hkldata.binned_df[D_lab].min() <= 0:
            min_D = hkldata.binned_df[D_lab][hkldata.binned_df[D_lab] > 0].min() * 0.1
            logger.writeln("WARNING: negative {} is detected from initial estimates. Replacing it using minimum positive value {:.2e}".format(D_lab, min_D))
            hkldata.binned_df[D_lab].where(hkldata.binned_df[D_lab] > 0, min_D, inplace=True) # arbitrary
        
    logger.writeln("Initial estimates:")
    logger.writeln(hkldata.binned_df.to_string())
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    for i_bin, idxes in hkldata.binned():
        x0 = [transD_inv(hkldata.binned_df[lab][i_bin]) for lab in D_labs] + [transS_inv(hkldata.binned_df.S[i_bin])] #+ [0,0,0,0,0,0]
        def target(x):
            DFc = (transD(x[:len(fc_labs)]) * hkldata.df.loc[idxes, fc_labs]).sum(axis=1)
            ll = ext.ll_int(hkldata.df.I[idxes], hkldata.df.SIGI[idxes], k_ani[idxes], transS(x[-1]) * hkldata.df.epsilon[idxes],
                            numpy.abs(DFc), hkldata.df.centric[idxes]+1)
            return numpy.nansum(ll)
        def grad(x):
            r = ext.ll_int_der1_params(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes], transS(x[-1]),
                                       hkldata.df[fc_labs].to_numpy()[idxes], transD(x[:len(fc_labs)]),
                                       hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes])
            g = numpy.zeros(len(fc_labs)+1)
            g[:len(fc_labs)] = numpy.nansum(r[:,:len(fc_labs)], axis=0) # D
            g[-1] = numpy.nansum(r[:,-2]) # S
            g[:len(fc_labs)] *= transD_deriv(x[:len(fc_labs)])
            g[-1] *= transS_deriv(x[-1])
            return g

        print("Bin", i_bin)
        res = scipy.optimize.minimize(fun=target, x0=x0, jac=grad)
        print(res)
        
        for i, lab in enumerate(D_labs):
            hkldata.binned_df.loc[i_bin, lab] = transD(res.x[i])
        hkldata.binned_df.loc[i_bin, "S"] = transS(res.x[-1])

    logger.writeln("Refined estimates:")
    logger.writeln(hkldata.binned_df.to_string())

    # Refine b_aniso
    adpdirs = utils.model.adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=True)
    SMattolist = lambda B: [B.u11, B.u22, B.u33, B.u12, B.u13, B.u23]

    def target_ani(x):
        b_aniso = gemmi.SMat33d(*numpy.dot(x, adpdirs))
        k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
        ret = 0.
        for i_bin, idxes in hkldata.binned():
            Ds = [hkldata.binned_df[lab][i_bin] for lab in D_labs]
            Fcs = [hkldata.df[lab].to_numpy()[idxes] for lab in fc_labs]
            DFc = calc_DFc(Ds, Fcs)
            ll = ext.ll_int(hkldata.df.I[idxes], hkldata.df.SIGI[idxes], k_ani[idxes],
                            hkldata.binned_df.S[i_bin] * hkldata.df.epsilon[idxes],
                            numpy.abs(DFc), hkldata.df.centric[idxes]+1)
            ret += numpy.nansum(ll)
        return ret
    
    def shift_ani(x):
        b_aniso = gemmi.SMat33d(*numpy.dot(x, adpdirs))
        k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
        S2mat = hkldata.ssq_mat() # ssqmat
        g = numpy.zeros(6)
        H = numpy.zeros((6, 6))
        for i_bin, idxes in hkldata.binned():
            r = ext.ll_int_der1_params(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes],
                                       k_ani[idxes], hkldata.binned_df.S[i_bin],
                                       hkldata.df[fc_labs].to_numpy()[idxes], hkldata.binned_df.loc[i_bin, D_labs],
                                       hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes])
            S2 = S2mat[:,idxes]
            g += -numpy.nansum(S2 * r[:,-1] * k_ani[idxes], axis=1)
            H += numpy.nansum(numpy.matmul(S2[None,:].T, S2.T[:,None]) * ((r[:,-1] * k_ani[idxes])**2)[:,None,None], axis=0)
            
        g, H = numpy.dot(g, adpdirs.T), numpy.dot(adpdirs, numpy.dot(H, adpdirs.T))
        return -numpy.dot(g, numpy.linalg.pinv(H))

    logger.writeln("Refining B_aniso. Current = {}".format(b_aniso))
    B_converged = False
    for j in range(10):
        x = numpy.dot(SMattolist(b_aniso), numpy.linalg.pinv(adpdirs))
        f0 = target_ani(x)
        shift = shift_ani(x)
        for i in range(3):
            ss = shift / 2**i
            f1 = target_ani(x + ss)
            logger.writeln("{:2d} f0 = {:.3e} shift = {} f1 = {:.3e} dec? {}".format(j, f0, ss, f1, f1 < f0))
            if f1 < f0:
                b_aniso = gemmi.SMat33d(*numpy.dot(x+ss, adpdirs))
                if numpy.max(numpy.abs(ss)) < 1e-4: B_converged = True
                break
        else:
            B_converged = True
        if B_converged: break

    logger.writeln("Refined B_aniso = {}".format(b_aniso))

    return b_aniso
# determine_mli_params()

def calculate_maps_int(hkldata, b_aniso, fc_labs, D_labs, centric_and_selections, use="all"):
    nmodels = len(fc_labs)
    hkldata.df["FWT"] = 0j
    hkldata.df["DELFWT"] = 0j
    hkldata.df["FOM"] = numpy.nan
    Io = hkldata.df.I.to_numpy()
    sigIo = hkldata.df.SIGI.to_numpy()
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    eps = hkldata.df.epsilon.to_numpy()
    for i_bin, idxes in hkldata.binned():
        Ds = [max(0., hkldata.binned_df[lab][i_bin]) for lab in D_labs] # negative D is replaced with zero here
        S = hkldata.binned_df.S[i_bin]
        for c, work, test in centric_and_selections[i_bin]:
            if use == "all":
                cidxes = numpy.concatenate([work, test])
            else:
                cidxes = work if use == "work" else test
            if c == 0: # acentric
                k_num, k_den = 0.5, 0.
            else:
                k_num, k_den = 0., -0.5
            Fcs = [hkldata.df[lab].to_numpy()[cidxes] for lab in fc_labs]
            #DFc = (hkldata.binned_df.loc[i_bin, D_labs] * hkldata.df.loc[cidxes, fc_labs]).sum(axis=1)#.to_numpy() does not work
            DFc = calc_DFc(Ds, Fcs)
            to = Io[cidxes] / sigIo[cidxes] - sigIo[cidxes] / (c+1) / k_ani[cidxes]**2 / S / eps[cidxes]
            tf = k_ani[cidxes] * numpy.abs(DFc) / numpy.sqrt(sigIo[cidxes])
            sig1 = numpy.sqrt(k_ani[cidxes]) * S / sigIo[cidxes]
            f = ext.integ_J_ratio(k_num, k_den, True, to, tf, sig1, c+1) * numpy.sqrt(sigIo[cidxes]) / k_ani[cidxes]
            exp_ip = numpy.exp(numpy.angle(DFc)*1j)
            hkldata.df.loc[cidxes, "FWT"] = 2 * f * exp_ip - DFc
            hkldata.df.loc[cidxes, "DELFWT"] = f * exp_ip - DFc
# calculate_maps_int()

def merge_models(sts): # simply merge models. no fix in chain ids etc.
    model = gemmi.Model("1")
    for st in sts:
        for m in st:
            for c in m:
                model.add_chain(c)
    return model
# merge_models()

def process_input(hklin, labin, n_bins, free, xyzins, source, d_max=None, d_min=None):
    assert 1 < len(labin) < 4
    mtz = gemmi.read_mtz_file(hklin)
    logger.writeln("Input mtz: {}".format(hklin))
    logger.writeln("    Unit cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*mtz.cell.parameters))
    logger.writeln("  Space group: {}".format(mtz.spacegroup.hm))
    logger.writeln("")
    
    sts = [utils.fileio.read_structure(f) for f in xyzins]
    if sts:
        assert source in ["electron", "xray", "neutron"]
        logger.writeln("From model 1:")
        logger.writeln("    Unit cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*sts[0].cell.parameters))
        logger.writeln("  Space group: {}".format(sts[0].spacegroup_hm))
        logger.writeln("")
    
        if not mtz.cell.approx(sts[0].cell, 1e-3):
            logger.writeln("Warning: unit cell mismatch between model and mtz")
            logger.writeln("         using unit cell from mtz")

        for st in sts: st.cell = mtz.cell # mtz cell is used in any case

        sg_st = sts[0].find_spacegroup() # may be None
        sg_use = mtz.spacegroup
        if mtz.spacegroup != sg_st:
            logger.writeln("Warning: space group mismatch between model and mtz")
            if sg_st and mtz.spacegroup.point_group_hm() == sg_st.point_group_hm():
                logger.writeln("         using space group from model")
                sg_use = sg_st
            else:
                logger.writeln("         using space group from mtz")
            logger.writeln("")

        for st in sts: st.spacegroup_hm = sg_use.hm
        mtz.spacegroup = sg_use
        
    col_types = {x.label:x.type for x in mtz.columns}
    if labin[0] not in col_types:
        raise RuntimeError("MTZ coulumn not found: {}".format(labin[0]))

    if col_types[labin[0]] == "F":
        logger.writeln("Observation type: amplitude")
        newlabels = ["FP","SIGFP"]
        require_types = ["F", "Q"]
    elif col_types[labin[0]] == "J":
        logger.writeln("Observation type: intensity")
        newlabels = ["I","SIGI"]
        require_types = ["J", "Q"]
    else:
        raise RuntimeError("MTZ column {} is neither amplitude nor intensity".format(labin[0]))

        
    if len(labin) == 3: newlabels.append("FREE")
    hkldata = utils.hkl.hkldata_from_mtz(mtz, labin, newlabels=newlabels, require_types=require_types)
    hkldata.remove_nonpositive(newlabels[1])
    hkldata.switch_to_asu()
    #hkldata.df = hkldata.df.astype({name: 'float64' for name in ["I","SIGI"]})

    if (d_min, d_max).count(None) != 2:
        hkldata = hkldata.copy(d_min=d_min, d_max=d_max)
    d_min, d_max = hkldata.d_min_max()
        
    hkldata.complete()
    hkldata.sort_by_resolution()
    hkldata.calc_epsilon()
    hkldata.calc_centric()
    hkldata.setup_binning(n_bins=n_bins)
    logger.writeln("Data completeness: {:.2f}%".format(hkldata.completeness()*100.))

    fc_labs = []
    for i, st in enumerate(sts):
        lab = "FC{}".format(i)
        hkldata.df[lab] = utils.model.calc_fc_fft(st, d_min-1e-6,
                                                  source=source, mott_bethe=(source=="electron"),
                                                  miller_array=hkldata.miller_array())
        fc_labs.append(lab)

    # Create a centric selection table for faster look up
    centric_and_selections = {}
    stats = hkldata.binned_df.copy()
    stats["n_all"] = 0
    stats["n_obs"] = 0
    if "FREE" in hkldata.df:
        stats["n_work"] = 0
        stats["n_test"] = 0
        
    for i_bin, idxes in hkldata.binned():
        centric_and_selections[i_bin] = []
        n_obs = 0
        n_work, n_test = 0, 0
        for c, g2 in hkldata.df.loc[idxes].groupby("centric", sort=False):
            valid_sel = numpy.isfinite(g2[newlabels[0]])
            if "FREE" in g2:
                test_sel = (g2.FREE == free).fillna(False)
                test = g2.index[test_sel]
                work = g2.index[~test_sel]
                n_work += (valid_sel & ~test_sel).sum()
                n_test += (valid_sel & test_sel).sum()
            else:
                work = g2.index
                test = type(work)([])
            centric_and_selections[i_bin].append((c, work, test))
            n_obs += numpy.sum(valid_sel)
            
        stats.loc[i_bin, "n_obs"] = n_obs
        stats.loc[i_bin, "n_all"] = len(idxes)
        if "FREE" in hkldata.df:
            stats.loc[i_bin, "n_work"] = n_work
            stats.loc[i_bin, "n_test"] = n_test
            
    stats["completeness"] = stats["n_obs"] / stats["n_all"] * 100
    logger.writeln(stats.to_string())
    return hkldata, sts, fc_labs, centric_and_selections
# process_input()

def bulk_solvent_and_lsq_scales(hkldata, sts, fc_labs, use_solvent=True):
    scaling = gemmi.Scaling(hkldata.cell, hkldata.sg)
    scaling.use_solvent = use_solvent
    scaleto = hkldata.as_asu_data(label="FP", label_sigma="SIGFP")
    scaleto.value_array["sigma"] = 1. # I guess this would be better.
    fc_asu_total = hkldata.as_asu_data(data=hkldata.df[fc_labs].sum(axis=1).to_numpy())
    if not use_solvent:
        logger.writeln("Scaling Fc with no bulk solvent contribution")
        scaling.prepare_points(fc_asu_total, scaleto)
    else:
        logger.writeln("Calculating solvent contribution..")
        d_min = hkldata.d_min_max()[0] - 1e-6
        grid = gemmi.FloatGrid()
        spacing = min(1 / (2 * x / d_min + 1) / xr for x, xr in zip(sts[0].cell.parameters[:3],
                                                                    sts[0].cell.reciprocal().parameters[:3]))
        grid.setup_from(sts[0], spacing=min(0.4, spacing))
        masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)
        masker.put_mask_on_float_grid(grid, merge_models(sts))
        fmask_gr = gemmi.transform_map_to_f_phi(grid)
        hkldata.df["Fmask"] = fmask_gr.get_value_by_hkl(hkldata.miller_array())
        fmask_asu = hkldata.as_asu_data("Fmask")
        scaling.prepare_points(fc_asu_total, scaleto, fmask_asu)

    scaling.fit_isotropic_b_approximately()
    logger.writeln(" initial k,b = {:.2e} {:.2e}".format(scaling.k_overall, scaling.b_overall.u11))
    scaling.fit_parameters()
    b_aniso = scaling.b_overall
    b_iso = b_aniso.trace() / 3
    b_aniso = b_aniso.added_kI(-b_iso) # subtract isotropic contribution
    logger.writeln(" k_ov= {:.2e} B_iso= {:.2e} B_aniso= {}".format(scaling.k_overall, b_iso, b_aniso))
    k_iso = hkldata.debye_waller_factors(b_iso=b_iso)
    k_aniso = hkldata.debye_waller_factors(b_cart=b_aniso)
    hkldata.df["k_aniso"] = k_aniso # we need it later when calculating stats
    
    if use_solvent:
        fc_labs.append("Fbulk")
        solvent_scale = scaling.get_solvent_scale(0.25 / hkldata.d_spacings()**2)
        hkldata.df[fc_labs[-1]] = hkldata.df.Fmask * solvent_scale

    # Apply scales.
    #  - k_aniso^-1 is applied to FP (isotropize), 
    #    but k_aniso should be applied to FC when calculating R or CC
    #  - k_iso should be applied to FC
    hkldata.df.FP /= scaling.k_overall * k_aniso
    hkldata.df.SIGFP /= scaling.k_overall * k_aniso
    for lab in fc_labs: hkldata.df[lab] *= k_iso
    
    # total Fc
    hkldata.df["FC"] = hkldata.df[fc_labs].sum(axis=1)
    
    return scaling.k_overall, b_aniso
# bulk_solvent_and_lsq_scales()

def calculate_maps(hkldata, centric_and_selections, fc_labs, D_labs, log_out):
    nmodels = len(fc_labs)
    hkldata.df["FWT"] = 0j
    hkldata.df["DELFWT"] = 0j
    hkldata.df["FOM"] = numpy.nan
    hkldata.df["X"] = numpy.nan # for FOM
    stats_data = []
    for i_bin, _ in hkldata.binned():
        idxes = numpy.concatenate([sel[1] for sel in centric_and_selections[i_bin]]) # w/o missing reflections
        bin_d_min = hkldata.binned_df.d_min[i_bin]
        bin_d_max = hkldata.binned_df.d_max[i_bin]
        Ds = [max(0., hkldata.binned_df[lab][i_bin]) for lab in D_labs] # negative D is replaced with zero here
        DFcs = [numpy.log(Ds[i] * numpy.nanmean(numpy.abs(hkldata.df[lab].to_numpy()[idxes])))
                for i, lab in enumerate(fc_labs)]
        S = hkldata.binned_df.S[i_bin]
        
        # 0: acentric 1: centric
        mean_fom = [0, 0]
        nrefs = [0, 0]
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Fcs = [hkldata.df[lab].to_numpy()[cidxes] for lab in fc_labs]

            Fc = numpy.abs(hkldata.df.FC.to_numpy()[cidxes])
            phic = numpy.angle(hkldata.df.FC.to_numpy()[cidxes])
            expip = numpy.cos(phic) + 1j*numpy.sin(phic)
            Fo = hkldata.df.FP.to_numpy()[cidxes]
            SigFo = hkldata.df.SIGFP.to_numpy()[cidxes]
            epsilon = hkldata.df.epsilon.to_numpy()[cidxes]
            nrefs[c] = len(cidxes)

            DFc = calc_abs_DFc(Ds, Fcs)
            if c == 0:
                Sigma = 2 * SigFo**2 + epsilon * S
                X = 2 * Fo * DFc / Sigma
                m = gemmi.bessel_i1_over_i0(X)
                hkldata.df.loc[cidxes, "FWT"] = (2 * m * Fo - DFc) * expip
            else:
                Sigma = SigFo**2 + epsilon * S
                X = Fo * DFc / Sigma
                m = numpy.tanh(X)
                hkldata.df.loc[cidxes, "FWT"] = (m * Fo) * expip

            hkldata.df.loc[cidxes, "DELFWT"] = (m * Fo - DFc) * expip
            hkldata.df.loc[cidxes, "FOM"] = m
            hkldata.df.loc[cidxes, "X"] = X
            mean_fom[c] = numpy.nanmean(m)
            
            # Fill missing
            hkldata.df.loc[nidxes, "FWT"] = sum(Ds[i] * hkldata.df[lab].to_numpy()[nidxes] for i, lab in enumerate(fc_labs))

        k = hkldata.df.k_aniso.to_numpy()[idxes]
        Fc = hkldata.df.FC.to_numpy()[idxes] * k
        Fcs = [hkldata.df[lab].to_numpy()[idxes] * k for lab in fc_labs]
        Fo = hkldata.df.FP.to_numpy()[idxes] * k
        DFc = calc_abs_DFc(Ds, Fcs)
        r = numpy.nansum(numpy.abs(numpy.abs(Fc)-Fo)) / numpy.nansum(Fo)
        valid_sel = numpy.isfinite(Fo)
        cc = numpy.corrcoef(numpy.abs(Fc[valid_sel]), Fo[valid_sel])[1,0]
        stats_data.append([1/bin_d_min**2, i_bin, nrefs[0], nrefs[1], bin_d_max, bin_d_min,
                           numpy.log(numpy.nanmean(numpy.abs(Fo)**2)),
                           numpy.log(numpy.nanmean(numpy.abs(Fc)**2)),
                           numpy.log(numpy.nanmean(DFc**2)),
                           numpy.log(S), mean_fom[0], mean_fom[1], r, cc] + Ds + DFcs)

    s2lab = "1/resol^2"
    DFc_labs = ["log(Mn(|{}{}|))".format(dl,fl) for dl,fl in zip(D_labs, fc_labs)]
    cols = [s2lab, "bin", "n_a", "n_c", "d_max", "d_min",
            "log(Mn(|Fo|^2))", "log(Mn(|Fc|^2))", "log(Mn(|DFc|^2))",
            "log(Sigma)", "FOM_a", "FOM_c", "R", "CC(|Fo|,|Fc|)"] + D_labs + DFc_labs
    stats = pandas.DataFrame(stats_data, columns=cols)
    title_labs = [["log(Mn(|F|^2)) and variances", [s2lab, "log(Mn(|Fo|^2))", "log(Mn(|Fc|^2))", "log(Mn(|DFc|^2))", "log(Sigma)"]],
                  ["FOM", [s2lab, "FOM_a", "FOM_c"]],
                  ["D", [s2lab] + D_labs],
                  ["DFc", [s2lab] + DFc_labs],
                  ["R-factor", [s2lab, "R"]],
                  ["CC", [s2lab, "CC(|Fo|,|Fc|)"]],
                  ["number of reflections", [s2lab, "n_a", "n_c"]]]
    with open(log_out, "w") as ofs:
        ofs.write(utils.make_loggraph_str(stats, main_title="Statistics",
                                          title_labs=title_labs))
    logger.writeln("output log: {}".format(log_out))
# calculate_maps()

def main(args):
    if args.nbins < 1:
        raise SystemExit("--nbins must be > 0")

    hkldata, sts, fc_labs, centric_and_selections = process_input(hklin=args.hklin,
                                                                  labin=args.labin.split(","),
                                                                  n_bins=args.nbins,
                                                                  free=args.free,
                                                                  xyzins=sum(args.model, []),
                                                                  source=args.source,
                                                                  d_min=args.d_min)
    is_int = "I" in hkldata.df
    if is_int:
        #from servalcat.xtal import french_wilson as fw
        #B_aniso = fw.determine_Sigma_and_aniso(hkldata, centric_and_selections)
        #fw.french_wilson(hkldata, centric_and_selections, B_aniso, labout=["FP", "SIGFP"])
        # at the moment we need FP just for bulk solvent and scales
        hkldata.df["FP"] = numpy.sqrt(numpy.where(hkldata.df.I > 0, hkldata.df.I, 0))
        hkldata.df["SIGFP"] = 0.5 * hkldata.df.SIGI / numpy.where(hkldata.df.I > 0, hkldata.df.FP, numpy.nan)

    # Overall scaling & bulk solvent
    # FP/SIGFP will be scaled. Total FC will be added.
    k_overall, b_aniso = bulk_solvent_and_lsq_scales(hkldata, sts, fc_labs, use_solvent=not args.no_solvent)
    if is_int:
        # in intensity case, we try to refine b_aniso with ML. perhaps we should do it in amplitude case also
        hkldata.df.I /= k_overall**2
        hkldata.df.SIGI /= k_overall**2
    
    # Show R and CC
    if not is_int:
        fpa, fca, k = hkldata.as_numpy_arrays(["FP", "FC", "k_aniso"])
        fpa *= k
        fca = numpy.abs(fca) * k
        logger.writeln(" CC(Fo,Fc)= {:.4f}".format(numpy.corrcoef(fca, fpa)[0,1]))
        logger.writeln(" Rcryst= {:.4f}".format(utils.hkl.r_factor(fpa, fca)))

    # Estimate ML parameters
    D_labs = ["D{}".format(i) for i in range(len(fc_labs))]

    if is_int:
        assert not args.use_cc
        logger.writeln("Estimating sigma-A parameters using ML..")
        b_aniso = determine_mli_params(hkldata, fc_labs, D_labs, b_aniso, centric_and_selections, args.D_as_exp, args.S_as_exp, args.use)
        calculate_maps_int(hkldata, b_aniso, fc_labs, D_labs, centric_and_selections)
    else:
        if args.use_cc:
            logger.writeln("Estimating sigma-A parameters from CC..")
            determine_mlf_params_from_cc(hkldata, fc_labs, D_labs, centric_and_selections, args.use)
        else:
            logger.writeln("Estimating sigma-A parameters using ML..")
            determine_mlf_params(hkldata, fc_labs, D_labs, centric_and_selections, args.D_as_exp, args.S_as_exp, args.use)

        # Calculate maps
        log_out = "{}.log".format(args.output_prefix)
        calculate_maps(hkldata, centric_and_selections, fc_labs, D_labs, log_out)

    # Write mtz file
    if is_int:
        labs = ["I", "SIGI"]
    else:
        labs = ["FP", "SIGFP", "FOM"]
    labs.extend(["FWT", "DELFWT", "FC"])
    if not args.no_solvent:
        labs.append("Fbulk")
        labs.append("Fmask")
    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=labs, types={"FOM": "W", "FP":"F", "SIGFP":"Q"})
    logger.writeln("output mtz: {}".format(mtz_out))

    return hkldata
# main()
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
