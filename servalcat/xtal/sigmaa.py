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
    parser.add_argument('--nbins', type=int,
                        help="Number of bins (default: auto)")
    parser.add_argument('-s', '--source', choices=["electron", "xray", "neutron"], required=True,
                        help="Scattering factor choice")
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

class VarTrans:
    def __init__(self, D_as_exp, S_as_exp):
        if D_as_exp:
            self.D = numpy.exp # D = D(x)
            self.D_deriv = numpy.exp # dD/dx
            self.D_inv = numpy.log # x = D_inv(D)
        else:
            self.D = lambda x: x
            self.D_deriv = lambda x: 1
            self.D_inv = lambda x: x

        if S_as_exp:
            self.S = numpy.exp
            self.S_deriv = numpy.exp
            self.S_inv = numpy.log
        else:
            self.S = lambda x: x
            self.S_deriv = lambda x: 1
            self.S_inv = lambda x: x
# class VarTrans

class LsqScale:
    # parameter x = [k_overall, adp_pars, k_sol, B_sol]
    def __init__(self, hkldata, obs, fc_list, use_int=False, k_as_exp=False):
        assert 0 < len(fc_list) < 3
        self.use_int = use_int
        self.obs = obs
        self.calc = fc_list
        self.s2mat = hkldata.ssq_mat()
        self.s2 = 1. / hkldata.d_spacings().to_numpy()**2
        self.adpdirs = utils.model.adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=False)
        self.k_trans = lambda x: numpy.exp(x) if k_as_exp else x
        self.k_trans_der = lambda x: numpy.exp(x) if k_as_exp else 1
        self.k_trans_inv = lambda x: numpy.log(x) if k_as_exp else x
        self.k_sol_ini = 0.35 # same default as gemmi/scaling.hpp
        self.b_sol_ini = 46.
        if use_int:
            self.sqrt_obs = numpy.sqrt(numpy.maximum(self.obs, 0.1))
        
    def get_solvent_scale(self, k_sol, b_sol):
        return k_sol * numpy.exp(-b_sol * self.s2 / 4)
    
    def scaled_fc(self, x):
        fc0 = self.calc[0]
        if len(self.calc) == 2:
            fmask = self.calc[1]
            fbulk = self.get_solvent_scale(x[-2], x[-1]) * fmask
            fc = fc0 + fbulk
        else:
            fc = fc0
        nadp = self.adpdirs.shape[0]
        B = numpy.dot(x[1:nadp+1], self.adpdirs)
        kani = numpy.exp(numpy.dot(-B, self.s2mat))
        return self.k_trans(x[0]) * kani * fc

    def target(self, x):
        y = numpy.abs(self.scaled_fc(x))
        if self.use_int:
            y2 = y**2
            return numpy.nansum(((self.obs - y2) / (self.sqrt_obs + y))**2)
        else:
            return numpy.nansum((self.obs - y)**2)
        
    def grad(self, x):
        g = numpy.zeros_like(x)
        fc0 = self.calc[0]
        if len(self.calc) == 2:
            fmask = self.calc[1]
            temp_sol = numpy.exp(-x[-1] * self.s2 / 4)
            fbulk = x[-2] * temp_sol * fmask
            fc = fc0 + fbulk
        else:
            fc = fc0
        nadp = self.adpdirs.shape[0]
        B = numpy.dot(x[1:nadp+1], self.adpdirs)
        kani = numpy.exp(numpy.dot(-B, self.s2mat))
        fc_abs = numpy.abs(fc)
        k = self.k_trans(x[0])
        y = k * kani * fc_abs
        if self.use_int:
            y2 = y**2
            t1 = self.obs - y2
            t2 = self.sqrt_obs + y
            dfdy = -2 * t1 * (self.obs + y * (2 * self.sqrt_obs + y)) / t2**3
        else:
            dfdy = -2 * (self.obs - y)
        dfdb = numpy.nansum(-self.s2mat * k * fc_abs * kani * dfdy, axis=1)
        g[0] = numpy.nansum(kani * fc_abs * dfdy * self.k_trans_der(x[0]))
        g[1:nadp+1] = numpy.dot(dfdb, self.adpdirs.T)
        if len(self.calc) == 2:
            re_fmask_fcconj = (fmask * fc.conj()).real
            tmp = k * kani * temp_sol / fc_abs * re_fmask_fcconj
            g[-2] = numpy.nansum(tmp * dfdy)
            g[-1] = numpy.nansum(-tmp * dfdy * x[-2] * self.s2 / 4)
                
        return g

    def initial_kb(self):
        fc0 = self.calc[0]
        if len(self.calc) == 2:
            fmask = self.calc[1]
            fbulk = self.get_solvent_scale(self.k_sol_ini, self.b_sol_ini) * fmask
            fc = fc0 + fbulk
        else:
            fc = fc0
        sel = self.obs > 0
        f1p, f2p, s2p = self.obs[sel], numpy.abs(fc)[sel], self.s2[sel]
        if self.use_int: f2p *= f2p
        tmp = numpy.log(f2p) - numpy.log(f1p)
        # g = [dT/dk, dT/db]
        g = numpy.array([2 * numpy.sum(tmp), -numpy.sum(tmp*s2p)/2])
        H = numpy.zeros((2,2))
        H[0,0] = 2*len(f1p)
        H[1,1] = numpy.sum(s2p**2/8)
        H[0,1] = H[1,0] = -numpy.sum(s2p)/2
        x = -numpy.dot(numpy.linalg.inv(H), g)
        if self.use_int: x /= 2
        k = numpy.exp(x[0])
        b = x[1]
        logger.writeln(" initial k,b = {:.2e} {:.2e}".format(k, b))
        logger.writeln("           R = {:.4f}".format(utils.hkl.r_factor(f1p, f2p * k * numpy.exp(-b*self.s2[sel]/4))))
        return k, b
    
    def scale(self):
        use_sol = len(self.calc) == 2
        msg = "Scaling Fc to {} {} bulk solvent contribution".format("Io" if self.use_int else "Fo",
                                                                     "with" if use_sol else "without")
        logger.writeln(msg)
        k, b = self.initial_kb()
        x0 = [self.k_trans_inv(k)]
        x0.extend(numpy.dot([b,b,b,0,0,0], self.adpdirs.T))
        if use_sol:
            x0.extend([self.k_sol_ini, self.b_sol_ini])
        if 0:
            f0 = self.target(x0)
            ader = self.grad(x0)
            e = 1e-2
            nder = []
            for i in range(len(x0)):
                x = numpy.copy(x0)
                x[i] += e
                f1 = self.target(x)
                nder.append((f1 - f0) / e)
            print("ADER NDER RATIO")
            print(ader)
            print(nder)
            print(ader / nder)
            quit()
            
        res = scipy.optimize.minimize(fun=self.target, x0=x0, jac=self.grad)
        #logger.writeln(str(res))
        self.k_overall = self.k_trans(res.x[0])
        nadp = self.adpdirs.shape[0]
        b_overall = gemmi.SMat33d(*numpy.dot(res.x[1:nadp+1], self.adpdirs))
        self.b_iso = b_overall.trace() / 3
        self.b_aniso = b_overall.added_kI(-self.b_iso) # subtract isotropic contribution

        logger.writeln(" k_ov= {:.2e} B_iso= {:.2e} B_aniso= {}".format(self.k_overall, self.b_iso, self.b_aniso))
        if use_sol:
            self.k_sol = res.x[-2] 
            self.b_sol = res.x[-1]
            logger.writeln(" k_sol= {:.2e} B_sol= {:.2e}".format(self.k_sol, self.b_sol))
        calc = numpy.abs(self.scaled_fc(res.x))
        if self.use_int: calc *= calc            
        logger.writeln(" CC = {:.4f}".format(utils.hkl.correlation(self.obs, calc)))
        logger.writeln(" R  = {:.4f}".format(utils.hkl.r_factor(self.obs, calc)))

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
    trans = VarTrans(D_as_exp, S_as_exp)
    
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
        x0 = [trans.D_inv(hkldata.binned_df[lab][i_bin]) for lab in D_labs] + [trans.S_inv(hkldata.binned_df.S[i_bin])]
        def target(x):
            return mlf(hkldata.df, fc_labs, trans.D(x[:-1]), trans.S(x[-1]), centric_and_selections[i_bin], use)
        def grad(x):
            g = deriv_mlf_wrt_D_S(hkldata.df, fc_labs, trans.D(x[:-1]), trans.S(x[-1]), centric_and_selections[i_bin], use)
            g[:-1] *= trans.D_deriv(x[:-1])
            g[-1] *= trans.S_deriv(x[-1])
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
            hkldata.binned_df.loc[i_bin, lab] = trans.D(res.x[i])
        hkldata.binned_df.loc[i_bin, "S"] = trans.S(res.x[-1])

    logger.writeln("Refined estimates:")
    logger.writeln(hkldata.binned_df.to_string())
    return D_labs
# determine_mlf_params()

def determine_mli_params(hkldata, fc_labs, D_labs, b_aniso, centric_and_selections,
                         D_as_exp=False, S_as_exp=False, use="all", n_cycle=1):
    assert use in ("all", "work", "test")
    trans = VarTrans(D_as_exp, S_as_exp)
    def get_idxes(i_bin):
        if use == "all":
            return numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin] for i in (1,2)])
        else:
            i = 1 if use == "work" else 2
            return numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin]])
    
    # Initial values
    for lab in D_labs: hkldata.binned_df[lab] = 1.
    hkldata.binned_df["S"] = 10000.
    for i_bin, _ in hkldata.binned():
        idxes = get_idxes(i_bin)
        valid_sel = numpy.isfinite(hkldata.df.I[idxes]) # as there is no nan-safe numpy.corrcoef
        idxes = idxes[valid_sel]
        FC = numpy.abs(hkldata.df.FC.to_numpy()[idxes])
        I = hkldata.df.I.to_numpy()[idxes]
        D = numpy.corrcoef(I, FC**2)[1,0]
        hkldata.binned_df.loc[i_bin, D_labs[0]] = D
        hkldata.binned_df.loc[i_bin, "S"] = numpy.sqrt(numpy.var(I - (D * FC)**2)) / 4 * (2-D)

    for D_lab in D_labs:
        if hkldata.binned_df[D_lab].min() <= 0:
            min_D = hkldata.binned_df[D_lab][hkldata.binned_df[D_lab] > 0].min() * 0.1
            logger.writeln("WARNING: negative {} is detected from initial estimates. Replacing it using minimum positive value {:.2e}".format(D_lab, min_D))
            hkldata.binned_df[D_lab].where(hkldata.binned_df[D_lab] > 0, min_D, inplace=True) # arbitrary
        
    logger.writeln("Initial estimates:")
    logger.writeln(hkldata.binned_df.to_string())
    for i_cyc in range(n_cycle):
        k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
        for i_bin, _ in hkldata.binned():
            idxes = get_idxes(i_bin)
            def target(x):
                DFc = (trans.D(x[:len(fc_labs)]) * hkldata.df.loc[idxes, fc_labs]).sum(axis=1)
                ll = ext.ll_int(hkldata.df.I[idxes], hkldata.df.SIGI[idxes], k_ani[idxes], trans.S(x[-1]) * hkldata.df.epsilon[idxes],
                                numpy.abs(DFc), hkldata.df.centric[idxes]+1)
                return numpy.nansum(ll)
            def grad(x):
                r = ext.ll_int_der1_DS(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes], trans.S(x[-1]),
                                       hkldata.df[fc_labs].to_numpy()[idxes], trans.D(x[:len(fc_labs)]),
                                       hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes])
                g = numpy.zeros(len(fc_labs)+1)
                g[:len(fc_labs)] = numpy.nansum(r[:,:len(fc_labs)], axis=0) # D
                g[-1] = numpy.nansum(r[:,-1]) # S
                g[:len(fc_labs)] *= trans.D_deriv(x[:len(fc_labs)])
                g[-1] *= trans.S_deriv(x[-1])
                return g

            def shift_DS(x):
                r = ext.ll_int_der1_DS(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes], trans.S(x[-1]),
                                       hkldata.df[fc_labs].to_numpy()[idxes], trans.D(x[:len(fc_labs)]),
                                       hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes])
                g = numpy.zeros(len(fc_labs)+1)
                g[:len(fc_labs)] = numpy.nansum(r[:,:len(fc_labs)], axis=0) * trans.D_deriv(x[:len(fc_labs)]) # D
                g[-1] = numpy.nansum(r[:,-1]) * trans.S_deriv(x[-1]) # S
                tmp = numpy.hstack([r[:,:len(fc_labs)] * trans.D_deriv(x[:len(fc_labs)]),
                                    r[:,-1,None] * trans.S_deriv(x[-1])])
                H = numpy.nansum(numpy.matmul(tmp[:,:,None], tmp[:,None]), axis=0)
                return -numpy.dot(g, numpy.linalg.pinv(H))

            #print("Bin", i_bin)
            if 1:
                x0 = [trans.D_inv(hkldata.binned_df[lab][i_bin]) for lab in D_labs] + [trans.S_inv(hkldata.binned_df.S[i_bin])]
                res = scipy.optimize.minimize(fun=target, x0=x0, jac=grad)
                #print(res)
                for i, lab in enumerate(D_labs):
                    hkldata.binned_df.loc[i_bin, lab] = trans.D(res.x[i])
                hkldata.binned_df.loc[i_bin, "S"] = trans.S(res.x[-1])
            else:
                DS_converged = False
                for j in range(10):
                    x = [trans.D_inv(hkldata.binned_df[lab][i_bin]) for lab in D_labs] + [trans.S_inv(hkldata.binned_df.S[i_bin])]
                    f0 = target(x)
                    shift = shift_DS(x)
                    for i in range(3):
                        ss = shift / 2**i
                        f1 = target(x + ss)
                        #logger.writeln("{:2d} f0 = {:.3e} shift = {} df = {:.3e}".format(j, f0, ss, f1 - f0))
                        if f1 < f0:
                            for i, lab in enumerate(D_labs):
                                hkldata.binned_df.loc[i_bin, lab] = trans.D((x+ss)[i])
                            hkldata.binned_df.loc[i_bin, "S"] = trans.S((x+ss)[-1])
                            if numpy.max(numpy.abs(ss)) < 1e-4: DS_converged = True
                            break
                    else:
                        DS_converged = True
                    if DS_converged: break

        logger.writeln("Refined estimates:")
        logger.writeln(hkldata.binned_df.to_string())

        # Refine b_aniso
        adpdirs = utils.model.adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=True)
        SMattolist = lambda B: [B.u11, B.u22, B.u33, B.u12, B.u13, B.u23]

        def target_ani(x):
            b_aniso = gemmi.SMat33d(*numpy.dot(x, adpdirs))
            k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
            ret = 0.
            for i_bin, _ in hkldata.binned():
                idxes = get_idxes(i_bin)
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
            for i_bin, _ in hkldata.binned():
                idxes = get_idxes(i_bin)
                r = ext.ll_int_der1_ani(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes],
                                        k_ani[idxes], hkldata.binned_df.S[i_bin],
                                        hkldata.df[fc_labs].to_numpy()[idxes], hkldata.binned_df.loc[i_bin, D_labs],
                                        hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes])
                S2 = S2mat[:,idxes]
                g += -numpy.nansum(S2 * r[:,0], axis=1) # k_ani is already multiplied in r
                H += numpy.nansum(numpy.matmul(S2[None,:].T, S2.T[:,None]) * (r[:,0]**2)[:,None,None], axis=0)

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
                #logger.writeln("{:2d} f0 = {:.3e} shift = {} df = {:.3e}".format(j, f0, ss, f1 - f0))
                if f1 < f0:
                    b_aniso = gemmi.SMat33d(*numpy.dot(x+ss, adpdirs))
                    if numpy.max(numpy.abs(ss)) < 1e-4: B_converged = True
                    break
            else:
                B_converged = True
            if B_converged: break

        logger.writeln("Refined B_aniso = {}".format(b_aniso))
        logger.writeln("cycle {} f= {}".format(i_cyc, f1))
    return b_aniso
# determine_mli_params()

def calculate_maps_int(hkldata, b_aniso, fc_labs, D_labs, centric_and_selections, use="all"):
    nmodels = len(fc_labs)
    hkldata.df["FWT"] = 0j * numpy.nan
    hkldata.df["DELFWT"] = 0j * numpy.nan
    hkldata.df["FOM"] = numpy.nan
    Io = hkldata.df.I.to_numpy()
    sigIo = hkldata.df.SIGI.to_numpy()
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    eps = hkldata.df.epsilon.to_numpy()
    for i_bin, idxes in hkldata.binned():
        Ds = [max(0., hkldata.binned_df[lab][i_bin]) for lab in D_labs] # negative D is replaced with zero here
        S = hkldata.binned_df.S[i_bin]
        for c, work, test in centric_and_selections[i_bin]:
            cidxes = numpy.concatenate([work, test])
            if c == 0: # acentric
                k_num, k_den = 0.5, 0.
            else:
                k_num, k_den = 0., -0.5
            Fcs = [hkldata.df[lab].to_numpy()[cidxes] for lab in fc_labs]
            #DFc = (hkldata.binned_df.loc[i_bin, D_labs] * hkldata.df.loc[cidxes, fc_labs]).sum(axis=1)#.to_numpy() does not work
            DFc = calc_DFc(Ds, Fcs)
            to = Io[cidxes] / sigIo[cidxes] - sigIo[cidxes] / (c+1) / k_ani[cidxes]**2 / S / eps[cidxes]
            tf = k_ani[cidxes] * numpy.abs(DFc) / numpy.sqrt(sigIo[cidxes])
            sig1 = k_ani[cidxes]**2 * S * eps[cidxes] / sigIo[cidxes]
            f = ext.integ_J_ratio(k_num, k_den, True, to, tf, sig1, c+1) * numpy.sqrt(sigIo[cidxes]) / k_ani[cidxes]
            exp_ip = numpy.exp(numpy.angle(DFc)*1j)
            hkldata.df.loc[cidxes, "FWT"] = 2 * f * exp_ip - DFc
            hkldata.df.loc[cidxes, "DELFWT"] = f * exp_ip - DFc

            # remove reflections that should be hidden
            if use != "all":
                # usually use == "work"
                tohide = test if use == "work" else work
                hkldata.df.loc[tohide, "FWT"] = 0j * numpy.nan
                hkldata.df.loc[tohide, "DELFWT"] = 0j * numpy.nan
            fill_sel = numpy.isnan(hkldata.df["FWT"][cidxes].to_numpy())
            hkldata.df.loc[cidxes[fill_sel], "FWT"] = DFc[fill_sel]
# calculate_maps_int()

def merge_models(sts): # simply merge models. no fix in chain ids etc.
    st = sts[0].clone()
    del st[:]
    model = gemmi.Model("1")
    for st in sts:
        for m in st:
            for c in m:
                model.add_chain(c)
    st.add_model(model)
    return st
# merge_models()

def process_input(hklin, labin, n_bins, free, xyzins, source, d_max=None, d_min=None,
                  n_per_bin=None, use="all", max_bins=None):
    assert 1 < len(labin) < 4
    assert use in ("all", "work", "test")
    assert n_bins or n_per_bin #if n_bins not set, n_per_bin should be given

    if utils.fileio.is_mmhkl_file(hklin):
        mtz = utils.fileio.read_mmhkl(hklin)
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
        sts = [utils.fileio.read_structure(f) for f in xyzins]
    else:
        assert len(xyzins) == 1
        st, hkldata = utils.fileio.read_small_molecule_files([hklin, xyzins[0]])
        sts = [st]
        newlabels = hkldata.columns()
    
    if sts:
        assert source in ["electron", "xray", "neutron"]
        if not hkldata.cell.approx(sts[0].cell, 1e-3):
            logger.writeln("Warning: unit cell mismatch between model and reflection data")
            logger.writeln("         using unit cell from mtz")

        for st in sts: st.cell = hkldata.cell # mtz cell is used in any case

        sg_st = sts[0].find_spacegroup() # may be None
        sg_use = hkldata.sg
        if hkldata.sg != sg_st:
            logger.writeln("Warning: space group mismatch between model and mtz")
            if sg_st and sg_st.laue_str() == hkldata.sg.laue_str():
                logger.writeln("         using space group from model")
                sg_use = sg_st
            else:
                logger.writeln("         using space group from mtz")
            logger.writeln("")

        for st in sts:
            st.spacegroup_hm = sg_use.xhm()
            st.setup_cell_images()
        hkldata.sg = sg_use
        
    hkldata.remove_nonpositive(newlabels[1])
    hkldata.switch_to_asu()
    hkldata.remove_systematic_absences()
    #hkldata.df = hkldata.df.astype({name: 'float64' for name in ["I","SIGI"]})

    if (d_min, d_max).count(None) != 2:
        hkldata = hkldata.copy(d_min=d_min, d_max=d_max)
    d_min, d_max = hkldata.d_min_max()
        
    hkldata.complete()
    hkldata.sort_by_resolution()
    hkldata.calc_epsilon()
    hkldata.calc_centric()

    if n_bins is None:
        sel = hkldata.df[newlabels[0]].notna()
        if use == "work":
            sel &= hkldata.df.FREE != free
        elif use == "test":
            sel &= hkldata.df.FREE == free
        n_bins = utils.hkl.decide_n_bins(n_per_bin, 1/hkldata.d_spacings()[sel], max_bins=max_bins)
        logger.writeln("n_per_bin={} requested for {}. n_bins set to {}".format(n_per_bin, use, n_bins))

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
    snr = "I/sigma" if newlabels[0] == "I" else "F/sigma"
    stats[snr] = 0.
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
                test = type(work)([], dtype=work.dtype)
            centric_and_selections[i_bin].append((c, work, test))
            n_obs += numpy.sum(valid_sel)
            
        stats.loc[i_bin, "n_obs"] = n_obs
        stats.loc[i_bin, "n_all"] = len(idxes)
        obs = hkldata.df[newlabels[0]].to_numpy()[idxes]
        sigma = hkldata.df[newlabels[1]].to_numpy()[idxes]
        stats.loc[i_bin, snr] = numpy.nanmean(obs / sigma)
        if "FREE" in hkldata.df:
            stats.loc[i_bin, "n_work"] = n_work
            stats.loc[i_bin, "n_test"] = n_test
            
    stats["completeness"] = stats["n_obs"] / stats["n_all"] * 100
    logger.writeln(stats.to_string())
    return hkldata, sts, fc_labs, centric_and_selections
# process_input()

def calc_Fmask(st, d_min, miller_array):
    logger.writeln("Calculating solvent contribution..")
    grid = gemmi.FloatGrid()
    spacing = min(1 / (2 * x / d_min + 1) / xr for x, xr in zip(st.cell.parameters[:3],
                                                                st.cell.reciprocal().parameters[:3]))
    grid.setup_from(st, spacing=min(0.4, spacing))
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)
    masker.put_mask_on_float_grid(grid, st[0])
    fmask_gr = gemmi.transform_map_to_f_phi(grid)
    Fmask = fmask_gr.get_value_by_hkl(miller_array)
    return Fmask
# calc_Fmask()

def bulk_solvent_and_lsq_scales(hkldata, sts, fc_labs, use_solvent=True, use_int=False):
    fc_list = [hkldata.df[fc_labs].sum(axis=1).to_numpy()]
    if use_solvent:
        Fmask = calc_Fmask(merge_models(sts), hkldata.d_min_max()[0] - 1e-6, hkldata.miller_array())
        fc_list.append(Fmask)

    obs = hkldata.df["I" if use_int else "FP"].to_numpy()
    scaling = LsqScale(hkldata, obs, fc_list, use_int)
    scaling.scale()
    b_aniso = scaling.b_aniso
    b_iso = scaling.b_iso
    k_iso = hkldata.debye_waller_factors(b_iso=b_iso)
    k_aniso = hkldata.debye_waller_factors(b_cart=b_aniso)
    hkldata.df["k_aniso"] = k_aniso # we need it later when calculating stats
    
    if use_solvent:
        fc_labs.append("Fbulk")
        solvent_scale = scaling.get_solvent_scale(scaling.k_sol, scaling.b_sol)
        hkldata.df[fc_labs[-1]] = Fmask * solvent_scale

    # Apply scales.
    #  - k_aniso^-1 is applied to FP (isotropize), 
    #    but k_aniso should be applied to FC when calculating R or CC
    #  - k_iso should be applied to FC
    if use_int:
        # in intensity case, we try to refine b_aniso with ML. perhaps we should do it in amplitude case also
        hkldata.df.I /= scaling.k_overall**2
        hkldata.df.SIGI /= scaling.k_overall**2
    else:
        hkldata.df.FP /= scaling.k_overall * k_aniso
        hkldata.df.SIGFP /= scaling.k_overall * k_aniso
    for lab in fc_labs: hkldata.df[lab] *= k_iso
    # total Fc
    hkldata.df["FC"] = hkldata.df[fc_labs].sum(axis=1)
    return scaling.k_overall, b_aniso
# bulk_solvent_and_lsq_scales()

def calculate_maps(hkldata, centric_and_selections, fc_labs, D_labs, log_out, use="all"):
    nmodels = len(fc_labs)
    hkldata.df["FWT"] = 0j * numpy.nan
    hkldata.df["DELFWT"] = 0j * numpy.nan
    hkldata.df["FOM"] = numpy.nan
    hkldata.df["X"] = numpy.nan # for FOM
    stats_data = []
    for i_bin, idxes in hkldata.binned():
        bin_d_min = hkldata.binned_df.d_min[i_bin]
        bin_d_max = hkldata.binned_df.d_max[i_bin]
        Ds = [max(0., hkldata.binned_df[lab][i_bin]) for lab in D_labs] # negative D is replaced with zero here
        DFcs = [numpy.log(Ds[i] * numpy.nanmean(numpy.abs(hkldata.df[lab].to_numpy()[idxes])))
                for i, lab in enumerate(fc_labs)]
        S = hkldata.binned_df.S[i_bin]
        
        # 0: acentric 1: centric
        mean_fom = [0, 0]
        nrefs = [0, 0]
        for c, work, test in centric_and_selections[i_bin]:
            cidxes = numpy.concatenate([work, test])
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

            # remove reflections that should be hidden
            if use != "all":
                # usually use == "work"
                tohide = test if use == "work" else work
                hkldata.df.loc[tohide, "FWT"] = 0j * numpy.nan
                hkldata.df.loc[tohide, "DELFWT"] = 0j * numpy.nan
            fill_sel = numpy.isnan(hkldata.df["FWT"][cidxes].to_numpy())
            hkldata.df.loc[cidxes[fill_sel], "FWT"] = (DFc * expip)[fill_sel]

        k = hkldata.df.k_aniso.to_numpy()[idxes]
        Fc = hkldata.df.FC.to_numpy()[idxes] * k
        Fcs = [hkldata.df[lab].to_numpy()[idxes] * k for lab in fc_labs]
        Fo = hkldata.df.FP.to_numpy()[idxes] * k
        DFc = calc_abs_DFc(Ds, Fcs)
        r = numpy.nansum(numpy.abs(numpy.abs(Fc)-Fo)) / numpy.nansum(Fo)
        cc = utils.hkl.correlation(Fo, numpy.abs(Fc))
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
    n_per_bin = {"all": 500, "work": 500, "test": 50}[args.use]
    hkldata, sts, fc_labs, centric_and_selections = process_input(hklin=args.hklin,
                                                                  labin=args.labin.split(","),
                                                                  n_bins=args.nbins,
                                                                  free=args.free,
                                                                  xyzins=sum(args.model, []),
                                                                  source=args.source,
                                                                  d_min=args.d_min,
                                                                  n_per_bin=n_per_bin,
                                                                  use=args.use,
                                                                  max_bins=30)
    is_int = "I" in hkldata.df
    
    # Overall scaling & bulk solvent
    # FP/SIGFP will be scaled. Total FC will be added.
    k_overall, b_aniso = bulk_solvent_and_lsq_scales(hkldata, sts, fc_labs, use_solvent=not args.no_solvent,
                                                     use_int=is_int)
    # Estimate ML parameters
    D_labs = ["D{}".format(i) for i in range(len(fc_labs))]

    if is_int:
        assert not args.use_cc
        logger.writeln("Estimating sigma-A parameters using ML..")
        b_aniso = determine_mli_params(hkldata, fc_labs, D_labs, b_aniso, centric_and_selections, args.D_as_exp, args.S_as_exp, args.use)
        calculate_maps_int(hkldata, b_aniso, fc_labs, D_labs, centric_and_selections,
                           use={"all": "all", "work": "work", "test": "work"}[args.use])
    else:
        if args.use_cc:
            logger.writeln("Estimating sigma-A parameters from CC..")
            determine_mlf_params_from_cc(hkldata, fc_labs, D_labs, centric_and_selections, args.use)
        else:
            logger.writeln("Estimating sigma-A parameters using ML..")
            determine_mlf_params(hkldata, fc_labs, D_labs, centric_and_selections, args.D_as_exp, args.S_as_exp, args.use)

        # Calculate maps
        log_out = "{}.log".format(args.output_prefix)
        calculate_maps(hkldata, centric_and_selections, fc_labs, D_labs, log_out,
                       use={"all": "all", "work": "work", "test": "work"}[args.use])

    # Write mtz file
    if is_int:
        labs = ["I", "SIGI"]
    else:
        labs = ["FP", "SIGFP", "FOM"]
    labs.extend(["FWT", "DELFWT", "FC"])
    if not args.no_solvent:
        labs.append("Fbulk")
    if "FREE" in hkldata.df:
        labs.append("FREE")
    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=labs, types={"FOM": "W", "FP":"F", "SIGFP":"Q"})
    return hkldata
# main()
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
