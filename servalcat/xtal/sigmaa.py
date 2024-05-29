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
import time
import scipy.special
import scipy.optimize
from servalcat.utils import logger
from servalcat import utils
from servalcat import ext

"""
DFc = sum_j D_j F_c,j
The last Fc,n is bulk solvent contribution.
"""

integr = ext.IntensityIntegrator()

def add_arguments(parser):
    parser.description = 'Sigma-A parameter estimation for crystallographic data'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin',
                        help='MTZ column for F,SIGF,FREE')
    parser.add_argument('--free', type=int,
                        help='flag number for test set')
    parser.add_argument('--model', required=True, nargs="+", action="append",
                        help='Input atomic model file(s)')
    parser.add_argument("-d", '--d_min', type=float)
    parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int,
                        help="Number of bins (default: auto)")
    parser.add_argument('-s', '--source', choices=["electron", "xray", "neutron"], required=True,
                        help="Scattering factor choice")
    parser.add_argument('--D_trans', choices=["exp", "splus"],
                        help="estimate D with positivity constraint")
    parser.add_argument('--S_trans', choices=["exp", "splus"],
                        help="estimate variance of unexplained signal with positivity constraint")
    parser.add_argument('--no_solvent',  action='store_true',
                        help="Do not consider bulk solvent contribution")
    parser.add_argument('--use_cc',  action='store_true',
                        help="Use CC(|F1|,|F2|) to CC(F1,F2) conversion to derive D and S")
    parser.add_argument('--use', choices=["all", "work", "test"], default="all",
                        help="Which reflections to be used for the parameter estimate.")
    parser.add_argument('--mask',
                        help="A solvent mask (by default calculated from the coordinates)")
    parser.add_argument('--keep_charges',  action='store_true',
                        help="Use scattering factor for charged atoms. Use it with care.")
    parser.add_argument('-o','--output_prefix', default="sigmaa",
                        help='output file name prefix (default: %(default)s)')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def nanaverage(cc, w):
    sel = ~numpy.isnan(cc)
    if numpy.sum(w[sel]) == 0:
        return numpy.nan
    return numpy.average(cc[sel], weights=w[sel]) 

def calc_r_and_cc(hkldata, centric_and_selections):
    has_int = "I" in hkldata.df
    has_free = "FREE" in hkldata.df
    stats = hkldata.binned_df.copy()
    stats["n_obs"] = 0
    if has_free:
        stats[["n_work", "n_free"]] = 0
    rlab = "R2" if has_int else "R"
    cclab = "CCI" if has_int else "CCF"
    Fc = numpy.abs(hkldata.df.FC * hkldata.df.k_aniso)
    if has_int:
        obs = hkldata.df.I
        calc = Fc**2
    else:
        obs = hkldata.df.FP
        calc = Fc
    if has_free:
        for lab in (cclab, rlab):
            for suf in ("work", "free"):
                stats[lab+suf] = numpy.nan
    else:
        stats[cclab] = numpy.nan
        stats[rlab] = numpy.nan

    for i_bin, idxes in hkldata.binned():
        stats.loc[i_bin, "n_obs"] = numpy.sum(numpy.isfinite(obs[idxes]))
        if has_free:
            for j, suf in ((1, "work"), (2, "free")):
                idxes2 = numpy.concatenate([sel[j] for sel in centric_and_selections[i_bin]])
                stats.loc[i_bin, "n_"+suf] = numpy.sum(numpy.isfinite(obs[idxes2]))
                stats.loc[i_bin, cclab+suf] = utils.hkl.correlation(obs[idxes2], calc[idxes2])
                stats.loc[i_bin, rlab+suf] = utils.hkl.r_factor(obs[idxes2], calc[idxes2])
        else:
            stats.loc[i_bin, cclab] = utils.hkl.correlation(obs[idxes], calc[idxes])
            stats.loc[i_bin, rlab] = utils.hkl.r_factor(obs[idxes], calc[idxes])

    # Overall
    ret = {}
    if has_free:
        for suf in ("work", "free"):
            ret[cclab+suf+"avg"] = nanaverage(stats[cclab+suf], stats["n_"+suf])
        for j, suf in ((1, "work"), (2, "free")):
            idxes = numpy.concatenate([sel[j] for i_bin, _ in hkldata.binned() for sel in centric_and_selections[i_bin]])
            ret[rlab+suf] = utils.hkl.r_factor(obs[idxes], calc[idxes])
    else:
        ret[cclab+"avg"] = nanaverage(stats[cclab], stats["n_obs"])
        ret[rlab] = utils.hkl.r_factor(obs, calc)
        
    return stats, ret
# calc_r_and_cc()

class VarTrans:
    def __init__(self, D_trans, S_trans):
        # splus (softplus) appears to be better than exp
        # exp sometimes results in too large parameter value
        trans_funcs = {"exp": (numpy.exp, # D = f(x)
                               numpy.exp, # dD/dx
                               numpy.log), # x = f^-1(D)
                       "splus": (lambda x: numpy.logaddexp(0, x),
                                 scipy.special.expit, # lambda x: 1. / (1. + numpy.exp(-x))
                                 lambda x: x + numpy.log(-numpy.expm1(-x))),
                       None: (lambda x: x,
                              lambda x: 1,
                              lambda x: x)}
        
        self.D, self.D_deriv, self.D_inv = trans_funcs[D_trans]
        self.S, self.S_deriv, self.S_inv = trans_funcs[S_trans]
# class VarTrans

class LsqScale:
    # parameter x = [k_overall, adp_pars, k_sol, B_sol]
    def __init__(self, k_as_exp=False, func_type="log_cosh"):
        assert func_type in ("sq", "log_cosh")
        self.k_trans = lambda x: numpy.exp(x) if k_as_exp else x
        self.k_trans_der = lambda x: numpy.exp(x) if k_as_exp else 1
        self.k_trans_inv = lambda x: numpy.log(x) if k_as_exp else x
        self.func_type = func_type
        self.reset()
        
    def reset(self):
        self.k_sol = 0.35 # same default as gemmi/scaling.hpp # refmac seems to use 0.33 and 100? SCALE_LS_PART
        self.b_sol = 46.
        self.k_overall = None
        self.b_iso = None
        self.b_aniso = None
        self.stats = {}

    def set_data(self, hkldata, fc_list, use_int=False, sigma_cutoff=None):
        assert 0 < len(fc_list) < 3
        self.use_int = use_int
        if sigma_cutoff is not None:
            if use_int:
                sel = hkldata.df.I / hkldata.df.SIGI > sigma_cutoff
                self.labcut = "(I/SIGI>{})".format(sigma_cutoff)
            else:
                sel = hkldata.df.FP / hkldata.df.SIGFP > sigma_cutoff
                self.labcut = "(F/SIGF>{})".format(sigma_cutoff)
        else:
            sel = hkldata.df.index
            self.labcut = ""
        self.obs = hkldata.df["I" if use_int else "FP"].to_numpy()[sel]
        self.calc = [x[sel] for x in fc_list]
        self.s2mat = hkldata.ssq_mat()[:,sel]
        self.s2 = 1. / hkldata.d_spacings().to_numpy()[sel]**2
        self.adpdirs = utils.model.adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=False)
        if use_int:
            self.sqrt_obs = numpy.sqrt(self.obs)
        
    def get_solvent_scale(self, k_sol, b_sol, s2=None):
        if s2 is None: s2 = self.s2
        return k_sol * numpy.exp(-b_sol * s2 / 4)
    
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
            diff = self.sqrt_obs - y
            #y2 = y**2
            #diff = self.obs - y2
        else:
            diff = self.obs - y

        if self.func_type == "sq":
            return numpy.nansum(diff**2)
        elif self.func_type == "log_cosh":
            return numpy.nansum(gemmi.log_cosh(diff))
        else:
            raise RuntimeError("bad func_type")
        
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
            diff = self.sqrt_obs - y
            diff_der = -1
            #diff = self.obs - y**2
            #diff_der = -2 * y
        else:
            diff = self.obs - y
            diff_der = -1
        if self.func_type == "sq":
            dfdy = 2 * diff * diff_der
        elif self.func_type == "log_cosh":
            dfdy = numpy.tanh(diff) * diff_der
        else:
            raise RuntimeError("bad func_type")
        
        dfdb = numpy.nansum(-self.s2mat * k * fc_abs * kani * dfdy, axis=1)
        g[0] = numpy.nansum(kani * fc_abs * dfdy * self.k_trans_der(x[0]))
        g[1:nadp+1] = numpy.dot(dfdb, self.adpdirs.T)
        if len(self.calc) == 2:
            re_fmask_fcconj = (fmask * fc.conj()).real
            tmp = k * kani * temp_sol / fc_abs * re_fmask_fcconj
            g[-2] = numpy.nansum(tmp * dfdy)
            g[-1] = numpy.nansum(-tmp * dfdy * x[-2] * self.s2 / 4)

        return g

    def calc_shift(self, x):
        # TODO: sort out code duplication, if we use this.
        g = numpy.zeros((len(self.calc[0]), len(x)))
        H = numpy.zeros((len(x), len(x)))
        
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
            diff = self.sqrt_obs - y
            diff_der = -1
            diff_der2 = 0
        else:
            diff = self.obs - y
            diff_der = -1.
            diff_der2 = 0.
            
        if self.func_type == "sq":
            dfdy = 2 * diff * diff_der
            dfdy2 = 2 * diff_der**2 + 2 * diff * diff_der2
        elif self.func_type == "log_cosh":
            dfdy = numpy.tanh(diff) * diff_der
            #dfdy2 = 1 /numpy.cosh(diff)**2 * diff_der**2 + numpy.tanh(diff) * diff_der2 # problematic with large diff
            #dfdy2 = numpy.where(diff==0, 1., numpy.abs(numpy.tanh(diff)) / gemmi.log_cosh(diff)) * diff_der**2 + numpy.tanh(diff) * diff_der2
            dfdy2 = numpy.where(diff==0, 1., numpy.tanh(diff) / diff) * diff_der**2 + numpy.tanh(diff) * diff_der2
        else:
            raise RuntimeError("bad func_type")
        
        dfdb = -self.s2mat * k * fc_abs * kani
        g[:,0] = kani * fc_abs * self.k_trans_der(x[0])
        g[:,1:nadp+1] = numpy.dot(dfdb.T, self.adpdirs.T)
        if len(self.calc) == 2:
            re_fmask_fcconj = (fmask * fc.conj()).real
            tmp = k * kani * temp_sol / fc_abs * re_fmask_fcconj
            g[:,-2] = tmp
            g[:,-1] = -tmp * x[-2] * self.s2 / 4

        # XXX won't work with NaNs.
        H = numpy.dot(g.T, g * dfdy2[:,None])
        g = numpy.sum(dfdy[:,None] * g, axis=0)
        dx = -numpy.dot(g, numpy.linalg.pinv(H))
        return dx

    def initial_kb(self):
        fc0 = self.calc[0]
        if len(self.calc) == 2:
            fmask = self.calc[1]
            fbulk = self.get_solvent_scale(self.k_sol, self.b_sol) * fmask
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
        logger.writeln("           R{} = {:.4f}".format(self.labcut, utils.hkl.r_factor(f1p, f2p * k * numpy.exp(-b*self.s2[sel]/4))))
        return k, b
    
    def scale(self):
        use_sol = len(self.calc) == 2
        msg = "Scaling Fc to {} {} bulk solvent contribution".format("Io" if self.use_int else "Fo",
                                                                     "with" if use_sol else "without")
        logger.writeln(msg)
        if self.k_overall is None or self.b_iso is None:
            k, b = self.initial_kb()
        else:
            k, b = self.k_overall, self.b_iso
        if self.b_aniso is None:
            self.b_aniso = gemmi.SMat33d(b,b,b,0,0,0)
        x0 = [self.k_trans_inv(k)]
        bounds = [(0, None)]
        x0.extend(numpy.dot(self.b_aniso.elements_pdb(), self.adpdirs.T))
        bounds.extend([(None, None)]*(len(x0)-1))
        if use_sol:
            x0.extend([self.k_sol, self.b_sol])
            bounds.extend([(1e-4, None), (10., 400.)])
        if 0:
            f0 = self.target(x0)
            ader = self.grad(x0)
            e = 1e-4
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

        t0 = time.time()
        if 1:
            x = x0
            for i in range(40):
                x_ini = x.copy()
                f0 = self.target(x)
                dx = self.calc_shift(x)
                if numpy.max(numpy.abs(dx)) < 1e-6:
                    break
                for s in (1, 0.5, 0.25):
                    if 0:
                        with open("debug.dat", "w") as ofs:
                            for s in numpy.linspace(-2, 2, 100):
                                f1 = self.target(x+dx * s)
                                #print(dx, f0, f1, f0 - f1)
                                ofs.write("{:4e} {:4e}\n".format(s, f1))
                    shift = dx * s
                    x = x_ini + shift
                    if x[0] < 0: x[0] = x0[0]
                    if use_sol:
                        if x[-1] < 10: x[-1] = 10
                        elif x[-1] > 400: x[-1] = 400
                        if x[-2] < 1e-4: x[-2] = 1e-4
                    f1 = self.target(x)
                    if f1 < f0: break
                #logger.writeln("cycle {} {} {} {} {} {}".format(i, f0, f1, s, shift, (f0 - f1) / f0))
                if 0 < (f0 - f1) / f0 < 1e-6:
                    break
            res_x = x
            self.stats["fun"] = f1
            self.stats["x"] = x
        else:
            res = scipy.optimize.minimize(fun=self.target, x0=x0, jac=self.grad, bounds=bounds)
            #logger.writeln(str(res))
            logger.writeln(" finished in {} iterations ({} evaluations)".format(res.nit, res.nfev))
            res_x = res.x
            self.stats["fun"] = res.fun
            self.stats["x"] = res.x
        logger.writeln(" time: {:.3f} sec".format(time.time() - t0))
        self.k_overall = self.k_trans(res_x[0])
        nadp = self.adpdirs.shape[0]
        b_overall = gemmi.SMat33d(*numpy.dot(res_x[1:nadp+1], self.adpdirs))
        self.b_iso = b_overall.trace() / 3
        self.b_aniso = b_overall.added_kI(-self.b_iso) # subtract isotropic contribution

        logger.writeln(" k_ov= {:.2e} B_iso= {:.2e} B_aniso= {}".format(self.k_overall, self.b_iso, self.b_aniso))
        if use_sol:
            self.k_sol = res_x[-2] 
            self.b_sol = res_x[-1]
            logger.writeln(" k_sol= {:.2e} B_sol= {:.2e}".format(self.k_sol, self.b_sol))
        calc = numpy.abs(self.scaled_fc(res_x))
        if self.use_int: calc *= calc
        self.stats["cc"] = utils.hkl.correlation(self.obs, calc)
        self.stats["r"] = utils.hkl.r_factor(self.obs, calc)
        logger.writeln(" CC{} = {:.4f}".format(self.labcut, self.stats["cc"]))
        logger.writeln(" R{}  = {:.4f}".format(self.labcut, self.stats["r"]))
# class LsqScale

def calc_abs_DFc(Ds, Fcs):
    DFc = sum(Ds[i] * Fcs[i] for i in range(len(Ds)))
    return numpy.abs(DFc)
# calc_abs_DFc()

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
#@profile
def mlf(df, fc_labs, Ds, S, k_ani, idxes):
    Fcs = numpy.vstack([df[lab].to_numpy()[idxes] for lab in fc_labs]).T
    DFc = (Ds * Fcs).sum(axis=1)
    ll = numpy.nansum(ext.ll_amp(df.FP.to_numpy()[idxes], df.SIGFP.to_numpy()[idxes],
                                 k_ani[idxes], S * df.epsilon.to_numpy()[idxes],
                                 numpy.abs(DFc), df.centric.to_numpy()[idxes]+1))
    return numpy.nansum(ll)
# mlf()

#@profile
def deriv_mlf_wrt_D_S(df, fc_labs, Ds, S, k_ani, idxes):
    Fcs = [df[lab].to_numpy()[idxes] for lab in fc_labs]
    r = ext.ll_amp_der1_DS(df.FP.to_numpy()[idxes], df.SIGFP.to_numpy()[idxes], k_ani[idxes], S,
                           numpy.vstack(Fcs).T, Ds,
                           df.centric.to_numpy()[idxes]+1, df.epsilon.to_numpy()[idxes])
    g = numpy.zeros(len(fc_labs)+1)
    g[:len(fc_labs)] = numpy.nansum(r[:,:len(fc_labs)], axis=0) # D
    g[-1] = numpy.nansum(r[:,-1]) # S
    return g
# deriv_mlf_wrt_D_S()

#@profile
def mlf_shift_S(df, fc_labs, Ds, S, k_ani, idxes):
    Fcs = [df[lab].to_numpy()[idxes] for lab in fc_labs]
    r = ext.ll_amp_der1_DS(df.FP.to_numpy()[idxes], df.SIGFP.to_numpy()[idxes], k_ani[idxes], S,
                           numpy.vstack(Fcs).T, Ds,
                           df.centric.to_numpy()[idxes]+1, df.epsilon.to_numpy()[idxes])
    g = numpy.nansum(r[:,-1])
    H = numpy.nansum(r[:,-1]**2) # approximating expectation value of second derivative
    return -g / H
# mlf_shift_S()

def mli(df, fc_labs, Ds, S, k_ani, idxes):
    Fcs = numpy.vstack([df[lab].to_numpy()[idxes] for lab in fc_labs]).T
    DFc = (Ds * Fcs).sum(axis=1)
    ll = integr.ll_int(df.I.to_numpy()[idxes], df.SIGI.to_numpy()[idxes],
                       k_ani[idxes], S * df.epsilon.to_numpy()[idxes],
                       numpy.abs(DFc), df.centric.to_numpy()[idxes]+1)
    return numpy.nansum(ll)
# mli()

def deriv_mli_wrt_D_S(df, fc_labs, Ds, S, k_ani, idxes):
    Fcs = numpy.vstack([df[lab].to_numpy()[idxes] for lab in fc_labs]).T
    r = integr.ll_int_der1_DS(df.I.to_numpy()[idxes], df.SIGI.to_numpy()[idxes], k_ani[idxes], S,
                              Fcs, Ds,
                              df.centric.to_numpy()[idxes]+1, df.epsilon.to_numpy()[idxes])
    g = numpy.zeros(len(fc_labs)+1)
    g[:len(fc_labs)] = numpy.nansum(r[:,:len(fc_labs)], axis=0) # D
    g[-1] = numpy.nansum(r[:,-1]) # S
    return g
# deriv_mli_wrt_D_S()

def mli_shift_D(df, fc_labs, Ds, S, k_ani, idxes):
    Fcs = numpy.vstack([df[lab].to_numpy()[idxes] for lab in fc_labs]).T
    r = integr.ll_int_der1_DS(df.I.to_numpy()[idxes], df.SIGI.to_numpy()[idxes], k_ani[idxes], S,
                              Fcs, Ds,
                              df.centric.to_numpy()[idxes]+1, df.epsilon.to_numpy()[idxes])[:,:len(fc_labs)]
    g = numpy.nansum(r, axis=0)# * trans.D_deriv(x[:len(fc_labs)]) # D
    #tmp = numpy.hstack([r[:,:len(fc_labs)] #* trans.D_deriv(x[:len(fc_labs)]),
    #                    r[:,-1,None] * trans.S_deriv(x[-1])])
    H = numpy.nansum(numpy.matmul(r[:,:,None], r[:,None]), axis=0)
    return -numpy.dot(g, numpy.linalg.pinv(H))
# mli_shift_D()

def mli_shift_S(df, fc_labs, Ds, S, k_ani, idxes):
    Fcs = numpy.vstack([df[lab].to_numpy()[idxes] for lab in fc_labs]).T
    r = integr.ll_int_der1_DS(df.I.to_numpy()[idxes], df.SIGI.to_numpy()[idxes], k_ani[idxes], S,
                              Fcs, Ds,
                              df.centric.to_numpy()[idxes]+1, df.epsilon.to_numpy()[idxes])
    g = numpy.nansum(r[:,-1])
    H = numpy.nansum(r[:,-1]**2) # approximating expectation value of second derivative
    return -g / H
# mli_shift_S()

def determine_mlf_params_from_cc(hkldata, fc_labs, D_labs, centric_and_selections, use="all", smoothing="gauss"):
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
        k_ani = hkldata.df.k_aniso.to_numpy()[cidxes]
        Fo = hkldata.df.FP.to_numpy()[cidxes] * factor / k_ani
        mean_Fo2 = numpy.mean(Fo**2)
        SigFo = hkldata.df.SIGFP.to_numpy()[cidxes] / k_ani
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
    smooth_params(hkldata, D_labs, smoothing)
# determine_mlf_params_from_cc()

def initialize_ml_params(hkldata, use_int, D_labs, b_aniso, centric_and_selections, use):
    # Initial values
    for lab in D_labs: hkldata.binned_df[lab] = 1.
    hkldata.binned_df["S"] = 10000.
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    lab_obs = "I" if use_int else "FP"
    for i_bin, _ in hkldata.binned():
        if use == "all":
            idxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin] for i in (1,2)])
        else:
            i = 1 if use == "work" else 2
            idxes = numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin]])
        valid_sel = numpy.isfinite(hkldata.df.loc[idxes, lab_obs]) # as there is no nan-safe numpy.corrcoef
        if numpy.sum(valid_sel) < 2:
            continue
        idxes = idxes[valid_sel]
        if use_int:
            Io = hkldata.df.I.to_numpy()[idxes]
        else:
            Io = hkldata.df.FP.to_numpy()[idxes]**2
        Io /= k_ani[idxes]**2 
        Ic = numpy.abs(hkldata.df.FC.to_numpy()[idxes])**2
        mean_Io = numpy.mean(Io)
        mean_Ic = numpy.mean(Ic)
        cc = numpy.corrcoef(Io, Ic)[1,0]
        if cc > 0 and mean_Io > 0:
            D = numpy.sqrt(mean_Io / mean_Ic * cc)
        else:
            D = 0 # will be taken care later
        hkldata.binned_df.loc[i_bin, D_labs[0]] = D
        if mean_Io > 0:
            S = mean_Io - 2 * numpy.sqrt(mean_Io * mean_Ic * numpy.maximum(0, cc)) + mean_Ic
        else:
            S = numpy.std(Io) # similar initial to french_wilson
        hkldata.binned_df.loc[i_bin, "S"] = S

    for D_lab in D_labs:
        if hkldata.binned_df[D_lab].min() <= 0:
            min_D = hkldata.binned_df[D_lab][hkldata.binned_df[D_lab] > 0].min() * 0.1
            logger.writeln("WARNING: negative {} is detected from initial estimates. Replacing it using minimum positive value {:.2e}".format(D_lab, min_D))
            hkldata.binned_df[D_lab].where(hkldata.binned_df[D_lab] > 0, min_D, inplace=True) # arbitrary
        
    logger.writeln("Initial estimates:")
    logger.writeln(hkldata.binned_df.to_string())
# initialize_ml_params()

def determine_ml_params(hkldata, use_int, fc_labs, D_labs, b_aniso, centric_and_selections,
                        D_trans=None, S_trans=None, use="all", n_cycle=1, smoothing="gauss"):
    assert use in ("all", "work", "test")
    assert smoothing in (None, "gauss")
    logger.writeln("Estimating sigma-A parameters using {}..".format("intensities" if use_int else "amplitudes"))
    trans = VarTrans(D_trans, S_trans)
    lab_obs = "I" if use_int else "FP"
    def get_idxes(i_bin):
        if use == "all":
            return numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin] for i in (1,2)])
        else:
            i = 1 if use == "work" else 2
            return numpy.concatenate([sel[i] for sel in centric_and_selections[i_bin]])

    if not set(D_labs + ["S"]).issubset(hkldata.binned_df):
        initialize_ml_params(hkldata, use_int, D_labs, b_aniso, centric_and_selections, use)
        for dlab, fclab in zip(D_labs, fc_labs):
            hkldata.binned_df["Mn(|{}*{}|)".format(dlab, fclab)] = numpy.nan

    refpar = "all"
    for i_cyc in range(n_cycle):
        t0 = time.time()
        nfev_total = 0
        k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
        for i_bin, _ in hkldata.binned():
            idxes = get_idxes(i_bin)
            valid_sel = numpy.isfinite(hkldata.df.loc[idxes, lab_obs]) # as there is no nan-safe numpy.corrcoef
            if numpy.sum(valid_sel) < 5:
                logger.writeln("WARNING: bin {} has no sufficient reflections".format(i_bin))
                continue

            def target(x):
                if refpar == "all":
                    Ds = trans.D(x[:len(fc_labs)])
                    S = trans.S(x[-1])
                elif refpar == "D":
                    Ds = trans.D(x[:len(fc_labs)])
                    S = hkldata.binned_df.loc[i_bin, "S"]
                else:
                    Ds = [hkldata.binned_df.loc[i_bin, lab] for lab in D_labs]
                    S = trans.S(x[-1])
                f = mli if use_int else mlf
                return f(hkldata.df, fc_labs, Ds, S, k_ani, idxes)

            def grad(x):
                if refpar == "all":
                    Ds = trans.D(x[:len(fc_labs)])
                    S = trans.S(x[-1])
                    n_par = len(fc_labs)+1
                elif refpar == "D":
                    Ds = trans.D(x[:len(fc_labs)])
                    S = hkldata.binned_df.loc[i_bin, "S"]
                    n_par = len(fc_labs)
                else:
                    Ds = [hkldata.binned_df.loc[i_bin, lab] for lab in D_labs]
                    S = trans.S(x[-1])
                    n_par = 1
                calc_deriv = deriv_mli_wrt_D_S if use_int else deriv_mlf_wrt_D_S
                r = calc_deriv(hkldata.df, fc_labs, Ds, S, k_ani, idxes)
                g = numpy.zeros(n_par)
                if refpar in ("all", "D"):
                    g[:len(fc_labs)] = r[:len(fc_labs)]
                    g[:len(fc_labs)] *= trans.D_deriv(x[:len(fc_labs)])
                if refpar in ("all", "S"):
                    g[-1] = r[-1]
                    g[-1] *= trans.S_deriv(x[-1])
                return g

            if 0:
                refpar = "S"
                x0 = trans.S_inv(hkldata.binned_df.loc[i_bin, "S"])
                with open("s_line_{}.dat".format(i_bin), "w") as ofs:
                    for sval in numpy.linspace(1, x0*2, 100):
                        ofs.write("{:.4e} {:.10e} {:.10e}\n".format(sval,
                                                                    target([sval]),
                                                                    grad([sval])[0]))
                continue
            #print("Bin", i_bin)
            if 1: # refine D and S iteratively
                vals_last = None
                for ids in range(10):
                    refpar = "D"
                    x0 = numpy.array([trans.D_inv(hkldata.binned_df.loc[i_bin, lab]) for lab in D_labs])
                    vals_now = []
                    if 0:
                        f0 = target(x0)
                        nfev_total += 1
                        shift = mli_shift_D(hkldata.df, fc_labs, trans.D(x0), hkldata.binned_df.loc[i_bin, "S"], k_ani, idxes)
                        shift /= trans.D_deriv(x0)
                        #if abs(shift) < 1e-3: break
                        for itry in range(10):
                            x1 = x0 + shift
                            if (D_trans and any(x1 < -3)) or (not D_trans and any(x1 < 5e-2)):
                                #print(i_bin, cyc_s, trans.S(x0), trans.S(x1), shift, "BAD")
                                shift /= 2
                                continue
                            f1 = target(x1)
                            nfev_total += 1
                            if f1 > f0:
                                shift /= 2
                                continue
                            else: # good
                                for i, lab in enumerate(D_labs):
                                    hkldata.binned_df.loc[i_bin, lab] = trans.D(x1[i])
                                    vals_now.append(hkldata.binned_df.loc[i_bin, lab])
                                break
                        else:
                            break
                    else:
                        #print(mli_shift_D(hkldata.df, fc_labs, trans.D(x0), hkldata.binned_df.S[i_bin], k_ani, idxes))
                        res = scipy.optimize.minimize(fun=target, x0=x0, jac=grad,
                                                      bounds=((-5 if D_trans else 1e-5, None),)*len(x0))
                        nfev_total += res.nfev
                        #print(i_bin, "mini cycle", ids, refpar)
                        #print(res)
                        for i, lab in enumerate(D_labs):
                            hkldata.binned_df.loc[i_bin, lab] = trans.D(res.x[i])
                            vals_now.append(hkldata.binned_df.loc[i_bin, lab])
                    refpar = "S"
                    if 1:
                        for cyc_s in range(1):
                            x0 = trans.S_inv(hkldata.binned_df.loc[i_bin, "S"])
                            f0 = target([x0])
                            Ds = [hkldata.binned_df.loc[i_bin, lab] for lab in D_labs]
                            nfev_total += 1
                            calc_shift_S = mli_shift_S if use_int else mlf_shift_S
                            shift = calc_shift_S(hkldata.df, fc_labs, Ds, trans.S(x0), k_ani, idxes)
                            shift /= trans.S_deriv(x0)
                            if abs(shift) < 1e-3: break
                            for itry in range(10):
                                x1 = x0 + shift
                                if (S_trans and x1 < -3) or (not S_trans and x1 < 5e-2):
                                    #print(i_bin, cyc_s, trans.S(x0), trans.S(x1), shift, "BAD")
                                    shift /= 2
                                    continue
                                f1 = target([x1])
                                nfev_total += 1
                                if f1 > f0:
                                    shift /= 2
                                    continue
                                else: # good
                                    #print(i_bin, cyc_s, trans.S(x0), trans.S(x1), shift)
                                    hkldata.binned_df.loc[i_bin, "S"] = trans.S(x1)
                                    break
                            else:
                                #print("all bad")
                                break
                    else:
                        # somehow this does not work well.
                        x0 = [trans.S_inv(hkldata.binned_df.loc[i_bin, "S"])]
                        res = scipy.optimize.minimize(fun=target, x0=x0, jac=grad,
                                                      bounds=((-3 if S_trans else 5e-2, None),))
                        nfev_total += res.nfev
                        #print(i_bin, "mini cycle", ids, refpar)
                        #print(res)
                        hkldata.binned_df.loc[i_bin, "S"] = trans.S(res.x[-1])
                    vals_now.append(hkldata.binned_df.loc[i_bin, "S"])
                    vals_now = numpy.array(vals_now)
                    if vals_last is not None and numpy.all(numpy.abs((vals_last - vals_now) / vals_now) < 1e-2):
                        #logger.writeln("converged in mini cycle {}".format(ids+1))
                        break
                    vals_last = vals_now
            else:
                x0 = [trans.D_inv(hkldata.binned_df.loc[i_bin, lab]) for lab in D_labs] + [trans.S_inv(hkldata.binned_df.loc[i_bin, "S"])]
                res = scipy.optimize.minimize(fun=target, x0=x0, jac=grad,
                                              bounds=((-5 if D_trans else 1e-5, None), )*len(D_labs) + ((-3 if S_trans else 5e-2, None),))
                nfev_total += res.nfev
                #print(i_bin)
                #print(res)
                for i, lab in enumerate(D_labs):
                    hkldata.binned_df.loc[i_bin, lab] = trans.D(res.x[i])
                hkldata.binned_df.loc[i_bin, "S"] = trans.S(res.x[-1])

        for i_bin, idxes in hkldata.binned():
            for dlab, fclab in zip(D_labs, fc_labs):
                mean_dfc = numpy.nanmean(numpy.abs(hkldata.binned_df[dlab][i_bin] * hkldata.df[fclab][idxes]))
                hkldata.binned_df.loc[i_bin, "Mn(|{}*{}|)".format(dlab, fclab)] = mean_dfc
                
        logger.writeln("Refined estimates:")
        logger.writeln(hkldata.binned_df.to_string())
        logger.writeln("time: {:.1f} sec ({} evaluations)".format(time.time() - t0, nfev_total))

        if not use_int:
            break # did not implement MLF B_aniso optimization
        
        # Refine b_aniso
        adpdirs = utils.model.adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=True)
        SMattolist = lambda B: [B.u11, B.u22, B.u33, B.u12, B.u13, B.u23]

        def target_ani(x):
            b = gemmi.SMat33d(*numpy.dot(x, adpdirs))
            k_ani = hkldata.debye_waller_factors(b_cart=b)
            ret = 0.
            for i_bin, idxes in hkldata.binned():
                Ds = [hkldata.binned_df.loc[i_bin, lab] for lab in D_labs]
                ret += mli(hkldata.df, fc_labs, Ds, hkldata.binned_df.loc[i_bin, "S"], k_ani, idxes)
            return ret
        def grad_ani(x):
            b = gemmi.SMat33d(*numpy.dot(x, adpdirs))
            k_ani = hkldata.debye_waller_factors(b_cart=b)
            S2mat = hkldata.ssq_mat() # ssqmat
            g = numpy.zeros(6)
            for i_bin, idxes in hkldata.binned():
                r = integr.ll_int_der1_ani(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes],
                                           k_ani[idxes], hkldata.binned_df.loc[i_bin, "S"],
                                           hkldata.df[fc_labs].to_numpy()[idxes], hkldata.binned_df.loc[i_bin, D_labs],
                                           hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes])
                S2 = S2mat[:,idxes]
                g += -numpy.nansum(S2 * r[:,0], axis=1) # k_ani is already multiplied in r
            return numpy.dot(g, adpdirs.T)
        def shift_ani(x):
            b = gemmi.SMat33d(*numpy.dot(x, adpdirs))
            k_ani = hkldata.debye_waller_factors(b_cart=b)
            S2mat = hkldata.ssq_mat() # ssqmat
            g = numpy.zeros(6)
            H = numpy.zeros((6, 6))
            for i_bin, idxes in hkldata.binned():
                r = integr.ll_int_der1_ani(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes],
                                           k_ani[idxes], hkldata.binned_df.loc[i_bin, "S"],
                                           hkldata.df[fc_labs].to_numpy()[idxes], list(hkldata.binned_df.loc[i_bin, D_labs]),
                                           hkldata.df.centric.to_numpy()[idxes]+1, hkldata.df.epsilon.to_numpy()[idxes])
                S2 = S2mat[:,idxes]
                g += -numpy.nansum(S2 * r[:,0], axis=1) # k_ani is already multiplied in r
                H += numpy.nansum(numpy.matmul(S2[None,:].T, S2.T[:,None]) * (r[:,0]**2)[:,None,None], axis=0)

            g, H = numpy.dot(g, adpdirs.T), numpy.dot(adpdirs, numpy.dot(H, adpdirs.T))
            return -numpy.dot(g, numpy.linalg.pinv(H))

        logger.writeln("Refining B_aniso. Current = {}".format(b_aniso))
        if 0:
            x0 = numpy.dot(SMattolist(b_aniso), numpy.linalg.pinv(adpdirs))
            res = scipy.optimize.minimize(fun=target_ani, x0=x0, jac=grad_ani)
            print(res)
            b_aniso = gemmi.SMat33d(*numpy.dot(res.x, adpdirs))
            f1 = res.fun
        else:
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

    smooth_params(hkldata, D_labs, smoothing)
    return b_aniso
# determine_ml_params()

def smooth_params(hkldata, D_labs, smoothing):
    if smoothing is None or len(hkldata.binned()) < 2:
        for i, lab in enumerate(D_labs + ["S"]):
            hkldata.df[lab] = hkldata.binned_data_as_array(lab)

    elif smoothing == "gauss":
        bin_centers = (0.5 / hkldata.binned_df[["d_min", "d_max"]]**2).sum(axis=1).to_numpy()
        vals = ext.smooth_gauss(bin_centers,
                                hkldata.binned_df[D_labs + ["S"]].to_numpy(),
                                1./hkldata.df.d.to_numpy()**2,
                                100, # min(n_ref?)
                                (bin_centers[1] - bin_centers[0]))
        for i, lab in enumerate(D_labs + ["S"]):
            hkldata.df[lab] = vals[:, i]
        # Update smoothened average; this affects next refinement.
        # TODO: update Mn(|Dj*FCj|) as well.
        #for i_bin, idxes in hkldata.binned():
        #    for lab in D_labs + ["S"]:
        #        hkldata.binned_df.loc[i_bin, lab] = numpy.mean(hkldata.df[lab].to_numpy()[idxes])
    else:
        raise RuntimeError("unknown smoothing method: {}".format(smoothing))
# smooth_params()

def expected_F_from_int(Io, sigIo, k_ani, DFc, eps, c, S):
    if c == 0: # acentric
        k_num, k_den = 0.5, 0.
    else:
        k_num, k_den = 0., -0.5
    to = Io / sigIo - sigIo / (c+1) / k_ani**2 / S / eps
    tf = k_ani * numpy.abs(DFc) / numpy.sqrt(sigIo)
    sig1 = k_ani**2 * S * eps / sigIo
    f = ext.integ_J_ratio(k_num, k_den, True, to, tf, sig1, c+1, integr.exp2_threshold, integr.h, integr.N, integr.ewmax)
    f *= numpy.sqrt(sigIo) / k_ani
    m_proxy = ext.integ_J_ratio(k_num, k_num, True, to, tf, sig1, c+1, integr.exp2_threshold, integr.h, integr.N, integr.ewmax)
    return f, m_proxy
# expected_F_from_int()

def calculate_maps_int(hkldata, b_aniso, fc_labs, D_labs, centric_and_selections, use="all"):
    nmodels = len(fc_labs)
    hkldata.df["FWT"] = 0j * numpy.nan
    hkldata.df["DELFWT"] = 0j * numpy.nan
    hkldata.df["FOM"] = numpy.nan # FOM proxy, |<F>| / <|F|>
    has_ano = "I(+)" in hkldata.df and "I(-)" in hkldata.df
    if has_ano:
        hkldata.df["FAN"] = 0j * numpy.nan
        ano_data = hkldata.df[["I(+)", "SIGI(+)", "I(-)", "SIGI(-)"]].to_numpy()
    Io = hkldata.df.I.to_numpy()
    sigIo = hkldata.df.SIGI.to_numpy()
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    eps = hkldata.df.epsilon.to_numpy()
    Ds = numpy.vstack([hkldata.df[lab].to_numpy() for lab in D_labs]).T
    Fcs = numpy.vstack([hkldata.df[lab].to_numpy() for lab in fc_labs]).T
    DFc = (Ds * Fcs).sum(axis=1)
    hkldata.df["DFC"] = DFc
    for i_bin, idxes in hkldata.binned():
        for c, work, test in centric_and_selections[i_bin]:
            cidxes = numpy.concatenate([work, test])
            S = hkldata.df["S"].to_numpy()[cidxes]
            f, m_proxy = expected_F_from_int(Io[cidxes], sigIo[cidxes], k_ani[cidxes], DFc[cidxes], eps[cidxes], c, S)
            exp_ip = numpy.exp(numpy.angle(DFc[cidxes])*1j)
            if c == 0:
                hkldata.df.loc[cidxes, "FWT"] = 2 * f * exp_ip - DFc[cidxes]
            else:
                hkldata.df.loc[cidxes, "FWT"] = f * exp_ip
            hkldata.df.loc[cidxes, "DELFWT"] = f * exp_ip - DFc[cidxes]
            hkldata.df.loc[cidxes, "FOM"] = m_proxy
            if has_ano:
                f_p, _ = expected_F_from_int(ano_data[cidxes,0], ano_data[cidxes,1],
                                             k_ani[cidxes], DFc[cidxes], eps[cidxes], c, S)
                f_m, _ = expected_F_from_int(ano_data[cidxes,2], ano_data[cidxes,3],
                                             k_ani[cidxes], DFc[cidxes], eps[cidxes], c, S)
                hkldata.df.loc[cidxes, "FAN"] = (f_p - f_m) * exp_ip / 2j
            # remove reflections that should be hidden
            if use != "all":
                # usually use == "work"
                tohide = test if use == "work" else work
                hkldata.df.loc[tohide, "FWT"] = 0j * numpy.nan
                hkldata.df.loc[tohide, "DELFWT"] = 0j * numpy.nan
            fill_sel = numpy.isnan(hkldata.df["FWT"][cidxes].to_numpy())
            hkldata.df.loc[cidxes[fill_sel], "FWT"] = DFc[cidxes][fill_sel]
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

def decide_mtz_labels(mtz, find_free=True):
    dlabs = utils.hkl.mtz_find_data_columns(mtz)
    logger.writeln("Finding possible options from MTZ:")
    for typ in dlabs:
        for labs in dlabs[typ]:
            logger.writeln(" --labin '{}'".format(",".join(labs)))
    if dlabs["F"]: # F is preferred for now
        labin = dlabs["F"][0]
    elif dlabs["J"]:
        labin = dlabs["J"][0]
    elif dlabs["G"]:
        labin = dlabs["G"][0]
    elif dlabs["K"]:
        labin = dlabs["K"][0]
    else:
        raise RuntimeError("Data not found from mtz")
    if find_free:
        flabs = utils.hkl.mtz_find_free_columns(mtz)
        if flabs:
            labin += [flabs[0]]
    logger.writeln("MTZ columns automatically selected: {}".format(labin))
    return labin
# decide_mtz_labels()

def process_input(hklin, labin, n_bins, free, xyzins, source, d_max=None, d_min=None,
                  n_per_bin=None, use="all", max_bins=None, cif_index=0, keep_charges=False):
    if labin: assert 1 < len(labin) < 6
    assert use in ("all", "work", "test")
    assert n_bins or n_per_bin #if n_bins not set, n_per_bin should be given

    if len(xyzins) > 0 and type(xyzins[0]) is gemmi.Structure:
        sts = xyzins
    else:
        sts = []
    
    if type(hklin) is gemmi.Mtz or utils.fileio.is_mmhkl_file(hklin):
        if type(hklin) is gemmi.Mtz:
            mtz = hklin
        else:
            mtz = utils.fileio.read_mmhkl(hklin, cif_index=cif_index)
        if not sts:
            sts = [utils.fileio.read_structure(f) for f in xyzins]
    else:
        assert len(xyzins) == 1
        assert not sts
        st, mtz = utils.fileio.read_small_molecule_files([hklin, xyzins[0]])
        sts = [st]

    if not labin:
        labin = decide_mtz_labels(mtz)
    col_types = {x.label:x.type for x in mtz.columns}
    if labin[0] not in col_types:
        raise RuntimeError("MTZ column not found: {}".format(labin[0]))
    labs_and_types = {"F": ("amplitude", ["FP","SIGFP"], ["F", "Q"]),
                      "J": ("intensity", ["I","SIGI"], ["J", "Q"]),
                      "G": ("anomalous amplitude", ["F(+)","SIGF(+)", "F(-)", "SIGF(-)"], ["G", "L", "G", "L"]),
                      "K": ("anomalous intensity", ["I(+)","SIGI(+)", "I(-)", "SIGI(-)"], ["K", "M", "K", "M"])}
    if col_types[labin[0]] not in labs_and_types:
        raise RuntimeError("MTZ column {} is neither amplitude nor intensity".format(labin[0]))
    name, newlabels, require_types = labs_and_types[col_types[labin[0]]]
    logger.writeln("Observation type: {}".format(name))
    if len(newlabels) < len(labin): newlabels.append("FREE")
    hkldata = utils.hkl.hkldata_from_mtz(mtz, labin, newlabels=newlabels, require_types=require_types)
    if newlabels[0] == "F(+)":
        hkldata.merge_anomalous(newlabels[:4], ["FP", "SIGFP"])
        newlabels = ["FP", "SIGFP"] + newlabels[4:]
    elif newlabels[0] == "I(+)":
        hkldata.merge_anomalous(newlabels[:4], ["I", "SIGI"])
        newlabels = ["I", "SIGI"] + newlabels[4:]

    if hkldata.df.empty:
        raise RuntimeError("No data in hkl data")
    
    if sts:
        assert source in ["electron", "xray", "neutron"]
        for st in sts:
            if st[0].count_atom_sites() == 0:
                raise RuntimeError("No atom in the model")
        if not hkldata.cell.approx(sts[0].cell, 1e-3):
            logger.writeln("Warning: unit cell mismatch between model and reflection data")
            logger.writeln("         using unit cell from mtz")

        for st in sts: st.cell = hkldata.cell # mtz cell is used in any case

        sg_st = sts[0].find_spacegroup() # may be None
        sg_use = hkldata.sg
        if hkldata.sg != sg_st:
            if st.cell.is_crystal() and sg_st and sg_st.laue_str() != hkldata.sg.laue_str():
                raise RuntimeError("Crystal symmetry mismatch between model and data")
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

        if not keep_charges:
            utils.model.remove_charge(sts)
        utils.model.check_atomsf(sts, source)

    if newlabels[0] == "FP":
        hkldata.remove_nonpositive(newlabels[0])
    hkldata.remove_nonpositive(newlabels[1])
    hkldata.switch_to_asu()
    hkldata.remove_systematic_absences()
    #hkldata.df = hkldata.df.astype({name: 'float64' for name in ["I","SIGI","FP","SIGFP"] if name in hkldata.df})

    if (d_min, d_max).count(None) != 2:
        hkldata = hkldata.copy(d_min=d_min, d_max=d_max)
    if hkldata.df.empty:
        raise RuntimeError("No data left in hkl data")
    d_min, d_max = hkldata.d_min_max()
        
    hkldata.complete()
    hkldata.sort_by_resolution()
    hkldata.calc_epsilon()
    hkldata.calc_centric()

    if "FREE" in hkldata.df and free is None:
        free = hkldata.guess_free_number(newlabels[0])

    if n_bins is None:
        sel = hkldata.df[newlabels[0]].notna()
        if use == "work":
            sel &= hkldata.df.FREE != free
        elif use == "test":
            sel &= hkldata.df.FREE == free
        s_array = 1/hkldata.d_spacings()[sel]
        if len(s_array) == 0:
            raise RuntimeError("no reflections in {} set".format(use))
        n_bins = utils.hkl.decide_n_bins(n_per_bin, s_array, max_bins=max_bins)
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
    stats[newlabels[0]] = numpy.nan
    snr = "I/sigma" if newlabels[0] == "I" else "F/sigma"
    stats[snr] = numpy.nan
    if newlabels[0] == "I":
        stats["Mn(I)/Std(I)"] = numpy.nan
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
        if n_obs > 0:
            stats.loc[i_bin, snr] = numpy.nanmean(obs / sigma)
            mean_obs = numpy.nanmean(obs)
            stats.loc[i_bin, newlabels[0]] = mean_obs
            if newlabels[0] == "I":
                stats.loc[i_bin, "Mn(I)/Std(I)"] = mean_obs / numpy.nanstd(obs)
        if "FREE" in hkldata.df:
            stats.loc[i_bin, "n_work"] = n_work
            stats.loc[i_bin, "n_test"] = n_test
            
    stats["completeness"] = stats["n_obs"] / stats["n_all"] * 100
    logger.writeln(stats.to_string())
    return hkldata, sts, fc_labs, centric_and_selections, free
# process_input()

def calc_Fmask(st, d_min, miller_array):
    logger.writeln("Calculating solvent contribution..")
    grid = gemmi.FloatGrid()
    grid.setup_from(st, spacing=min(0.6, d_min / 2 - 1e-9))
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Refmac)
    masker.put_mask_on_float_grid(grid, st[0])
    fmask_gr = gemmi.transform_map_to_f_phi(grid)
    Fmask = fmask_gr.get_value_by_hkl(miller_array)
    return Fmask
# calc_Fmask()

def bulk_solvent_and_lsq_scales(hkldata, sts, fc_labs, use_solvent=True, use_int=False, mask=None, func_type="log_cosh"):
    fc_list = [hkldata.df[fc_labs].sum(axis=1).to_numpy()]
    if use_solvent:
        if mask is None:
            Fmask = calc_Fmask(merge_models(sts), hkldata.d_min_max()[0] - 1e-6, hkldata.miller_array())
        else:
            fmask_gr = gemmi.transform_map_to_f_phi(mask)
            Fmask = fmask_gr.get_value_by_hkl(hkldata.miller_array())
        fc_list.append(Fmask)

    scaling = LsqScale(func_type=func_type)
    scaling.set_data(hkldata, fc_list, use_int, sigma_cutoff=0)
    scaling.scale()
    b_iso = scaling.b_iso
    k_iso = hkldata.debye_waller_factors(b_iso=b_iso)
    k_aniso = hkldata.debye_waller_factors(b_cart=scaling.b_aniso)
    hkldata.df["k_aniso"] = k_aniso # we need it later when calculating stats
    
    if use_solvent:
        fc_labs.append("Fbulk")
        solvent_scale = scaling.get_solvent_scale(scaling.k_sol, scaling.b_sol,
                                                  1. / hkldata.d_spacings().to_numpy()**2)
        hkldata.df[fc_labs[-1]] = Fmask * solvent_scale

    # Apply scales
    if use_int:
        # in intensity case, we try to refine b_aniso with ML. perhaps we should do it in amplitude case also
        o_labs = ["I", "SIGI", "I(+)","SIGI(+)", "I(-)", "SIGI(-)"]
        hkldata.df[hkldata.df.columns.intersection(o_labs)] /= scaling.k_overall**2
    else:
        o_labs = ["FP", "SIGFP", "F(+)","SIGF(+)", "F(-)", "SIGF(-)"]
        hkldata.df[hkldata.df.columns.intersection(o_labs)] /= scaling.k_overall
    for lab in fc_labs: hkldata.df[lab] *= k_iso
    # total Fc
    hkldata.df["FC"] = hkldata.df[fc_labs].sum(axis=1)
    return scaling
# bulk_solvent_and_lsq_scales()

def calculate_maps(hkldata, b_aniso, centric_and_selections, fc_labs, D_labs, log_out, use="all"):
    nmodels = len(fc_labs)
    hkldata.df["FWT"] = 0j * numpy.nan
    hkldata.df["DELFWT"] = 0j * numpy.nan
    hkldata.df["FOM"] = numpy.nan
    hkldata.df["X"] = numpy.nan # for FOM
    has_ano = "F(+)" in hkldata.df and "F(-)" in hkldata.df
    if has_ano:
        hkldata.df["FAN"] = 0j * numpy.nan
    stats_data = []
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    Ds = numpy.vstack([hkldata.df[lab].to_numpy() for lab in D_labs]).T
    Fcs = numpy.vstack([hkldata.df[lab].to_numpy() for lab in fc_labs]).T
    DFc = (Ds * Fcs).sum(axis=1)
    hkldata.df["DFC"] = DFc
    for i_bin, idxes in hkldata.binned():
        bin_d_min = hkldata.binned_df.d_min[i_bin]
        bin_d_max = hkldata.binned_df.d_max[i_bin]
        # 0: acentric 1: centric
        mean_fom = [numpy.nan, numpy.nan]
        nrefs = [0, 0]
        for c, work, test in centric_and_selections[i_bin]:
            cidxes = numpy.concatenate([work, test])
            S = hkldata.df["S"].to_numpy()[cidxes]
            expip = numpy.exp(numpy.angle(DFc[cidxes])*1j)
            Fo = hkldata.df.FP.to_numpy()[cidxes] / k_ani[cidxes]
            SigFo = hkldata.df.SIGFP.to_numpy()[cidxes] / k_ani[cidxes]
            epsilon = hkldata.df.epsilon.to_numpy()[cidxes]
            nrefs[c] = numpy.sum(numpy.isfinite(Fo))
            DFc_abs = numpy.abs(DFc[cidxes])
            if c == 0:
                Sigma = 2 * SigFo**2 + epsilon * S
                X = 2 * Fo * DFc_abs / Sigma
                m = gemmi.bessel_i1_over_i0(X)
                hkldata.df.loc[cidxes, "FWT"] = (2 * m * Fo - DFc_abs) * expip
            else:
                Sigma = SigFo**2 + epsilon * S
                X = Fo * DFc_abs / Sigma
                m = numpy.tanh(X)
                hkldata.df.loc[cidxes, "FWT"] = (m * Fo) * expip

            hkldata.df.loc[cidxes, "DELFWT"] = (m * Fo - DFc_abs) * expip
            hkldata.df.loc[cidxes, "FOM"] = m
            hkldata.df.loc[cidxes, "X"] = X
            if has_ano:
                Fo_dano = (hkldata.df["F(+)"].to_numpy()[cidxes] - hkldata.df["F(-)"].to_numpy()[cidxes]) / k_ani[cidxes]
                hkldata.df.loc[cidxes, "FAN"] = m * Fo_dano * expip / 2j
            if nrefs[c] > 0: mean_fom[c] = numpy.nanmean(m)

            # remove reflections that should be hidden
            if use != "all":
                # usually use == "work"
                tohide = test if use == "work" else work
                hkldata.df.loc[tohide, "FWT"] = 0j * numpy.nan
                hkldata.df.loc[tohide, "DELFWT"] = 0j * numpy.nan
            fill_sel = numpy.isnan(hkldata.df["FWT"][cidxes].to_numpy())
            hkldata.df.loc[cidxes[fill_sel], "FWT"] = DFc[cidxes][fill_sel]

        Fc = hkldata.df.FC.to_numpy()[idxes] * k_ani[idxes]
        Fo = hkldata.df.FP.to_numpy()[idxes]
        mean_DFc2 = numpy.nanmean(numpy.abs((Ds[idxes,:] * Fcs[idxes,:]).sum(axis=1) * k_ani[idxes])**2)
        mean_log_DFcs = numpy.log(numpy.nanmean(numpy.abs(Ds[idxes,:] * Fcs[idxes,:] * k_ani[idxes,None]), axis=0)).tolist()
        mean_Ds = numpy.nanmean(Ds[idxes,:], axis=0).tolist()
        if sum(nrefs) > 0:
            r = numpy.nansum(numpy.abs(numpy.abs(Fc)-Fo)) / numpy.nansum(Fo)
            cc = utils.hkl.correlation(Fo, numpy.abs(Fc))
            mean_Fo2 = numpy.nanmean(numpy.abs(Fo)**2)
        else:
            r, cc, mean_Fo2 = numpy.nan, numpy.nan, numpy.nan
        stats_data.append([i_bin, nrefs[0], nrefs[1], bin_d_max, bin_d_min,
                           numpy.log(mean_Fo2),
                           numpy.log(numpy.nanmean(numpy.abs(Fc)**2)),
                           numpy.log(mean_DFc2),
                           numpy.log(numpy.mean(hkldata.df["S"].to_numpy()[idxes])),
                           mean_fom[0], mean_fom[1], r, cc] + mean_Ds + mean_log_DFcs)

    DFc_labs = ["log(Mn(|{}{}|))".format(dl,fl) for dl,fl in zip(D_labs, fc_labs)]
    cols = ["bin", "n_a", "n_c", "d_max", "d_min",
            "log(Mn(|Fo|^2))", "log(Mn(|Fc|^2))", "log(Mn(|DFc|^2))",
            "log(Sigma)", "FOM_a", "FOM_c", "R", "CC(|Fo|,|Fc|)"] + D_labs + DFc_labs
    stats = pandas.DataFrame(stats_data, columns=cols)
    title_labs = [["log(Mn(|F|^2)) and variances", ["log(Mn(|Fo|^2))", "log(Mn(|Fc|^2))", "log(Mn(|DFc|^2))", "log(Sigma)"]],
                  ["FOM", ["FOM_a", "FOM_c"]],
                  ["D", D_labs],
                  ["DFc", DFc_labs],
                  ["R-factor", ["R"]],
                  ["CC", ["CC(|Fo|,|Fc|)"]],
                  ["number of reflections", ["n_a", "n_c"]]]
    with open(log_out, "w") as ofs:
        ofs.write(utils.make_loggraph_str(stats, main_title="Statistics",
                                          title_labs=title_labs,
                                          s2=1/stats["d_min"]**2))
    logger.writeln("output log: {}".format(log_out))
# calculate_maps()

def main(args):
    n_per_bin = {"all": 500, "work": 500, "test": 50}[args.use]
    try:
        hkldata, sts, fc_labs, centric_and_selections,free = process_input(hklin=args.hklin,
                                                                           labin=args.labin.split(",") if args.labin else None,
                                                                           n_bins=args.nbins,
                                                                           free=args.free,
                                                                           xyzins=sum(args.model, []),
                                                                           source=args.source,
                                                                           d_max=args.d_max,
                                                                           d_min=args.d_min,
                                                                           n_per_bin=n_per_bin,
                                                                           use=args.use,
                                                                           max_bins=30,
                                                                           keep_charges=args.keep_charges)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    is_int = "I" in hkldata.df

    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None
    
    # Overall scaling & bulk solvent
    # FP/SIGFP will be scaled. Total FC will be added.
    lsq = bulk_solvent_and_lsq_scales(hkldata, sts, fc_labs, use_solvent=not args.no_solvent,
                                      use_int=is_int, mask=mask)
    b_aniso = lsq.b_aniso
    # stats
    stats, overall = calc_r_and_cc(hkldata, centric_and_selections)
    for lab in "R", "CC":
        logger.writeln(" ".join("{} = {:.4f}".format(x, overall[x]) for x in overall if x.startswith(lab)))

    # Estimate ML parameters
    D_labs = ["D{}".format(i) for i in range(len(fc_labs))]

    if args.use_cc:
        assert not is_int
        logger.writeln("Estimating sigma-A parameters from CC..")
        determine_mlf_params_from_cc(hkldata, fc_labs, D_labs, centric_and_selections, args.use)
    else:
        b_aniso = determine_ml_params(hkldata, is_int, fc_labs, D_labs, b_aniso, centric_and_selections, args.D_trans, args.S_trans, args.use)
    if is_int:
        calculate_maps_int(hkldata, b_aniso, fc_labs, D_labs, centric_and_selections,
                           use={"all": "all", "work": "work", "test": "work"}[args.use])
    else:
        log_out = "{}.log".format(args.output_prefix)
        calculate_maps(hkldata, b_aniso, centric_and_selections, fc_labs, D_labs, log_out,
                       use={"all": "all", "work": "work", "test": "work"}[args.use])

    # Write mtz file
    if is_int:
        labs = ["I", "SIGI", "FOM"]
    else:
        labs = ["FP", "SIGFP", "FOM"]
    labs.extend(["FWT", "DELFWT", "FC", "DFC"])
    if "FAN" in hkldata.df:
        labs.append("FAN")
    if not args.no_solvent:
        labs.append("Fbulk")
    if "FREE" in hkldata.df:
        labs.append("FREE")
    labs += D_labs + ["S"]
    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=labs, types={"FOM": "W", "FP":"F", "SIGFP":"Q"})
    return hkldata
# main()
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
