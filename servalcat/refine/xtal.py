"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import json
import scipy.sparse
from servalcat.utils import logger
from servalcat.xtal import sigmaa
from servalcat import utils
from servalcat import ext
b_to_u = utils.model.b_to_u
u_to_b = utils.model.u_to_b

def calc_bin_stats(hkldata, centric_and_selections):
    has_int = "I" in hkldata.df
    has_free = "FREE" in hkldata.df
    stats = hkldata.binned_df[["d_max", "d_min"]].copy()
    stats["1/resol^2"] = 1 / stats.d_min**2
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
        for suf in ("work", "free"):
            stats[cclab+suf] = numpy.nan
            stats[rlab+suf] = numpy.nan
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
    return stats
# calc_bin_stats()

def calc_cc_avg(stats):
    cc_labs = [x for x in stats if x.startswith("CC")]
    ret = {x+"avg" : numpy.nan for x in cc_labs}
    for lab in cc_labs:
        if lab.endswith("work"):
            weights = stats["n_work"]
        elif lab.endswith("free"):
            weights = stats["n_free"]
        else:
            weights = stats["n_obs"]
        ret[lab+"avg"] = numpy.average(stats[lab], weights=weights)
    return ret
# calc_cc_avg()

class LL_Xtal:
    def __init__(self, hkldata, centric_and_selections, free, st, monlib, source="xray", mott_bethe=True,
                 use_solvent=False, use_in_est="all", use_in_target="all"):
        assert source in ("electron", "xray", "neutron")
        self.source = source
        self.mott_bethe = False if source != "electron" else mott_bethe
        self.hkldata = hkldata
        self.is_int = "I" in self.hkldata.df
        self.centric_and_selections = centric_and_selections
        self.free = free
        self.st = st
        self.monlib = monlib
        self.d_min = hkldata.d_min_max()[0]
        self.fc_labs = ["FC0"]
        self.use_solvent = use_solvent
        if use_solvent:
            self.fc_labs.append("FCbulk")
            self.hkldata.df["FCbulk"] = 0j
        self.D_labs = ["D{}".format(i) for i in range(len(self.fc_labs))]
        self.k_overall = numpy.ones(len(self.hkldata.df.index))
        self.b_aniso = None
        self.hkldata.df["k_aniso"] = 1.
        self.use_in_est = use_in_est
        self.use_in_target = use_in_target
        self.ll = None
        logger.writeln("will use {} reflections for parameter estimation".format(self.use_in_est))
        logger.writeln("will use {} reflections for refinement".format(self.use_in_target))

    def update_ml_params(self):
        self.b_aniso = sigmaa.determine_ml_params(self.hkldata, self.is_int, self.fc_labs, self.D_labs, self.b_aniso,
                                                   self.centric_and_selections, use=self.use_in_est,
                                                   D_trans="splus", S_trans="splus")
        self.hkldata.df["k_aniso"] = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        for lab in self.D_labs + ["S"]:
            self.hkldata.binned_df[lab].where(self.hkldata.binned_df[lab] > 0, 0.01, inplace=True)
            self.hkldata.binned_df[lab].where(self.hkldata.binned_df[lab] < numpy.inf, 1, inplace=True)
        #determine_mlf_params_from_cc(self.hkldata, self.fc_labs, self.D_labs,
        #                             self.centric_and_selections)


    def update_fc(self):
        if self.st.ncs:
            st = self.st.clone()
            st.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
        else:
            st = self.st

        self.hkldata.df[self.fc_labs[0]] = utils.model.calc_fc_fft(st, self.d_min - 1e-6,
                                                                   monlib=self.monlib,
                                                                   source=self.source,
                                                                   mott_bethe=self.mott_bethe,
                                                                   miller_array=self.hkldata.miller_array())
        self.hkldata.df["FC"] = self.hkldata.df[self.fc_labs].sum(axis=1)
        
    def overall_scale(self, min_b=0.1):
        fc_list = [self.hkldata.df[self.fc_labs[0]].to_numpy()]
        if self.use_solvent:
            Fmask = sigmaa.calc_Fmask(self.st, self.d_min - 1e-6, self.hkldata.miller_array())
            fc_list.append(Fmask)

        scaling = sigmaa.LsqScale(self.hkldata, fc_list, self.is_int, sigma_cutoff=0)
        scaling.scale()
        self.b_aniso = scaling.b_aniso
        b = scaling.b_iso
        min_b_iso = utils.model.minimum_b(self.st[0]) # actually min of aniso too
        tmp = min_b_iso + b
        if tmp < min_b: # perhaps better only adjust b_iso that went too small, but we need to recalculate Fc
            logger.writeln(" Adjusting overall B to avoid too small value")
            b += min_b - tmp
        logger.writeln(" Applying overall B to model: {:.2f}".format(b))
        utils.model.shift_b(self.st[0], b)
        k_iso = self.hkldata.debye_waller_factors(b_iso=b)
        self.hkldata.df["k_aniso"] = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        if self.use_solvent:
            solvent_scale = scaling.get_solvent_scale(scaling.k_sol, scaling.b_sol,
                                                      1. / self.hkldata.d_spacings().to_numpy()**2)
            self.hkldata.df[self.fc_labs[-1]] = Fmask * solvent_scale
        if self.is_int:
            self.hkldata.df["I"] /= scaling.k_overall**2
            self.hkldata.df["SIGI"] /= scaling.k_overall**2
        else:
            self.hkldata.df["FP"] /= scaling.k_overall
            self.hkldata.df["SIGFP"] /= scaling.k_overall

        for lab in self.fc_labs: self.hkldata.df[lab] *= k_iso
        self.hkldata.df["FC"] = self.hkldata.df[self.fc_labs].sum(axis=1)
    # overall_scale()

    def calc_target(self): # -LL target for MLF or MLI
        ret = 0
        k_aniso = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        # in MLF, df.k_aniso is used
            
        for i_bin, _ in self.hkldata.binned():
            Ds = [self.hkldata.binned_df.loc[i_bin, lab] for lab in self.D_labs]
            if self.is_int:
                if self.use_in_target == "all":
                    idxes = numpy.concatenate([sel[i] for sel in self.centric_and_selections[i_bin] for i in (1,2)])
                else:
                    i = 1 if self.use_in_target == "work" else 2
                    idxes = numpy.concatenate([sel[i] for sel in self.centric_and_selections[i_bin]])
                ret += sigmaa.mli(self.hkldata.df,
                                  self.fc_labs,
                                  Ds,
                                  self.hkldata.binned_df.S[i_bin],
                                  k_aniso,
                                  idxes)
            else:
                ret += sigmaa.mlf(self.hkldata.df,
                                  self.fc_labs,
                                  Ds,
                                  self.hkldata.binned_df.S[i_bin],
                                  self.centric_and_selections[i_bin],
                                  use=self.use_in_target)

        return ret * 2 # friedel mates
    # calc_target()

    def calc_stats(self, bin_stats=False):
        if self.is_int:
            calc_r = lambda sel: utils.hkl.r_factor(self.hkldata.df.I[sel],
                                                    numpy.abs(self.hkldata.df.FC[sel] * self.hkldata.df.k_aniso[sel])**2)
            rlab = "R2"
            cclab = "CCI"
        else:
            calc_r = lambda sel: utils.hkl.r_factor(self.hkldata.df.FP[sel],
                                                    numpy.abs(self.hkldata.df.FC[sel] * self.hkldata.df.k_aniso[sel]))
            rlab = "R"
            cclab = "CCF"
        ret = {"summary": {}}
        stats = calc_bin_stats(self.hkldata, self.centric_and_selections)
        ret["summary"].update(calc_cc_avg(stats))
        if "FREE" in self.hkldata.df:
            test_sel = (self.hkldata.df.FREE == self.free).fillna(False)
            r_free = calc_r(test_sel)
            r_work = calc_r(~test_sel)
            logger.writeln("{}_work = {:.4f} {}_free = {:.4f}".format(rlab, r_work, rlab, r_free))
            ret["summary"]["{}work".format(rlab)] = r_work
            ret["summary"]["{}free".format(rlab)] = r_free
            cc_free = ret["summary"]["{}freeavg".format(cclab)]
            cc_work = ret["summary"]["{}workavg".format(cclab)]
            logger.writeln("{}avg_work = {:.4f} {}avg_free = {:.4f}".format(cclab, cc_work, cclab, cc_free))
        else:
            r = calc_r(slice(None))
            cc = ret["summary"]["{}avg".format(cclab)]
            logger.writeln("{} = {:.4f}".format(rlab, r))
            logger.writeln("{}avg = {:.4f}".format(cclab, cc))
            ret["summary"][rlab] = r
        ret["summary"]["-LL"] = self.calc_target()
        if bin_stats:
            ret["bin_stats"] = stats
        return ret

    def calc_grad(self, refine_xyz, adp_mode, refine_h, specs):
        dll_dab = numpy.zeros(len(self.hkldata.df.FC), dtype=numpy.complex128)
        d2ll_dab2 = numpy.empty(len(self.hkldata.df.index))
        d2ll_dab2[:] = numpy.nan
        blur = utils.model.determine_blur_for_dencalc(self.st, self.d_min / 3) # TODO need more work
        logger.writeln("blur for deriv= {:.2f}".format(blur))
        k_ani = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        for i_bin, _ in self.hkldata.binned():
            bin_d_min = self.hkldata.binned_df.d_min[i_bin]
            bin_d_max = self.hkldata.binned_df.d_max[i_bin]
            Ds = [max(0., self.hkldata.binned_df[lab][i_bin]) for lab in self.D_labs] # negative D is replaced with zero here
            S = self.hkldata.binned_df.S[i_bin]
            for c, work, test in self.centric_and_selections[i_bin]:
                if self.use_in_target == "all":
                    cidxes = numpy.concatenate([work, test])
                else:
                    cidxes = work if self.use_in_target == "work" else test
                Fcs = [self.hkldata.df[lab].to_numpy()[cidxes] for lab in self.fc_labs]
                Fc = sigmaa.calc_DFc(Ds, Fcs) # sum(D * Fc)
                expip = numpy.exp(1j * numpy.angle(Fc))
                epsilon = self.hkldata.df.epsilon.to_numpy()[cidxes]
                Fc_abs = numpy.abs(Fc)

                if self.is_int:
                    Io = self.hkldata.df.I.to_numpy()
                    sigIo = self.hkldata.df.SIGI.to_numpy()
                    to = Io[cidxes] / sigIo[cidxes] - sigIo[cidxes] / (c+1) / k_ani[cidxes]**2 / S / epsilon
                    tf = k_ani[cidxes] * Fc_abs / numpy.sqrt(sigIo[cidxes])
                    sig1 = numpy.sqrt(k_ani[cidxes]) * S / sigIo[cidxes]
                    if c == 0: # acentric
                        k_num, k_den = 0.5, 0.
                    else:
                        k_num, k_den = 0., -0.5
                    r = ext.integ_J_ratio(k_num, k_den, True, to, tf, sig1, c+1) * numpy.sqrt(sigIo[cidxes])
                    dll_dab[cidxes] =  (2-c) * (Fc_abs - r / k_ani[cidxes]) / epsilon / S  * Ds[0] * expip
                    #d2ll_dab2[cidxes] = (2-c)**2 / S / epsilon * Ds[0]**2 # approximation
                    #d2ll_dab2[cidxes] = ((2-c) / S / epsilon + ((2-c) * r / k_ani[cidxes] / epsilon / S)**2) * Ds[0]**2
                    d2ll_dab2[cidxes] =  ((2-c) * (Fc_abs - r / k_ani[cidxes]) / epsilon / S  * Ds[0])**2
                else:
                    Fo = self.hkldata.df.FP.to_numpy()[cidxes]
                    SigFo = self.hkldata.df.SIGFP.to_numpy()[cidxes]
                    if c == 0: # acentric
                        Sigma = 2 * SigFo**2 + epsilon * S * k_ani[cidxes]**2
                        X = 2 * Fo * Fc_abs * k_ani[cidxes] / Sigma
                        m = gemmi.bessel_i1_over_i0(X)
                        g = (2 * k_ani[cidxes]**2 * Fc_abs / Sigma - m * 2 * Fo * k_ani[cidxes] / Sigma) * Ds[0]  # XXX assuming 0 is atomic structure
                        dll_dab[cidxes] = g * expip
                        d2ll_dab2[cidxes] = (2 * k_ani[cidxes]**2 / Sigma - (1 - m / X - m**2) * (2 * Fo * k_ani[cidxes] / Sigma)**2) * Ds[0]**2
                    else:
                        Sigma = SigFo**2 + epsilon * S * k_ani[cidxes]**2
                        X = Fo * Fc_abs * k_ani[cidxes] / Sigma
                        #X = X.astype(numpy.float64)
                        m = numpy.tanh(X)
                        g = (Fc_abs * k_ani[cidxes]**2 / Sigma - m * Fo * k_ani[cidxes] / Sigma) * Ds[0]
                        dll_dab[cidxes] = g * expip
                        d2ll_dab2[cidxes] = (k_ani[cidxes]**2 / Sigma - (Fo * k_ani[cidxes] / (Sigma * numpy.cosh(X)))**2) * Ds[0]**2

        if self.mott_bethe:
            dll_dab *= self.hkldata.d_spacings()**2 * gemmi.mott_bethe_const()
            d2ll_dab2 *= gemmi.mott_bethe_const()**2

        # we need V for Hessian and V**2/n for gradient.
        d2ll_dab2 *= self.hkldata.cell.volume
        dll_dab_den = self.hkldata.fft_map(data=dll_dab * self.hkldata.debye_waller_factors(b_iso=-blur))
        dll_dab_den.array[:] *= self.hkldata.cell.volume**2 / dll_dab_den.point_count
        #asu = dll_dab_den.masked_asu()
        #dll_dab_den.array[:] *= 1 - asu.mask_array # 0 to use
        
        self.ll = ext.LL(self.st, self.mott_bethe, refine_xyz, adp_mode, refine_h)
        self.ll.set_ncs([x.tr for x in self.st.ncs if not x.given])
        if self.source == "neutron":
            self.ll.calc_grad_n92(dll_dab_den, blur)
        else:
            self.ll.calc_grad_it92(dll_dab_den, blur)

        # second derivative
        d2dfw_table = ext.TableS3(*self.hkldata.d_min_max())
        valid_sel = numpy.isfinite(d2ll_dab2)
        d2dfw_table.make_table(1./self.hkldata.d_spacings().to_numpy()[valid_sel], d2ll_dab2[valid_sel])
        if self.source == "neutron":
            self.ll.make_fisher_table_diag_fast_n92(d2dfw_table)
            self.ll.fisher_diag_from_table_n92()
        else:
            self.ll.make_fisher_table_diag_fast_it92(d2dfw_table)
            self.ll.fisher_diag_from_table_it92()
        #json.dump(dict(b=ll.table_bs, pp1=ll.pp1, bb=ll.bb),
        #          open("ll_fisher.json", "w"), indent=True)
        #a, (b,c) = ll.fisher_for_coo()
        #json.dump(([float(x) for x in a], ([int(x) for x in b], [int(x) for x in c])), open("fisher.json", "w"))

        self.ll.spec_correction(specs)
