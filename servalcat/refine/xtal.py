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
integr = sigmaa.integr

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
        self.scaling = sigmaa.LsqScale()
        logger.writeln("will use {} reflections for parameter estimation".format(self.use_in_est))
        logger.writeln("will use {} reflections for refinement".format(self.use_in_target))

    def update_ml_params(self):
        self.b_aniso = sigmaa.determine_ml_params(self.hkldata, self.is_int, self.fc_labs, self.D_labs, self.b_aniso,
                                                   self.centric_and_selections, use=self.use_in_est,
                                                  )#D_trans="splus", S_trans="splus")
        self.hkldata.df["k_aniso"] = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        #determine_mlf_params_from_cc(self.hkldata, self.fc_labs, self.D_labs,
        #                             self.centric_and_selections)


    def update_fc(self):
        if self.st.ncs:
            st = self.st.clone()
            st.expand_ncs(gemmi.HowToNameCopiedChain.Dup, merge_dist=0)
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

        self.scaling.set_data(self.hkldata, fc_list, self.is_int, sigma_cutoff=0)
        self.scaling.scale()
        self.b_aniso = self.scaling.b_aniso
        b = self.scaling.b_iso
        min_b_iso = self.st[0].calculate_b_aniso_range()[0] # actually min of aniso too
        tmp = min_b_iso + b
        if tmp < min_b: # perhaps better only adjust b_iso that went too small, but we need to recalculate Fc
            logger.writeln(" Adjusting overall B to avoid too small value")
            b += min_b - tmp
        logger.writeln(" Applying overall B to model: {:.2f}".format(b))
        utils.model.shift_b(self.st[0], b)
        k_iso = self.hkldata.debye_waller_factors(b_iso=b)
        self.hkldata.df["k_aniso"] = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        if self.use_solvent:
            solvent_scale = self.scaling.get_solvent_scale(self.scaling.k_sol, self.scaling.b_sol,
                                                      1. / self.hkldata.d_spacings().to_numpy()**2)
            self.hkldata.df[self.fc_labs[-1]] = Fmask * solvent_scale
        if self.is_int:
            o_labs = self.hkldata.df.columns.intersection(["I", "SIGI",
                                                           "I(+)","SIGI(+)", "I(-)", "SIGI(-)"])
            self.hkldata.df[o_labs] /= self.scaling.k_overall**2
        else:
            o_labs = self.hkldata.df.columns.intersection(["FP", "SIGFP",
                                                           "F(+)","SIGF(+)", "F(-)", "SIGF(-)"])
            self.hkldata.df[o_labs] /= self.scaling.k_overall

        for lab in self.fc_labs: self.hkldata.df[lab] *= k_iso
        self.hkldata.df["FC"] = self.hkldata.df[self.fc_labs].sum(axis=1)

        # for next cycle
        self.scaling.k_overall = 1.
        self.scaling.b_iso = 0.
    # overall_scale()

    def calc_target(self): # -LL target for MLF or MLI
        ret = 0
        k_aniso = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        f = sigmaa.mli if self.is_int else sigmaa.mlf
        for i_bin, _ in self.hkldata.binned():
            if self.use_in_target == "all":
                idxes = numpy.concatenate([sel[i] for sel in self.centric_and_selections[i_bin] for i in (1,2)])
            else:
                i = 1 if self.use_in_target == "work" else 2
                idxes = numpy.concatenate([sel[i] for sel in self.centric_and_selections[i_bin]])
            ret += f(self.hkldata.df,
                     self.fc_labs,
                     numpy.vstack([self.hkldata.df[lab].to_numpy()[idxes] for lab in self.D_labs]).T,
                     self.hkldata.df.S.to_numpy()[idxes],
                     k_aniso,
                     idxes)
        return ret * 2 # friedel mates
    # calc_target()

    def calc_stats(self, bin_stats=False):
        stats, overall = sigmaa.calc_r_and_cc(self.hkldata, self.centric_and_selections)
        ret = {"summary": overall}
        ret["summary"]["-LL"] = self.calc_target()
        if bin_stats:
            ret["bin_stats"] = stats
        for lab in "R", "CC":
            logger.writeln(" ".join("{} = {:.4f}".format(x, overall[x]) for x in overall if x.startswith(lab)))
        return ret

    def calc_grad(self, atom_pos, refine_xyz, adp_mode, refine_occ, refine_h, specs=None):
        dll_dab = numpy.zeros(len(self.hkldata.df.FC), dtype=numpy.complex128)
        d2ll_dab2 = numpy.empty(len(self.hkldata.df.index))
        d2ll_dab2[:] = numpy.nan
        blur = utils.model.determine_blur_for_dencalc(self.st, self.d_min / 3) # TODO need more work
        logger.writeln("blur for deriv= {:.2f}".format(blur))
        k_ani = self.hkldata.debye_waller_factors(b_cart=self.b_aniso)
        for i_bin, _ in self.hkldata.binned():
            for c, work, test in self.centric_and_selections[i_bin]:
                if self.use_in_target == "all":
                    cidxes = numpy.concatenate([work, test])
                else:
                    cidxes = work if self.use_in_target == "work" else test
                epsilon = self.hkldata.df.epsilon.to_numpy()[cidxes]
                Fcs = numpy.vstack([self.hkldata.df[lab].to_numpy()[cidxes] for lab in self.fc_labs]).T
                Ds = numpy.vstack([self.hkldata.df[lab].to_numpy()[cidxes] for lab in self.D_labs]).T
                S = self.hkldata.df["S"].to_numpy()[cidxes]
                Fc = (Ds * Fcs).sum(axis=1)
                Fc_abs = numpy.abs(Fc)
                expip = numpy.exp(1j * numpy.angle(Fc))
                if self.is_int:
                    Io = self.hkldata.df.I.to_numpy()
                    sigIo = self.hkldata.df.SIGI.to_numpy()
                    to = Io[cidxes] / sigIo[cidxes] - sigIo[cidxes] / (c+1) / k_ani[cidxes]**2 / S / epsilon
                    tf = k_ani[cidxes] * Fc_abs / numpy.sqrt(sigIo[cidxes])
                    sig1 = k_ani[cidxes]**2 * epsilon * S / sigIo[cidxes]
                    k_num = 0.5 if c == 0 else 0. # acentric:0.5, centric: 0.
                    r = ext.integ_J_ratio(k_num, k_num - 0.5, True, to, tf, sig1, c+1,
                                          integr.exp2_threshold, integr.h, integr.N, integr.ewmax)
                    r *= numpy.sqrt(sigIo[cidxes]) / k_ani[cidxes]
                    g = (2-c) * (Fc_abs - r) / epsilon / S  * Ds[:,0]
                    dll_dab[cidxes] = g * expip
                    #d2ll_dab2[cidxes] = (2-c)**2 / S / epsilon * Ds[0]**2 # approximation
                    #d2ll_dab2[cidxes] = ((2-c) / S / epsilon + ((2-c) * r / k_ani[cidxes] / epsilon / S)**2) * Ds[0]**2
                    d2ll_dab2[cidxes] =  g**2
                else:
                    Fo = self.hkldata.df.FP.to_numpy()[cidxes] / k_ani[cidxes]
                    SigFo = self.hkldata.df.SIGFP.to_numpy()[cidxes] / k_ani[cidxes]
                    if c == 0: # acentric
                        Sigma = 2 * SigFo**2 + epsilon * S
                        X = 2 * Fo * Fc_abs / Sigma
                        m = gemmi.bessel_i1_over_i0(X)
                        g = 2 * (Fc_abs - m * Fo) / Sigma * Ds[:,0]  # XXX assuming 0 is atomic structure
                        dll_dab[cidxes] = g * expip
                        d2ll_dab2[cidxes] = (2 / Sigma - (1 - m / X - m**2) * (2 * Fo / Sigma)**2) * Ds[:,0]**2
                    else:
                        Sigma = SigFo**2 + epsilon * S
                        X = Fo * Fc_abs / Sigma
                        #X = X.astype(numpy.float64)
                        m = numpy.tanh(X)
                        g = (Fc_abs - m * Fo) / Sigma * Ds[:,0]
                        dll_dab[cidxes] = g * expip
                        d2ll_dab2[cidxes] = (1. / Sigma - (Fo / (Sigma * numpy.cosh(X)))**2) * Ds[:,0]**2

        if self.mott_bethe:
            dll_dab *= self.hkldata.d_spacings()**2 * gemmi.mott_bethe_const()
            d2ll_dab2 *= gemmi.mott_bethe_const()**2

        # we need V**2/n for gradient.
        dll_dab_den = self.hkldata.fft_map(data=dll_dab * self.hkldata.debye_waller_factors(b_iso=-blur))
        dll_dab_den.array[:] *= self.hkldata.cell.volume**2 / dll_dab_den.point_count
        #asu = dll_dab_den.masked_asu()
        #dll_dab_den.array[:] *= 1 - asu.mask_array # 0 to use
        
        self.ll = ext.LL(self.st, atom_pos, self.mott_bethe, refine_xyz, adp_mode, refine_occ, refine_h)
        self.ll.set_ncs([x.tr for x in self.st.ncs if not x.given])
        if self.source == "neutron":
            self.ll.calc_grad_n92(dll_dab_den, blur)
        else:
            self.ll.calc_grad_it92(dll_dab_den, blur)

        # second derivative
        if self.source == "neutron":
            self.ll.make_fisher_table_diag_direct_n92(1./self.hkldata.d_spacings().to_numpy(),
                                                      d2ll_dab2)
            self.ll.fisher_diag_from_table_n92()
        else:
            self.ll.make_fisher_table_diag_direct_it92(1./self.hkldata.d_spacings().to_numpy(),
                                                       d2ll_dab2)
            self.ll.fisher_diag_from_table_it92()
        #json.dump(dict(b=ll.table_bs, pp1=ll.pp1, bb=ll.bb),
        #          open("ll_fisher.json", "w"), indent=True)
        #a, (b,c) = ll.fisher_for_coo()
        #json.dump(([float(x) for x in a], ([int(x) for x in b], [int(x) for x in c])), open("fisher.json", "w"))
        if specs is not None:
            self.ll.spec_correction(specs)
