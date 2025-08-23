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
from servalcat import utils
from servalcat.spa import fofc
from servalcat.spa import fsc
from servalcat import ext
b_to_u = utils.model.b_to_u
u_to_b = utils.model.u_to_b

def calc_D_and_S(hkldata, lab_obs): # simplified version of fofc.calc_D_and_S()
    bdf = hkldata.binned_df["ml"]
    bdf["D"] = 0.
    bdf["S"] = 0.
    for i_bin, idxes in hkldata.binned("ml"):
        Fo = hkldata.df[lab_obs].to_numpy()[idxes]
        Fc = hkldata.df.FC.to_numpy()[idxes]
        bdf.loc[i_bin, "D"] = numpy.nansum(numpy.real(Fo * numpy.conj(Fc))) / numpy.sum(numpy.abs(Fc)**2)
        bdf.loc[i_bin, "S"] = numpy.nanmean(numpy.abs(Fo - bdf.D[i_bin] * Fc)**2)
# calc_D_and_S()

class LL_SPA:
    def __init__(self, hkldata, st, monlib, lab_obs, source="electron", mott_bethe=True):
        assert source in ("electron", "xray", "custom")
        self.source = source
        self.mott_bethe = False if source != "electron" else mott_bethe
        self.hkldata = hkldata
        self.lab_obs = lab_obs
        self.st = st
        self.monlib = monlib
        self.d_min_max = hkldata.d_min_max()
        self.ll = None
        self.b_aniso = None

    def refine_id(self):
        if self.source in ("electron", "custom"):
            # XXX when custom, it's actually unknown..
            return "ELECTRON MICROSCOPY"
        return "NON-EM SPA" # does not happen, I guess

    def update_ml_params(self):
        # FIXME make sure D > 0
        calc_D_and_S(self.hkldata, self.lab_obs)

    def update_fc(self):
        if self.st.ncs:
            st = self.st.clone()
            st.expand_ncs(gemmi.HowToNameCopiedChain.Short, merge_dist=0)
        else:
            st = self.st

        self.hkldata.df["FC"] = utils.model.calc_fc_fft(st, self.d_min_max[0] - 1e-6,
                                                        monlib=self.monlib,
                                                        source=self.source,
                                                        mott_bethe=self.mott_bethe,
                                                        miller_array=self.hkldata.miller_array())

    def prepare_target(self):
        pass

    def overall_scale(self, min_b=0.5):
        k, b = self.hkldata.scale_k_and_b(lab_ref=self.lab_obs, lab_scaled="FC")
        min_b_iso = self.st[0].calculate_b_aniso_range()[0] # actually min of aniso too
        tmp = min_b_iso + b
        if tmp < min_b: # perhaps better only adjust b_iso that went too small, but we need to recalculate Fc
            logger.writeln("Adjusting overall B to avoid too small value")
            b += min_b - tmp
        logger.writeln("Applying overall B to model: {:.2f}".format(b))
        utils.model.shift_b(self.st[0], b)
        # adjust Fc
        k_iso = self.hkldata.debye_waller_factors(b_iso=b)
        self.hkldata.df["FC"] *= k_iso
        # adjust Fo
        self.hkldata.df[self.lab_obs] /= k
    # overall_scale()

    def calc_target(self): # -LL target for SPA
        ret = 0
        for i_bin, idxes in self.hkldata.binned("ml"):
            Fo = self.hkldata.df[self.lab_obs].to_numpy()[idxes]
            DFc = self.hkldata.df.FC.to_numpy()[idxes] * self.hkldata.binned_df["ml"].D[i_bin]
            S = self.hkldata.binned_df["ml"].S[i_bin]
            ret += numpy.nansum(numpy.abs(Fo - DFc)**2) / S + numpy.log(S) * len(idxes)
        return ret * 2 # friedel mates
    # calc_target()

    def calc_stats(self, bin_stats=False):
        # ignore bin_stats for now. better stats are calculated after refinement
        stats = fsc.calc_fsc_all(self.hkldata, labs_fc=["FC"], lab_f=self.lab_obs)
        fsca = fsc.fsc_average(stats.ncoeffs, stats.fsc_FC_full)
        logger.writeln("FSCaverage = {:.4f}".format(fsca))
        ret = {"summary": {"FSCaverage": fsca, "-LL": self.calc_target()}}
        # XXX in fsc object, _full is misleading - it's not full in cross validation mode
        ret["bin_stats"] = stats
        ret["ml"] = self.hkldata.binned_df["ml"].copy()
        return ret

    def calc_grad(self, refine_params, specs):
        dll_dab = numpy.empty_like(self.hkldata.df[self.lab_obs])
        d2ll_dab2 = numpy.zeros(len(self.hkldata.df.index))
        blur = utils.model.determine_blur_for_dencalc(self.st, self.d_min_max[0] / 3) # TODO need more work
        logger.writeln("blur for deriv= {:.2f}".format(blur))
        for i_bin, idxes in self.hkldata.binned("ml"):
            D = self.hkldata.binned_df["ml"].D[i_bin]
            S = self.hkldata.binned_df["ml"].S[i_bin]
            Fc = self.hkldata.df.FC.to_numpy()[idxes]
            Fo = self.hkldata.df[self.lab_obs].to_numpy()[idxes]
            dll_dab[idxes] = -2 * D / S * (Fo - D * Fc)#.conj()
            d2ll_dab2[idxes] = 2 * D**2 / S

        if self.mott_bethe:
            dll_dab *= self.hkldata.d_spacings()**2 * gemmi.mott_bethe_const()
            d2ll_dab2 *= gemmi.mott_bethe_const()**2

        # we need V for Hessian and V**2/n for gradient.
        d2ll_dab2 *= self.hkldata.cell.volume
        dll_dab_den = self.hkldata.fft_map(data=dll_dab * self.hkldata.debye_waller_factors(b_iso=-blur))
        dll_dab_den.array[:] *= self.hkldata.cell.volume**2 / dll_dab_den.point_count
        self.ll = ext.LL(self.st, refine_params, self.mott_bethe)
        self.ll.set_ncs([x.tr for x in self.st.ncs if not x.given])
        if self.source == "custom":
            self.ll.calc_grad_custom(dll_dab_den, blur)
        else:
            self.ll.calc_grad_it92(dll_dab_den, blur)

        # second derivative
        d2dfw_table = ext.TableS3(*self.hkldata.d_min_max())
        d2dfw_table.make_table(1./self.hkldata.d_spacings(), d2ll_dab2)
        if self.source == "custom":
            self.ll.make_fisher_table_diag_fast_custom(d2dfw_table, 1.)
            self.ll.fisher_diag_from_table_custom()
        else:
            self.ll.make_fisher_table_diag_fast_it92(d2dfw_table)
            self.ll.fisher_diag_from_table_it92()
        #json.dump(dict(b=self.ll.table_bs, pp1=self.ll.pp1, bb=self.ll.bb),
        #          open("ll_fisher.json", "w"), indent=True)
        #a, (b,c) = ll.fisher_for_coo()
        #json.dump(([float(x) for x in a], ([int(x) for x in b], [int(x) for x in c])), open("fisher.json", "w"))
        #logger.writeln("disabling spec_correction in spa target")
        if specs is not None:
            self.ll.spec_correction(specs, use_rr=False)
