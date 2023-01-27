"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import scipy.sparse
from servalcat.utils import logger
from servalcat import utils
from servalcat.spa import fofc
from servalcat.spa import fsc

class LL_SPA:
    def __init__(self, hkldata, st, monlib, source="electron", mott_bethe=True):
        self.source = source
        self.mott_bethe = False if source != "electron" else mott_bethe
        self.hkldata = hkldata
        self.st = st
        self.monlib = monlib
        self.d_min = hkldata.d_min_max()[0]
        self.update_fc()
        self.calc_fsc()

    def update_ml_params(self):
        # FIXME S should include variance of noise.
        # FIXME make sure D > 0 and S > 0
        # following function needs half maps - but they are actually not needed absolutely
        fofc.calc_D_and_S(self.hkldata)
        # quick fix
        self.hkldata.binned_df.S += self.hkldata.binned_df.var_noise
        logger.writeln(self.hkldata.binned_df.to_string(columns=["d_max", "d_min", "D", "S"]))

    def update_fc(self):
        if self.st.ncs:
            st = self.st.clone()
            st.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
        else:
            st = self.st

        self.hkldata.df["FC"] = utils.model.calc_fc_fft(st, self.d_min - 1e-6,
                                                        cutoff=1e-7,
                                                        monlib=self.monlib,
                                                        source=self.source,
                                                        mott_bethe=self.mott_bethe,
                                                        miller_array=self.hkldata.miller_array())

    def overall_scale(self):
        k, b = self.hkldata.scale_k_and_b(lab_ref="FP", lab_scaled="FC")
        logger.writeln("Applying overall B to model: {:.2f}".format(b))
        for cra in self.st[0].all():
            # aniso not considered!
            cra.atom.b_iso += b

        # adjust Fc
        k_iso = self.hkldata.debye_waller_factors(b_iso=b)
        self.hkldata.df["FC"] *= k_iso
    # overall_scale()

    def calc_target(self): # -LL target for SPA
        ret = 0
        for i_bin, idxes in self.hkldata.binned():
            Fo = self.hkldata.df.FP.to_numpy()[idxes]
            DFc = self.hkldata.df.FC.to_numpy()[idxes] * self.hkldata.binned_df.D[i_bin]
            ret += numpy.sum(numpy.abs(Fo - DFc)**2) / self.hkldata.binned_df.S[i_bin]
        return ret * 2 # friedel mates
    # calc_target()

    def calc_fsc(self):
        stats = fsc.calc_fsc_all(self.hkldata, labs_fc=["FC"], lab_f="FP")
        fsca = fsc.fsc_average(stats.ncoeffs, stats.fsc_FC_full)
        logger.writeln("FSCaverage = {:.4f}".format(fsca))
        return stats

    def calc_grad(self, refine_xyz, refine_adp):
        dll_dab = numpy.empty_like(self.hkldata.df.FP)
        d2ll_dab2 = numpy.zeros(len(self.hkldata.df.index))
        for i_bin, idxes in self.hkldata.binned():
            D = self.hkldata.binned_df.D[i_bin]
            S = self.hkldata.binned_df.S[i_bin]
            Fc = self.hkldata.df.FC.to_numpy()[idxes]
            Fo = self.hkldata.df.FP.to_numpy()[idxes]
            dll_dab[idxes] = -2 * D / S * (Fo - D * Fc)#.conj()
            d2ll_dab2[idxes] = 2 * D**2 / S

        if self.mott_bethe:
            dll_dab *= self.hkldata.d_spacings()**2 * gemmi.mott_bethe_const()
            d2ll_dab2 *= gemmi.mott_bethe_const()**2

        # strangely, we need V for Hessian and V**2/n for gradient.
        d2ll_dab2 *= self.hkldata.cell.volume
        dll_dab_den = self.hkldata.fft_map(data=dll_dab)
        dll_dab_den.array[:] *= self.hkldata.cell.volume**2 / dll_dab_den.point_count

        #atoms = [x.atom for x in self.st[0].all()]
        atoms = [None for _ in range(self.st[0].count_atom_sites())]
        for cra in self.st[0].all(): atoms[cra.atom.serial-1] = cra.atom
        ll = gemmi.LLX(self.hkldata.cell, self.hkldata.sg, atoms, self.mott_bethe)
        ll.set_ncs([x.tr for x in self.st.ncs if not x.given])
        vn = ll.calc_grad(dll_dab_den, refine_xyz, refine_adp)
        d2dfw_table = gemmi.TableS3(*self.hkldata.d_min_max())
        d2dfw_table.make_table(1./self.hkldata.d_spacings(), d2ll_dab2)

        b_iso_min = min(cra.atom.b_iso for cra in self.st[0].all())
        b_iso_max = max(cra.atom.b_iso for cra in self.st[0].all())
        elems = set(cra.atom.element for cra in self.st[0].all())
        b_sf_min = 0 #min(min(e.it92.b) for e in elems) # because there is constants
        b_sf_max = max(max(e.it92.b) for e in elems)
        ll.make_fisher_table_diag_fast(b_iso_min + b_sf_min, b_iso_max + b_sf_max, d2dfw_table)
        am = ll.fisher_diag_from_table(refine_xyz, refine_adp)
        return numpy.array(vn), scipy.sparse.diags(am)
