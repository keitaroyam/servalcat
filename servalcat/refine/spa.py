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
b_to_u = utils.model.b_to_u
u_to_b = utils.model.u_to_b

def calc_D_and_S(hkldata): # simplified version of fofc.calc_D_and_S()
    bdf = hkldata.binned_df
    bdf["D"] = 0.
    bdf["S"] = 0.
    for i_bin, idxes in hkldata.binned():
        Fo = hkldata.df.FP.to_numpy()[idxes]
        Fc = hkldata.df.FC.to_numpy()[idxes]
        bdf.loc[i_bin, "D"] = numpy.nansum(numpy.real(Fo * numpy.conj(Fc))) / numpy.sum(numpy.abs(Fc)**2)
        bdf.loc[i_bin, "S"] = numpy.nanmean(numpy.abs(Fo - bdf.D[i_bin] * Fc)**2)
# calc_D_and_S()

class LL_SPA:
    def __init__(self, hkldata, st, monlib, source="electron", mott_bethe=True):
        assert source in ("electron", "xray")
        self.source = source
        self.mott_bethe = False if source != "electron" else mott_bethe
        self.hkldata = hkldata
        self.st = st
        self.monlib = monlib
        self.d_min = hkldata.d_min_max()[0]

    def update_ml_params(self):
        # FIXME make sure D > 0
        calc_D_and_S(self.hkldata)
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

    def overall_scale(self, min_b=0.5):
        k, b = self.hkldata.scale_k_and_b(lab_ref="FP", lab_scaled="FC")
        min_b_iso = utils.model.minimum_b(self.st[0]) # actually min of aniso too
        tmp = min_b_iso + b
        if tmp < min_b: # perhaps better only adjust b_iso that went too small, but we need to recalculate Fc
            logger.writeln("Adjusting overall B to avoid too small value")
            b += min_b - tmp
        logger.writeln("Applying overall B to model: {:.2f}".format(b))
        for cra in self.st[0].all():
            cra.atom.b_iso += b
            if cra.atom.aniso.nonzero():
                cra.atom.aniso.u11 += b * b_to_u
                cra.atom.aniso.u22 += b * b_to_u
                cra.atom.aniso.u33 += b * b_to_u

        # adjust Fc
        k_iso = self.hkldata.debye_waller_factors(b_iso=b)
        self.hkldata.df["FC"] *= k_iso
    # overall_scale()

    def calc_target(self): # -LL target for SPA
        ret = 0
        for i_bin, idxes in self.hkldata.binned():
            Fo = self.hkldata.df.FP.to_numpy()[idxes]
            DFc = self.hkldata.df.FC.to_numpy()[idxes] * self.hkldata.binned_df.D[i_bin]
            ret += numpy.nansum(numpy.abs(Fo - DFc)**2) / self.hkldata.binned_df.S[i_bin]
        return ret * 2 # friedel mates
    # calc_target()

    def calc_stats(self):
        stats = fsc.calc_fsc_all(self.hkldata, labs_fc=["FC"], lab_f="FP")
        fsca = fsc.fsc_average(stats.ncoeffs, stats.fsc_FC_full)
        logger.writeln("FSCaverage = {:.4f}".format(fsca))
        return {"fsc": stats, "summary": {"FSCaverage": fsca, "-LL": self.calc_target()}}

    def calc_grad(self, refine_xyz, adp_mode, refine_h, specs): # specs not used
        dll_dab = numpy.empty_like(self.hkldata.df.FP)
        d2ll_dab2 = numpy.zeros(len(self.hkldata.df.index))
        blur = utils.model.determine_blur_for_dencalc(self.st, self.d_min / 3) # TODO need more work
        logger.writeln("blur for deriv= {:.2f}".format(blur))
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
        dll_dab_den = self.hkldata.fft_map(data=dll_dab * self.hkldata.debye_waller_factors(b_iso=-blur))
        dll_dab_den.array[:] *= self.hkldata.cell.volume**2 / dll_dab_den.point_count
        #asu = dll_dab_den.masked_asu()
        #dll_dab_den.array[:] *= 1 - asu.mask_array # 0 to use
        #atoms = [x.atom for x in self.st[0].all()]
        atoms = [None for _ in range(self.st[0].count_atom_sites())]
        for cra in self.st[0].all(): atoms[cra.atom.serial-1] = cra.atom
        ll = gemmi.LLX(self.st.cell, self.hkldata.sg, atoms, self.mott_bethe, refine_xyz, adp_mode, refine_h)
        ll.set_ncs([x.tr for x in self.st.ncs if not x.given])
        vn = ll.calc_grad(dll_dab_den, blur)
        d2dfw_table = gemmi.TableS3(*self.hkldata.d_min_max())
        d2dfw_table.make_table(1./self.hkldata.d_spacings(), d2ll_dab2)

        b_iso_all = [cra.atom.aniso.trace() / 3 * u_to_b if cra.atom.aniso.nonzero() else cra.atom.b_iso
                     for cra in self.st[0].all()]
        b_iso_min = min(b_iso_all)
        b_iso_max = max(b_iso_all)
        elems = set(cra.atom.element for cra in self.st[0].all())
        b_sf_min = 0 #min(min(e.it92.b) for e in elems) # because there is constants
        b_sf_max = max(max(e.it92.b) for e in elems)
        fisher_b_min = b_iso_min + b_sf_min
        fisher_b_max = 2 * (b_iso_max + b_sf_max)
        logger.writeln("preparing fast Fisher table for B= {:.2f} - {:.2f}".format(fisher_b_min, fisher_b_max))
        ll.make_fisher_table_diag_fast(fisher_b_min, fisher_b_max, d2dfw_table)
        #json.dump(dict(b=ll.table_bs, pp1=ll.pp1, bb=ll.bb),
        #          open("ll_fisher.json", "w"), indent=True)
        #a, (b,c) = ll.fisher_for_coo()
        #json.dump(([float(x) for x in a], ([int(x) for x in b], [int(x) for x in c])), open("fisher.json", "w"))
        coo = scipy.sparse.coo_matrix(ll.fisher_for_coo())
        lil = coo.tolil()
        rows, cols = lil.nonzero()
        lil[cols,rows] = lil[rows,cols]
        return numpy.array(vn), lil
