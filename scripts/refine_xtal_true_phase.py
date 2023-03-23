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
import os
import shutil
import argparse
import scipy.optimize
from servalcat.utils import logger
from servalcat import utils
from servalcat.xtal.sigmaa import process_input, calculate_maps, calc_DFc, calc_abs_DFc
from servalcat.refine import refine_xtal
from servalcat.refine.xtal import LL_Xtal
from servalcat.refine.refine import Geom, Refine
b_to_u = utils.model.b_to_u
logger.set_file("refine_xtal_using_true_phase.log")

# refine_xtal_norefmac with estimation of D and Sigma using true phase information

def update_ml_params(cls):
    assert len(cls.fc_labs) == 2
    #use = cls.use_in_est
    use = "all"
    for i_bin, _ in cls.hkldata.binned():
        if use == "all":
            idxes = numpy.concatenate([sel[i] for sel in cls.centric_and_selections[i_bin] for i in (1,2)])
        else:
            i = 1 if use == "work" else 2
            idxes = numpy.concatenate([sel[i] for sel in cls.centric_and_selections[i_bin]])

        idxes = idxes[numpy.isfinite(cls.hkldata.df.FP[idxes])]
        FC = cls.hkldata.df.FC.to_numpy()[idxes]
        FC0 = cls.hkldata.df[cls.fc_labs[0]].to_numpy()[idxes]
        FC1 = cls.hkldata.df[cls.fc_labs[1]].to_numpy()[idxes]
        FP = cls.hkldata.df.FP.to_numpy()[idxes]
        varFP = cls.hkldata.df.SIGFP.to_numpy()[idxes]**2
        c = cls.hkldata.df.centric.to_numpy()[idxes] + 1
        eps = cls.hkldata.df.epsilon.to_numpy()[idxes]
        Phitrue = cls.hkldata.df.PHItrue.to_numpy()[idxes]
        Ftrue = FP * numpy.exp(1j * Phitrue)
        Phic = numpy.angle(FC)
        m_true = numpy.cos(Phic - Phitrue)
        logger.writeln("debug: bin {} <fom_true> = {:.4f}".format(i_bin, numpy.nanmean(m_true)))
        # Determine D
        #re_fc1_fc2 = (FC0 * FC1.conj()).real
        #a = numpy.array([[numpy.abs(FC0)**2, re_fc1_fc2],
        #                 [re_fc1_fc2, numpy.abs(FC1)**2]]) / c
        #b = m_true * (3 - c) * FP * (numpy.array([FC0, FC1]) * numpy.exp(-1j * Phic)).real / 2
        #Ds = numpy.linalg.solve(numpy.nansum(a, axis=-1), numpy.nansum(b, axis=-1))
        def calc_a(Fk, Fj):
            ret = numpy.sqrt(numpy.sum(numpy.abs(Fj)**2) * numpy.sum(numpy.abs(Fk)**2))
            ret /= numpy.sum(FP**2)
            ret *= numpy.corrcoef(Fj, Fk)[0,1].real
            return ret
        def calc_b(Fj):
            ret = numpy.sqrt(numpy.sum(numpy.abs(Fj)**2)/numpy.sum(FP**2))
            ret *= numpy.corrcoef(Ftrue, Fj)[0,1].real
            return ret
        a = numpy.array([[calc_a(FC0,FC0), calc_a(FC0,FC1)],
                         [calc_a(FC0,FC1), calc_a(FC1,FC1)]])
        b = numpy.array([calc_b(FC0), calc_b(FC1)])
        Ds = numpy.linalg.solve(a, b)

        # Determine S
        DFc = calc_abs_DFc(Ds, [FC0, FC1])
        S = numpy.nansum(eps * (FP**2 + DFc**2) / c - m_true * (3-c)*eps*FP*DFc - eps * (3-c) * varFP / c) / numpy.nansum(eps**2 / c)

        """
        # Optimize just in case
        x0 = [transD_inv(x) for x in Ds] + [transS_inv(S)]
        x0 = list(Ds) + [S]
        def f(x):
            g = numpy.zeros(3)
            Sigma = (3-c) * varFP + eps * x[2]
            DFc = calc_DFc(x[:2], [FC0, FC1])
            DFc_abs = numpy.abs(DFc)
            tmp = (2 / c - m_true * (3 - c) * FP / DFc_abs) / Sigma
            g[0] = numpy.nansum(tmp * (FC0 * DFc.conj()).real)
            g[1] = numpy.nansum(tmp * (FC1 * DFc.conj()).real)
            g[2] = numpy.nansum(eps / c / Sigma - eps * (FP**2 + DFc_abs**2) / Sigma**2 + m_true * (3 - c) * eps * FP * DFc_abs / Sigma**2)
            return g
        def fprime(x):
            g = numpy.zeros(3)
            Sigma = (3-c) * varFP + eps * x[2]
            DFc = calc_DFc(x[:2], [FC0, FC1])
            DFc_abs = numpy.abs(DFc)
            g[0] = numpy.nansum(numpy.abs(FC0)**2 / c / Sigma)
            g[1] = numpy.nansum(numpy.abs(FC1)**2 / c / Sigma)
            g[2] = numpy.nansum(eps**2 * (-1/c/Sigma**2 + 2*(FP**2+DFc_abs**2)/c/Sigma**3 -2*m_true*(3-c)*FP*DFc_abs/Sigma**3))
            return g
        
        print(x0, f(x0))
        x = scipy.optimize.newton(func=f, fprime=fprime, x0=x0, maxiter=10000)
        #x = scipy.optimize.newton(func=f, x0=x0, maxiter=10000)
        print(x, f(x))
        Ds = transD(x[:2])
        S = transS(x[-1])
        """
        for l, d in zip(cls.D_labs, Ds):
            cls.hkldata.binned_df.loc[i_bin, l] = d
        cls.hkldata.binned_df.loc[i_bin, "S"] = S

    logger.writeln(cls.hkldata.binned_df.to_string())
    for lab in cls.D_labs + ["S"]:
        cls.hkldata.binned_df[lab].where(cls.hkldata.binned_df[lab] > 0, 0., inplace=True) # 0 would be ok?
        cls.hkldata.binned_df[lab].where(cls.hkldata.binned_df[lab] < numpy.inf, 1, inplace=True)


def main(args):
    if not args.output_prefix:
        args.output_prefix = utils.fileio.splitext(os.path.basename(args.model))[0] + "_refined"

    keywords = []
    if args.keywords or args.keyword_file:
        if args.keywords: keywords = sum(args.keywords, [])
        if args.keyword_file: keywords.extend(l for f in sum(args.keyword_file, []) for l in open(f))

    hkldata, sts, fc_labs, centric_and_selections = process_input(hklin=args.hklin,
                                                                  labin=args.labin,
                                                                  n_bins=args.nbins,
                                                                  free=args.free,
                                                                  xyzins=[args.model],
                                                                  source=args.source,
                                                                  d_min=args.d_min)
    mtz = gemmi.read_mtz_file(args.hklin_true)
    df = utils.hkl.hkldata_from_mtz(mtz, [args.labin_true], newlabels=["PHItrue"], require_types=["P"]).df
    hkldata.df = hkldata.df.merge(df)
    hkldata.df["PHItrue"] = numpy.deg2rad(hkldata.df.PHItrue)
    
    st = sts[0]
    monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                   stop_for_unknowns=False)
    h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                "yes":gemmi.HydrogenChange.NoChange,
                "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
    try:
        topo = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                 check_hydrogen=(args.hydrogen=="yes"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    # initialize ADP
    if args.adp != "fix":
        utils.model.reset_adp(st[0], args.bfactor, args.adp == "aniso")
    
    geom = Geom(st, topo, monlib, shake_rms=args.randomize, sigma_b=args.sigma_b, refmac_keywords=keywords,
                jellybody_only=args.jellyonly)
    geom.geom.adpr_max_dist = args.max_dist_for_adp_restraint
    if args.jellybody or args.jellyonly:
        geom.geom.ridge_sigma, geom.geom.ridge_dmax = args.jellybody_params

    ll = LL_Xtal(hkldata, centric_and_selections, args.free, st, monlib, source=args.source, use_solvent=not args.no_solvent)
    ll.update_ml_params = lambda : update_ml_params(ll)
    
    refiner = Refine(st, geom, ll=ll,
                     refine_xyz=not args.fix_xyz,
                     adp_mode=dict(fix=0, iso=1, aniso=2)[args.adp],
                     refine_h=args.refine_h,
                     unrestrained=args.unrestrained)

    refiner.run_cycles(args.ncycle, weight=args.weight)
    utils.fileio.write_model(refiner.st, args.output_prefix, pdb=True, cif=True)

    calculate_maps(ll.hkldata, centric_and_selections, ll.fc_labs, ll.D_labs, args.output_prefix + "_stats.log")

    # Write mtz file
    labs = ["FP", "SIGFP", "FOM", "FWT", "DELFWT", "FC"]
    if not args.no_solvent:
        labs.append("FCbulk")
    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=labs, types={"FOM": "W", "FP":"F", "SIGFP":"Q"})
    logger.writeln("output mtz: {}".format(mtz_out))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hklin_true", required=True)
    parser.add_argument("--labin_true", required=True)
    refine_xtal.add_arguments(parser)
    args = parser.parse_args()
    main(args)
