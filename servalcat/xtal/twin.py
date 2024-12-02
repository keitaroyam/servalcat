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
from servalcat.utils import logger
from servalcat import utils
from servalcat import ext

def find_twin_domains_from_data(hkldata, max_oblique=5, min_alpha=0.05):
    logger.writeln("Finding possible twin operators from data")
    ops = gemmi.find_twin_laws(hkldata.cell, hkldata.sg, max_oblique, False)
    logger.writeln(f" {len(ops)} possible twin operator(s) found")
    #for op in ops:
    #    logger.writeln(f"  {op.triplet()}")
    if not ops:
        return
    twin_data = ext.TwinData()
    twin_data.setup(hkldata.miller_array(), hkldata.df.bin, hkldata.sg, hkldata.cell, ops)
    if "I" in hkldata.df:
        Io = hkldata.df.I.to_numpy()
    else:
        Io = hkldata.df.FP.to_numpy()**2
    alphas = []
    ccs, nums = [], []
    for i_bin, bin_idxes in hkldata.binned():
        ratios = [1.]
        ccs.append([])
        nums.append([])
        for i_op, op in enumerate(ops):
            ii = numpy.array(twin_data.pairs(i_op, i_bin))
            val = numpy.all(numpy.isfinite(Io[ii]), axis=1)
            if numpy.sum(val) == 0:
                cc = numpy.nan
            else:
                cc = numpy.corrcoef(Io[ii][val].T)[0,1]
            rr = (1 - numpy.sqrt(1 - cc**2)) / cc
            ratios.append(rr)
            ccs[-1].append(cc)
            nums[-1].append(len(val))
        alphas.append(numpy.array(ratios) / numpy.nansum(ratios))
    alphas = numpy.maximum(0, numpy.mean(alphas, axis=0))
    alphas /= numpy.nansum(alphas)
    ccs = numpy.array(ccs)
    nums = numpy.array(nums)
    tmp = [{"Operator": gemmi.Op().triplet(),
            "R_twin_obs": 0,
            "CC_mean": 1,
            "Alpha_from_CC": alphas[0]}]
    for i_op, op in enumerate(ops):
        ii = numpy.array(twin_data.pairs(i_op))
        val = numpy.all(numpy.isfinite(Io[ii]), axis=1)
        if numpy.sum(val) == 0:
            r_obs = numpy.nan
        else:
            r_obs = numpy.sum(numpy.abs(Io[ii][val, 0] - Io[ii][val, 1])) / numpy.sum(Io[ii][val])
        tmp.append({"Operator": op.triplet(),
                    "CC_mean": numpy.sum(nums[:,i_op] * ccs[:,i_op]) / numpy.sum(nums[:,i_op]),
                    "R_twin_obs": r_obs,
                    "Alpha_from_CC": alphas[i_op+1],
                    })
    df = pandas.DataFrame(tmp)
    logger.writeln(df.to_string(float_format="%.2f"))

    sel_idxes = [i for i, a in enumerate(alphas) if i > 0 and a > min_alpha]
    if not sel_idxes:
        logger.writeln(" No twinning detected")
        return
    
    if len(sel_idxes) + 1 != len(alphas):
        ops = [ops[i-1] for i in sel_idxes]
        logger.writeln(" Twin operators after filtering small fractions")
        alphas = numpy.array([alphas[0]] + [alphas[i] for i in sel_idxes])
        alphas /= numpy.sum(alphas)
        df = pandas.DataFrame({"Operator": [x.triplet() for x in [gemmi.Op()]+ops],
                               "Alpha": alphas})
        logger.writeln(df.to_string(float_format="%.2f"))
        twin_data = ext.TwinData()
        twin_data.setup(hkldata.miller_array(), hkldata.df.bin, hkldata.sg, hkldata.cell, ops)
    twin_data.alphas = alphas
    if "I" not in hkldata.df:
        logger.writeln('Generating "observed" intensities for twin refinement: Io = Fo**2, SigIo = 2*F*SigFo')
        hkldata.df["I"] = hkldata.df.FP**2
        hkldata.df["SIGI"] = 2 * hkldata.df.FP * hkldata.df.SIGFP
    return twin_data

# find_twin_domains_from_data()

def estimate_twin_fractions_from_model(twin_data, hkldata):
    logger.writeln("Estimating twin fractions")
    Ic = numpy.abs(twin_data.f_calc.sum(axis=1))**2
    Ic_all = Ic[twin_data.twin_related(hkldata.sg)]
    rr = twin_data.obs_related_asu()
    tmp = []
    for i_bin, bin_idxes in hkldata.binned():
        cc_o_c = []
        i_tmp = Ic_all[numpy.asarray(twin_data.bin)==i_bin,:]
        P = numpy.corrcoef(i_tmp.T)
        iobs = hkldata.df.I.to_numpy()[bin_idxes]
        ic_bin = Ic[rr[bin_idxes,:]]
        val = numpy.isfinite(iobs) & numpy.isfinite(ic_bin).all(axis=1)
        iobs, ic_bin = iobs[val], ic_bin[val,:]
        cc_o_c = [numpy.corrcoef(iobs, ic_bin[:,i])[0,1] for i in range(len(twin_data.ops)+1)]
        frac_est = numpy.dot(numpy.linalg.pinv(P), cc_o_c)
        tmp.append(frac_est.tolist())

    df = pandas.DataFrame(tmp)
    df.iloc[:,:] /= df.sum(axis=1).to_numpy()[:,None]
    mean_alphas = numpy.maximum(0, df.mean())
    mean_alphas /= numpy.sum(mean_alphas)
    logger.write(" Estimated fractions from data-model correlations: ")
    logger.writeln(" ".join("%.2f"%x for x in mean_alphas))
    twin_data.alphas = mean_alphas

