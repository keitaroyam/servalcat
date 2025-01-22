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
        logger.writeln("")
        return None, None
    twin_data = ext.TwinData()
    twin_data.setup(hkldata.miller_array(), hkldata.df.bin, hkldata.sg, hkldata.cell, ops)
    if "I" in hkldata.df:
        Io = hkldata.df.I.to_numpy()
    else:
        Io = hkldata.df.FP.to_numpy()**2
    ccs, nums = [], []
    tmp = []
    for i_bin, bin_idxes in hkldata.binned():
        ratios = [1.]
        ccs.append([])
        nums.append([])
        rs = []
        for i_op, op in enumerate(ops):
            ii = numpy.array(twin_data.pairs(i_op, i_bin))
            val = numpy.all(numpy.isfinite(Io[ii]), axis=1)
            if numpy.sum(val) == 0:
                cc = r = numpy.nan
            else:
                cc = numpy.corrcoef(Io[ii][val].T)[0,1]
                r = numpy.sum(numpy.abs(Io[ii][val, 0] - Io[ii][val, 1])) / numpy.sum(Io[ii][val])
            ratio = (1 - numpy.sqrt(1 - cc**2)) / cc
            ratios.append(ratio)
            ccs[-1].append(cc)
            rs.append(r)
            nums[-1].append(len(val))
        tmp.append(rs + ccs[-1] + nums[-1] + (numpy.array(ratios) / numpy.nansum(ratios)).tolist()[1:])
    df = pandas.DataFrame(tmp, columns=[f"{n}_op{i+1}" for n in ("R", "CC", "num", "raw_est") for i in range(len(ops))])
    with logger.with_prefix(" "):
        logger.writeln(df.to_string(float_format="%.4f"))
    ccs = numpy.array(ccs)
    nums = numpy.array(nums)
    tmp = [{"Operator": gemmi.Op().triplet(),
            "R_twin_obs": 0,
            "CC_mean": 1}]
    for i_op, op in enumerate(ops):
        ii = numpy.array(twin_data.pairs(i_op))
        val = numpy.all(numpy.isfinite(Io[ii]), axis=1)
        if numpy.sum(val) == 0:
            r_obs = numpy.nan
        else:
            r_obs = numpy.sum(numpy.abs(Io[ii][val, 0] - Io[ii][val, 1])) / numpy.sum(Io[ii][val])
        cc = numpy.sum(nums[:,i_op] * ccs[:,i_op]) / numpy.sum(nums[:,i_op])
        tmp.append({"Operator": op.triplet(),
                    "CC_mean": cc, 
                    "R_twin_obs": r_obs,
                    })
    df = pandas.DataFrame(tmp)
    df["Alpha_from_CC"] = (1 - numpy.sqrt(1 - df["CC_mean"]**2)) / df["CC_mean"]
    df["Alpha_from_CC"] /= numpy.nansum(df["Alpha_from_CC"])
    logger.writeln("\n Initial twin fraction estimates:")
    with logger.with_prefix(" "):
        logger.writeln(df.to_string(float_format="%.2f"))

    sel = df["Alpha_from_CC"].to_numpy() > min_alpha
    if sel[1:].sum() == 0:
        logger.writeln(" No twinning detected\n")
        return None, None
    
    if not sel.all():
        ops = [ops[i] for i in range(len(ops)) if sel[i+1]]
        logger.writeln(f"\n Twin operators after filtering small fractions (<= {min_alpha})")
        df = df[sel]
        df["Alpha_from_CC"] /= numpy.nansum(df["Alpha_from_CC"])
        with logger.with_prefix(" "):
            logger.writeln(df.to_string(float_format="%.2f"))
        twin_data = ext.TwinData()
        twin_data.setup(hkldata.miller_array(), hkldata.df.bin, hkldata.sg, hkldata.cell, ops)
    twin_data.alphas = df["Alpha_from_CC"].tolist()
    if "I" not in hkldata.df:
        logger.writeln('Generating "observed" intensities for twin refinement: Io = Fo**2, SigIo = 2*F*SigFo')
        hkldata.df["I"] = hkldata.df.FP**2
        hkldata.df["SIGI"] = 2 * hkldata.df.FP * hkldata.df.SIGFP
    logger.writeln("")
    return twin_data, df

# find_twin_domains_from_data()

def estimate_twin_fractions_from_model(twin_data, hkldata):
    logger.writeln("Estimating twin fractions")
    Ic = numpy.abs(twin_data.f_calc.sum(axis=1))**2
    idx_all = twin_data.twin_related(hkldata.sg)
    Ic_all = Ic[idx_all]
    Ic_all[(idx_all < 0).any(axis=1)] = numpy.nan
    rr = twin_data.obs_related_asu()
    tmp = []
    P_list, cc_oc_list, weight_list = [], [], []
    n_ops = len(twin_data.ops) + 1
    tidxes = numpy.triu_indices(n_ops, 1)
    for i_bin, bin_idxes in hkldata.binned():
        i_tmp = Ic_all[numpy.asarray(twin_data.bin)==i_bin,:]
        i_tmp = i_tmp[numpy.isfinite(i_tmp).all(axis=1)]
        P = numpy.corrcoef(i_tmp.T)
        iobs = hkldata.df.I.to_numpy()[bin_idxes]
        ic_bin = Ic[rr[bin_idxes,:]]
        val = numpy.isfinite(iobs) & numpy.isfinite(ic_bin).all(axis=1) & numpy.all(rr[bin_idxes,:]>=0, axis=1)
        iobs, ic_bin = iobs[val], ic_bin[val,:]
        cc_oc = [numpy.corrcoef(iobs, ic_bin[:,i])[0,1] for i in range(n_ops)]
        P_list.append(P)
        cc_oc_list.append(cc_oc)
        weight_list.append(numpy.sum(val))
        frac_est = numpy.dot(numpy.linalg.pinv(P), cc_oc)
        frac_est /= frac_est.sum()
        tmp.append(P[tidxes].tolist() + cc_oc + [weight_list[-1]] + frac_est.tolist())

    P = numpy.average(P_list, axis=0, weights=weight_list)
    cc_oc = numpy.average(cc_oc_list, axis=0, weights=weight_list)
    frac_est = numpy.dot(numpy.linalg.pinv(P), cc_oc)
    frac_est = numpy.maximum(0, frac_est)
    frac_est /= frac_est.sum()
    df = pandas.DataFrame(tmp, columns=[f"cc_{i+1}_{j+1}" for i, j in zip(*tidxes)] +
                          [f"cc_o_{i+1}" for i in range(n_ops)] +
                          ["nref"] + [f"raw_est_{i+1}" for i in range(n_ops)])
    with logger.with_prefix(" "):
        logger.writeln(df.to_string(float_format="%.4f"))
    logger.write(" Final twin fraction estimate: ")
    logger.writeln(" ".join("%.2f"%x for x in frac_est))
    twin_data.alphas = frac_est
    return df
