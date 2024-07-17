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

def find_twin_domains(hkldata, fc_labs, max_oblique=5, min_alpha=0.05):#, ops=None):
    logger.writeln("Finding possible twin operators")
    ops = gemmi.find_twin_laws(hkldata.cell, hkldata.sg, max_oblique, False)
    logger.writeln(f" {len(ops)} possible twin operator(s) found")
    #for op in ops:
    #    logger.writeln(f"  {op.triplet()}")
    if not ops:
        return
    tw = ext.TwinData()
    tw.setup(hkldata.miller_array().to_numpy(), hkldata.df.bin, hkldata.sg, hkldata.cell, ops)
    asu_idxes = tw.idx_of_asu(hkldata.miller_array(), inv=True)
    Fcs = numpy.vstack([hkldata.df[lab].to_numpy() for lab in fc_labs]).T
    Ic_org = numpy.abs(Fcs.sum(axis=1))**2
    Ic = Ic_org[asu_idxes]
    Ic[asu_idxes < 0] = numpy.nan ####### this confusion should be sorted!
    Ic_all = Ic[tw.twin_related(hkldata.sg, ops)]
    rr = tw.obs_related_asu()
    tmp = []
    for i_bin, bin_idxes in hkldata.binned():
        cc_o_c = []
        i_tmp = Ic_all[numpy.asarray(tw.bin)==i_bin,:]
        i_tmp = i_tmp[numpy.isfinite(i_tmp).all(axis=1),:]
        P = numpy.corrcoef(i_tmp.T)
        iobs = hkldata.df.I.to_numpy()[bin_idxes]
        ic_bin = Ic[rr[bin_idxes,:]]
        val = numpy.isfinite(iobs) & numpy.isfinite(ic_bin).all(axis=1)
        iobs, ic_bin = iobs[val], ic_bin[val,:]
        cc_o_c = [numpy.corrcoef(iobs, ic_bin[:,i])[0,1] for i in range(len(ops)+1)]
        frac_est = numpy.dot(numpy.linalg.pinv(P), cc_o_c)
        tmp.append(frac_est.tolist())

    df = pandas.DataFrame(tmp)
    #print("Estimated twin fractions from CC")
    #print(df.to_string(float_format="%.4f"))
    #print("Normalized:")
    df.iloc[:,:] /= df.sum(axis=1).to_numpy()[:,None]
    #print(df.to_string(float_format="%.4f"))
    mean_alphas = df.mean().tolist()
    logger.write(" Estimated fractions from data/model correlations: ")
    logger.writeln(" ".join("%.2f"%x for x in mean_alphas))
    tmp = [{"Operator": gemmi.Op().triplet(),
            "R_twin_obs": 0,
            "R_twin_calc": 0,
            "Alpha_from_CC": mean_alphas[0]}]
    for i, op in enumerate(ops):
        idxes = numpy.array(tw.pairs(i))
        ii = hkldata.df.I.to_numpy()[idxes]
        sel = numpy.all(numpy.isfinite(ii), axis=1)
        r_obs = numpy.sum(numpy.abs(ii[sel, 0] - ii[sel, 1])) / numpy.sum(ii[sel])
        iic = Ic_org[idxes]
        sel = numpy.all(numpy.isfinite(iic), axis=1)
        r_calc = numpy.sum(numpy.abs(iic[sel, 0] - iic[sel, 1])) / numpy.sum(iic[sel])
        tmp.append({"Operator": op.triplet(),
                    #"CC_twin": cc, # must be bin-average
                    "R_twin_obs": r_obs,
                    "R_twin_calc": r_calc,
                    "Alpha_from_CC": mean_alphas[i+1],
        })
    df = pandas.DataFrame(tmp)
    logger.writeln(df.to_string(float_format="%.2f"))

    sel_idxes = [i for i, a in enumerate(mean_alphas) if i > 0 and a > min_alpha]
    if len(sel_idxes) != len(mean_alphas):
        ops = [ops[i-1] for i in sel_idxes]
        logger.writeln(" Twin operators after filtering small fractions")
        alphas = [mean_alphas[0]] + [mean_alphas[i] for i in sel_idxes]
        sum_alphas = sum(alphas)
        alphas = [a / sum_alphas for a in alphas]
        df = pandas.DataFrame({"Operator": [x.triplet() for x in [gemmi.Op()]+ops],
                               "Alpha": alphas})
        logger.writeln(df.to_string(float_format="%.2f"))
        tw = ext.TwinData()
        tw.setup(hkldata.miller_array().to_numpy(), hkldata.df.bin, hkldata.sg, hkldata.cell, ops)
    else:
        alphas = mean_alphas

    tw.alphas = alphas
    return tw
