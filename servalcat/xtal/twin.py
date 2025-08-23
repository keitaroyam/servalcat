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
import scipy.optimize
from servalcat.utils import logger
from servalcat import utils
from servalcat import ext

def find_twin_domains_from_data(hkldata, max_oblique=5, min_cc=0.2):
    logger.writeln("Finding possible twin operators from data")
    ops = gemmi.find_twin_laws(hkldata.cell, hkldata.sg, max_oblique, False)
    logger.writeln(f" {len(ops)} possible twin operator(s) found")
    #for op in ops:
    #    logger.writeln(f"  {op.triplet()}")
    if not ops:
        logger.writeln("")
        return None, None
    twin_data = ext.TwinData()
    twin_data.setup(hkldata.miller_array(), hkldata.df.bin_ml, hkldata.sg, hkldata.cell, ops)
    if "I" in hkldata.df:
        Io = hkldata.df.I.to_numpy()
    else:
        Io = hkldata.df.FP.to_numpy()**2
    ccs, nums = [], []
    tmp = []
    for i_bin, bin_idxes in hkldata.binned("ml"):
        ccs.append([])
        nums.append([])
        rs = []
        for i_op, op in enumerate(ops):
            cc = r = numpy.nan
            ii = numpy.array(twin_data.pairs(i_op, i_bin))
            val = numpy.all(numpy.isfinite(Io[ii]), axis=1) if ii.size != 0 else []
            if numpy.sum(val) != 0:
                cc = numpy.corrcoef(Io[ii][val].T)[0,1]
                r = numpy.sum(numpy.abs(Io[ii][val, 0] - Io[ii][val, 1])) / numpy.sum(Io[ii][val])
            ccs[-1].append(cc)
            rs.append(r)
            nums[-1].append(len(val))
        tmp.append(rs + ccs[-1] + nums[-1])
    df = pandas.DataFrame(tmp, columns=[f"{n}_op{i+1}" for n in ("R", "CC", "num") for i in range(len(ops))])
    with logger.with_prefix(" "):
        logger.writeln(df.to_string(float_format="%.4f"))
    ccs = numpy.array(ccs)
    nums = numpy.array(nums)
    tmp = [{"Operator": "h,k,l",
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
        tmp.append({"Operator": op.as_hkl().triplet(),
                    "CC_mean": cc, 
                    "R_twin_obs": r_obs,
                    })
    df = pandas.DataFrame(tmp)
    with logger.with_prefix(" "):
        logger.writeln(df.to_string(float_format="%.2f"))

    sel = df["CC_mean"].to_numpy() > min_cc
    if sel[1:].sum() == 0:
        logger.writeln(" No possible twinning detected\n")
        return None, None
    
    if 0:#not sel.all():
        ops = [ops[i] for i in range(len(ops)) if sel[i+1]]
        logger.writeln(f"\n Twin operators after filtering small correlations (<= {min_cc})")
        df = df[sel]
        with logger.with_prefix(" "):
            logger.writeln(df.to_string(float_format="%.2f"))
        twin_data = ext.TwinData()
        twin_data.setup(hkldata.miller_array(), hkldata.df.bin_ml, hkldata.sg, hkldata.cell, ops)
    twin_data.alphas = [1. / len(twin_data.alphas) for _ in range(len(twin_data.alphas)) ]
    if "I" not in hkldata.df:
        logger.writeln('Generating "observed" intensities for twin refinement: Io = Fo**2, SigIo = 2*F*SigFo')
        hkldata.df["I"] = hkldata.df.FP**2
        hkldata.df["SIGI"] = 2 * hkldata.df.FP * hkldata.df.SIGFP
    logger.writeln("")
    return twin_data, df

# find_twin_domains_from_data()

def estimate_twin_fractions_from_model(twin_data, hkldata, min_alpha=0.02):
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
    if "CC*" in hkldata.binned_df["ml"]:
        logger.writeln(" data-correlations are corrected using CC*")
    for i_bin, bin_idxes in hkldata.binned("ml"): # XXX
        i_tmp = Ic_all[numpy.asarray(twin_data.bin)==i_bin,:]
        i_tmp = i_tmp[numpy.isfinite(i_tmp).all(axis=1)]
        P = numpy.corrcoef(i_tmp.T)
        iobs = hkldata.df.I.to_numpy()[bin_idxes]
        ic_bin = Ic[rr[bin_idxes,:]]
        val = numpy.isfinite(iobs) & numpy.isfinite(ic_bin).all(axis=1) & numpy.all(rr[bin_idxes,:]>=0, axis=1)
        iobs, ic_bin = iobs[val], ic_bin[val,:]
        cc_star = hkldata.binned_df["ml"]["CC*"][i_bin] if "CC*" in hkldata.binned_df["ml"] else 1
        if cc_star < 0.5:
            break
        cc_oc = [numpy.corrcoef(iobs, ic_bin[:,i])[0,1] / cc_star for i in range(n_ops)]
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
    logger.writeln(" ".join("%.4f"%x for x in frac_est))
    twin_data.alphas = frac_est

    if numpy.logical_and(0 < frac_est, frac_est < min_alpha).any():
        frac_est[frac_est < min_alpha] = 0.
        frac_est /= frac_est.sum()
        logger.write(" Small fraction removed: ")
        logger.writeln(" ".join("%.4f"%x for x in frac_est))
        twin_data.alphas = frac_est

    return df

def mlopt_twin_fractions(hkldata, twin_data, b_aniso):
    k_ani2_inv = 1 / hkldata.debye_waller_factors(b_cart=b_aniso)**2
    Io = hkldata.df.I.to_numpy(copy=True) * k_ani2_inv
    sigIo = hkldata.df.SIGI.to_numpy(copy=True) * k_ani2_inv
    def fun(x):
        twin_data.alphas = x
        twin_data.est_f_true(Io, sigIo, 100)
        ret = twin_data.ll(Io, sigIo)
        return ret
    def grad(x):
        twin_data.alphas = x
        twin_data.est_f_true(Io, sigIo, 100)
        return twin_data.ll_der_alpha(Io, sigIo, True)
    if 0:
        bak = [_ for _ in twin_data.alphas]
        with open("alpha_ll.csv", "w") as ofs:
            ofs.write("a,ll,ll_new,der1,der2,der_new1,der_new2\n")
            for a in numpy.linspace(0., 1.0, 100):
                x = [a, 1-a]
                twin_data.alphas = x
                twin_data.est_f_true(Io, sigIo, 100)
                f_new = twin_data.ll(Io, sigIo)
                f = twin_data.ll_rice()
                der = twin_data.ll_der_alpha(Io, sigIo, False)
                #der = [x - der[-1] for x in der[:-1]]
                der_new = twin_data.ll_der_alpha(Io, sigIo, True)
                #der_new = [x - der_new[-1] for x in der_new[:-1]]
                ofs.write(f"{a},{f},{f_new},{der[0]},{der[1]},{der_new[0]},{der_new[1]}\n")
            ofs.write("\n")
        twin_data.alphas = bak
    if 0:
        x0 = [x for x in twin_data.alphas]
        f0 = fun(x0)
        ader = grad(x0)
        
        print(f"{ader=}")
        for e in (1e-2, 1e-3, 1e-4, 1e-5):
            nder = []
            for i in range(len(x0)):
                x = [_ for _ in x0]
                x[i] += e
                f1 = fun(x)
                nder.append((f1 - f0) / e)
            print(f"{e=} {nder=}")
    
    logger.writeln("ML twin fraction refinement..")
    num_params = len(twin_data.alphas)
    A = numpy.ones((1, num_params))
    linear_constraint = scipy.optimize.LinearConstraint(A, [1.0], [1.0])
    bounds = scipy.optimize.Bounds(numpy.zeros(num_params), numpy.ones(num_params))
    logger.writeln(" starting with " + " ".join("%.4f"%x for x in twin_data.alphas))
    logger.writeln(f" f0= {fun(twin_data.alphas)}")
    res = scipy.optimize.minimize(fun=fun, x0=twin_data.alphas,
                                  bounds=bounds,
                                  constraints=[linear_constraint],
                                  jac=grad,
                                  #callback=lambda *x: logger.writeln(f"callback {x}"),
                                  )
    logger.writeln(" finished in {} iterations ({} evaluations)".format(res.nit, res.nfev))
    logger.writeln(f" f = {res.fun}")
    # ensure constraints
    alphas = numpy.clip(res.x, 0, 1)
    twin_data.alphas = list(alphas / alphas.sum())
    logger.write(" ML twin fraction estimate: ")
    logger.writeln(" ".join("%.4f"%x for x in twin_data.alphas))
