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
import time
import os
from servalcat.utils import logger
from servalcat.xtal import sigmaa
from servalcat import utils
from servalcat import ext

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)

integr = sigmaa.integr

def add_arguments(parser):
    parser.description = 'Convert intensity to amplitude'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--hklin_index', type=int, default=0,
                        help='block index if hklin is mmcif file (default: %(default)d)')
    parser.add_argument('--labin', 
                        help='MTZ column for I,SIGI')
    parser.add_argument('--d_min', type=float)
    parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int,
                        help="Number of bins (default: auto)")
    parser.add_argument('-o','--output_prefix',
                        help='output file name prefix')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def line_search_Sigma(S0, ln_shift, g, f0, compute_ll_func, c1=1e-4):
    """
    Line search with expansion (if initial step is too small)
    and backtracking (if initial step is too large).
    """
    directional_grad = g * ln_shift  # Always < 0 for a descent direction
    alpha = 1.0
    S_trial = max(1e-6, S0 * numpy.exp(alpha * ln_shift))
    f_trial = compute_ll_func(S_trial)
    df = f0 - f_trial
    # Check if alpha = 1.0 satisfies Armijo sufficient decrease
    is_armijo = df >= -c1 * alpha * directional_grad
    # When data are noisy, the likelihood becomes nearly flat
    is_flat_region = (df > 0) and (S0 < 1e-5 or abs(directional_grad) < 1e-3)
    if is_armijo or is_flat_region: # expansion
        best_S, best_f, best_alpha = S_trial, f_trial, alpha
        for _ in range(3):
            alpha_next = alpha * 2.0
            S_next = max(1e-6, S0 * numpy.exp(alpha_next * ln_shift))
            f_next = compute_ll_func(S_next)
            # Keep expanding as long as likelihood improves
            if f_next < best_f:
                best_f = f_next
                best_S = S_next
                best_alpha = alpha_next
                alpha = alpha_next
            else:
                break
        return best_S, best_f, True, best_alpha
    else: # backtracking
        for _ in range(6): # Test down to alpha = 0.5^6 ≈ 0.0156
            alpha *= 0.5
            S_trial = max(1e-6, S0 * numpy.exp(alpha * ln_shift))
            f_trial = compute_ll_func(S_trial)
            df = f0 - f_trial
            is_armijo = df >= -c1 * alpha * directional_grad
            is_flat_region = (df > 0) and (S0 < 1e-5 or abs(directional_grad) < 1e-3)
            if is_armijo or is_flat_region:
                return S_trial, f_trial, True, alpha
                
        # Line search failed to find a lower point
        return S0, f0, False, 0.0

def optimize_Sigma(hkldata, B, max_cycle=10, gtol=1e-4, ftol=1e-7):
    S_converged = [False for _ in hkldata.binned("ml")]
    k_ani = hkldata.debye_waller_factors(b_cart=B)
    Io = hkldata.df.I.to_numpy()
    sigIo = hkldata.df.SIGI.to_numpy()
    epsilon = hkldata.df.epsilon.to_numpy()
    c = hkldata.df.centric.to_numpy() + 1
    llweight = hkldata.df.llweight.to_numpy()
    
    for i, (i_bin, idxes) in enumerate(hkldata.binned("ml")):
        #logger.writeln("Bin {}".format(i_bin))
        zeros_bin = numpy.zeros(len(idxes))
        S0 = hkldata.binned_df["ml"].loc[i_bin, "S0"]
        for j in range(max_cycle):
            #print(f"{i_bin=} {j=} {S0=}")
            compute_bin_ll = lambda x: numpy.nansum(integr.ll_int(Io[idxes], sigIo[idxes], k_ani[idxes],
                                                                  x * epsilon[idxes],
                                                                  zeros_bin, c[idxes],
                                                                  llweight[idxes]))
            f0 = compute_bin_ll(S0)
            g, H = ll_shift_bin_S(Io[idxes], sigIo[idxes], k_ani[idxes],
                                  S0, c[idxes], epsilon[idxes],
                                  llweight[idxes])
            ln_shift = numpy.clip(-g / (H + 1e-6), -4., 4.)
            S_next, f_next, step_accepted, alpha_used = line_search_Sigma(S0=S0,
                                                                          ln_shift=ln_shift,
                                                                          g=g,
                                                                          f0=f0,
                                                                          compute_ll_func=compute_bin_ll)
            #print(i_bin, "S_next, f_next, step_accepted, alpha_used", S_next, f_next, step_accepted, alpha_used)
            
            if step_accepted:
                df = f0 - f_next  # Positive value indicates decrease in objective
                S0 = S_next
                f0 = f_next
            
                # Check function convergence ONLY on full/expanded steps
                if alpha_used >= 1.0 and df < ftol * (abs(f0) + 1.0):
                    S_converged[i] = True
                    break
            else:
                break  # Exit cycle if search cannot find a valid descent direction
        hkldata.binned_df["ml"].loc[i_bin, "S0"] = S0
    #print(f"{S_converged=}")
    return S_converged
# optimize_Sigma

def determine_initial_Sigma(hkldata, B):
    k_ani = hkldata.debye_waller_factors(b_cart=B)
    hkldata.binned_df["ml"]["S0"] = 1.
    I_over_eps = hkldata.df.I.to_numpy() / hkldata.df.epsilon.to_numpy() / k_ani**2
    for i_bin, idxes in hkldata.binned("ml"):
        #S0 = max(numpy.nanmean(I_over_eps[idxes]), 1e-3)
        # var(I) = var_signal + var_noise, so this overestimates S0,
        # but it should be better than having negative values (in noisy shell)
        S0 = numpy.nanmean(I_over_eps[idxes])
        hkldata.binned_df["ml"].loc[i_bin, "S0"] = max(1e-3, S0)
    logger.writeln("Initial estimates:")
    logger.writeln(hkldata.binned_df["ml"].to_string())
# determine_initial_Sigma()

def determine_Sigma_and_aniso(hkldata, xtol=1e-4, ftol_rel=1e-5):
    B = gemmi.SMat33d(0,0,0,0,0,0)
    determine_initial_Sigma(hkldata, B)

    SMattolist = lambda B: [B.u11, B.u22, B.u33, B.u12, B.u13, B.u23]
    adpdirs = utils.model.adp_constraints(hkldata.sg.operations(), hkldata.cell, tr0=True)
    logger.writeln("ADP free parameters = {}".format(adpdirs.shape[0]))
    ssqmat = hkldata.ssq_mat()
    cycle_data = [[0] + SMattolist(B) + list(hkldata.binned_df["ml"].S0)]
    for icyc in range(100):
        #logger.writeln("Refine B")
        B_converged = False
        t0 = time.time()
        args=(ssqmat, hkldata, adpdirs)
        for j in range(10):
            x = numpy.dot(SMattolist(B), numpy.linalg.pinv(adpdirs))
            f0 = ll_all_B(gemmi.SMat33d(*numpy.dot(x, adpdirs)), hkldata)
            shift = ll_shift_B(x, *args)
            # Check if shift is already negligible
            if numpy.max(numpy.abs(shift)) < xtol:
                B_converged = True
                break
            
            ftol = ftol_rel * (abs(f0) + 1.0)

            for i in range(3):
                ss = shift / 2**i
                f1 = ll_all_B(gemmi.SMat33d(*numpy.dot(x + ss, adpdirs)), hkldata)
                df = f0 - f1  # Positive df means f1 < f0 (improvement)
                #print("determine_Sigma_and_aniso", df, df >= 0 or -df < ftol, df < ftol or numpy.max(numpy.abs(ss)) < xtol)
                if df >= 0:
                    B = gemmi.SMat33d(*numpy.dot(x+ss, adpdirs))
                    if df < ftol or numpy.max(numpy.abs(ss)) < xtol:
                        B_converged = True
                    break
                elif -df < ftol: # tolerable increase
                    B_converged = True
                    break
            else:
                B_converged = False
                break
            if B_converged:
                break

        #logger.writeln("time= {}".format(time.time() - t0))
        #logger.writeln("B_aniso= {}".format(B))
        #logger.writeln("Refine S0")
        f0 = ll_all_B(B, hkldata)
        S_converged = optimize_Sigma(hkldata, B)
        f1 = ll_all_B(B, hkldata)
        logger.writeln(f"cyc= {icyc:2d} f= {f1:4e} df= {f0-f1:4e} tol= {ftol_rel * (abs(f1) + 1.):4e}")
        #logger.writeln("Refined estimates in cycle {}:".format(icyc))
        #logger.writeln(hkldata.binned_df["ml"].to_string())
        #logger.writeln("B_aniso= {}".format(B))
        cycle_data.append([icyc] + SMattolist(B) + list(hkldata.binned_df["ml"].S0))
        if B_converged and 0 <= f0 - f1 < ftol_rel * (abs(f1) + 1.):
            logger.writeln("Converged in cycle {}".format(icyc))
            logger.writeln("Refined estimates:")
            logger.writeln(hkldata.binned_df["ml"].to_string())
            logger.writeln("B_aniso= {}".format(B))
            break

    #with open("fw_cycles.dat", "w") as ofs:
    #    ofs.write("cycle B11 B22 B33 B12 B13 B23 " + " ".join("S0{}".format(i) for i in hkldata.binned_df["ml"].index) + "\n")
    #    for data in cycle_data:
    #        ofs.write("{:2d} ".format(data[0]+1))
    #        ofs.write(" ".join("{:.4e}".format(x) for x in data[1:]))
    #        ofs.write("\n")
        
    return B

def ll_all_B(B, hkldata):
    #B = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k_ani = hkldata.debye_waller_factors(b_cart=B)
    ret = 0.
    for i_bin, idxes in hkldata.binned("ml"):
        ret += numpy.nansum(integr.ll_int(hkldata.df.I.to_numpy()[idxes], hkldata.df.SIGI.to_numpy()[idxes], k_ani[idxes],
                                          hkldata.binned_df["ml"].S0[i_bin] * hkldata.df.epsilon.to_numpy()[idxes],
                                          numpy.zeros(len(idxes)), hkldata.df.centric.to_numpy()[idxes]+1,
                                          hkldata.df.llweight.to_numpy()[idxes]))
    return ret

def ll_shift_bin_S(Io, sigIo, k_ani, S0, c, eps, llw, exp_trans=True):
    tmp = integr.ll_int_fw_der1_S(Io, sigIo, k_ani, S0, c, eps, llw)
    g = numpy.nansum(tmp)
    H = numpy.nansum(tmp**2)
    if exp_trans:
        # mathematically, H + g should be used, but this could be unstable
        return g * S0, H * S0**2
    else:
        return g, H

def ll_shift_B(x, ssqmat, hkldata, adpdirs):
    b_aniso = gemmi.SMat33d(*numpy.dot(x, adpdirs))
    k_ani = hkldata.debye_waller_factors(b_cart=b_aniso)
    Io = hkldata.df.I.to_numpy()
    sigIo = hkldata.df.SIGI.to_numpy()
    c = hkldata.df.centric.to_numpy() + 1
    epsilon = hkldata.df.epsilon.to_numpy()
    llw = hkldata.df.llweight.to_numpy()
    r = numpy.empty(len(Io)) * numpy.nan
    for i_bin, idxes in hkldata.binned("ml"):
        r[idxes] = integr.ll_int_fw_der1_ani(Io[idxes], sigIo[idxes],
                                             k_ani[idxes], hkldata.binned_df["ml"].S0[i_bin],
                                             c[idxes], epsilon[idxes], llw[idxes])
    g = -numpy.nansum(ssqmat * r, axis=1)
    H = numpy.nansum(numpy.matmul(ssqmat[None,:].T, ssqmat.T[:,None]) * (r**2)[:,None,None], axis=0)
    g, H = numpy.dot(g, adpdirs.T), numpy.dot(adpdirs, numpy.dot(H, adpdirs.T))
    return -numpy.dot(g, numpy.linalg.pinv(H))

def expected_F_from_int(Io, sigo, k_ani, eps, c, S0):
    to = Io / sigo - sigo / c / k_ani**2 / S0 / eps
    tf = numpy.zeros(Io.size)
    sig1 = numpy.ones(Io.size)
    k_num = numpy.where(c == 1,  0.5, 0.)
    F = numpy.sqrt(sigo) * ext.integ_J_ratio(k_num, k_num - 0.5, False, to, tf, sig1, c,
                                             integr.exp2_threshold, integr.h, integr.N, integr.ewmax)
    Fsq = sigo * ext.integ_J_ratio(k_num + 0.5, k_num - 0.5, False, to, tf, sig1, c,
                                   integr.exp2_threshold, integr.h, integr.N, integr.ewmax)
    varF = numpy.maximum(0.0, Fsq - F**2)
    return F, numpy.sqrt(varF)

def french_wilson(hkldata, B_aniso, labout=None):
    if labout is None: labout = ["F", "SIGF"]
    k_ani = hkldata.debye_waller_factors(b_cart=B_aniso)
    has_ano = "I(+)" in hkldata.df and "I(-)" in hkldata.df
    if has_ano:
        ano_data = hkldata.df[["I(+)", "SIGI(+)", "I(-)", "SIGI(-)"]].to_numpy()
        if len(labout) == 2:
            labout += ["F(+)", "SIGF(+)", "F(-)", "SIGF(-)"]
    hkldata.df[labout] = numpy.nan
    for i_bin, idxes in hkldata.binned("ml"):
        S0 = hkldata.binned_df["ml"].S0[i_bin]
        c = hkldata.df.centric.to_numpy()[idxes] + 1 # 1 for acentric, 2 for centric
        Io = hkldata.df.I.to_numpy()[idxes]
        sigo = hkldata.df.SIGI.to_numpy()[idxes]
        eps = hkldata.df.epsilon.to_numpy()[idxes]
        F, sigF = expected_F_from_int(Io, sigo, k_ani[idxes], eps, c, S0)
        hkldata.df.loc[idxes, labout[0]] = F
        hkldata.df.loc[idxes, labout[1]] = sigF
        if has_ano:
            Fp, sigFp = expected_F_from_int(ano_data[idxes,0], ano_data[idxes,1], k_ani[idxes], eps, c, S0)
            Fm, sigFm = expected_F_from_int(ano_data[idxes,2], ano_data[idxes,3], k_ani[idxes], eps, c, S0)
            hkldata.df.loc[idxes, labout[2]] = Fp
            hkldata.df.loc[idxes, labout[3]] = sigFp
            hkldata.df.loc[idxes, labout[4]] = Fm
            hkldata.df.loc[idxes, labout[5]] = sigFm

def set_prefix(args):
    if not args.output_prefix:
        args.output_prefix = utils.fileio.splitext(os.path.basename(args.hklin))[0] + "_fw"
# set_prefix()

def main(args):
    set_prefix(args)
    try:
        mtz = utils.fileio.read_mmhkl(args.hklin, cif_index=args.hklin_index)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    if not args.labin:
        labin = sigmaa.decide_mtz_labels(mtz, require=("K", "J"))
    else:
        labin = args.labin.split(",")
    try:
        hkldata, _, _, _, _ = sigmaa.process_input(hklin=mtz,
                                                   labin=labin,
                                                   n_bins_ml=args.nbins,
                                                   free=None,
                                                   xyzin=None,
                                                   d_min=args.d_min,
                                                   n_per_mlbin=500,
                                                   max_mlbins=30,
                                                   cif_index=args.hklin_index)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    
    B_aniso = determine_Sigma_and_aniso(hkldata)
    french_wilson(hkldata, B_aniso)
    mtz_out = args.output_prefix+".mtz"
    lab_out = ["F", "SIGF", "I", "SIGI"]
    labo_types = {"F":"F", "SIGF":"Q", "I":"J", "SIGI":"Q"}
    if "I(+)" in hkldata.df and "I(-)" in hkldata.df:
        lab_out += ["F(+)", "SIGF(+)", "F(-)", "SIGF(-)"]
        labo_types.update({"F(+)":"G", "SIGF(+)":"L", "F(-)":"G", "SIGF(-)":"L"})
    if len(labin) == 3:
        lab_out.append("FREE")
        labo_types[lab_out[-1]] = "I"
    hkldata.write_mtz(mtz_out, lab_out, types=labo_types)
    return B_aniso, hkldata
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
