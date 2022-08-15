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
import scipy.special
import scipy.optimize
from servalcat.utils import logger
from servalcat import utils

def add_arguments(parser):
    parser.description = 'Convert intensity to amplitude'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin', required=True,
                        help='MTZ column for I,SIGI')
    parser.add_argument('--d_min', type=float)
    parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int, default=20,
                        help="Number of bins (default: %(default)d)")
    parser.add_argument('-o','--output_prefix', default="servalcat_fw",
                        help='output file name prefix (default: %(default)s)')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

# TODO this function can be generalised and merged with sigmaa.process_input
def process_input(hklin, labin, n_bins, d_max=None, d_min=None):
    assert len(labin) == 2
    
    mtz = gemmi.read_mtz_file(hklin)
    logger.write("Input mtz: {}".format(hklin))
    logger.write("    Unit cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*mtz.cell.parameters))
    logger.write("  Space group: {}".format(mtz.spacegroup.hm))
    logger.write("")
    
    hkldata = utils.hkl.hkldata_from_mtz(mtz, labin, newlabels=["I","SIGI"])
    hkldata.df = hkldata.df.astype({name: 'float64' for name in ["I","SIGI"]})

    # TODO perhaps we should switch type to float64
    if (d_min, d_max).count(None) != 2:
        hkldata = hkldata.copy(d_min=d_min, d_max=d_max)
    d_min, d_max = hkldata.d_min_max()
    
    hkldata.complete()
    hkldata.sort_by_resolution()
    hkldata.calc_epsilon()
    hkldata.calc_centric()
    hkldata.setup_binning(n_bins=n_bins)
    logger.write("Data completeness: {:.2%}".format(hkldata.completeness()))

    # Create a centric selection table for faster look up
    centric_and_selections = {}
    for i_bin, idxes in hkldata.binned():
        centric_and_selections[i_bin] = []
        for c, g2 in hkldata.df.loc[idxes].groupby("centric", sort=False):
            valid_sel = numpy.isfinite(g2.I) & (g2.SIGI > 0)
            vidxes = g2.index[valid_sel]
            nidxes = g2.index[~valid_sel] # missing reflections
            centric_and_selections[i_bin].append((c, vidxes, nidxes))
    
    return hkldata, centric_and_selections
# process_input()

def determine_Sigma_and_aniso(hkldata, centric_and_selections):
    # initial estimate
    hkldata.binned_df["S"] = 1.
    k_aniso = numpy.zeros(6)
    I_over_eps = hkldata.df.I.to_numpy() / hkldata.df.epsilon.to_numpy()
    for i_bin, idxes in hkldata.binned():
        hkldata.binned_df.loc[i_bin, "S"] = numpy.nanmean(I_over_eps[idxes])

    logger.write("Initial estimates:")
    logger.write(hkldata.binned_df.to_string())

    return k_aniso

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
#@profile
def J(k, z, mode="pbdv"):
    """
    Calculate J(k,z) = int_0^inf t^k exp(-(t-z)^2/2) dt
    """
    zsq = z**2
    if mode == "pbdv":
        return scipy.special.gamma(k+1) * numpy.exp(-zsq/4) * scipy.special.pbdv(-k-1, -z)[0]
    elif mode == "1f1":
        ret = scipy.special.gamma((k+1)/2) * scipy.special.hyp1f1(-k/2, 0.5, -zsq/2)
        ret += numpy.sqrt(2) * z * scipy.special.gamma(k/2 + 1) * scipy.special.hyp1f1((1-k)/2, 1.5, -zsq/2)
        ret *= 2**((k-1) / 2)
        return ret
    elif mode == "1f1_u":
        selp = z >= 0
        seln = ~selp
        retp = scipy.special.gamma((k+1)/2) * scipy.special.hyp1f1(-k/2, 0.5, -zsq[selp]/2)
        retp += numpy.sqrt(2) * z[selp] * scipy.special.gamma(k/2 + 1) * scipy.special.hyp1f1((1-k)/2, 1.5, -zsq[selp]/2)
        retp *= 2**((k-1) / 2)

        retn = 2**(-k/2-1) * scipy.special.gamma(k + 1)
        retn *= numpy.exp(-zsq[seln]/2) * numpy.sqrt(2)
        retn *= scipy.special.hyperu(k/2+0.5, 0.5, zsq[seln]/2)

        ret = numpy.zeros(len(z))
        ret[selp] = retp
        ret[seln] = retn
        return ret
    elif mode == "laplace_x2": # Laplace approximation with t=x^2
        xk = numpy.sqrt(0.5 * z + 0.5 * numpy.sqrt(zsq + 4 * k + 2))
        f_xk = 0.5 * xk**4 - xk**2 * z - (2 * k + 1) * numpy.log(xk)
        fpp_xk = 4 * numpy.sqrt(zsq + 4*k + 2)
        return 2 * numpy.exp(-0.5*zsq - f_xk) * numpy.sqrt(0.5 * numpy.pi / fpp_xk) * (scipy.special.erf(xk*numpy.sqrt(0.5 * fpp_xk)) + 1)
    elif mode == "laplace_x4": # Laplace approximation with t=x^4
        xk4 = 0.5 * (z + numpy.sqrt(zsq + 4 * k + 3))
        xk = xk4**0.25
        f_xk = 0.5 * xk4**2 - z * xk4 - (4 * k + 3) * numpy.log(xk)
        fpp_xk = 28 * xk4 * numpy.sqrt(xk4) - 12 * z * numpy.sqrt(xk4) + (4 * k + 3) / numpy.sqrt(xk4)
        return 4 * numpy.exp(-0.5*zsq - f_xk) * numpy.sqrt(0.5 * numpy.pi / fpp_xk) * (scipy.special.erf(xk * numpy.sqrt(0.5 * fpp_xk)) + 1)
    else:
        raise Exception("bad mode")
# J()

def french_wilson(hkldata, centric_and_selections, k_aniso):
    hkldata.df["F"] = numpy.nan
    hkldata.df["SIGF"] = numpy.nan
    hkldata.df["to1"] = numpy.nan

    for i_bin, idxes in hkldata.binned():
        S = hkldata.binned_df.S[i_bin]
        for c, cidxes, nidxes in centric_and_selections[i_bin]:
            Io = hkldata.df.I.to_numpy()[cidxes]
            sigo = hkldata.df.SIGI.to_numpy()[cidxes]
            epsS = hkldata.df.epsilon.to_numpy()[cidxes] * S
            
            if c == 0: # acentric
                to1 = Io / sigo - sigo / epsS
                J_0 = J(0., to1)
                F = numpy.sqrt(sigo) * J(0.5, to1) / J_0
                Fsq = sigo * J(1., to1) / J_0
            else: # centric
                to1 = Io / sigo - 0.5 * sigo / epsS
                J_minus_half = J(-0.5, to1)
                F = numpy.sqrt(sigo) * J(0., to1) / J_minus_half
                Fsq = sigo * J(0.5, to1) / J_minus_half

            print("bin=",i_bin, "cen=", c, "min_to1=", numpy.min(to1))
            varF = Fsq - F**2
            hkldata.df.loc[cidxes, "F"] = F
            hkldata.df.loc[cidxes, "SIGF"] = numpy.sqrt(varF)
            hkldata.df.loc[cidxes, "to1"] = to1

def main(args):
    if args.nbins < 1:
        raise SystemExit("--nbins must be > 0")

    hkldata, centric_and_selections = process_input(hklin=args.hklin,
                                                    labin=args.labin.split(","),
                                                    n_bins=args.nbins,
                                                    d_min=args.d_min,
                                                    d_max=args.d_max)
    print(hkldata.df)
    k_aniso = determine_Sigma_and_aniso(hkldata, centric_and_selections)
    french_wilson(hkldata, centric_and_selections, k_aniso)
    print(hkldata.df)

    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=["F","SIGF","I","SIGI","d","bin","centric","to1"],
                      types={"F":"F", "SIGF":"Q"})
    logger.write("output mtz: {}".format(mtz_out))
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
