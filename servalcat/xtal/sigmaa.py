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
    parser.description = 'Sigma-A parameter estimation for crystallographic data'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin', required=True,
                        help='MTZ column for F,SIGF')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument("-d", '--d_min', type=float)
    #parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int, default=20,
                        help="Number of bins (default: %(default)d)")
    parser.add_argument('-s', '--source', choices=["electron", "xray", "neutron"], default="xray")
    parser.add_argument('-o','--output_prefix', default="sigmaa",
                        help='output file name prefix (default: %(default)s)')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def fom_acentric(Fo, varFo, Fc, D, S, epsilon):
    Sigma = 2 * varFo + epsilon * S
    return gemmi.bessel_i1_over_i0(2 * Fo * D * Fc / Sigma)
# fom_acentric()

def fom_centric(Fo, varFo, Fc, D, S, epsilon):
    Sigma = varFo + epsilon * S
    return numpy.tanh(Fo * D * Fc / Sigma)
# fom_centric()

def mlf_acentric(Fo, varFo, Fc, D, S, epsilon):
    # https://doi.org/10.1107/S0907444911001314
    # eqn (4)
    Sigma = 2 * varFo + epsilon * S
    ret = numpy.log(2) + numpy.log(Fo) - numpy.log(Sigma)
    ret += -(Fo**2 + D**2*Fc**2)/Sigma
    ret += gemmi.log_bessel_i0(2*Fo*D*Fc/Sigma)
    return -ret
# mlf_acentric()

def deriv_mlf_wrt_D_S_acentric(Fo, varFo, Fc, D, S, epsilon):
    deriv = numpy.zeros(2)
    Sigma = 2 * varFo + epsilon * S
    Fo2 = Fo**2
    Fc2 = Fc**2
    i1_i0_x = gemmi.bessel_i1_over_i0(2*Fo*D*Fc/Sigma) # m
    deriv[0] = -numpy.sum(-2*D*Fc2/Sigma + i1_i0_x*2*Fo*Fc/Sigma)
    deriv[1] = -numpy.sum((-1/Sigma + (Fo2 + D**2 * Fc2 - i1_i0_x * 2 * Fo * D * Fc) / Sigma**2) * epsilon)
    return deriv
# deriv_mlf_wrt_D_S_acentric()

def mlf_centric(Fo, varFo, Fc, D, S, epsilon):
    # https://doi.org/10.1107/S0907444911001314
    # eqn (4)
    Sigma = varFo + epsilon * S
    ret = 0.5 * (numpy.log(2 / numpy.pi) - numpy.log(Sigma))
    ret += -0.5 * (Fo**2 + D**2 * Fc**2) / Sigma
    ret += gemmi.log_cosh(Fo * D * Fc / Sigma)
    return -ret
# mlf_centric()

def deriv_mlf_wrt_D_S_centric(Fo, varFo, Fc, D, S, epsilon):
    deriv = numpy.zeros(2)
    Sigma = varFo + epsilon * S
    Fo2 = Fo**2
    Fc2 = Fc**2
    tanh_x = numpy.tanh(Fo*D*Fc/Sigma)
    deriv[0] = -numpy.sum(-D*Fc2/Sigma + tanh_x*Fo*Fc/Sigma)
    deriv[1] = -numpy.sum((-0.5 / Sigma + (0.5*(Fo2+D**2*Fc2) - tanh_x * Fo*D*Fc)/Sigma**2)*epsilon)
    return deriv
# deriv_mlf_wrt_D_S_centric()

def mlf(df, D, S):
    ret = 0.
    params = lambda s: (s.FP.to_numpy(), s.SIGFP.to_numpy()**2, abs(s.FC.to_numpy()), D, S, s.epsilon.to_numpy())
    for c, g in df.groupby("centric", sort=False):
        if c == 0:
            ret += numpy.sum(mlf_acentric(*params(g)))
        else:
            ret += numpy.sum(mlf_centric(*params(g)))
    return ret
# mlf()

def deriv_mlf_wrt_D_S(df, D, S):
    params = lambda s: (s.FP.to_numpy(), s.SIGFP.to_numpy()**2, abs(s.FC.to_numpy()), D, S, s.epsilon.to_numpy())
    ret = []
    for c, g in df.groupby("centric", sort=False):
        if c == 0:
            ret.append(deriv_mlf_wrt_D_S_acentric(*params(g)))
        else:
            ret.append(deriv_mlf_wrt_D_S_centric(*params(g)))
    return sum(ret)
# deriv_mlf_wrt_D_S()

def calc_fom(df, D, S):
    ret = pandas.Series(index=df.index)
    params = lambda s: (s.FP.to_numpy(), s.SIGFP.to_numpy()**2, abs(s.FC.to_numpy()), D, S, s.epsilon.to_numpy())
    for c, g in df.groupby("centric", sort=False):
        if c == 0:
            ret[g.index] = fom_acentric(*params(g))
        else:
            ret[g.index] = fom_centric(*params(g))
    return ret
# calc_fom()

def write_mtz(hkldata, mtz_out):
    map_labs = "FWT", "DELFWT", "FC", "Fmask"
    other_labs = ["FOM", "FP"]
    other_types = ["W", "F"]
    data = numpy.empty((len(hkldata.df.index), len(map_labs)*2+len(other_labs)+3))
    data[:,:3] = hkldata.df[["H","K","L"]]
    for i, lab in enumerate(map_labs):
        data[:,3+i*2] = numpy.abs(hkldata.df[lab])
        data[:,3+i*2+1] = numpy.angle(hkldata.df[lab], deg=True)

    for i, lab in enumerate(other_labs):
        data[:,3+len(map_labs)*2+i] = hkldata.df[lab]
        
    mtz = gemmi.Mtz()
    mtz.spacegroup = hkldata.sg
    mtz.cell = hkldata.cell
    mtz.add_dataset('HKL_base')
    for label in ['H', 'K', 'L']: mtz.add_column(label, 'H')

    for lab in map_labs:
        mtz.add_column(lab, "F")
        mtz.add_column(("PH"+lab).replace("FWT", "WT"), "P")
    for lab, typ in zip(other_labs, other_types):
        mtz.add_column(lab, typ)

    mtz.set_data(data)
    mtz.write_to_file(mtz_out)

def determine_mlf_params(hkldata):
    # Initial values
    hkldata.binned_df["D"] = 1.
    hkldata.binned_df["S"] = 10000.
    for i_bin, idxes in hkldata.binned():
        FC = numpy.abs(hkldata.df.FC.to_numpy()[idxes])
        FP = hkldata.df.FP.to_numpy()[idxes]
        D = numpy.corrcoef(FP, FC)[1,0]
        hkldata.binned_df.loc[i_bin, "D"] = D
        hkldata.binned_df.loc[i_bin, "S"] = numpy.var(FP - D * FC)

    logger.write("Initial estimates:")
    logger.write(hkldata.binned_df.to_string())

    # test derivative
    if 0:
        gana = grad(x0)
        e = 1e-4
        for i in range(len(x0)):
            tmp = x0.copy()
            f0 = target(tmp)
            tmp[i] += e
            fe = target(tmp)
            gnum = (fe-f0)/e
            print(i, gnum, gana[i], gana[i]-gnum)
        
    for i_bin, idxes in hkldata.binned():
        D = hkldata.binned_df.D[i_bin]
        S = hkldata.binned_df.S[i_bin]
        x0 = [D, S]
        def target(x):
            return mlf(hkldata.df.loc[idxes], x[0], x[1])
        def grad(x):
            return deriv_mlf_wrt_D_S(hkldata.df.loc[idxes], x[0], x[1])

        #print("Bin", i_bin)
        res = scipy.optimize.minimize(fun=target, jac=grad, x0=x0)
        #print(res)
        
        hkldata.binned_df.loc[i_bin, "D"] = res.x[0]
        hkldata.binned_df.loc[i_bin, "S"] = res.x[1]

    logger.write("Refined estimates:")
    logger.write(hkldata.binned_df.to_string())
# determine_mlf_params()

# TODO isotropize map
# TODO add missing reflections
def main(args):
    if args.nbins < 1:
        logger.error("--nbins must be > 0")
        return
    
    st = utils.fileio.read_structure(args.model)
    mtz = gemmi.read_mtz_file(args.hklin)
    d_min = args.d_min
    if d_min is None: d_min = mtz.resolution_high()
    labin = args.labin.split(",")
    assert len(labin) == 2
    scaleto = mtz.get_value_sigma(*labin)
    fp = mtz.get_float(labin[0])
    sigfp = mtz.get_float(labin[1])

    logger.write("Calculating solvent contribution..")
    grid = gemmi.FloatGrid()
    grid.setup_from(st, spacing=0.4)
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)
    masker.put_mask_on_float_grid(grid, st[0])
    fmask_asu = gemmi.transform_map_to_f_phi(grid).prepare_asu_data(dmin=d_min)

    fc_asu = utils.model.calc_fc_fft(st, d_min, source=args.source, mott_bethe=args.source=="electron")

    logger.write("Scaling Fc..")
    scaling = gemmi.Scaling(st.cell, st.find_spacegroup())
    scaling.use_solvent = True
    scaling.prepare_points(fc_asu, scaleto, fmask_asu)
    scaling.fit_isotropic_b_approximately()
    scaling.fit_parameters()
    b_aniso = scaling.b_overall
    logger.write(" k_ov= {:.2e} B= {}".format(scaling.k_overall, b_aniso))
    scaling.scale_data(fc_asu, fmask_asu)

    hkldata = utils.hkl.hkldata_from_asu_data(fc_asu, "FC")
    hkldata.merge_asu_data(fp, "FP")
    hkldata.merge_asu_data(sigfp, "SIGFP")
    fca = numpy.abs(hkldata.df.FC.to_numpy())
    fpa = hkldata.df.FP.to_numpy()
    logger.write(" CC(Fo,Fc)= {:.4f}".format(numpy.corrcoef(fca, fpa)[0,1]))
    logger.write(" Rcryst= {:.4f}".format(numpy.sum(numpy.abs(fca-fpa))/numpy.sum(fpa)))

    hkldata.calc_epsilon()
    hkldata.calc_centric()
    hkldata.setup_binning(n_bins=args.nbins)

    logger.write("Estimating sigma-A parameters..")
    determine_mlf_params(hkldata)

    log_out = "{}.log".format(args.output_prefix)
    ofs = open(log_out, "w")
    ofs.write("""$TABLE: Statistics :
$GRAPHS
: log(Mn(|F|^2)) and variances :A:1,7,8,9,11:
: weights :A:1,10,12,13:
: number of reflections :A:1,3,4:
$$
1/resol^2 bin n_a n_c d_max d_min log(Mn(|Fo|^2)) log(Mn(|Fc|^2)) log(Mn(|DFc|^2)) D log(Sigma) FOM_a FOM_c
$$
$$
""")
    tmpl = "{:.4f} {:3d} {:7d} {:7d} {:7.3f} {:7.3f} {:.4e} {:.4e} {:4e} {: .4f}   {: .4e} {:.4f} {:.4f}\n"

    hkldata.df["FWT"] = 0j
    hkldata.df["DELFWT"] = 0j
    hkldata.df["FOM"] = 0.
    for i_bin, idxes in hkldata.binned():
        bin_d_min = hkldata.binned_df.d_min[i_bin]
        bin_d_max = hkldata.binned_df.d_max[i_bin]
        D = hkldata.binned_df.D[i_bin]
        S = hkldata.binned_df.S[i_bin]

        # 0: acentric 1: centric
        mean_fom = [0, 0]
        nrefs = [0, 0]
        fom_func = [fom_acentric, fom_centric]
        for c, g2 in hkldata.df.loc[idxes].groupby("centric", sort=False):
            Fc = numpy.abs(g2.FC.to_numpy())
            phic = numpy.angle(g2.FC.to_numpy())
            expip = numpy.cos(phic) + 1j*numpy.sin(phic)
            Fo = g2.FP.to_numpy()
            
            m = fom_func[c](Fo, g2.SIGFP**2, Fc, D, S, g2.epsilon.to_numpy())
            mean_fom[c] = numpy.mean(m)
            nrefs[c] = len(g2.index)

            hkldata.df.loc[g2.index, "FOM"] = m
            hkldata.df.loc[g2.index, "DELFWT"] = (m * Fo - D * Fc ) * expip
            if c == 0:
                hkldata.df.loc[g2.index, "FWT"] = (2 * m * Fo - D * Fc) * expip
            else:
                hkldata.df.loc[g2.index, "FWT"] = (m * Fo) * expip
        
        ofs.write(tmpl.format(1/bin_d_min**2, i_bin, nrefs[0], nrefs[1], bin_d_max, bin_d_min,
                              numpy.log(numpy.average(numpy.abs(Fo)**2)),
                              numpy.log(numpy.average(numpy.abs(Fc)**2)),
                              numpy.log(D**2*numpy.average(numpy.abs(Fc)**2)),
                              D, numpy.log(S), mean_fom[0], mean_fom[1]))
    ofs.close()
    logger.write("output log: {}".format(log_out))
    
    hkldata.merge_asu_data(fmask_asu, "Fmask")
    mtz_out = args.output_prefix+".mtz"
    write_mtz(hkldata, mtz_out)
    logger.write("output mtz: {}".format(mtz_out))

    return hkldata
# main()
if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
