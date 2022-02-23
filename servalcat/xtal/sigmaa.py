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

"""
DFc = sum_j D_j F_c,j
The last Fc,n is bulk solvent contribution.
"""

def add_arguments(parser):
    parser.description = 'Sigma-A parameter estimation for crystallographic data'
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin', required=True,
                        help='MTZ column for F,SIGF')
    parser.add_argument('--model', required=True, nargs="+", action="append",
                        help='Input atomic model file(s)')
    parser.add_argument("-d", '--d_min', type=float)
    #parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int, default=20,
                        help="Number of bins (default: %(default)d)")
    parser.add_argument('-s', '--source', choices=["electron", "xray", "neutron"], default="xray")
    parser.add_argument('--D_as_exp',  action='store_true',
                        help="estimate D through exp(x) as a positivity constraint")
    parser.add_argument('-o','--output_prefix', default="sigmaa",
                        help='output file name prefix (default: %(default)s)')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def calc_abs_DFc(Ds, Fcs):
    DFc = sum(Ds[i] * Fcs[i] for i in range(len(Ds)))
    return numpy.abs(DFc)
# calc_abs_DFc()

def deriv_DFc2_and_DFc_dDj(Ds, Fcs):
    """
    [(d/dDj |sum(Dk * Fc,k)|^2,
      d/dDj |sum(Dk * Fc,k)|), ....] for j = 0 .. N-1
    """
    
    DFc = sum(Ds[i] * Fcs[i] for i in range(len(Ds)))
    DFc_abs = numpy.abs(DFc)
    
    ret = []
    for j in range(len(Ds)):
        rsq = numpy.real(Fcs[j] * DFc.conj())
        ret.append((2 * rsq,
                    rsq / DFc_abs))
    return DFc_abs, ret
# deriv_DFc2_and_DFc_dDj()

def fom_acentric(Fo, varFo, Fcs, Ds, S, epsilon):
    Sigma = 2 * varFo + epsilon * S
    return gemmi.bessel_i1_over_i0(2 * Fo * calc_abs_DFc(Ds, Fcs) / Sigma)
# fom_acentric()

def fom_centric(Fo, varFo, Fcs, Ds, S, epsilon):
    Sigma = varFo + epsilon * S
    return numpy.tanh(Fo * calc_abs_DFc(Ds, Fcs) / Sigma)
# fom_centric()

def mlf_acentric(Fo, varFo, Fcs, Ds, S, epsilon):
    # https://doi.org/10.1107/S0907444911001314
    # eqn (4)
    Sigma = 2 * varFo + epsilon * S
    DFc = calc_abs_DFc(Ds, Fcs)
    ret = numpy.log(2) + numpy.log(Fo) - numpy.log(Sigma)
    ret += -(Fo**2 + DFc**2)/Sigma
    ret += gemmi.log_bessel_i0(2*Fo*DFc/Sigma)
    return -ret
# mlf_acentric()

def deriv_mlf_wrt_D_S_acentric(Fo, varFo, Fcs, Ds, S, epsilon):
    deriv = numpy.zeros(1+len(Ds))
    Sigma = 2 * varFo + epsilon * S
    Fo2 = Fo**2
    DFc, tmp = deriv_DFc2_and_DFc_dDj(Ds, Fcs)
    i1_i0_x = gemmi.bessel_i1_over_i0(2*Fo*DFc/Sigma) # m
    for i, (sqder, der) in enumerate(tmp):
        deriv[i] = -numpy.sum(-sqder / Sigma + i1_i0_x * 2 * Fo * der / Sigma)
    
    deriv[-1] = -numpy.sum((-1/Sigma + (Fo2 + DFc**2 - i1_i0_x * 2 * Fo * DFc) / Sigma**2) * epsilon)
    return deriv
# deriv_mlf_wrt_D_S_acentric()

def mlf_centric(Fo, varFo, Fcs, Ds, S, epsilon):
    # https://doi.org/10.1107/S0907444911001314
    # eqn (4)
    Sigma = varFo + epsilon * S
    DFc = calc_abs_DFc(Ds, Fcs)
    ret = 0.5 * (numpy.log(2 / numpy.pi) - numpy.log(Sigma))
    ret += -0.5 * (Fo**2 + DFc**2) / Sigma
    ret += gemmi.log_cosh(Fo * DFc / Sigma)
    return -ret
# mlf_centric()

def deriv_mlf_wrt_D_S_centric(Fo, varFo, Fcs, Ds, S, epsilon):
    deriv = numpy.zeros(1+len(Ds))
    Sigma = varFo + epsilon * S
    Fo2 = Fo**2
    DFc, tmp = deriv_DFc2_and_DFc_dDj(Ds, Fcs)
    tanh_x = numpy.tanh(Fo*DFc/Sigma)
    for i, (sqder, der) in enumerate(tmp):
        deriv[i] = -numpy.sum(-0.5 * sqder / Sigma + tanh_x * Fo * der / Sigma)
    deriv[-1] = -numpy.sum((-0.5 / Sigma + (0.5*(Fo2+DFc**2) - tanh_x * Fo*DFc)/Sigma**2)*epsilon)
    return deriv
# deriv_mlf_wrt_D_S_centric()

#import line_profiler
#profile = line_profiler.LineProfiler()
#import atexit
#atexit.register(profile.print_stats)
#@profile
def mlf(df, Ds, S, centric_sel):
    ret = 0.
    func = (mlf_acentric, mlf_centric)
    for c, cidxes in centric_sel:
        Fcs = [df["FC{}".format(i)].to_numpy()[cidxes] for i in range(len(Ds))]
        ret += numpy.sum(func[c](df.FP.to_numpy()[cidxes], df.SIGFP.to_numpy()[cidxes]**2, Fcs, Ds, S, df.epsilon.to_numpy()[cidxes]))
    return ret
# mlf()

#@profile
def deriv_mlf_wrt_D_S(df, Ds, S, centric_sel):
    ret = []
    func = (deriv_mlf_wrt_D_S_acentric, deriv_mlf_wrt_D_S_centric)
    for c, cidxes in centric_sel:
        Fcs = [df["FC{}".format(i)].to_numpy()[cidxes] for i in range(len(Ds))]
        ret.append((func[c](df.FP.to_numpy()[cidxes], df.SIGFP.to_numpy()[cidxes]**2, Fcs, Ds, S, df.epsilon.to_numpy()[cidxes])))
    return sum(ret)
# deriv_mlf_wrt_D_S()

def calc_fom(df, Ds, S, centric_sel):
    ret = pandas.Series(index=df.index)
    func = (fom_acentric, fom_centric)
    for c, cidxes in centric_sel:
        Fcs = [df["FC{}".format(i)].to_numpy()[cidxes] for i in range(len(Ds))]
        ret[cidxes] = func[c](df.FP.to_numpy()[cidxes], df.SIGFP.to_numpy()[cidxes]**2, Fcs, Ds, S, df.epsilon.to_numpy()[cidxes])

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

def determine_mlf_params(hkldata, nmodels, centric_and_selections, D_as_exp=False):
    if D_as_exp:
        transD = numpy.exp # D = transD(x)
        transD_deriv = numpy.exp # dD/dx
        transD_inv = numpy.log # x = transD_inv(D)
    else:
        transD = lambda x: x
        transD_deriv = lambda x: 1
        transD_inv = lambda x: x
    
    # Initial values
    for i in range(nmodels):
        hkldata.binned_df["D{}".format(i)] = 1.

    hkldata.binned_df["S"] = 10000.
    for i_bin, idxes in hkldata.binned():
        FC = numpy.abs(hkldata.df.FC.to_numpy()[idxes])
        FP = hkldata.df.FP.to_numpy()[idxes]
        D = numpy.corrcoef(FP, FC)[1,0]
        hkldata.binned_df.loc[i_bin, "D0"] = D
        hkldata.binned_df.loc[i_bin, "S"] = numpy.var(FP - D * FC)

    logger.write("Initial estimates:")
    logger.write(hkldata.binned_df.to_string())

    for i_bin, idxes in hkldata.binned():
        x0 = [transD_inv(hkldata.binned_df["D{}".format(i)][i_bin]) for i in range(nmodels)] + [hkldata.binned_df.S[i_bin]]
        def target(x):
            return mlf(hkldata.df, transD(x[:-1]), x[-1], centric_and_selections[i_bin])
        def grad(x):
            g = deriv_mlf_wrt_D_S(hkldata.df, transD(x[:-1]), x[-1], centric_and_selections[i_bin])
            g[:-1] *= transD_deriv(x[:-1])
            return g

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
                print("DERIV:", i, gnum, gana[i], gana[i]/gnum)

        #print("Bin", i_bin)
        res = scipy.optimize.minimize(fun=target, x0=x0, jac=grad)
        #print(res)
        
        for i in range(nmodels):
            hkldata.binned_df.loc[i_bin, "D{}".format(i)] = transD(res.x[i])
        hkldata.binned_df.loc[i_bin, "S"] = res.x[-1]

    logger.write("Refined estimates:")
    logger.write(hkldata.binned_df.to_string())
# determine_mlf_params()

def merge_models(sts): # simply merge models. no fix in chain ids etc.
    model = gemmi.Model("1")
    for st in sts:
        for m in st:
            for c in m:
                model.add_chain(c)
    return model
# merge_models()

# TODO isotropize map
# TODO add missing reflections
def main(args):
    if args.nbins < 1:
        logger.error("--nbins must be > 0")
        return

    args.model = sum(args.model, [])
    sts = []
    for xyzin in args.model:
        sts.append(utils.fileio.read_structure(xyzin))

    for st in sts[1:]:
        if st.cell.parameters != sts[0].cell.parameters:
            logger.write("WARNING: resetting cell to 1st model.")
            st.cell = sts[0].cell
        if st.find_spacegroup() != sts[0].find_spacegroup():
            logger.write("WARNING: resetting space group to 1st model.")
            st.spacegroup_hm = sts[0].spacegroup_hm
        
    nmodels = len(sts) + 1 # bulk
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
    grid.setup_from(sts[0], spacing=0.4)
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)
    masker.put_mask_on_float_grid(grid, merge_models(sts))
    fmask_asu = gemmi.transform_map_to_f_phi(grid).prepare_asu_data(dmin=d_min)

    # TODO no need to make multiple AsuData (just inefficient)
    fc_asu = [utils.model.calc_fc_fft(st, d_min, source=args.source, mott_bethe=args.source=="electron") for st in sts]
    if len(fc_asu) == 1:
        fc_asu_total = fc_asu[0]
    else:
        fc_asu_total = type(fc_asu[0])(fc_asu[0].unit_cell, fc_asu[0].spacegroup, fc_asu[0].miller_array, fc_asu[0].value_array)
        for asu in fc_asu[1:]:
            fc_asu_total.value_array[:] += asu.value_array
        
    logger.write("Scaling Fc..")
    scaling = gemmi.Scaling(sts[0].cell, sts[0].find_spacegroup())
    scaling.use_solvent = True
    scaling.prepare_points(fc_asu_total, scaleto, fmask_asu)
    scaling.fit_isotropic_b_approximately()
    scaling.fit_parameters()
    b_aniso = scaling.b_overall
    logger.write(" k_ov= {:.2e} B= {}".format(scaling.k_overall, b_aniso))

    # TODO 'merge' not needed; they must have same hkl array (really? what if missing data?)
    hkldata = utils.hkl.hkldata_from_asu_data(fp, "FP")
    hkldata.merge_asu_data(sigfp, "SIGFP")
    for i, asu in enumerate(fc_asu):
        hkldata.merge_asu_data(asu, "FC{}".format(i))
    hkldata.merge_asu_data(fmask_asu, "FC{}".format(nmodels-1)) # will become Fbulk

    overall_scale = scaling.get_overall_scale_factor(hkldata.miller_array())
    solvent_scale = scaling.get_solvent_scale(0.25 / hkldata.d_spacings()**2)
    hkldata.df["FC{}".format(nmodels-1)] *= solvent_scale
    for i in range(nmodels):
        hkldata.df["FC{}".format(i)] *= overall_scale
    
    # total
    hkldata.df["FC"] = 0j
    for i in range(nmodels):
        hkldata.df.FC += hkldata.df["FC{}".format(i)]

    fca = numpy.abs(hkldata.df.FC.to_numpy())
    fpa = hkldata.df.FP.to_numpy()
    logger.write(" CC(Fo,Fc)= {:.4f}".format(numpy.corrcoef(fca, fpa)[0,1]))
    logger.write(" Rcryst= {:.4f}".format(numpy.sum(numpy.abs(fca-fpa))/numpy.sum(fpa)))

    hkldata.calc_epsilon()
    hkldata.calc_centric()
    hkldata.setup_binning(n_bins=args.nbins)

    # Create a centric selection table for faster look up
    centric_and_selections = {}
    for i_bin, idxes in hkldata.binned():
        centric_and_selections[i_bin] = []
        for c, g2 in hkldata.df.loc[idxes].groupby("centric", sort=False):
            centric_and_selections[i_bin].append((c, g2.index))
    
    logger.write("Estimating sigma-A parameters..")
    determine_mlf_params(hkldata, nmodels, centric_and_selections, args.D_as_exp)

    log_out = "{}.log".format(args.output_prefix)
    ofs = open(log_out, "w")
    ofs.write("""$TABLE: Statistics :
$GRAPHS
: log(Mn(|F|^2)) and variances :A:1,7,8,9,10:
: FOM :A:1,11,12:
: D :A:1,{Dns}:
: DFc :A:1,{DFcns}:
: number of reflections :A:1,3,4:
$$
1/resol^2 bin n_a n_c d_max d_min log(Mn(|Fo|^2)) log(Mn(|Fc|^2)) log(Mn(|DFc|^2)) log(Sigma) FOM_a FOM_c {Ds} {DFcs}
$$
$$
""".format(Dns=",".join(map(str, range(13, 13+nmodels))),
           Ds=" ".join(["D{}".format(i) for i in range(nmodels)]),
           DFcns=",".join(map(str, range(13+nmodels, 13+nmodels*2))),
           DFcs=" ".join(["log(Mn(|D{}Fc{}|))".format(i,i) for i in range(nmodels)]),
           ))
    tmpl = "{:.4f} {:3d} {:7d} {:7d} {:7.3f} {:7.3f} {:.4e} {:.4e} {:4e}"
    tmpl += "{: .4f} " * (nmodels * 2)
    tmpl += "{: .4e} {:.4f} {:.4f}\n"

    hkldata.df["FWT"] = 0j
    hkldata.df["DELFWT"] = 0j
    hkldata.df["FOM"] = 0.
    for i_bin, idxes in hkldata.binned():
        bin_d_min = hkldata.binned_df.d_min[i_bin]
        bin_d_max = hkldata.binned_df.d_max[i_bin]
        Ds = [max(0., hkldata.binned_df["D{}".format(i)][i_bin]) for i in range(nmodels)] # negative D is replaced with zero here
        DFcs = [numpy.log(Ds[i] * numpy.average(numpy.abs(hkldata.df["FC{}".format(i)].to_numpy()[idxes]))) for i in range(nmodels)]
        S = hkldata.binned_df.S[i_bin]

        # 0: acentric 1: centric
        mean_fom = [0, 0]
        nrefs = [0, 0]
        fom_func = [fom_acentric, fom_centric]
        for c, cidxes in centric_and_selections[i_bin]:
            Fcs = [hkldata.df["FC{}".format(i)].to_numpy()[cidxes] for i in range(len(Ds))]

            Fc = numpy.abs(hkldata.df.FC.to_numpy()[cidxes])
            phic = numpy.angle(hkldata.df.FC.to_numpy()[cidxes])
            expip = numpy.cos(phic) + 1j*numpy.sin(phic)
            Fo = hkldata.df.FP.to_numpy()[cidxes]
            
            m = fom_func[c](Fo, hkldata.df.SIGFP.to_numpy()[cidxes]**2, Fcs, Ds, S, hkldata.df.epsilon.to_numpy()[cidxes])
            mean_fom[c] = numpy.mean(m)
            nrefs[c] = len(cidxes)

            DFc = calc_abs_DFc(Ds, Fcs)
            hkldata.df.loc[cidxes, "FOM"] = m
            hkldata.df.loc[cidxes, "DELFWT"] = (m * Fo - DFc) * expip
            if c == 0:
                hkldata.df.loc[cidxes, "FWT"] = (2 * m * Fo - DFc) * expip
            else:
                hkldata.df.loc[cidxes, "FWT"] = (m * Fo) * expip

        Fc = hkldata.df.FC.to_numpy()[idxes]
        Fcs = [hkldata.df["FC{}".format(i)].to_numpy()[idxes] for i in range(len(Ds))]
        Fo = hkldata.df.FP.to_numpy()[idxes]
        DFc = calc_abs_DFc(Ds, Fcs)
        ofs.write(tmpl.format(1/bin_d_min**2, i_bin, nrefs[0], nrefs[1], bin_d_max, bin_d_min,
                              numpy.log(numpy.average(numpy.abs(Fo)**2)),
                              numpy.log(numpy.average(numpy.abs(Fc)**2)),
                              numpy.log(numpy.average(DFc**2)),
                              numpy.log(S), mean_fom[0], mean_fom[1], *Ds, *DFcs)) # no python2 support!
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
