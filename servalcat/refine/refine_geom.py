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
import scipy.optimize
import scipy.sparse
from servalcat.utils import logger
from servalcat import utils
from servalcat.refine.refine import Refine

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

def add_arguments(parser):
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--ligand', nargs="*", action="append",
                        help="restraint dictionary cif file(s)")
    parser.add_argument('--ncycle', type=int, default=10,
                        help="number of CG cycles (default: %(default)d)")
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no"],
                        help="all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input. "
                        "Default: %(default)s")
    parser.add_argument('--randomize', type=float, default=0,
                        help='Shake coordinates with specified rmsd')

# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

@profile
def update_fc(hkldata, st, d_min, monlib):
    if st.ncs:
        logger.write("Expanding NCS")
        st = st.clone()
        st.expand_ncs(gemmi.HowToNameCopiedChain.Dup)

    fc_asu = utils.model.calc_fc_fft(st, d_min, cutoff=1e-7, monlib=monlib, source="electron")
    if "FC" in hkldata.df: del hkldata.df["FC"]
    hkldata.merge_asu_data(fc_asu, "FC")
# update_fc()

@profile
def calc_target(hkldata, st=None, d_min=None, monlib=None): # -LL target for SPA
    if st is not None:
        update_fc(hkldata, st, d_min, monlib)

    ret = 0
    for i_bin, g in hkldata.binned():
        Fo = g.FP.to_numpy()
        DFc = g.FC.to_numpy() * hkldata.binned_df.D[i_bin]
        ret += numpy.sum(numpy.abs(Fo-DFc)**2) / hkldata.binned_df.S[i_bin]
    return ret
# calc_target()

@profile
def calc_grad(hkldata, st, monlib, d_min):
    f0 = calc_target(hkldata)
    logger.write("-LL= {:.3e}".format(f0))

    dll_dab = numpy.empty_like(hkldata.df.FP)
    d2ll_dab2 = numpy.zeros(len(hkldata.df.index))
    for i_bin, g in hkldata.binned():
        D = hkldata.binned_df.D[i_bin]
        S = hkldata.binned_df.S[i_bin]
        Fc = g.FC.to_numpy()
        Fo = g.FP.to_numpy()
        dll_dab[g.index] = -2*D/S*(Fo - D*Fc)#.conj()
        d2ll_dab2[g.index] = 2*D**2/S

    # only when Mott-Bethe; 1/s**2
    dll_dab *= hkldata.d_spacings()**2 / (2*numpy.pi**2*0.529177210903)
    d2ll_dab2 *= 1 / (2*numpy.pi**2*0.529177210903)
    #dll_dab *= hkldata.cell.volume / (2 * numpy.pi)**1.5  *2 # why this *2??
    #dll_dab *= dll_dab_den.point_count
        
    dll_dab_asu = gemmi.ComplexAsuData(hkldata.cell, hkldata.sg, hkldata.miller_array(), dll_dab)
    dll_dab_den = dll_dab_asu.transform_f_phi_to_map(sample_rate=3, exact_size=(0, 0, 0)) # must be the same sampling as Fc?
    numpy.array(dll_dab_den, copy=False)[:] *= hkldata.cell.volume**2/dll_dab_den.point_count/2 # why this factor??

    atoms = [x.atom for x in st[0].all()]
    ll = LL(hkldata.cell, hkldata.sg, atoms)
    ll.set_ncs([x.tr for x in st.ncs if not x.given])
    logger.write("LL.ncs= {}".format(ll.ncs))
    ll.calc_grad(dll_dab_den)
    #print(ll.vn)
    #print(hkldata.df)
    d2dfw_table = TableS3(*hkldata.d_min_max())
    d2dfw_table.make_table(1./hkldata.d_spacings(), d2ll_dab2)

    if 0:
        with open("d2dfw_smooth.dat", "w") as ofs:
            ofs.write("x y\n")
            for x, y in zip(d2dfw_table.s3_values, d2dfw_table.y_values):
                ofs.write("{} {}\n".format(x, y))
        with open("d2dfw_org.dat", "w") as ofs:
            ofs.write("x y\n")
            bin_limits = dict(hkldata.bin_and_limits())
            for i_bin, g in hkldata.binned():
                bin_d_max, bin_d_min = bin_limits[i_bin]
                D = hkldata.binned_df.D[i_bin]
                S = hkldata.binned_df.S[i_bin]
                s3_mean = (1/bin_d_max**3 + 1/bin_d_min**3)/2
                y = 2*D**2/S / (2*numpy.pi**2*0.529177210903)
                ofs.write("{} {}\n".format(s3_mean, y))

    b_iso_min = min((cra.atom.b_iso for cra in st[0].all()))
    b_iso_max = max((cra.atom.b_iso for cra in st[0].all()))
    elems = set(cra.atom.element for cra in st[0].all())
    b_sf_min = 0 #min(min(e.it92.b) for e in elems) # because there is constants
    b_sf_max = max(max(e.it92.b) for e in elems)
    ll.make_fisher_table_diag_fast(b_iso_min+b_sf_min, b_iso_max+b_sf_max, d2dfw_table)
    ll.fisher_diag_from_table()
    return ll.vn, ll.am

    # Test hessian
    if 0:
        am_fast = ll.am
        ll.setup_vn_am();
        ll.calc_fisher_diagonal_naive(hkldata.s_array(), d2ll_dab2)
        am_slow = ll.am

        with open("am_slow_fast.dat", "w") as ofs:
            ofs.write("slow fast\n")
            for x, y in zip (am_slow, am_fast):
                ofs.write("{} {}\n".format(x, y))

        sel = numpy.arange(0, len(ll.am), 6)
        print(numpy.array(am_fast)[sel] / numpy.array(am_slow)[sel])
        logger.write("V={}".format(st.cell.volume))
        print(ll.pp1)
        #return ll.vn, ll.am
        quit()
    
    # Test e
    if 0:
        for e in numpy.logspace(-1,-8,num=15):
            k = j = 0
            bak = getattr(atoms[j].pos, "xyz"[k])
            setattr(atoms[j].pos, "xyz"[k], bak+e)
            update_fc(hkldata, st, d_min, monlib)
            setattr(atoms[j].pos, "xyz"[k], bak)
            f1 = calc_target(hkldata)
            print(e, (f1-f0)/e)
    
    e = 1e-3
    ng = []
    j = 0
    f0 = calc_target(hkldata, st, d_min, monlib)
    logger.write("-LL= {:.3e}".format(f0))
    for k in range(3):
        bak = getattr(atoms[j].pos, "xyz"[k])
        setattr(atoms[j].pos, "xyz"[k], bak+e)
        #update_fc(hkldata, st, d_min, monlib)
        f1 = calc_target(hkldata, st, d_min, monlib)
        ng.append((f1-f0)/e)
        print("f0=", f0, "f1=", f1)
        setattr(atoms[j].pos, "xyz"[k], bak)
        
    logger.write("num grad= {}".format(ng))
    logger.write("ana grad= {}".format(ll.vn[j*3:j*3+3]))
    logger.write("num/ana= {}".format(numpy.array(ng)/ll.vn[j*3:j*3+3]))
    logger.write("V/n= {}/{}={}".format(st.cell.volume,dll_dab_den.point_count,st.cell.volume/dll_dab_den.point_count))
    quit()
# calc_grad()

@profile
def set_x(atoms, x):
    for i in range(len(x)//3):
        atoms[i].pos.fromlist(x[3*i:3*i+3])

@profile
def test_refine_grad(hkldata, st, monlib, d_min):
    atoms = [x.atom for x in st[0].all()]

    def f(x):
        set_x(atoms, x)
        return calc_target(hkldata, st, d_min, monlib)
    def grad(x):
        set_x(atoms, x)
        return calc_grad(hkldata, st, monlib, d_min)[0]

    x0 = sum([a.pos.tolist() for a in atoms], [])
    
    res = scipy.optimize.minimize(fun=f,
                                  jac=grad,
                                  x0=x0,
                                  options=dict(maxiter=50))
    logger.write(str(res))
    set_x(atoms, res.x)
    utils.fileio.write_model(st, "refined", pdb=True, cif=True)

@profile
def test_refine(hkldata, st, monlib, d_min, weight=1):
    enerlib = utils.restraints.load_ener_lib()
    r = utils.restraints.Restraints(st, monlib, enerlib)
    geom = r.calc_geom()
    N = len(r.atoms)*3
    gw = 1
    
    ll_g, ll_a = calc_grad(hkldata, st, monlib, d_min)
    vn = numpy.array(ll_g)*weight + numpy.array(geom.vn, copy=False) * gw
    geom_am = numpy.array(geom.am, copy=False)
    geom_am[:] *= gw
    #numpy.array(geom.am, copy=False)[:] *= gw
    logger.write("geom_am= {}".format(geom.am))
    geom_am[:2*N] += numpy.array(ll_a, copy=False) * weight

    coo = scipy.sparse.coo_matrix(geom.for_coo_matrix(), shape=(N, N))
    lil = coo.tolil()
    rows, cols = lil.nonzero()
    lil[cols,rows] = lil[rows,cols]
    diag = lil.diagonal()
    print("diagonal min=", numpy.min(diag))
    diag[diag<=0] = 1.
    diag = numpy.sqrt(diag)
    rdiag = 1./diag # sk
    rdmat = scipy.sparse.diags(rdiag)
    rdmat.dot(lil).dot(rdmat)
    x0 = numpy.zeros(3*len(r.atoms))
    for i, a in enumerate(r.atoms):
        x0[3*i:3*(i+1)] = a.pos.tolist()

    def f(x):
        target_geom = r.calc_target(x) # set x
        return calc_target(hkldata, st, d_min, monlib) * weight + target_geom * gw

    #print("x0=", x0)
    f0 = f(x0)
    logger.write("f0= {:.4e}".format(f0))
    csc = lil.tocsc()
    #ilu = scipy.sparse.linalg.spilu(csc)
    #print(rdmat.dot(lil).dot(rdmat).toarray())
    gamma = 0.

    Pinv = scipy.sparse.coo_matrix(geom.precondition_eigen_coo(1e-4), shape=(N, N)) # did not work if <= 1e-7
    dx, r.gamma = cgsolve_rm(A=lil, v=vn, M=Pinv, gamma=r.gamma)

    if 0: # to check hessian scale
        with open("minimise_line.dat", "w") as ofs:
            ofs.write("s f\n")
            for s in numpy.arange(-2, 2, 0.1):
                fval = f(x0+s*dx)
                ofs.write("{} {}\n".format(s, fval))
        quit()

    
    for i in range(3):
        dx2 = scale_shifts(dx, 1/2**i)
        f1 = f(x0-dx2)
        print("f1, ", i, "=", f1)
        logger.write("f1, {}= {:.4e}".format(i, f1))
        if f1 < f0: break


    r.show_all(True)
    #utils.fileio.write_model(st, "refined", pdb=True, cif=True)

def main(args):
    st = utils.fileio.read_structure(args.model)

    if not st.cell.is_crystal():
        sg = st.find_spacegroup()
        if sg is None or sg.number == 1:
            logger.writeln("This is non-crystal.")
            #st.cell = utils.model.box_from_model(st[0], 10) # XXX padding should be defined from ADP
        else:
            raise SystemExit("Space group is given but cell is non-crystallographic")

    logger.write("NCS= {}".format([x for x in st.ncs]))
    st2 = st.clone()
    if st.ncs:
        logger.write("Take NCS constraints into account.")
        st2.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
        utils.fileio.write_model(st2, file_name="input_expanded.pdb")

    if args.ligand: args.ligand = sum(args.ligand, [])
    monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                   stop_for_unknowns=True,
                                                   check_hydrogen=(args.hydrogen=="yes"))
    h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                "yes":gemmi.HydrogenChange.NoChange,
                "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=logger,
                                  reorder=True, ignore_unknown_links=False) # we should remove logger here??

    refiner = Refine(st, topo, monlib)

    if args.randomize > 0:
        numpy.random.seed(0)
        from servalcat.utils import model
        utils.model.shake_structure(refiner.st, args.randomize, copy=False)

    for i in range(args.ncycle):
        logger.writeln("==== CYCLE {:2d}".format(i))
        refiner.run_cycle()
        utils.fileio.write_model(refiner.st, "refined_{:02d}".format(i), pdb=True)#, cif=True)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
