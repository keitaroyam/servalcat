"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import pandas
import scipy.sparse
from servalcat.utils import logger
from servalcat import utils
from . import cgsolve

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

def scale_shifts(dx, scale, n_atoms):
    #ave_shift = numpy.mean(dx)
    #max_shift = numpy.maximum(dx)
    #rms_shift = numpy.std(dx)
    shift_allow_high =  0.5
    shift_allow_low  = -0.5
    shift_max_allow_B = 30.0
    shift_min_allow_B = -30.0
    dx = scale * dx
    dxx = dx[:n_atoms*3]
    dxx[dxx > shift_allow_high] = shift_allow_high
    dxx[dxx < shift_allow_low] = shift_allow_low
    dxb = dx[n_atoms*3:]
    dxb[dxb > shift_max_allow_B] = shift_max_allow_B
    dxb[dxb < shift_min_allow_B] = shift_min_allow_B
    return dx

class Refine:
    def __init__(self, st, topo, monlib, ll=None, refine_adp=True):
        self.monlib = monlib
        self.st = st # clone()?
        self.atoms = [None for _ in range(self.st[0].count_atom_sites())]
        for cra in self.st[0].all(): self.atoms[cra.atom.serial-1] = cra.atom
        self.lookup = {x.atom: x for x in self.st[0].all()}
        self.ll = ll
        self.geom = gemmi.Geometry(self.st)
        # shake here if needed
        self.geom.load_topo(topo)
        # load external restraints here
        self.geom.finalize_restraints()
        print("   bonds =", len(self.geom.bonds))
        print("  angles =", len(self.geom.angles))
        print("torsions =", len(self.geom.torsions))
        self.gamma = 0
        self.outlier_sigmas = dict(bond=5, angle=5, torsion=5, vdw=5, chir=5, plane=5)
        self.use_nucleus = False
        self.refine_adp = False if self.ll is None else refine_adp 

    def set_x(self, x):
        for i in range(len(self.atoms)):
            self.atoms[i].pos.fromlist(x[3*i:3*i+3]) # faster than substituting pos.x,pos.y,pos.z
            if self.refine_adp: # only isotropic for now
                self.atoms[i].b_iso = max(0.5, x[len(self.atoms) * 3 + i]) # minimum B = 0.5

    def get_x(self):
        x = numpy.zeros(len(self.atoms) * (4 if self.refine_adp else 3))
        for i, a in enumerate(self.atoms):
            x[3*i:3*(i+1)] = a.pos.tolist()
            if self.refine_adp: # only isotropic for now
                x[len(self.atoms) * 3 + i] = self.atoms[i].b_iso

        return x
    @profile
    def calc_target(self, w=1, wadp=1, target_only=False):
        N = len(self.atoms) * (4 if self.refine_adp else 3)
        if self.geom is not None:
            geom_x = self.geom.calc(self.use_nucleus, target_only)
            geom_a = self.geom.calc_adp_restraint(target_only, wadp) if self.refine_adp else 0
            logger.writeln(" geom_x = {}".format(geom_x))
            logger.writeln(" geom_a = {}".format(geom_a))
            geom = geom_x + geom_a
            if not target_only:
                # for now we do not implement adp restraints
                g_vn = numpy.zeros(N)
                #g_vn = numpy.array(self.geom.target.vn) # don't want copy?
                g_vn[:len(self.geom.target.vn)] = self.geom.target.vn
                coo = scipy.sparse.coo_matrix(self.geom.target.am_for_coo(), shape=(N, N))
                #print("am=", coo)
                lil = coo.tolil()
                rows, cols = lil.nonzero()
                lil[cols,rows] = lil[rows,cols]
                g_am = lil
        else:
            geom = 0

        if self.ll is not None:
            self.ll.update_fc()
            ll = self.ll.calc_target()
            if not target_only:
                l_vn, l_am = self.ll.calc_grad(self.refine_adp)
        else:
            ll = 0

        f =  w * ll + geom

        if not target_only:
            if self.geom is not None and self.ll is not None:
                vn = w * l_vn + g_vn
                am = w * l_am + g_am
            elif self.geom is None:
                vn = w * l_vn
                am = w * l_am
            elif self.ll is None:
                vn = g_vn
                am = g_am
        else:
            vn, am = None, None

        return f, vn, am

    def show_model_stats(self):
        f0 = self.geom.calc(self.use_nucleus, True)
        chirsstr = {gemmi.ChiralityType.Positive:"positive",
                    gemmi.ChiralityType.Negative:"negative",
                    gemmi.ChiralityType.Both:"both"}

        items = dict(bond=self.geom.reporting.bonds,
                     angle=self.geom.reporting.angles,
                     torsion=self.geom.reporting.torsions,
                     chir=self.geom.reporting.chirs,
                     plane=self.geom.reporting.planes,
                     vdw=self.geom.reporting.vdws)
        labs = dict(bond="Bond distances",
                    angle="Bond angles",
                    torsion="Torsion angles",
                    chir="Chiral centres",
                    plane="Planar groups",
                    vdw="VDW repulsions")
        sdata = []
        skeys = []
        for k in items:
            if k == "torsion":
                sum_r, sum_s, n = {}, {}, {}
            else:
                sum_r, sum_s = 0., 0 # sum of squared resid and sigma
                n = 0
            outliers = []
            for rep in items[k]:
                if k in ("chir", "plane", "vdw"):
                    g, resid = rep
                    closest = g
                else:
                    g, closest, resid = rep
                if k == "bond":
                    ideal = closest.value_nucleus if self.use_nucleus else closest.value
                    sigma = closest.sigma_nucleus if self.use_nucleus else closest.sigma
                elif k == "chir":
                    ideal, sigma = g.value, g.sigma
                elif k == "plane":
                    ideal, sigma = 0, g.sigma
                else:
                    ideal, sigma = closest.value, closest.sigma

                if k == "plane":
                    sum_r += numpy.sum(numpy.square(resid))
                    sum_s += sigma * len(resid)
                    n += len(resid)
                    for a, r in zip(g.atoms, resid):
                        r = abs(r)
                        if r / sigma > self.outlier_sigmas[k]:
                            outliers.append((a, closest, r, ideal, r / sigma))
                else:
                    if k == "torsion":
                        p = closest.period
                        sum_r[p] = sum_r.get(p, 0) + resid**2
                        sum_s[p] = sum_s.get(p, 0) + sigma
                        n[p] = n.get(p, 0) + 1
                    elif k != "bond" or g.type < 2:
                        sum_r += resid**2
                        sum_s += sigma
                        n += 1
                    if abs(resid) / sigma > self.outlier_sigmas[k]:
                        outliers.append((g, closest, resid+ideal, ideal, resid / sigma))

            if k == "torsion":
                for p in sorted(sum_r):
                    rmsd = numpy.sqrt(sum_r[p] / n[p]) if n[p] > 0 else 0.
                    mean_sig = sum_s[p] / n[p] if n[p] > 0 else 0.
                    sdata.append([n[p], rmsd, mean_sig])
                    skeys.append("{}, period {} refined".format(labs[k], p))
            else:
                rmsd = numpy.sqrt(sum_r / n) if n > 0 else 0.
                mean_sig = sum_s / n if n > 0 else 0.
                sdata.append([n, rmsd, mean_sig])
                skeys.append("{} refined".format(labs[k]))
            if outliers:
                outliers.sort(key=lambda x: -abs(x[-1]))
                odata = []
                for g, closest, val, ideal, z in outliers:
                    odata.append({})
                    if k == "plane":
                        odata[-1]["atom"] = str(self.lookup[g])
                    else:
                        odata[-1].update({"atom{}".format(i+1):str(self.lookup[a]) for i, a in enumerate(g.atoms)})
                    odata[-1]["value"] = val
                    if k != "plane": odata[-1]["ideal"] = ideal
                    if k == "torsion": odata[-1]["per"] = closest.period
                    if k == "chir": odata[-1]["sign"] = chirsstr[g.sign]
                    if k in ("vdw", "bond"): odata[-1]["type"] = g.type
                    if k == "bond" and g.type == 2: odata[-1]["alpha"] = g.alpha
                    if k == "vdw": odata[-1]["sym"] = g.sym_idx
                    odata[-1]["z"] = z

                df = pandas.DataFrame(odata)
                logger.writeln("{} outliers".format(k))
                logger.writeln(df.to_string(float_format="{:.3f}".format) + "\n")

        df = pandas.DataFrame(sdata, index=skeys, columns=["N restraints", "rmsd", "Av(sigma)"])
        logger.writeln(df.to_string(float_format="{:.3f}".format) + "\n")

    @profile
    def run_cycle(self, weight=1, adp_weight=1):
        if 0: # test of grad
            self.ll.update_fc()
            x0 = self.get_x()
            f0 = self.ll.calc_target()
            ader = self.ll.calc_grad(self.refine_adp)[0]
            for e in 1e-3, 1e-4, 1e-5, 1e-6:
                x1 = numpy.copy(x0)
                x1[0] += e
                self.set_x(x1)
                self.ll.update_fc()
                f1 = self.ll.calc_target()
                nder = (f1 - f0) / e
                print("e=", e)
                print("NUM DER=", nder)
                print("ANA DER=", ader[0])
                print("ratio=", nder/ader[0])
            quit()

        # TODO overall scaling
        self.geom.setup_vdw(self.monlib.ener_lib)
        print("vdws =", len(self.geom.vdws))
        self.geom.setup_target(True, self.refine_adp)
        f0, vn, am = self.calc_target(weight, adp_weight)
        x0 = self.get_x()
        logger.writeln("f0= {:.4e}".format(f0))

        if 0:
            assert not self.refine_adp # not supported!
            logger.writeln("Preconditioning using eigen")
            #Pinv = scipy.sparse.coo_matrix(self.geom.target.precondition_eigen_coo(1e-4), shape=(N, N)) # did not work if <= 1e-7
            N = len(self.atoms) * 3
            tmp = am.tocoo()
            Pinv = scipy.sparse.coo_matrix(gemmi.precondition_eigen_coo(tmp.data, tmp.row, tmp.col, N, 1e-4),
                                           shape=(N, N))
            M = Pinv
            #a = Pinv.T.dot(lil).dot(Pinv)
            #logger.writeln("cond(Pinv^-1 A Pinv)= {}".format(numpy.linalg.cond(a.todense())))
            #logger.writeln("cond(A)= {}".format(numpy.linalg.cond(lil.todense())))
            #quit()
            #M = scipy.sparse.identity(N)
        else:
            diag = am.diagonal()
            print("diagonal min=", numpy.min(diag))
            diag[diag<=0] = 1.
            diag = numpy.sqrt(diag)
            rdiag = 1./diag # sk
            rdmat = scipy.sparse.diags(rdiag)
            M = rdmat
            
        dx, self.gamma = cgsolve.cgsolve_rm(A=am, v=vn, M=M, gamma=self.gamma)
        dxx = dx[:len(self.atoms)*3]
        logger.writeln("dx = {}".format(dxx))
        logger.writeln("min(dx) = {}".format(numpy.min(dxx)))
        logger.writeln("max(dx) = {}".format(numpy.max(dxx)))
        logger.writeln("mean(dx)= {}".format(numpy.mean(dxx)))
        if self.refine_adp:
            db = dx[len(self.atoms)*3:]
            logger.writeln("dB = {}".format(db))
            logger.writeln("min(dB) = {}".format(numpy.min(db)))
            logger.writeln("max(dB) = {}".format(numpy.max(db)))
            logger.writeln("mean(dB)= {}".format(numpy.mean(db)))
            
        if 0: # to check hessian scale
            with open("minimise_line.dat", "w") as ofs:
                ofs.write("s f\n")
                for s in numpy.arange(-2, 2, 0.1):
                    fval = f(x0+s*dx)
                    ofs.write("{} {}\n".format(s, fval))
            quit()

        for i in range(3):
            dx2 = scale_shifts(dx, 1/2**i, len(self.atoms))
            self.set_x(x0 - dx2)
            f1, _, _ = self.calc_target(weight, adp_weight, target_only=True)
            logger.writeln("f1, {}= {:.4e}".format(i, f1))
            if f1 < f0: break
        else:
            logger.writeln("function not minimised")
            self.set_x(x0)

        self.show_model_stats()
        utils.model.adp_analysis(self.st)
# class Refine
