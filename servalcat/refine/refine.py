"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import gemmi
import numpy
import pandas
import scipy.sparse
import servalcat # for version
from servalcat.utils import logger
from servalcat import utils
from servalcat.refmac import exte
from servalcat.refmac.refmac_keywords import parse_keywords
from . import cgsolve
u_to_b = utils.model.u_to_b
b_to_u = utils.model.b_to_u

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

class Geom:
    def __init__(self, st, topo, monlib, sigma_b=30, shake_rms=0, refmac_keywords=None):
        self.st = st
        self.lookup = {x.atom: x for x in self.st[0].all()}
        self.specs = utils.model.find_special_positions(self.st)
        self.geom = gemmi.Geometry(self.st, monlib.ener_lib)
        self.sigma_b = sigma_b
        if shake_rms > 0:
            numpy.random.seed(0)
            utils.model.shake_structure(self.st, shake_rms, copy=False)
            utils.fileio.write_model(self.st, "shaken", pdb=True, cif=True)
        self.geom.load_topo(topo)
        self.use_nucleus = False
        self.calc_kwds = {"use_nucleus": self.use_nucleus}
        if refmac_keywords:
            exte.read_external_restraints(refmac_keywords, self.st, self.geom)
            kwds = parse_keywords(refmac_keywords)
            for k in ("wbond", "wangle", "wtors", "wplane", "wchir", "wvdw"):
                if k in kwds:
                    self.calc_kwds[k] = kwds[k]
                    logger.writeln("setting geometry weight {}= {}".format(k, kwds[k]))
        self.geom.finalize_restraints()
        self.outlier_sigmas = dict(bond=5, angle=5, torsion=5, vdw=5, chir=5, plane=5, staca=5, stacd=5)
        self.parents = {}
        self.check_chemtypes(os.path.join(monlib.path(), "ener_lib.cif"), topo)
    # __init__()

    def check_chemtypes(self, enerlib_path, topo):
        block = gemmi.cif.read(enerlib_path).sole_block()
        all_types = set(block.find_values("_lib_atom.type"))
        for ci in topo.chain_infos:
            for ri in ci.res_infos:
                cc_all = {x: ri.get_final_chemcomp(x) for x in set(a.altloc for a in ri.res)}
                for a in ri.res:
                    cca = cc_all[a.altloc].find_atom(a.name)
                    if cca is None: # I believe it won't happen..
                        logger.writeln("WARNING: restraint for {} not found.".format(self.lookup[a]))
                    elif cca.chem_type not in all_types:
                        raise RuntimeError("Energy type {} of {} not found in ener_lib.".format(cca.chem_type,
                                                                                                self.lookup[a]))
    def set_h_parents(self):
        self.parents = {}
        for bond in self.geom.bonds:
            if bond.atoms[0].is_hydrogen():
                self.parents[bond.atoms[0]] = bond.atoms[1]
            elif bond.atoms[1].is_hydrogen():
                self.parents[bond.atoms[1]] = bond.atoms[0]
    # set_h_parents()

    def calc(self, target_only):
        return self.geom.calc(check_only=target_only, **self.calc_kwds)
    def calc_adp_restraint(self, target_only):
        return self.geom.calc_adp_restraint(target_only, self.sigma_b)
    def calc_target(self, target_only, refine_xyz, adp_mode, N):
        self.geom.clear_target()
        geom_x = self.calc(target_only) if refine_xyz else 0
        geom_a = self.calc_adp_restraint(target_only) if adp_mode > 0 else 0
        logger.writeln(" geom_x = {}".format(geom_x))
        logger.writeln(" geom_a = {}".format(geom_a))
        geom = geom_x + geom_a
        if not target_only:
            g_vn = numpy.array(self.geom.target.vn) # don't want copy?            
            coo = scipy.sparse.coo_matrix(self.geom.target.am_for_coo(), shape=(N, N))
            lil = coo.tolil()
            rows, cols = lil.nonzero()
            lil[cols,rows] = lil[rows,cols]
            g_am = lil
            diag = g_am.diagonal()
            logger.writeln("diag(restr) min= {:3e} max= {:3e}".format(numpy.min(diag),
                                                                      numpy.max(diag)))
        else:
            g_vn, g_am = None, None

        return geom, g_vn, g_am
        
    def show_model_stats(self, show_outliers=True):
        f0 = self.calc(True)
        ret = {"outliers": {}}
        if show_outliers:
            get_table = dict(bond=self.geom.reporting.get_bond_outliers,
                             angle=self.geom.reporting.get_angle_outliers,
                             torsion=self.geom.reporting.get_torsion_outliers,
                             chir=self.geom.reporting.get_chiral_outliers,
                             plane=self.geom.reporting.get_plane_outliers,
                             staca=self.geom.reporting.get_stacking_angle_outliers,
                             stacd=self.geom.reporting.get_stacking_dist_outliers,
                             vdw=self.geom.reporting.get_vdw_outliers,
                             )
            labs = dict(bond="Bond distances",
                        angle="Bond angles",
                        torsion="Torsion angles",
                        chir="Chiral centres",
                        plane="Planar groups",
                        staca="Stacking plane angles",
                        stacd="Stacking plane distances",
                        vdw="VDW repulsions")

            for k in get_table:
                kwgs = {"min_z": self.outlier_sigmas[k]}
                if k == "bond": kwgs["use_nucleus"] = self.use_nucleus
                table = get_table[k](**kwgs)
                if table["z"]:
                    for kk in table:
                        if kk.startswith(("atom", "plane")):
                            table[kk] = [str(self.lookup[x]) for x in table[kk]]
                    df = pandas.DataFrame(table)
                    df = df.reindex(df.z.abs().sort_values(ascending=False).index)
                    ret["outliers"][k] = df
                    if k == "bond":
                        df0 = df[df.type < 2].drop(columns=["type", "alpha"])
                        if len(df0.index) > 0:
                            logger.writeln(" *** {} outliers (Z >= {}) ***\n".format(labs[k], self.outlier_sigmas[k]))
                            logger.writeln(df0.to_string(float_format="{:.3f}".format, index=False) + "\n")
                        df0 = df[df.type == 2].drop(columns=["type"])
                        if len(df0.index) > 0:
                            logger.writeln(" *** External bond outliers (Z >= {}) ***\n".format(self.outlier_sigmas[k]))
                            logger.writeln(df0.to_string(float_format="{:.3f}".format, index=False) + "\n")
                    else:
                        logger.writeln(" *** {} outliers (Z >= {}) ***\n".format(labs[k], self.outlier_sigmas[k]))
                        logger.writeln(df.to_string(float_format="{:.3f}".format, index=False) + "\n")

        df = pandas.DataFrame(self.geom.reporting.get_summary_table(self.use_nucleus))
        df = df.set_index("Restraint type").rename_axis(index=None)
        ret["summary"] = df
        logger.writeln(df.to_string(float_format="{:.3f}".format) + "\n")
        return ret
        
class Refine:
    def __init__(self, st, geom, ll=None, refine_xyz=True, adp_mode=1, refine_h=False, unrestrained=False):
        assert adp_mode in (0, 1, 2) # 0=fix, 1=iso, 2=aniso
        assert geom is not None
        self.st = st # clone()?
        self.atoms = [None for _ in range(self.st[0].count_atom_sites())]
        for cra in self.st[0].all(): self.atoms[cra.atom.serial-1] = cra.atom
        self.geom = geom
        self.ll = ll
        self.gamma = 0
        self.adp_mode = 0 if self.ll is None else adp_mode
        self.refine_xyz = refine_xyz
        self.unrestrained = unrestrained
        self.refine_h = refine_h
        self.h_inherit_parent_adp = self.adp_mode > 0 and not self.refine_h and self.st[0].has_hydrogen()
        if self.h_inherit_parent_adp:
            self.geom.set_h_parents()
    # __init__()

    def scale_shifts(self, dx, scale):
        n_atoms = len(self.atoms)
        #ave_shift = numpy.mean(dx)
        #max_shift = numpy.maximum(dx)
        #rms_shift = numpy.std(dx)
        shift_allow_high =  0.5
        shift_allow_low  = -0.5
        shift_max_allow_B = 30.0
        shift_min_allow_B = -30.0
        dx = scale * dx
        offset_b = 0
        if self.refine_xyz:
            dxx = dx[:n_atoms*3]
            dxx[dxx > shift_allow_high] = shift_allow_high
            dxx[dxx < shift_allow_low] = shift_allow_low
            offset_b = n_atoms*3
        if self.adp_mode == 1:
            dxb = dx[offset_b:]
            dxb[dxb > shift_max_allow_B] = shift_max_allow_B
            dxb[dxb < shift_min_allow_B] = shift_min_allow_B
        elif self.adp_mode == 2: # FIXME aniso. need to find eigenvalues..
            dxb = dx[offset_b:]
            dxb[dxb > shift_max_allow_B] = shift_max_allow_B
            dxb[dxb < shift_min_allow_B] = shift_min_allow_B
            
        return dx

    def n_params(self):
        n_atoms = len(self.atoms)
        n_params = 0
        if self.refine_xyz: n_params += 3 * n_atoms
        if self.adp_mode == 1:
            n_params += n_atoms
        elif self.adp_mode == 2:
            n_params += 6 * n_atoms
        return n_params

    def set_x(self, x):
        n_atoms = len(self.atoms)
        offset_b = n_atoms * 3 if self.refine_xyz else 0
        for i in range(len(self.atoms)):
            if self.refine_xyz:
                self.atoms[i].pos.fromlist(x[3*i:3*i+3]) # faster than substituting pos.x,pos.y,pos.z
            if self.adp_mode == 1:
                self.atoms[i].b_iso = max(0.5, x[offset_b + i]) # minimum B = 0.5
            elif self.adp_mode == 2:
                a = x[offset_b + 6 * i: offset_b + 6 * (i+1)]
                a = gemmi.SMat33d(*a)
                M = numpy.array(a.as_mat33())
                v, Q = numpy.linalg.eigh(M) # eig() may return complex due to numerical precision?
                v = numpy.maximum(v, 0.5) # avoid NPD with minimum B = 0.5
                M2 = Q.dot(numpy.diag(v)).dot(Q.T)
                self.atoms[i].b_iso = M2.trace() / 3
                M2 *= b_to_u
                self.atoms[i].aniso = gemmi.SMat33f(M2[0,0], M2[1,1], M2[2,2], M2[0,1], M2[0,2], M2[1,2])

        # Copy B of hydrogen from parent
        if self.h_inherit_parent_adp:
            for h in self.geom.parents:
                p = self.geom.parents[h]
                h.b_iso = p.b_iso
                h.aniso = p.aniso

    def get_x(self):
        n_atoms = len(self.atoms)
        offset_b = n_atoms * 3 if self.refine_xyz else 0
        x = numpy.zeros(self.n_params())
        for i, a in enumerate(self.atoms):
            if self.refine_xyz:
                x[3*i:3*(i+1)] = a.pos.tolist()
            if self.adp_mode == 1:
                x[offset_b + i] = self.atoms[i].b_iso
            elif self.adp_mode == 2:
                x[offset_b + 6*i : offset_b + 6*(i+1)] = self.atoms[i].aniso.elements_pdb()
                x[offset_b + 6*i : offset_b + 6*(i+1)] *= u_to_b

        return x
    #@profile
    def calc_target(self, w=1, target_only=False):
        N = self.n_params()
        geom, g_vn, g_am = self.geom.calc_target(target_only,
                                                 not self.unrestrained and self.refine_xyz,
                                                 self.adp_mode, N)
        if self.ll is not None:
            self.ll.update_fc()
            ll = self.ll.calc_target()
            if not target_only:
                l_vn, l_am = self.ll.calc_grad(self.refine_xyz, self.adp_mode, self.refine_h, self.geom.specs)
                diag = l_am.diagonal()
                logger.writeln("diag(data) min= {:3e} max= {:3e}".format(numpy.min(diag),
                                                                         numpy.max(diag)))
        else:
            ll = 0

        f =  w * ll + geom

        if not target_only:
            if self.ll is not None:
                vn = w * l_vn + g_vn
                am = w * l_am + g_am
            else:
                vn = g_vn
                am = g_am
        else:
            vn, am = None, None

        return f, vn, am

    #@profile
    def run_cycle(self, weight=1):
        if self.ll is not None:
            self.ll.update_fc()
            self.ll.overall_scale()
            self.ll.update_ml_params()

        self.geom.geom.setup_nonbonded() # if refine_xyz=False, no need to do it every time
        logger.writeln("vdws = {}".format(len(self.geom.geom.vdws)))
        self.geom.geom.setup_target(self.refine_xyz, self.adp_mode)
            
        if 0: # test of grad
            self.ll.update_fc()
            x0 = self.get_x()
            f0,ader,_ = self.calc_target(weight)
            i = 1
            for e in 1e-1,1e-2,1e-3, 1e-4, 1e-5:
                x1 = numpy.copy(x0)
                x1[i] += e
                self.set_x(x1)
                self.ll.update_fc()
                f1,_,_ = self.calc_target(weight, target_only=True)
                nder = (f1 - f0) / e
                print("e=", e)
                print("NUM DER=", nder)
                print("ANA DER=", ader[i])
                print("ratio=", nder/ader[i])
            quit()

        f0, vn, am = self.calc_target(weight)
        x0 = self.get_x()
        logger.writeln("f0= {:.4e}".format(f0))
        
        if 0:
            assert self.adp_mode == 0 # not supported!
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
            logger.writeln("diagonal min= {:3e} max= {:3e}".format(numpy.min(diag),
                                                                   numpy.max(diag)))
            #for i in numpy.where(diag <= 0)[0]:
            #    if self.refine_xyz:
            #        if i >= len(self.atoms)*3:
            #            j = i - len(self.atoms)*3
            #            print("diag<=0: {} B {}".format(i, self.geom.lookup[self.atoms[j]]))
            #        else:
            #            j = i // 3
            #            print("diag<=0: {} x {}".format(i, self.geom.lookup[self.atoms[j]]))
            diag[diag<=0] = 1.
            diag = numpy.sqrt(diag)
            rdiag = 1./diag # sk
            rdmat = scipy.sparse.diags(rdiag)
            M = rdmat
            
        dx, self.gamma = cgsolve.cgsolve_rm(A=am, v=vn, M=M, gamma=self.gamma)
        if self.refine_xyz:
            dxx = dx[:len(self.atoms)*3]
            #logger.writeln("dx = {}".format(dxx))
            logger.writeln("min(dx) = {}".format(numpy.min(dxx)))
            logger.writeln("max(dx) = {}".format(numpy.max(dxx)))
            logger.writeln("mean(dx)= {}".format(numpy.mean(dxx)))
        if self.adp_mode > 0: # TODO for aniso
            db = dx[len(self.atoms)*3 if self.refine_xyz else 0:]
            #logger.writeln("dB = {}".format(db))
            logger.writeln("min(dB) = {}".format(numpy.min(db)))
            logger.writeln("max(dB) = {}".format(numpy.max(db)))
            logger.writeln("mean(dB)= {}".format(numpy.mean(db)))
            
        if 0: # to check hessian scale
            with open("minimise_line.dat", "w") as ofs:
                ofs.write("s f\n")
                for s in numpy.arange(-2, 2, 0.1):
                    self.set_x(x0 + s * dx)
                    fval = self.calc_target(weight, target_only=True)[0]
                    ofs.write("{} {}\n".format(s, fval))
            quit()

        ret = True # success
        for i in range(3):
            dx2 = self.scale_shifts(dx, 1/2**i)
            self.set_x(x0 - dx2)
            f1, _, _ = self.calc_target(weight, target_only=True)
            logger.writeln("f1, {}= {:.4e}".format(i, f1))
            if f1 < f0: break
        else:
            ret = False
            logger.writeln("WARNING: function not minimised")
            #self.set_x(x0) # Refmac accepts it even when function increases

        return ret
    
    def run_cycles(self, ncycles, weight=1, debug=False):
        stats = [{"Ncyc": 0}]
        if not self.unrestrained:
            stats[-1]["geom"] = self.geom.show_model_stats(show_outliers=True)["summary"]
        if self.ll is not None:
            # redundant calculations that will happen in run_cycle() as well
            self.ll.update_fc()
            self.ll.overall_scale()
            self.ll.update_ml_params()
            stats[-1]["data"] = self.ll.calc_stats()["summary"]
            
        for i in range(ncycles):
            logger.writeln("\n====== CYCLE {:2d} ======\n".format(i+1))
            self.run_cycle(weight=weight) # check ret?
            stats.append({"Ncyc": i+1})
            if debug: utils.fileio.write_model(self.st, "refined_{:02d}".format(i+1), pdb=True)#, cif=True)
            if self.refine_xyz and not self.unrestrained:
                stats[-1]["geom"] = self.geom.show_model_stats(show_outliers=(i==ncycles-1))["summary"]
            if self.adp_mode > 0:
                utils.model.adp_analysis(self.st)
            logger.writeln("")

            if self.ll is not None:
                self.ll.overall_scale()
                stats[-1]["data"] = self.ll.calc_stats()["summary"]

        df = pandas.DataFrame({"Ncyc": range(ncycles+1)})
        if self.ll is not None:
            df["FSCaverage"] = [s["data"].get("FSCaverage", numpy.nan) for s in stats]
            df["R"] = [s["data"].get("R", numpy.nan) for s in stats]
            df["-LL"] = [s["data"].get("-LL", numpy.nan) for s in stats]
        if self.refine_xyz and not self.unrestrained:
            df["rmsBOND"] =[s["geom"]["r.m.s.d."].get("Bond distances, non H") for s in stats]
            df["zBOND"] = [s["geom"]["r.m.s.Z"].get("Bond distances, non H") for s in stats]
            df["rmsANGL"] = [s["geom"]["r.m.s.d."].get("Bond angles, non H") for s in stats]
            df["zANGL"] = [s["geom"]["r.m.s.Z"].get("Bond angles, non H") for s in stats]

        forplot = []
        if self.ll is not None:
            forplot.append(["FSC or R", ["Ncyc", "FSCaverage", "R"]])
            forplot.append(["-LL", ["Ncyc", "-LL"]])
        if self.refine_xyz and not self.unrestrained:
            forplot.extend([["Geometry", ["Ncyc", "rmsBOND", "rmsANGL"]],
                            ["Geometry Z", ["Ncyc", "zBOND", "zANGL"]]])

        lstr = utils.make_loggraph_str(df, "stats vs cycle", forplot,
                                       float_format="{:.4f}".format)
        logger.writeln(lstr)
        self.update_meta()
        return stats

    def update_meta(self):
        # TODO write stats. probably geom.reporting.get_summary_table should return with _refine_ls_restr.type names
        self.st.raw_remarks = []
        si = gemmi.SoftwareItem()
        si.classification = gemmi.SoftwareItem.Classification.Refinement
        si.pdbx_ordinal = 1
        si.name = "Servalcat"
        si.version = servalcat.__version__
        si.date = servalcat.__date__
        self.st.meta.software = [si]

        self.st.meta.refinement = []
        #ri = gemmi.RefinementInfo()
        #rr = gemmi.RefinementInfo.Restr("")
        #ri.restr_stats.append(rr)
        #st.meta.refinement = [ri]
        
# class Refine
