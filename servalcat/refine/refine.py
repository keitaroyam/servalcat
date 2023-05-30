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
from servalcat import ext
from . import cgsolve
u_to_b = utils.model.u_to_b
b_to_u = utils.model.b_to_u

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

class Geom:
    def __init__(self, st, topo, monlib, sigma_b=10, shake_rms=0,
                 refmac_keywords=None, unrestrained=False, use_nucleus=False):
        self.st = st
        self.atoms = [None for _ in range(self.st[0].count_atom_sites())]
        for cra in self.st[0].all(): self.atoms[cra.atom.serial-1] = cra.atom
        self.lookup = {x.atom: x for x in self.st[0].all()}
        self.geom = ext.Geometry(self.st, monlib.ener_lib)
        self.specs = utils.model.find_special_positions(self.st)
        #cs_count = len(self.st.find_spacegroup().operations())
        for atom, images, matp, mata in self.specs:
            #n_sym = len([x for x in images if x < cs_count]) + 1
            n_sym = len(images) + 1
            self.geom.specials.append(ext.Geometry.Special(atom, matp, mata, n_sym))
        self.sigma_b = sigma_b
        self.unrestrained = unrestrained
        if shake_rms > 0:
            numpy.random.seed(0)
            utils.model.shake_structure(self.st, shake_rms, copy=False)
            utils.fileio.write_model(self.st, "shaken", pdb=True, cif=True)
        if not self.unrestrained:
            self.geom.load_topo(topo)
            self.check_chemtypes(os.path.join(monlib.path(), "ener_lib.cif"), topo)
        self.use_nucleus = use_nucleus
        self.calc_kwds = {"use_nucleus": self.use_nucleus}
        if refmac_keywords:
            exte.read_external_restraints(refmac_keywords, self.st, self.geom)
            kwds = parse_keywords(refmac_keywords)
            for k in ("wbond", "wangle", "wtors", "wplane", "wchir", "wvdw"):
                if k in kwds:
                    self.calc_kwds[k] = kwds[k]
                    logger.writeln("setting geometry weight {}= {}".format(k, kwds[k]))
        self.geom.finalize_restraints()
        self.outlier_sigmas = dict(bond=5, angle=5, torsion=5, vdw=5, chir=5, plane=5, staca=5, stacd=5, per_atom=5)
        self.parents = {}
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
    def setup_nonbonded(self, refine_xyz):
        skip_critical_dist = not refine_xyz or self.unrestrained
        self.geom.setup_nonbonded(skip_critical_dist=skip_critical_dist)
    def calc(self, target_only):
        return self.geom.calc(check_only=target_only, **self.calc_kwds)
    def calc_adp_restraint(self, target_only):
        return self.geom.calc_adp_restraint(target_only, self.sigma_b)
    def calc_target(self, target_only, refine_xyz, adp_mode):
        self.geom.clear_target()
        geom_x = self.calc(target_only) if refine_xyz else 0
        geom_a = self.calc_adp_restraint(target_only) if adp_mode > 0 else 0
        logger.writeln(" geom_x = {}".format(geom_x))
        logger.writeln(" geom_a = {}".format(geom_a))
        geom = geom_x + geom_a
        if not target_only:
            self.geom.spec_correction()
        return geom
        
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

        # Per-atom score
        peratom = self.geom.reporting.per_atom_score(len(self.atoms), self.use_nucleus, "mean")
        df = pandas.DataFrame(peratom)
        df.insert(0, "atom", [str(self.lookup[x]) for x in self.atoms])
        df = df[df["total"] >= self.outlier_sigmas["per_atom"]]
        if show_outliers and len(df.index) > 0:
            df.sort_values("total", ascending=False, inplace=True)
            ret["outliers"]["per_atom"] = df
            logger.writeln(" *** Per-atom violations (Z >= {}) ***\n".format(self.outlier_sigmas["per_atom"]))
            logger.writeln(df.to_string(float_format="{:.2f}".format, index=False) + "\n")

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
        self.atoms = geom.atoms # not a copy
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
        elif self.adp_mode == 2:
            dxb = dx[offset_b:]
            for i in range(len(dxb)//6):
                j = i * 6
                a = numpy.array([[dxb[j],   dxb[j+3], dxb[j+4]],
                                 [dxb[j+3], dxb[j+1], dxb[j+5]],
                                 [dxb[j+4], dxb[j+5], dxb[j+2]]])
                v, Q = numpy.linalg.eigh(a)
                v[v > shift_max_allow_B] = shift_max_allow_B
                v[v < shift_min_allow_B] = shift_min_allow_B
                a = Q.dot(numpy.diag(v)).dot(Q.T)
                dxb[j:j+6] = a[0,0], a[1,1], a[2,2], a[0,1], a[0,2], a[1,2]
            
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

        if self.ll is not None:
            self.ll.update_fc()
        
        self.geom.setup_nonbonded(self.refine_xyz) # if refine_xyz=False, no need to do it every time
        logger.writeln("vdws = {}".format(len(self.geom.geom.vdws)))

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
        geom = self.geom.calc_target(target_only,
                                     not self.unrestrained and self.refine_xyz,
                                     self.adp_mode)
        if self.ll is not None:
            ll = self.ll.calc_target()
            if not target_only:
                self.ll.calc_grad(self.refine_xyz, self.adp_mode, self.refine_h, self.geom.geom.specials)
        else:
            ll = 0

        f =  w * ll + geom
        return f

    #@profile
    def run_cycle(self, weight=1):
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

        f0 = self.calc_target(weight)
        x0 = self.get_x()
        logger.writeln("f0= {:.4e}".format(f0))
        if 1:
            use_ic = False # incomplete cholesky. problematic at least in geometry optimisation case
            logger.writeln("using cgsolve in c++, ic={}".format(use_ic))
            cgsolver = ext.CgSolve(self.geom.geom.target, None if self.ll is None else self.ll.ll)
            if use_ic:
                cgsolver.gamma = 0
                cgsolver.max_gamma_cyc = 1
            else:
                cgsolver.gamma = self.gamma
            dx = cgsolver.solve(weight, logger, use_ic)
            self.gamma = cgsolver.gamma
        else:
            logger.writeln("using cgsolve in py")
            am = self.geom.geom.target.am_spmat
            vn = numpy.array(self.geom.geom.target.vn)
            if self.ll is not None:
                am += self.ll.ll.fisher_spmat * weight
                vn += numpy.array(self.ll.ll.vn) * weight
            diag = am.diagonal()
            diag[diag<=0] = 1.
            diag = numpy.sqrt(diag)
            rdiag = 1./diag # sk
            M = scipy.sparse.diags(rdiag)
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
            f1 = self.calc_target(weight, target_only=True)
            logger.writeln("f1, {}= {:.4e}".format(i, f1))
            if f1 < f0: break
        else:
            ret = False
            logger.writeln("WARNING: function not minimised")
            #self.set_x(x0) # Refmac accepts it even when function increases

        return ret
    
    def run_cycles(self, ncycles, weight=1, debug=False):
        stats = [{"Ncyc": 0}]
        self.geom.setup_nonbonded(self.refine_xyz)
        logger.writeln("vdws = {}".format(len(self.geom.geom.vdws)))
        if self.refine_xyz and not self.unrestrained:
            stats[-1]["geom"] = self.geom.show_model_stats(show_outliers=True)["summary"]
        if self.ll is not None:
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
            if self.ll is not None:
                self.ll.overall_scale()
                self.ll.update_ml_params()
                llstats = self.ll.calc_stats(bin_stats=True)#(i==ncycles-1))
                stats[-1]["data"] = llstats["summary"]
                if "bin_stats" in llstats:
                    df = llstats["bin_stats"]
                    forplot = []
                    rlabs = [x for x in df if x.startswith("R")]
                    cclabs = [x for x in df if x.startswith("CC")]
                    if "fsc_model" in df: forplot.append(["FSC", ["1/resol^2", "fsc_model"]])
                    if rlabs: forplot.append(["R", ["1/resol^2"] + rlabs])
                    if cclabs: forplot.append(["CC", ["1/resol^2"] + cclabs])
                    lstr = utils.make_loggraph_str(df, "Data stats in cycle {}".format(i+1), forplot,
                                                   float_format="{:.4f}".format)
                    logger.writeln(lstr)
            if self.adp_mode > 0:
                utils.model.adp_analysis(self.st)
            logger.writeln("")

        # Make table
        data_keys, geom_keys = set(), set()
        tmp = []
        for d in stats:
            x = {"Ncyc": d["Ncyc"]}
            if "data" in d:
                x.update(d["data"])
                data_keys.update(d["data"])
            if "geom" in d:
                for k, n, l in (("r.m.s.d.", "Bond distances, non H", "rmsBOND"),
                                ("r.m.s.Z", "Bond distances, non H", "zBOND"),
                                ("r.m.s.d.", "Bond angles, non H", "rmsANGL"),
                                ("r.m.s.Z", "Bond angles, non H", "zANGL")):
                    if k in d["geom"] and n in d["geom"][k]:
                        x[l] = d["geom"][k].get(n)
                        geom_keys.add(l)
            tmp.append(x)
        df = pandas.DataFrame(tmp)
        forplot = []
        if "FSCaverage" in data_keys:
            forplot.append(["FSC", ["Ncyc", "FSCaverage"]])
        r_keys = [x for x in data_keys if x.startswith("R")]
        if r_keys:
            forplot.append(["R", ["Ncyc"] + r_keys])
        cc_keys = [x for x in data_keys if x.startswith("CC")]
        if cc_keys:
            forplot.append(["CC", ["Ncyc"] + cc_keys])
        if "-LL" in data_keys:
            forplot.append(["-LL", ["Ncyc", "-LL"]])
        rms_keys = [x for x in geom_keys if x.startswith("rms")]
        if rms_keys:
            forplot.append(["Geometry", ["Ncyc"] + rms_keys])
        z_keys = [x for x in geom_keys if x.startswith("z")]
        if z_keys:
            forplot.append(["Geometry Z", ["Ncyc"] + z_keys])

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
