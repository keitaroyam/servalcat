"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import re
import gemmi
import numpy
import json
import pandas
import scipy.sparse
import servalcat # for version
from servalcat.utils import logger
from servalcat import utils
from servalcat.refmac import exte
from servalcat import ext
from . import cgsolve
u_to_b = utils.model.u_to_b
b_to_u = utils.model.b_to_u

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

class Geom:
    def __init__(self, st, topo, monlib, adpr_w=1, shake_rms=0,
                 params=None, unrestrained=False, use_nucleus=False,
                 ncslist=None, atom_pos=None):
        self.st = st
        self.atoms = [None for _ in range(self.st[0].count_atom_sites())]
        for cra in self.st[0].all(): self.atoms[cra.atom.serial-1] = cra.atom
        if atom_pos is not None:
            self.atom_pos = atom_pos
        else:
            self.atom_pos = list(range(len(self.atoms)))
        self.n_refine_atoms = max(self.atom_pos) + 1
        self.lookup = {x.atom: x for x in self.st[0].all()}
        self.geom = ext.Geometry(self.st, self.atom_pos, monlib.ener_lib)
        self.specs = utils.model.find_special_positions(self.st)
        #cs_count = len(self.st.find_spacegroup().operations())
        for atom, images, matp, mata in self.specs:
            #n_sym = len([x for x in images if x < cs_count]) + 1
            n_sym = len(images) + 1
            self.geom.specials.append(ext.Geometry.Special(atom, matp, mata, n_sym))
        self.adpr_w = adpr_w
        self.occr_w = 1.
        self.unrestrained = unrestrained
        if shake_rms > 0:
            numpy.random.seed(0)
            utils.model.shake_structure(self.st, shake_rms, copy=False)
            #utils.fileio.write_model(self.st, "shaken", pdb=True, cif=True)
        self.use_nucleus = use_nucleus
        self.calc_kwds = {"use_nucleus": self.use_nucleus}
        if params is None:
            params = {}
        exte.read_external_restraints(params.get("exte", []), self.st, self.geom)
        for k in ("wbond", "wangle", "wtors", "wplane", "wchir", "wvdw", "wncs"):
            if k in params:
                self.calc_kwds[k] = params[k]
                logger.writeln("setting geometry weight {}= {}".format(k, params[k]))
        inc_tors, exc_tors = utils.restraints.make_torsion_rules(params.get("restr", {}))
        rtors = utils.restraints.select_restrained_torsions(monlib, inc_tors, exc_tors)
        self.geom.mon_tors_names = rtors["monomer"]
        self.geom.link_tors_names = rtors["link"]
        self.group_occ = GroupOccupancy(self.st, params.get("occu"))
        if not self.unrestrained:
            self.geom.load_topo(topo)
            self.check_chemtypes(os.path.join(monlib.path(), "ener_lib.cif"), topo)
        self.geom.finalize_restraints()
        self.outlier_sigmas = dict(bond=5, angle=5, torsion=5, vdw=5, ncs=5, chir=5, plane=5, staca=5, stacd=5, per_atom=5)
        self.parents = {}
        self.ncslist = ncslist
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
        self.geom.setup_nonbonded(skip_critical_dist=skip_critical_dist, group_idxes=self.group_occ.group_idxes)
        if self.ncslist:
            self.geom.setup_ncsr(self.ncslist)
    def calc(self, target_only):
        return self.geom.calc(check_only=target_only, **self.calc_kwds)
    def calc_adp_restraint(self, target_only):
        return self.geom.calc_adp_restraint(target_only, self.adpr_w)
    def calc_occ_restraint(self, target_only):
        return self.geom.calc_occ_restraint(target_only, self.occr_w)
    def calc_target(self, target_only, refine_xyz, adp_mode, use_occr):
        self.geom.clear_target()
        geom_x = self.calc(target_only) if refine_xyz else 0
        geom_a = self.calc_adp_restraint(target_only) if adp_mode > 0 else 0
        geom_q = self.calc_occ_restraint(target_only) if use_occr > 0 else 0
        logger.writeln(" geom_x = {}".format(geom_x))
        logger.writeln(" geom_a = {}".format(geom_a))
        logger.writeln(" geom_q = {}".format(geom_q))
        geom = geom_x + geom_a + geom_q
        if not target_only:
            self.geom.spec_correction()
        return geom
        
    def show_model_stats(self, refine_xyz=True, adp_mode=1, use_occr=False, show_outliers=True):
        if refine_xyz:
            self.calc(True)
        if adp_mode > 0:
            self.calc_adp_restraint(True)
        if use_occr:
            self.calc_occ_restraint(True)
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
                             #ncs=self.geom.reporting.get_ncsr_outliers, # not useful?
                             )
            labs = dict(bond="Bond distances",
                        angle="Bond angles",
                        torsion="Torsion angles",
                        chir="Chiral centres",
                        plane="Planar groups",
                        staca="Stacking plane angles",
                        stacd="Stacking plane distances",
                        vdw="VDW repulsions",
                        ncs="Local NCS restraints")

            for k in get_table:
                kwgs = {"min_z": self.outlier_sigmas[k]}
                if k == "bond": kwgs["use_nucleus"] = self.use_nucleus
                table = get_table[k](**kwgs)
                if table["z"]:
                    for kk in table:
                        if kk.startswith(("atom", "plane", "1_atom", "2_atom")):
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

def show_binstats(df, cycle_number):
    forplot = []
    rlabs = [x for x in df if x.startswith("R")]
    fsclabs = [x for x in df if x.startswith("fsc")]
    cclabs = [x for x in df if x.startswith("CC")]
    dlabs = [x for x in df if re.search("^D[0-9]*", x)]
    if "fsc_model" in df: forplot.append(["FSC", ["fsc_model"]])
    if rlabs: forplot.append(["R", rlabs])
    if fsclabs: forplot.append(["FSC", fsclabs])
    if cclabs: forplot.append(["CC", cclabs])
    if dlabs: forplot.append(["ML parameters - D", dlabs])
    if "S" in df: forplot.append(["ML parameters - Sigma", ["S"]])
    lstr = utils.make_loggraph_str(df, "Data stats in cycle {}".format(cycle_number), forplot,
                                   s2=1/df["d_min"]**2,
                                   float_format="{:.4f}".format)
    logger.writeln(lstr)
# show_binstats()

def convert_stats_to_dicts(stats):
    tmp = []
    for s in stats: # stats must be a list of dict
        tmp.append({})
        for k in s:
            if k == "geom":
                tmp[-1]["geom"] = {"summary": s["geom"]["summary"].to_dict()}
                for kk in s["geom"]["outliers"]:
                    tmp[-1]["geom"].setdefault("outliers", {})[kk] = s["geom"]["outliers"][kk].to_dict(orient="records")
            else:
                tmp[-1][k] = s[k]
    return tmp
# convert_stats_to_dicts()

def write_stats_json_safe(stats, json_out):
    tmp = convert_stats_to_dicts(stats)
    out_tmp = json_out + ".part"
    with open(out_tmp, "w") as ofs:
        json.dump(tmp, ofs, indent=2)
    os.replace(out_tmp, json_out)
    logger.writeln(f"Refinement statistics saved: {json_out}")
# write_stats_json_safe()

class GroupOccupancy:
    # TODO max may not be one. should check multiplicity
    def __init__(self, st, params):
        self.groups = []
        self.consts = []
        self.group_idxes = [0 for _ in range(st[0].count_atom_sites())]
        self.ncycle = 0
        if not params or not params.get("groups"):
            return
        logger.writeln("Occupancy groups:")
        self.atom_pos = [-1 for _ in range(st[0].count_atom_sites())]
        count = 0
        for igr in params["groups"]:
            self.groups.append([[], []]) # list of [indexes, atoms]
            n_curr = count
            for sel in params["groups"][igr]:
                sel_chains = sel.get("chains")
                sel_from = sel.get("resi_from")
                sel_to = sel.get("resi_to")
                sel_seq = sel.get("resi")
                sel_atom = sel.get("atom")
                sel_alt = sel.get("alt")
                for chain in st[0]:
                    if sel_chains and chain.name not in sel_chains:
                        continue
                    flag = False
                    for res in chain:
                        if sel_seq and res.seqid != sel_seq:
                            continue
                        if sel_from and res.seqid == sel_from:
                            flag = True
                        if sel_from and not flag:
                            continue
                        for atom in res:
                            if sel_atom and atom.name != sel_atom:
                                continue
                            if sel_alt and atom.altloc != sel_alt:
                                continue
                            self.atom_pos[atom.serial-1] = count
                            self.groups[-1][0].append(count)
                            self.groups[-1][1].append(atom)
                            self.group_idxes[atom.serial-1] = len(self.groups)
                            count += 1
                        if sel_to and res.seqid == sel_to:
                            flag = False
            logger.writeln(" id= {} atoms= {}".format(igr, count - n_curr))

        igr_idxes = {igr:i for i, igr in enumerate(params["groups"])}
        self.consts = [(is_comp, [igr_idxes[g] for g in gids])
                       for is_comp, gids in params["const"]]
        self.ncycle = params.get("ncycle", 5)
    # __init__()
    
    def constraint(self, x):
        # x: occupancy parameters
        ret = []
        for is_comp, ids in self.consts:
            x_sum = numpy.sum(x[ids])
            if is_comp or x_sum > 1:
                ret.append(x_sum - 1)
            else:
                ret.append(0.)
        return numpy.array(ret)
        
    def ensure_constraints(self):
        vals = []
        for _, atoms in self.groups:
            occ = numpy.mean([a.occ for a in atoms])
            vals.append(occ)
        for is_comp, idxes in self.consts:
            sum_occ = sum(vals[i] for i in idxes)
            if not is_comp and sum_occ < 1:
                sum_occ = 1. # do nothing
            for i in idxes:
                #logger.writeln("Imposing constraints: {} {}".format(vals[i], vals[i]/sum_occ))
                vals[i] /= sum_occ
        for occ, (_, atoms) in zip(vals, self.groups):
            for a in atoms: a.occ = occ
        
    def get_x(self):
        return numpy.array([atoms[0].occ for _, atoms in self.groups])

    def set_x(self, x):
        for p, (_, atoms) in zip(x, self.groups):
            for a in atoms:
                a.occ = p

    def target(self, x, ll, ls, u):
        self.set_x(x)
        ll.update_fc()
        c = self.constraint(x)
        f = ll.calc_target() - numpy.dot(ls, c) + 0.5 * u * numpy.sum(c**2)
        return f
    
    def grad(self, x, ll, ls, u, refine_h):
        c = self.constraint(x)
        ll.calc_grad(self.atom_pos, refine_xyz=False, adp_mode=0, refine_occ=True, refine_h=refine_h, specs=None)
        #print("grad=", ll.ll.vn)
        #print("diag=", ll.ll.am)
        assert len(ll.ll.vn) == len(ll.ll.am)
        vn = []
        diag = []
        for idxes, atoms in self.groups:
            if not refine_h:
                idxes = [i for i, a in zip(idxes, atoms) if not a.is_hydrogen()]
            vn.append(numpy.sum(numpy.array(ll.ll.vn)[idxes]))
            diag.append(numpy.sum(numpy.array(ll.ll.am)[idxes]))
        vn, diag = numpy.array(vn), numpy.array(diag)
        for i, (is_comp, idxes) in enumerate(self.consts):
            dcdx = numpy.zeros(len(self.groups))
            dcdx[idxes] = 1.
            if is_comp or c[i] != 0:
                vn -= (ls[i] - u * c[i]) * dcdx
            diag += u * dcdx**2

        return vn, diag
        
    def refine(self, ll, refine_h, alpha=1.1):
        # Refinement of grouped occupancies using augmented Lagrangian
        # f(x) = LL(x) - sum_j (lambda_j c_j(x)) + u/2 sum_j (c_j(x))^2
        # with c_j(x) = 0 constraints
        if not self.groups:
            return
        logger.writeln("\n== Group occupancy refinement ==")
        self.ensure_constraints() # make sure constrained groups have the same occupancies.
        ls = 0 * numpy.ones(len(self.consts)) # Lagrange multiplier
        u = 10000. # penalty parameter. in Refmac 1/0.01**2
        x0 = self.get_x()
        #logger.writeln("  parameters: {}".format(len(x0)))
        f0 = self.target(x0, ll, ls, u)
        ret = []
        for cyc in range(self.ncycle):
            ret.append({"Ncyc": cyc+1, "f0": f0})
            logger.writeln("occ_{}_f0= {:.4e}".format(cyc, f0))
            vn, diag = self.grad(x0, ll, ls, u, refine_h)
            diag[diag < 1e-6] = 1.
            dx = -vn / diag
            if 0:
                ofs = open("debug.dat", "w")
                for scale in (-1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2):
                    self.set_x(x0 + scale * dx)
                    ll.update_fc()
                    c = self.constraint(x0 + dx)
                    f = ll.calc_target() + numpy.dot(ls, c) + 0.5 * u * numpy.sum(c**2)
                    ofs.write("{} {}\n".format(scale, f))
                ofs.close()
                import scipy.optimize
                print(scipy.optimize.line_search(f=lambda x: self.target(x, ll, ls, u),
                                                 myfprime= lambda x: self.grad(ll, ls, u, refine_h)[0],
                                                 xk= x0,
                                                 pk= dx))
                quit()

            scale = 1
            for i in range(3):
                scale = 1/2**i
                f1 = self.target(x0 + dx * scale, ll, ls, u)
                logger.writeln("occ_{}_f1, {}= {:.4e}".format(cyc, i, f1))
                if f1 < f0: break
            else:
                logger.writeln("WARNING: function not minimised")
                #self.set_x(x0) # Refmac accepts it even when function increases
            c = self.constraint(x0 + dx * scale)
            ret[-1]["f1"] = f1
            ret[-1]["shift_scale"] = scale
            f0 = f1
            x0 = x0 + dx * scale
            ls -= u * c
            u = alpha * u
            ret[-1]["const_viol"] = list(c)
            ret[-1]["lambda_new"] = list(ls)
        self.ensure_constraints()
        ll.update_fc()
        f = ll.calc_target()
        logger.writeln("final -LL= {}".format(f))
        return ret


class Refine:
    def __init__(self, st, geom, ll=None, refine_xyz=True, adp_mode=1, refine_h=False, refine_occ=False,
                 unrestrained=False, params=None):
        assert adp_mode in (0, 1, 2) # 0=fix, 1=iso, 2=aniso
        assert geom is not None
        self.st = st # clone()?
        self.st_traj = None
        self.atoms = geom.atoms # not a copy
        self.geom = geom
        self.ll = ll
        self.gamma = 0
        self.adp_mode = 0 if self.ll is None else adp_mode
        self.refine_xyz = refine_xyz
        self.refine_occ = refine_occ
        self.use_occr = self.refine_occ # for now?
        self.unrestrained = unrestrained
        self.refine_h = refine_h
        self.h_inherit_parent_adp = self.adp_mode > 0 and not self.refine_h and self.st[0].has_hydrogen()
        if self.h_inherit_parent_adp:
            self.geom.set_h_parents()
        if params and params.get("write_trajectory"):
            self.st_traj = self.st.clone()
            self.st_traj[-1].name = "0"
        assert self.geom.group_occ.groups or self.n_params() > 0
    # __init__()
    
    def print_weights(self): # TODO unfinished
        logger.writeln("Geometry weights")
        g = self.geom.geom
        if self.adp_mode > 0:
            logger.writeln(" ADP restraints")
            logger.writeln("  weight: {}".format(self.geom.adpr_w))
            logger.writeln("  mode: {}".format(g.adpr_mode))
            if g.adpr_mode == "diff":
                logger.writeln("  sigmas: {}".format(" ".join("{:.2f}".format(x) for x in g.adpr_diff_sigs)))
            elif g.adpr_mode == "kldiv":
                logger.writeln("  sigmas: {}".format(" ".join("{:.2f}".format(x) for x in g.adpr_kl_sigs)))
            else:
                raise LookupError("unknown adpr_mode")

    def scale_shifts(self, dx, scale):
        n_atoms = self.geom.n_refine_atoms
        #ave_shift = numpy.mean(dx)
        #max_shift = numpy.maximum(dx)
        #rms_shift = numpy.std(dx)
        shift_allow_high =  1.0
        shift_allow_low  = -1.0
        shift_max_allow_B = 30.0
        shift_min_allow_B = -30.0
        shift_max_allow_q = 0.5
        shift_min_allow_q = -0.5
        dx = scale * dx
        offset_b = n_atoms * 3 if self.refine_xyz else 0
        offset_q = offset_b + n_atoms * {0: 0, 1: 1, 2: 6}[self.adp_mode]
        if self.refine_xyz:
            dxx = dx[:offset_b]
            logger.writeln("min(dx) = {}".format(numpy.min(dxx)))
            logger.writeln("max(dx) = {}".format(numpy.max(dxx)))
            logger.writeln("mean(dx)= {}".format(numpy.mean(dxx)))
            dxx[dxx > shift_allow_high] = shift_allow_high
            dxx[dxx < shift_allow_low] = shift_allow_low
        if self.adp_mode == 1:
            dxb = dx[offset_b:offset_q]
            logger.writeln("min(dB) = {}".format(numpy.min(dxb)))
            logger.writeln("max(dB) = {}".format(numpy.max(dxb)))
            logger.writeln("mean(dB)= {}".format(numpy.mean(dxb)))
            dxb[dxb > shift_max_allow_B] = shift_max_allow_B
            dxb[dxb < shift_min_allow_B] = shift_min_allow_B
        elif self.adp_mode == 2:
            dxb = dx[offset_b:offset_q]
            # TODO this is misleading
            logger.writeln("min(dB) = {}".format(numpy.min(dxb)))
            logger.writeln("max(dB) = {}".format(numpy.max(dxb)))
            logger.writeln("mean(dB)= {}".format(numpy.mean(dxb)))
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
        if self.refine_occ:
            dxq = dx[offset_q:]
            logger.writeln("min(dq) = {}".format(numpy.min(dxq)))
            logger.writeln("max(dq) = {}".format(numpy.max(dxq)))
            logger.writeln("mean(dq)= {}".format(numpy.mean(dxq)))
            dxq[dxq > shift_max_allow_q] = shift_max_allow_q
            dxq[dxq < shift_min_allow_q] = shift_min_allow_q

        return dx

    def n_params(self):
        n_atoms = self.geom.n_refine_atoms
        n_params = 0
        if self.refine_xyz: n_params += 3 * n_atoms
        if self.adp_mode == 1:
            n_params += n_atoms
        elif self.adp_mode == 2:
            n_params += 6 * n_atoms
        if self.refine_occ:
            n_params += n_atoms
        return n_params

    def set_x(self, x):
        n_atoms = self.geom.n_refine_atoms
        offset_b = n_atoms * 3 if self.refine_xyz else 0
        offset_q = offset_b + n_atoms * {0: 0, 1: 1, 2: 6}[self.adp_mode]
        max_occ = {}
        if self.refine_occ and self.geom.specs:
            max_occ = {atom: 1./(len(images)+1) for atom, images, _, _ in self.geom.specs}
        for i, j in enumerate(self.geom.atom_pos):
            if j < 0: continue
            if self.refine_xyz:
                self.atoms[i].pos.fromlist(x[3*j:3*j+3]) # faster than substituting pos.x,pos.y,pos.z
            if self.adp_mode == 1:
                self.atoms[i].b_iso = max(0.5, x[offset_b + j]) # minimum B = 0.5
            elif self.adp_mode == 2:
                a = x[offset_b + 6 * j: offset_b + 6 * (j+1)]
                a = gemmi.SMat33d(*a)
                M = numpy.array(a.as_mat33())
                v, Q = numpy.linalg.eigh(M) # eig() may return complex due to numerical precision?
                v = numpy.maximum(v, 0.5) # avoid NPD with minimum B = 0.5
                M2 = Q.dot(numpy.diag(v)).dot(Q.T)
                self.atoms[i].b_iso = M2.trace() / 3
                M2 *= b_to_u
                self.atoms[i].aniso = gemmi.SMat33f(M2[0,0], M2[1,1], M2[2,2], M2[0,1], M2[0,2], M2[1,2])
            if self.refine_occ:
                self.atoms[i].occ = min(max_occ.get(self.atoms[i], 1), max(1e-3, x[offset_q + j]))

        # Copy B of hydrogen from parent
        if self.h_inherit_parent_adp:
            for h in self.geom.parents:
                p = self.geom.parents[h]
                h.b_iso = p.b_iso
                h.aniso = p.aniso

        if self.ll is not None:
            self.ll.update_fc()
        
        self.geom.setup_nonbonded(self.refine_xyz) # if refine_xyz=False, no need to do it every time
        self.geom.geom.setup_target(self.refine_xyz, self.adp_mode, self.refine_occ, self.use_occr)
        logger.writeln("vdws = {}".format(len(self.geom.geom.vdws)))

    def get_x(self):
        n_atoms = self.geom.n_refine_atoms
        offset_b = n_atoms * 3 if self.refine_xyz else 0
        offset_q = offset_b + n_atoms * {0: 0, 1: 1, 2: 6}[self.adp_mode]
        x = numpy.zeros(self.n_params())
        for i, j in enumerate(self.geom.atom_pos):
            if j < 0: continue
            a = self.atoms[i]
            if self.refine_xyz:
                x[3*j:3*(j+1)] = a.pos.tolist()
            if self.adp_mode == 1:
                x[offset_b + j] = self.atoms[i].b_iso
            elif self.adp_mode == 2:
                x[offset_b + 6*j : offset_b + 6*(j+1)] = self.atoms[i].aniso.elements_pdb()
                x[offset_b + 6*j : offset_b + 6*(j+1)] *= u_to_b
            if self.refine_occ:
                x[offset_q + j] = a.occ

        return x
    #@profile
    def calc_target(self, w=1, target_only=False):
        N = self.n_params()
        geom = self.geom.calc_target(target_only,
                                     not self.unrestrained and self.refine_xyz,
                                     self.adp_mode, self.use_occr)
        if self.ll is not None:
            ll = self.ll.calc_target()
            logger.writeln(" ll= {}".format(ll))
            if not target_only:
                self.ll.calc_grad(self.geom.atom_pos, self.refine_xyz, self.adp_mode, self.refine_occ,
                                  self.refine_h, self.geom.geom.specials)
        else:
            ll = 0

        f =  w * ll + geom
        return f

    #@profile
    def run_cycle(self, weight=1):
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

        if 0: # to check hessian scale
            with open("minimise_line.dat", "w") as ofs:
                ofs.write("s f\n")
                for s in numpy.arange(-2, 2, 0.1):
                    dx2 = self.scale_shifts(dx, s)
                    self.set_x(x0 + dx2)
                    fval = self.calc_target(weight, target_only=True)[0]
                    ofs.write("{} {}\n".format(s, fval))
            quit()

        ret = True # success
        shift_scale = 1
        for i in range(3):
            shift_scale = 1/2**i
            dx2 = self.scale_shifts(dx, shift_scale)
            self.set_x(x0 - dx2)
            f1 = self.calc_target(weight, target_only=True)
            logger.writeln("f1, {}= {:.4e}".format(i, f1))
            if f1 < f0: break
        else:
            ret = False
            logger.writeln("WARNING: function not minimised")
            #self.set_x(x0) # Refmac accepts it even when function increases

        return ret, shift_scale, f1

    def run_cycles(self, ncycles, weight=1, weight_adjust=False, debug=False,
                   weight_adjust_bond_rmsz_range=(0.5, 1.), stats_json_out=None):
        self.print_weights()
        stats = [{"Ncyc": 0}]
        self.geom.setup_nonbonded(self.refine_xyz)
        self.geom.geom.setup_target(self.refine_xyz, self.adp_mode, self.refine_occ, self.use_occr)
        logger.writeln("vdws = {}".format(len(self.geom.geom.vdws)))
        stats[-1]["geom"] = self.geom.show_model_stats(refine_xyz=self.refine_xyz and not self.unrestrained,
                                                       adp_mode=self.adp_mode,
                                                       use_occr=self.refine_occ,
                                                       show_outliers=True)
        if self.ll is not None:
            self.ll.update_fc()
            self.ll.overall_scale()
            self.ll.update_ml_params()
            llstats = self.ll.calc_stats(bin_stats=True)
            stats[-1]["data"] = {"summary": llstats["summary"],
                                 "binned": llstats["bin_stats"].to_dict(orient="records")}
            show_binstats(llstats["bin_stats"], 0)
        if self.adp_mode > 0:
            utils.model.adp_analysis(self.st)
        if stats_json_out:
            write_stats_json_safe(stats, stats_json_out)
        occ_refine_flag = self.ll is not None and self.geom.group_occ.groups and self.geom.group_occ.ncycle > 0

        for i in range(ncycles):
            logger.writeln("\n====== CYCLE {:2d} ======\n".format(i+1))
            logger.writeln(f" weight = {weight:.4e}")
            if self.refine_xyz or self.adp_mode > 0 or self.refine_occ:
                is_ok, shift_scale, fval = self.run_cycle(weight=weight)
                stats.append({"Ncyc": len(stats), "shift_scale": shift_scale, "fval": fval, "fval_decreased": is_ok,
                              "weight": weight})
            elif occ_refine_flag:
                stats.append({"Ncyc": len(stats)})
            if occ_refine_flag:
                stats[-1]["occ_refine"] = self.geom.group_occ.refine(self.ll, self.refine_h)
            if debug: utils.fileio.write_model(self.st, "refined_{:02d}".format(i+1), pdb=True)#, cif=True)
            stats[-1]["geom"] = self.geom.show_model_stats(refine_xyz=self.refine_xyz and not self.unrestrained,
                                                           adp_mode=self.adp_mode,
                                                           use_occr=self.refine_occ,
                                                           show_outliers=(i==ncycles-1))
            if self.ll is not None:
                self.ll.overall_scale()
                f0 = self.ll.calc_target()
                self.ll.update_ml_params()
                llstats = self.ll.calc_stats(bin_stats=True)#(i==ncycles-1))
                if llstats["summary"]["-LL"] > f0:
                    logger.writeln("WARNING: -LL has increased after ML parameter optimization:"
                                   "{} to {}".format(f0, llstats["summary"]["-LL"]))
                stats[-1]["data"] = {"summary": llstats["summary"],
                                     "binned": llstats["bin_stats"].to_dict(orient="records")}
                show_binstats(llstats["bin_stats"], i+1)
            if self.adp_mode > 0:
                utils.model.adp_analysis(self.st)
            if (weight_adjust and self.refine_xyz and not self.unrestrained and self.ll is not None and
                len(stats) > 2 and "Bond distances, non H" in stats[-1]["geom"]["summary"].index):
                rmsz = stats[-1]["geom"]["summary"]["r.m.s.Z"]["Bond distances, non H"]
                rmsz0 = stats[-2]["geom"]["summary"]["r.m.s.Z"]["Bond distances, non H"]
                if rmsz > weight_adjust_bond_rmsz_range[1] and rmsz > rmsz0:
                    weight /= 1.1
                elif rmsz < weight_adjust_bond_rmsz_range[0] and rmsz0 < weight_adjust_bond_rmsz_range[0] and rmsz < rmsz0:
                    weight *= 1.3
                elif rmsz > 1.5 * rmsz0:
                    weight /= 1.1
            if self.st_traj is not None:
                self.st_traj.add_model(self.st[0])
                self.st_traj[-1].name = str(i+1)
            if stats_json_out:
                write_stats_json_safe(stats, stats_json_out)

            logger.writeln("")

        # Make table
        data_keys, geom_keys = set(), set()
        tmp = []
        for d in stats:
            x = {"Ncyc": d["Ncyc"]}
            if "data" in d and "summary" in d["data"]:
                x.update(d["data"]["summary"])
                data_keys.update(d["data"]["summary"])
            if "geom" in d:
                for k, n, l in (("r.m.s.d.", "Bond distances, non H", "rmsBOND"),
                                ("r.m.s.Z", "Bond distances, non H", "zBOND"),
                                ("r.m.s.d.", "Bond angles, non H", "rmsANGL"),
                                ("r.m.s.Z", "Bond angles, non H", "zANGL")):
                    if k in d["geom"]["summary"] and n in d["geom"]["summary"][k]:
                        x[l] = d["geom"]["summary"][k].get(n)
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
        self.update_meta(stats[-1])
        return stats

    def update_meta(self, stats):
        # TODO write stats. probably geom.reporting.get_summary_table should return with _refine_ls_restr.type names
        # should remove st.mod_residues?
        self.st.helices.clear()
        self.st.sheets.clear()
        raw_remarks = [f'REMARK   3',
                       f'REMARK   3 REFINEMENT.',
                       f'REMARK   3   PROGRAM     : SERVALCAT {servalcat.__version__}',
                       f'REMARK   3   AUTHORS     : YAMASHITA,MURSHUDOV',
                       f'REMARK   3',
                       ]
        si = gemmi.SoftwareItem()
        si.classification = gemmi.SoftwareItem.Classification.Refinement
        si.name = "Servalcat"
        si.version = servalcat.__version__
        si.date = servalcat.__date__
        self.st.meta.software = [si]

        ri = gemmi.RefinementInfo()
        if "geom" in stats:
            restr_stats = []
            raw_remarks.append("REMARK   3  RMS DEVIATIONS FROM IDEAL VALUES        COUNT    RMS    WEIGHT")
            for k, n, l, pl in (("r.m.s.d.", "Bond distances, non H", "s_bond_nonh_d", "BOND LENGTHS REFINED ATOMS        (A)"),
                                ("r.m.s.d.", "Bond angles, non H", "s_angle_nonh_d", "BOND ANGLES REFINED ATOMS   (DEGREES)")):
                if k in stats["geom"]["summary"] and n in stats["geom"]["summary"][k]:
                    rr = gemmi.RefinementInfo.Restr(l)
                    rr.dev_ideal = stats["geom"]["summary"][k].get(n)
                    rr.count = stats["geom"]["summary"]["N restraints"].get(n)
                    rr.weight = stats["geom"]["summary"]["Mn(sigma)"].get(n)
                    restr_stats.append(rr)
                    raw_remarks.append(f"REMARK   3   {pl}:{rr.count:6d} ;{rr.dev_ideal:6.3f} ;{rr.weight:6.3f}")
            ri.restr_stats = restr_stats
            raw_remarks.append("REMARK   3")
        self.st.meta.refinement = [ri]
        self.st.raw_remarks = raw_remarks

# class Refine
