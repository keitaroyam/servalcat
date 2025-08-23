"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import time
import re
import gemmi
import numpy
import json
import pandas
import scipy.sparse
from dataclasses import dataclass, field
from typing import List, Dict
import omegaconf
import servalcat # for version
from servalcat.utils import logger
from servalcat import utils
from servalcat.refmac import exte
from servalcat import ext
from . import cgsolve
u_to_b = utils.model.u_to_b
b_to_u = utils.model.b_to_u
Type = ext.RefineParams.Type
#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

"""
atom_selection:
  xyz:
    include: []
    exclude: []
    exclude_restraint: []
  adp:
    include: []
    exclude: []
    exclude_restraint: []
  occ:
    include: []
    exclude: []
    exclude_restraint: []
  dfrac:
    include: []
    exclude: []

occ_groups:
  - id: 1
    selections:
      - sel_1
      - sel_2
occ_group_constraints:
  - ids: [1, 2]
    complete: true

initialisation:
  adp:
    '*': 50
  occ: {}
  dfrac:
    '[H]': 1.0
"""

@dataclass
class SelectionConfig:
    include: List[str] = field(default_factory=list, metadata={"help": "List of gemmi Selection to include"})
    exclude: List[str] = field(default_factory=list, metadata={"help": "List of gemmi Selection to exclude"})
    exclude_restraint: List[str] = field(default_factory=list, metadata={"help": "List of gemmi Selection to exclude from restraints"})
@dataclass
class OccGroupItem:
    id: int
    selections: List[str] = field(default_factory=list)
    
@dataclass
class OccGroupConstItem:
    ids: List[int] = field(default_factory=list)
    complete: bool = True
    
@dataclass
class RefineConfig:
    atom_selection: Dict[str, SelectionConfig] = field(
        default_factory=lambda: {
            "xyz": SelectionConfig(include=["*"]),
            "adp": SelectionConfig(include=["*"]),
            "occ": SelectionConfig(),
            "dfrac": SelectionConfig(),
        },
        metadata={"help": "Configuration for atom selection during refinement"}
    )
    occ_groups: List[OccGroupItem] = field(default_factory=list)
    occ_group_constraints: List[OccGroupConstItem] = field(default_factory=list)
    occ_group_const_mu: float = 10
    occ_group_const_mu_update_factor: float = 1.1
    occ_group_const_mu_update_tol_rel: float = 0.25
    occ_group_const_mu_update_tol_abs: float = 0.01
    initialisation: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "adp": {},
            "occ": {},
            "dfrac": {}
        },
        metadata={"help": ""}
    )
    write_trajectory: bool = False

def load_config(yaml_file, args, refmac_params):
    cfg = omegaconf.OmegaConf.create({"refine": RefineConfig()})
    if yaml_file:
        conf = omegaconf.OmegaConf.load(yaml_file)
        try:
            cfg = omegaconf.OmegaConf.merge(cfg, conf)
        except omegaconf.errors.ValidationError as e:
            raise SystemExit(f"Error while parsing {yaml_file}.\n{e.msg}")
            
    # load from args
    rcfg = cfg.refine
    if getattr(args, "write_trajectory", False):
        rcfg.write_trajectory = True
    if getattr(args, "fix_xyz", False):
        rcfg.atom_selection.xyz.include = []
        rcfg.atom_selection.xyz.exclude = []
    if getattr(args, "adp", None) == "fix":
        rcfg.atom_selection.adp.include = []
        rcfg.atom_selection.adp.exclude = []
    if getattr(args, "refine_all_occ", False):
        rcfg.atom_selection.occ.include = ["*"]
        rcfg.atom_selection.occ.exclude = []
    if getattr(args, "refine_dfrac", False):
        rcfg.atom_selection.dfrac.include = ["*"] # or H?
        rcfg.atom_selection.dfrac.exclude = []

    # load Refmac params (unfinished)
    if refmac_params.get("occu", {}).get("groups"):
        rgroups = refmac_params["occu"]["groups"]
        rconst = refmac_params["occu"].get("const", [])
        occ_grs = rcfg.occ_groups
        occ_cnst = rcfg.occ_group_constraints
        occ_incl = rcfg.atom_selection.occ.include
        for igr in rgroups:
            occ_grs.append(OccGroupItem(igr))
            for sel in rgroups[igr]:
                for chain in sel.get("chains", ["*"]):
                    if "resi_from" in sel and "resi_to" in sel:
                        resi = f"{sel['resi_from']}-{sel['resi_to']}"
                    else:
                        resi = sel.get("resi", "")
                    atom = sel.get("atom", "*")
                    alt = sel.get("alt", "")
                    if alt: alt = ":" + alt
                    selstr = f"//{chain}/{resi}/{atom}{alt}"
                    occ_grs[-1]["selections"].append(selstr)
                    #occ_incl.append(selstr) # to do this, GroupOccupancy needs to be removed
        for cmpl, ids in rconst:
            occ_cnst.append(OccGroupConstItem(ids, cmpl))
    
    logger.writeln("Config loaded")
    logger.writeln("--")
    logger.write(omegaconf.OmegaConf.to_yaml(cfg))
    logger.writeln("--")
    return cfg.refine
# load_config()

def RefineParams(st, refine_xyz=False, adp_mode=0, refine_occ=False,
                 refine_dfrac=False, use_q_b_mixed=True, cfg=None,
                 exclude_h_ll=True): # FIXME refine_dfrac/exclude_h_ll and cfg
    assert adp_mode in (0, 1, 2) # 0=fix, 1=iso, 2=aniso
    if refine_dfrac and not st[0].has_hydrogen():
        raise RuntimeError("Hydrogen must be present when deuterium fraction refinement is requested")
    ret = ext.RefineParams(use_aniso=(adp_mode == 2), use_q_b_mixed=use_q_b_mixed)
    ret.set_model(st[0])
    if cfg:
        # occupancy groups
        occ_groups = []
        group_ids = {}
        for occ_gr in cfg.occ_groups:
            occ_groups.append([])
            group_ids[occ_gr.id] = len(occ_groups) - 1
            for s in occ_gr.selections:
                sel = gemmi.Selection(s)
                nsel = 0
                for model in sel.models(st):
                    for chain in sel.chains(model):
                        for residue in sel.residues(chain):
                            for atom in sel.atoms(residue):
                                occ_groups[-1].append(atom)
                                nsel += 1
                if nsel == 0:
                    logger.writeln(f"Warning: no atom found for the selection {s}")
        ret.set_occ_groups(occ_groups)
        for o in cfg.occ_group_constraints:
            ret.occ_group_constraints.append((o.complete, [group_ids[x] for x in o.ids]))

        # selections
        sele = cfg.atom_selection
        ext.set_refine_flags(st[0],
                             sele.xyz.include, sele.xyz.exclude,
                             sele.adp.include, sele.adp.exclude,
                             sele.occ.include, sele.occ.exclude,
                             sele.dfrac.include, sele.dfrac.exclude)
        ret.set_params_from_flags()

        for t, p in ((Type.X, sele.xyz), (Type.B, sele.adp), (Type.Q, sele.occ)):
            for ex_sel in p.exclude_restraint:
                sel = gemmi.Selection(ex_sel)
                for model in sel.models(st):
                    for chain in sel.chains(model):
                        for residue in sel.residues(chain):
                            for atom in sel.atoms(residue):
                                ret.add_geom_exclusion(atom.serial-1, t)
    else:
        ret.set_params(refine_xyz=refine_xyz, refine_adp=adp_mode > 0,
                       refine_occ=refine_occ, refine_dfrac=refine_dfrac)
    if exclude_h_ll:
        if refine_dfrac:
            for t in (Type.X, Type.B, Type.Q):
                if ret.is_refined(t):
                    ret.exclude_h_ll(t)
        else:
            ret.exclude_h_ll()

    logger.writeln("Number of refinement parameters:")
    df = pandas.DataFrame(ret.params_summary())
    logger.writeln(df.to_string() + "\n")
    return ret

class Geom:
    def __init__(self, st, topo, monlib, refine_params, adpr_w=1, occr_w=1, shake_rms=0,
                 params=None, unrestrained=False, use_nucleus=False,
                 ncslist=None, atom_pos=None):
        self.st = st
        self.params = refine_params
        self.lookup = {x.atom: x for x in self.st[0].all()}
        try:
            self.geom = ext.Geometry(self.st, self.params, monlib.ener_lib)
        except TypeError as e:
            raise SystemExit(f"An error occurred while creating the Geometry object:\n{e}\n\n"
                             "This likely indicates an installation issue. "
                             "Please verify that you have the correct version of gemmi installed and that both gemmi and servalcat were compiled in the same environment.")
        self.specs = utils.model.find_special_positions(self.st)
        #cs_count = len(self.st.find_spacegroup().operations())
        for atom, images, matp, mata in self.specs:
            #n_sym = len([x for x in images if x < cs_count]) + 1
            n_sym = len(images) + 1
            self.geom.specials.append(ext.Geometry.Special(atom, matp, mata, n_sym))
        self.adpr_w = adpr_w
        self.occr_w = occr_w
        self.unrestrained = unrestrained
        if shake_rms > 0:
            numpy.random.seed(0)
            utils.model.shake_structure(self.st, shake_rms, copy=False)
            #utils.fileio.write_model(self.st, "shaken", pdb=True, cif=True)
        self.use_nucleus = use_nucleus
        self.calc_kwds = {"use_nucleus": self.use_nucleus}
        if params is None:
            params = {}
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
        exte.read_external_restraints(params.get("exte", []), self.st, self.geom)
        self.geom.finalize_restraints()
        self.outlier_sigmas = dict(bond=5, angle=5, torsion=5, vdw=5, ncs=5, chir=5, plane=5, staca=5, stacd=5, per_atom=5, interval=5)
        self.parents = {}
        self.ncslist = ncslist
        self.const_ls, self.const_u = [], []
    # __init__()

    def set_h_parents(self):
        self.parents = {}
        for bond in self.geom.bonds:
            if bond.atoms[0].is_hydrogen():
                self.parents[bond.atoms[0]] = bond.atoms[1]
            elif bond.atoms[1].is_hydrogen():
                self.parents[bond.atoms[1]] = bond.atoms[0]
    # set_h_parents()

    def setup_target(self):
        self.geom.setup_target(self.params.is_refined(Type.Q))
    def setup_nonbonded(self):
        skip_critical_dist = not self.params.is_refined(Type.X) or self.unrestrained
        self.geom.setup_nonbonded(skip_critical_dist=skip_critical_dist)
        if self.ncslist:
            self.geom.setup_ncsr(self.ncslist)
    def setup_occ_constraint(self, lambda_ini=0., u_ini=100.):
        self.const_ls = [lambda_ini for _ in self.params.occ_group_constraints]
        self.const_u = [u_ini for _ in self.params.occ_group_constraints]
    def calc(self, target_only):
        if self.params.is_refined(Type.X) and not self.unrestrained:
            return self.geom.calc(check_only=target_only, **self.calc_kwds)
        return 0
    def calc_adp_restraint(self, target_only):
        if self.params.is_refined(Type.B):
            return self.geom.calc_adp_restraint(target_only, self.adpr_w)
        return 0
    def calc_occ_restraint(self, target_only):
        if self.params.is_refined(Type.Q):
            return self.geom.calc_occ_restraint(target_only, self.occr_w)
        return 0
    def update_occ_consts(self, consts_prev, alpha=1.1, eta=0.25, tol=0.01):
        consts = self.params.occ_constraints()
        self.const_ls = [self.const_ls[i] - self.const_u[i] * consts[i]
                         for i in range(len(consts))]
        self.const_u = [u * (1 if abs(c) < max(tol, eta * abs(c_prev)) else alpha)
                        for u, c, c_prev in zip(self.const_u, consts, consts_prev)]
        return consts
    def calc_target(self, target_only):
        self.geom.clear_target()
        geom_x = self.calc(target_only) 
        geom_a = self.calc_adp_restraint(target_only)
        geom_q = self.calc_occ_restraint(target_only)
        geom_c = self.geom.calc_occ_constraint(target_only, self.const_ls, self.const_u)
        logger.writeln(" geom_x = {}".format(geom_x))
        logger.writeln(" geom_a = {}".format(geom_a))
        logger.writeln(" geom_q = {}".format(geom_q))
        logger.writeln(" geom_c = {}".format(geom_c))
        geom = geom_x + geom_a + geom_q + geom_c
        if not target_only:
            self.geom.spec_correction()
        return geom
        
    def show_model_stats(self, show_outliers=True):
        self.calc(True)
        self.calc_adp_restraint(True)
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
                             interval=self.geom.reporting.get_interval_outliers,
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
                        interval="Interval",
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
        if 0:
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
    datalabs = [x for x in ("Mn(Io)", "Mn(Ic)", "Mn(Fo)", "Mn(Fc)") if x in df]
    rlabs = [x for x in df if x.startswith("R")]
    fsclabs = [x for x in df if x.startswith("fsc")]
    cclabs = [x for x in df if x.startswith("CC")]
    dlabs = [x for x in df if re.search("^D[0-9]*", x)]
    if datalabs: forplot.append(["Mean I/F vs. Resolution", datalabs])
    if "fsc_model" in df: forplot.append(["FSC", ["fsc_model"]])
    if rlabs: forplot.append(["R", rlabs])
    if fsclabs: forplot.append(["FSC", fsclabs])
    if cclabs: forplot.append(["CC", cclabs])
    if dlabs: forplot.append(["ML parameters - D", dlabs])
    if "S" in df: forplot.append(["ML parameters - Sigma", ["S"]])
    if "Cmpl" in df: forplot.append(["Data completeness", ["Cmpl"]])
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
    for i in range(10):
        try:
            # On Windows, this fails when another process open the file
            os.replace(out_tmp, json_out)
            break
        except PermissionError:
            logger.writeln(f"{json_out} locked. retrying..")
            time.sleep(0.5)
    else:
        raise RuntimeError(f"Cannot write {json_out}")
    logger.writeln(f"Refinement statistics saved: {json_out}")
# write_stats_json_safe()

def print_h_options(h_change, h_present, refine_h, hout, geom_only):
    if not h_present:
        h_change = gemmi.HydrogenChange.Remove
    logger.writeln("Hydrogen related options")
    logger.write(" use in refinement{}: hydrogen atoms ".format("" if geom_only else "/map calculation"))
    logger.writeln({gemmi.HydrogenChange.ReAddButWater: "have been (re)generated",
                    gemmi.HydrogenChange.ReAdd:         "(including water) have been (re)generated",
                    gemmi.HydrogenChange.ReAddKnown:    "(except for rotatable) have been (re) generated",
                    gemmi.HydrogenChange.NoChange:      "from the input model have been retained",
                    gemmi.HydrogenChange.Remove:        "have either been removed or were not present"}[h_change])
    if h_present:
        logger.write(" target: hydrogen atoms will be ")
        if geom_only or not refine_h:
            logger.writeln("just optimized according to geometric restraints")
        else:
            logger.writeln("refined against experimental data")
    logger.writeln(" in output model: " + ("written" if hout and h_present else "not written"))
    logger.writeln("")
# print_hydrogen_options()

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
        atom_idxes = []
        for igr in params["groups"]:
            self.groups.append([]) # list of atoms
            n_curr = len(atom_idxes)
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
                            atom_idxes.append(atom.serial-1)
                            self.groups[-1].append(atom)
                            self.group_idxes[atom.serial-1] = len(self.groups)
                        if sel_to and res.seqid == sel_to:
                            flag = False
            logger.writeln(" id= {} atoms= {}".format(igr, len(atom_idxes) - n_curr))

        igr_idxes = {igr:i for i, igr in enumerate(params["groups"])}
        self.consts = [(is_comp, [igr_idxes[g] for g in gids])
                       for is_comp, gids in params["const"]]
        self.ncycle = params.get("ncycle", 5)
        self.params = ext.RefineParams()
        self.params.set_model(st[0])
        self.params.set_params_selected(atom_idxes, refine_occ=True)
        self.params.exclude_h_ll() # should be reasonable
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
        for atoms in self.groups:
            occ = numpy.mean([a.occ for a in atoms])
            occ = min(1, max(1e-3, occ))
            vals.append(occ)
        for is_comp, idxes in self.consts:
            sum_occ = sum(vals[i] for i in idxes)
            if not is_comp and sum_occ < 1:
                sum_occ = 1. # do nothing
            for i in idxes:
                logger.writeln("Imposing constraints: {} {}".format(vals[i], vals[i]/sum_occ))
                vals[i] /= sum_occ
        for occ, atoms in zip(vals, self.groups):
            for a in atoms: a.occ = occ
        
    def get_x(self):
        return numpy.array([atoms[0].occ for atoms in self.groups])

    def set_x(self, x):
        for p, atoms in zip(x, self.groups):
            for a in atoms:
                a.occ = p
                #a.occ = max(1, min(1e-3, p))

    def target(self, x, ll, ls, u):
        self.set_x(x)
        ll.update_fc()
        c = self.constraint(x)
        f = ll.calc_target() - numpy.dot(ls, c) + 0.5 * u * numpy.sum(c**2)
        return f
    
    def grad(self, x, ll, ls, u):
        c = self.constraint(x)
        ll.calc_grad(self.params, specs=None)
        #print("grad=", ll.ll.vn)
        #print("diag=", ll.ll.am)
        assert len(ll.ll.vn) == len(ll.ll.am)
        vn = []
        diag = []
        atom_to_param = self.params.atom_to_param(Type.Q)
        for atoms in self.groups: # idxes
            idxes = [atom_to_param[a.serial-1] for a in atoms if not self.params.is_excluded_ll(a.serial-1, Type.Q)]
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
        
    def refine(self, ll, alpha=1.1):
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
            vn, diag = self.grad(x0, ll, ls, u)
            #diag0 = diag.copy()
            diag[diag < 1e-6] = 1.
            dx = -vn / diag
            #logger.writeln(f"debug {cyc=} {dx=} {vn=} {diag=} {diag0=}")
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
                                                 myfprime= lambda x: self.grad(ll, ls, u)[0],
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
    def __init__(self, st, geom, cfg, refine_params, ll=None, unrestrained=False):
        assert geom is not None
        self.st = st # clone()?
        self.st_traj = None
        self.params = refine_params
        self.geom = geom
        self.ll = ll
        self.gamma = 0
        self.unrestrained = unrestrained
        #self.h_inherit_parent_adp = self.params.is_refined(Type.B) and not self.refine_h and self.st[0].has_hydrogen()
        #if self.h_inherit_parent_adp:
        #    self.geom.set_h_parents()
        self.cfg = cfg
        if self.cfg.write_trajectory:
            self.st_traj = self.st.clone()
            self.st_traj[-1].num = 0
        assert self.geom.group_occ.groups or self.params.n_params() > 0
    # __init__()
    
    def print_weights(self): # TODO unfinished
        logger.writeln("Geometry weights")
        g = self.geom.geom
        if self.params.is_refined(Type.B):
            logger.writeln(" ADP restraints")
            logger.writeln("  weight: {}".format(self.geom.adpr_w))
            logger.writeln("  mode: {}".format(g.adpr_mode))
            if g.adpr_mode == "diff":
                logger.writeln("  sigmas: {}".format(" ".join("{:.2f}".format(x) for x in g.adpr_diff_sigs)))
            elif g.adpr_mode == "kldiv":
                logger.writeln("  sigmas: {}".format(" ".join("{:.2f}".format(x) for x in g.adpr_kl_sigs)))
            else:
                raise LookupError("unknown adpr_mode")
        if self.params.is_refined(Type.Q):
            logger.writeln(" Occupancy restraints")
            logger.writeln("  weight: {}".format(self.geom.occr_w))

    def scale_shifts(self, dx, scale):
        shift_allow_high =  1.0
        shift_allow_low  = -1.0
        shift_max_allow_B = 30.0
        shift_min_allow_B = -30.0
        shift_max_allow_q = 0.5
        shift_min_allow_q = -0.5
        shift_max_allow_d = 0.5
        shift_min_allow_d = -0.5
        dx = scale * dx
        dxx = dx[self.params.vec_selection(Type.X)]
        dxb = dx[self.params.vec_selection(Type.B)]
        dxq = dx[self.params.vec_selection(Type.Q)]
        dxd = dx[self.params.vec_selection(Type.D)]
        if len(dxx) > 0:
            logger.writeln("min(dx) = {}".format(numpy.min(dxx)))
            logger.writeln("max(dx) = {}".format(numpy.max(dxx)))
            logger.writeln("mean(dx)= {}".format(numpy.mean(dxx)))
            dxx[dxx > shift_allow_high] = shift_allow_high
            dxx[dxx < shift_allow_low] = shift_allow_low
        if len(dxb) > 0:
            # TODO this is misleading in anisotropic case
            logger.writeln("min(dB) = {}".format(numpy.min(dxb)))
            logger.writeln("max(dB) = {}".format(numpy.max(dxb)))
            logger.writeln("mean(dB)= {}".format(numpy.mean(dxb)))
            # FIXME we should'nt apply eigen decomp to dxb
            if self.params.aniso:
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
            else:
                dxb[dxb > shift_max_allow_B] = shift_max_allow_B
                dxb[dxb < shift_min_allow_B] = shift_min_allow_B
        if len(dxq) > 0:
            logger.writeln("min(dq) = {}".format(numpy.min(dxq)))
            logger.writeln("max(dq) = {}".format(numpy.max(dxq)))
            logger.writeln("mean(dq)= {}".format(numpy.mean(dxq)))
            dxq[dxq > shift_max_allow_q] = shift_max_allow_q
            dxq[dxq < shift_min_allow_q] = shift_min_allow_q
        if len(dxd) > 0:
            logger.writeln("min(dd) = {}".format(numpy.min(dxd)))
            logger.writeln("max(dd) = {}".format(numpy.max(dxd)))
            logger.writeln("mean(dd)= {}".format(numpy.mean(dxd)))
            dxd[dxd > shift_max_allow_d] = shift_max_allow_d
            dxd[dxd < shift_min_allow_d] = shift_min_allow_d

        return dx

    def set_x(self, x):
        self.params.set_x(x, min_b=0.5)
        max_occ = {}
        if self.params.is_refined(Type.Q) and self.geom.specs:
            max_occ = {atom: 1./(len(images)+1) for atom, images, _, _ in self.geom.specs}
        for a in self.params.atoms:
            a.occ = min(max_occ.get(a, 1), max(0, a.occ))
        # Copy B of hydrogen from parent
        #if self.h_inherit_parent_adp:
        #    for h in self.geom.parents:
        #        p = self.geom.parents[h]
        #        h.b_iso = p.b_iso
        #        h.aniso = p.aniso

        if self.ll is not None:
            self.ll.update_fc()
        
        self.geom.setup_nonbonded() # if refine_xyz=False, no need to do it every time
        self.geom.setup_target()
        logger.writeln("vdws = {}".format(len(self.geom.geom.vdws)))
        logger.writeln(f"atoms = {len(self.params.atoms)}")
        logger.writeln(f"pairs = {self.geom.geom.target.n_pairs()}")

    def get_x(self):
        return numpy.array(self.params.get_x())
    
    #@profile
    def calc_target(self, w=1, target_only=False):
        geom = self.geom.calc_target(target_only)
        if self.ll is not None:
            ll = self.ll.calc_target()
            logger.writeln(" ll= {}".format(ll))
            if not target_only:
                self.ll.calc_grad(self.params, self.geom.geom.specials)
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
        if 0:
            logger.writeln(f"geom_vec=\n{self.geom.geom.target.vn}")
            logger.writeln(f"geom_mat=\n{self.geom.geom.target.am_spmat}")
            logger.writeln(f"  ll_vec=\n{self.ll.ll.vn}")
            logger.writeln(f"  ll_mat=\n{self.ll.ll.fisher_spmat}")
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
                    fval = self.calc_target(weight, target_only=True)
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
        self.geom.setup_nonbonded()
        self.geom.setup_target()
        self.geom.setup_occ_constraint(u_ini=self.cfg.occ_group_const_mu)
        logger.writeln("vdws = {}".format(len(self.geom.geom.vdws)))
        logger.writeln(f"atoms = {len(self.params.atoms)}")
        logger.writeln(f"pairs = {self.geom.geom.target.n_pairs()}")
        stats[-1]["geom"] = self.geom.show_model_stats(show_outliers=True)
        if self.params.occ_group_constraints:
            stats[-1]["occ_const"] = {"lambda": self.geom.const_ls,
                                      "mu": self.geom.const_u,
                                      "violation": self.params.occ_constraints(),
                                      "occ": self.params.constrained_occ_values()
                                      }
        if self.ll is not None:
            self.ll.update_fc()
            self.ll.overall_scale()
            self.ll.update_ml_params()
            self.ll.prepare_target()
            llstats = self.ll.calc_stats(bin_stats=True)
            stats[-1]["data"] = {"summary": llstats["summary"],
                                 "binned": llstats["bin_stats"].to_dict(orient="records"),
                                 "ml": llstats["ml"].to_dict(orient="records")}
            if "twin_alpha" in llstats:
                stats[-1]["twin_alpha"] = llstats["twin_alpha"]
            show_binstats(llstats["bin_stats"], 0)
        if self.params.is_refined(Type.B):
            utils.model.adp_analysis(self.st)
        if stats_json_out:
            write_stats_json_safe(stats, stats_json_out)
        occ_refine_flag = self.ll is not None and self.geom.group_occ.groups and self.geom.group_occ.ncycle > 0

        for i in range(ncycles):
            logger.writeln("\n====== CYCLE {:2d} ======\n".format(i+1))
            logger.writeln(f" weight = {weight:.4e}")
            if self.params.is_refined_any():
                is_ok, shift_scale, fval = self.run_cycle(weight=weight)
                stats.append({"Ncyc": len(stats), "shift_scale": shift_scale, "fval": fval, "fval_decreased": is_ok,
                              "weight": weight})
            elif occ_refine_flag:
                stats.append({"Ncyc": len(stats)})
            if occ_refine_flag:
                stats[-1]["occ_refine"] = self.geom.group_occ.refine(self.ll)
            if debug: utils.fileio.write_model(self.st, "refined_{:02d}".format(i+1), pdb=True)#, cif=True)
            stats[-1]["geom"] = self.geom.show_model_stats(show_outliers=(i==ncycles-1))
            if self.params.occ_group_constraints:
                viols = self.geom.update_occ_consts(consts_prev=stats[-2]["occ_const"]["violation"],
                                                    alpha=self.cfg.occ_group_const_mu_update_factor,
                                                    eta=self.cfg.occ_group_const_mu_update_tol_rel,
                                                    tol=self.cfg.occ_group_const_mu_update_tol_abs)
                stats[-1]["occ_const"] = {"lambda": self.geom.const_ls,
                                          "mu": self.geom.const_u,
                                          "violation": viols,
                                          "occ": self.params.constrained_occ_values()
                                          }
            # TODO add stats[-1]["occ_constraints"] and hide stdout
            if self.ll is not None:
                if i == ncycles - 1: # last cycle
                    self.params.ensure_occ_constraints()
                self.ll.overall_scale()
                f0 = self.ll.calc_target()
                self.ll.update_ml_params()
                self.ll.prepare_target()
                llstats = self.ll.calc_stats(bin_stats=True)#(i==ncycles-1))
                if llstats["summary"]["-LL"] > f0:
                    logger.writeln("WARNING: -LL has increased after ML parameter optimization:"
                                   "{} to {}".format(f0, llstats["summary"]["-LL"]))
                stats[-1]["data"] = {"summary": llstats["summary"],
                                     "binned": llstats["bin_stats"].to_dict(orient="records"),
                                     "ml": llstats["ml"].to_dict(orient="records")}
                if "twin_alpha" in llstats:
                    stats[-1]["twin_alpha"] = llstats["twin_alpha"]
                show_binstats(llstats["bin_stats"], i+1)
            if self.params.is_refined(Type.B):
                utils.model.adp_analysis(self.st)
            if (weight_adjust and self.params.is_refined(Type.X) and not self.unrestrained and self.ll is not None and
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
                self.st_traj[-1].num = len(self.st_traj) - 1
            if stats_json_out:
                write_stats_json_safe(stats, stats_json_out)

            logger.writeln("")

        # Make tables
        if self.params.occ_group_constraints:
            tmp = []
            for icyc, s in enumerate(stats):
                con = s["occ_const"]
                d = {"Ncyc": icyc}
                d.update({f"lambda_{i+1}":l for i,l in enumerate(con["lambda"])})
                d.update({f"mu_{i+1}":l for i,l in enumerate(con["mu"])})
                d.update({f"violation_{i+1}":l for i,l in enumerate(con["violation"])})
                d.update({f"occ_{i+1}_{j+1}":q for i, l in enumerate(con["occ"])
                          for j, q in enumerate(l)})
                tmp.append(d)
            df = pandas.DataFrame(tmp)
            forplot = [
                ["Lagrange multiplier", 
                 ["Ncyc"] + [x for x in df if x.startswith("lambda_")]],
                ["Penalty parameter mu",
                 ["Ncyc"] + [x for x in df if x.startswith("mu")]],
                ["Group constrained occupancies",
                 ["Ncyc"] + [x for x in df if x.startswith("occ_")]],
                ["Constraint violations",
                 ["Ncyc"] + [x for x in df if x.startswith("violation_")]],
            ]
            lstr = utils.make_loggraph_str(df, "group occupancies vs cycle", forplot,
                                           float_format="{:.4f}".format)
            logger.writeln(lstr)
            
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
        return stats

# class Refine

def update_meta(st, stats, ll=None):
    # TODO write stats. probably geom.reporting.get_summary_table should return with _refine_ls_restr.type names
    # should remove st.mod_residues?
    st.helices.clear()
    st.sheets.clear()
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
    st.meta.software = [si]

    ri = gemmi.RefinementInfo()
    if "geom" in stats:
        restr_stats = []
        raw_remarks.append("REMARK   3  RMS DEVIATIONS FROM IDEAL VALUES        COUNT    RMS    WEIGHT")
        for k, n, l, pl in (("r.m.s.d.", "Bond distances, non H", "s_bond_nonh_d",             "BOND LENGTHS REFINED ATOMS        (A)"),
                            ("r.m.s.d.", "Bond angles, non H", "s_angle_nonh_deg",             "BOND ANGLES REFINED ATOMS   (DEGREES)"),
                            ("r.m.s.d.", "Torsion angles, period 1", "s_dihedral_angle_1_deg", "TORSION ANGLES, PERIOD 1    (DEGREES)"),
                            ("r.m.s.d.", "Torsion angles, period 2", "s_dihedral_angle_2_deg", "TORSION ANGLES, PERIOD 2    (DEGREES)"),
                            ("r.m.s.d.", "Torsion angles, period 3", "s_dihedral_angle_3_deg", "TORSION ANGLES, PERIOD 3    (DEGREES)"),
                            ("r.m.s.d.", "Torsion angles, period 6", "s_dihedral_angle_6_deg", "TORSION ANGLES, PERIOD 6    (DEGREES)"),
                            ("r.m.s.d.", "Chiral centres", "s_chiral_restr",                   "CHIRAL-CENTER RESTRAINTS       (A**3)"),
                            ("r.m.s.d.", "Planar groups", "s_planes",                          "GENERAL PLANES REFINED ATOMS      (A)"),
                            ("r.m.s.d.", "VDW nonbonded", "s_nbd",                             ""),
                            ("r.m.s.d.", "VDW torsion", "s_nbtor",                             ""),
                            ("r.m.s.d.", "VDW hbond", "s_hbond_nbd",                           ""),
                            ("r.m.s.d.", "VDW metal", "s_metal_ion",                           ""),
                            ("r.m.s.d.", "VDW dummy", "s_dummy_nbd",                           ""),
                            ("r.m.s.d.", "VDW nonbonded, symmetry", "s_symmetry_nbd",          ""),
                            ("r.m.s.d.", "VDW torsion, symmetry", "s_symmetry_nbtor",          ""),
                            ("r.m.s.d.", "VDW hbond, symmetry", "s_symmetry_hbond_nbd",        ""),
                            ("r.m.s.d.", "VDW metal, symmetry", "s_symmetry_metal_ion",        ""),
                            ("r.m.s.d.", "VDW dummy, symmetry", "s_symmetry_dummy_nbd",        "")):
            if k in stats["geom"]["summary"] and n in stats["geom"]["summary"][k]:
                rr = gemmi.RefinementInfo.Restr(l)
                rr.dev_ideal = round(stats["geom"]["summary"][k].get(n), 4)
                rr.count = stats["geom"]["summary"]["N restraints"].get(n)
                rr.weight = round(stats["geom"]["summary"]["Mn(sigma)"].get(n), 4)
                restr_stats.append(rr)
                if pl:
                    raw_remarks.append(f"REMARK   3   {pl}:{rr.count:6d} ;{rr.dev_ideal:6.3f} ;{rr.weight:6.3f}")
        ri.restr_stats = restr_stats
        raw_remarks.append("REMARK   3")
    if ll is not None:
        ri.id = ll.refine_id()
        ri.mean_b = round(numpy.mean([cra.atom.b_iso for cra in st[0].all()]), 2)
        if ll.b_aniso is not None:
            ri.aniso_b = ll.b_aniso
        for k, kd, nd in (("Rwork", "r_work", 4), ("Rfree", "r_free", 4), ("R", "r_all", 4),
                          ("FSCaverage", "fsc_work", 4),
                          ("FSCaverage_half1", "fsc_work", 4), ("FSCaverage_half2", "fsc_free", 4)):
            if k in stats["data"]["summary"]:
                setattr(ri, kd, round(stats["data"]["summary"][k], nd))
        bins = []
        n_all = 0
        for b in stats["data"]["binned"]:
            bri = gemmi.BasicRefinementInfo()
            bri.resolution_high = round(b["d_min"], 3)
            bri.resolution_low = round(b["d_max"], 3)
            for k, kd, nd in (("Rwork", "r_work", 4), ("Rfree", "r_free", 4),
                              ("R1work", "r_work", 4), ("R1free", "r_free", 4),
                              ("R", "r_all", 4), ("R1", "r_all", 4),
                              ("CCI", "cc_intensity_work", 4), ("CCF", "cc_fo_fc_work", 4),
                              ("CCIwork", "cc_intensity_work", 4), ("CCIfree", "cc_intensity_free", 4),
                              ("CCFwork", "cc_fo_fc_work", 4), ("CCFfree", "cc_fo_fc_free", 4),
                              ("fsc_FC_full", "fsc_work", 4), ("fsc_model", "fsc_work", 4),
                              ("fsc_model_half1", "fsc_work", 4), ("fsc_model_half2", "fsc_free", 4),
                              ("n_work", "work_set_count", 0), ("n_free", "rfree_set_count", 0),
                              ("n_obs", "reflection_count", 0), ("ncoeffs", "reflection_count", 0)):
                if k in b: setattr(bri, kd, round(b[k], nd))
            if "n_all" in b and "n_obs" in b:
                bri.completeness = round(b["n_obs"] / b["n_all"] * 100, 2)
                n_all += b["n_all"]
            bins.append(bri)
        ri.rfree_set_count = max(-1, sum(b.rfree_set_count for b in bins))
        ri.work_set_count = max(-1, sum(b.work_set_count for b in bins))
        ri.reflection_count = max(-1, sum(b.reflection_count for b in bins))
        ri.resolution_high = round(min(b.resolution_high for b in bins), 3)
        ri.resolution_low = round(max(b.resolution_low for b in bins), 3)
        if ri.reflection_count > 0 and n_all > 0:
            ri.completeness = round(ri.reflection_count / n_all * 100, 2)
        ri.bins = bins
        if ri.rfree_set_count > 0:
            ri.cross_validation_method = "THROUGHOUT"
    st.meta.refinement = [ri]
    st.raw_remarks = raw_remarks
# update_meta()
