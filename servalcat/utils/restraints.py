"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
import os
import io
import re
import gemmi
import numpy
import pandas
import string
import random

default_proton_scale = 1.13 # scale of X-proton distance to X-H(e) distance

def decide_new_mod_id(mod_id, mods):
    # Refmac only allows up to 8 characters
    letters = string.digits + string.ascii_lowercase
    if len(mod_id) < 8:
        for l in letters:
            new_id = "{}{}{}".format(mod_id, "" if len(mod_id)==7 else "-", l)
            if new_id not in mods:
                return new_id

    # give up keeping original name
    while True: # XXX risk of infinite loop.. less likely though
        new_id = "mod" + "".join([random.choice(letters) for _ in range(4)])
        if new_id not in mods:
            return new_id
# decide_new_mod_id()

def rename_cif_modification_if_necessary(doc, known_ids):
    # FIXME Problematic if other file refers to modification that is renamed in this function - but how can we know?
    trans = {}
    for b in doc:
        for row in b.find("_chem_mod.", ["id"]):
            mod_id = row.str(0)
            if mod_id in known_ids:
                new_id = decide_new_mod_id(mod_id, known_ids)
                trans[mod_id] = new_id
                row[0] = new_id # modify id
                logger.writeln("INFO:: renaming modification id {} to {}".format(mod_id, new_id))

    # modify ids in mod_* blocks
    for mod_id in trans:
        b = doc.find_block("mod_{}".format(mod_id))
        if not b: # should raise error?
            logger.writeln("WARNING:: inconsistent mod description for {} in {}".format(mod_id, f))
            continue
        b.name = "mod_{}".format(trans[mod_id]) # modify name
        for item in b:
            for tag in item.loop.tags:
                if tag.endswith(".mod_id"):
                    for row in b.find(tag[:tag.rindex(".")+1], ["mod_id"]):
                        row[0] = trans[mod_id]

    # Update mod id in links
    if trans:
        for b in doc:
            for row in b.find("_chem_link.", ["mod_id_1", "mod_id_2"]):
                for i in range(2):
                    if row.str(i) in trans:
                        row[i] = trans[row.str(i)]

    return trans
# rename_cif_modification_if_necessary()

def load_monomer_library(st, monomer_dir=None, cif_files=None, stop_for_unknowns=False, check_hydrogen=False,
                         ignore_monomer_dir=False):
    resnames = st[0].get_all_residue_names()

    if monomer_dir is None and not ignore_monomer_dir:
        if "CLIBD_MON" not in os.environ:
            logger.error("WARNING: CLIBD_MON is not set")
        else:
            monomer_dir = os.environ["CLIBD_MON"]

    if cif_files is None:
        cif_files = []
        
    if monomer_dir and not ignore_monomer_dir:
        if not os.path.isdir(monomer_dir):
            logger.error("ERROR: not a directory: {}".format(monomer_dir))
            return

        logger.writeln("Reading monomers from {}".format(monomer_dir))
        monlib = gemmi.read_monomer_lib(monomer_dir, resnames, ignore_missing=True)
    else:
        monlib = gemmi.MonLib()

    for f in cif_files:
        logger.writeln("Reading monomer: {}".format(f))
        doc = gemmi.cif.read(f)
        for b in doc:
            if b.find_values("_chem_comp_atom.atom_id"):
                name = b.name.replace("comp_", "")
                if name in monlib.monomers:
                    logger.writeln("WARNING:: updating monomer {} using {}".format(name, f))
                    del monlib.monomers[name]

                # Check if bond length values are included
                # This is to fail if cif file is e.g. from PDB website
                if not b.find_values("_chem_comp_bond.value_dist"):
                    raise RuntimeError("{} does not contain bond length value for {}. You need to generate restraints (e.g. using acedrg).".format(f, name))
                    
            for row in b.find("_chem_link.", ["id"]):
                link_id = row.str(0)
                if link_id in monlib.links:
                    logger.writeln("WARNING:: updating link {} using {}".format(link_id, f))
                    del monlib.links[link_id]

        # If modification id is duplicated, need to rename
        rename_cif_modification_if_necessary(doc, monlib.modifications)
        
        monlib.insert_chemcomps(doc)
        monlib.insert_chemlinks(doc)
        monlib.insert_chemmods(doc)
        
        for b in doc:
            for row in b.find("_chem_comp.", ["id", "group"]):
                if row.str(0) in monlib.monomers:
                    monlib.monomers[row.str(0)].set_group(row.str(1))

    not_loaded = set(resnames).difference(monlib.monomers)
    if not_loaded:
        logger.writeln("WARNING: monomers not loaded: {}".format(" ".join(not_loaded)))
        
    logger.writeln("Monomer library loaded: {} monomers, {} links, {} modifications".format(len(monlib.monomers),
                                                                                          len(monlib.links),
                                                                                          len(monlib.modifications)))
    logger.writeln("       Monomers: {}".format(" ".join([x for x in monlib.monomers])))
    logger.writeln("          Links: {}".format(" ".join([x for x in monlib.links])))
    logger.writeln("  Modifications: {}".format(" ".join([x for x in monlib.modifications])))
    logger.writeln("")

    logger.writeln("Checking if unknown atoms exist..")

    unknown_cc = set()
    for chain in st[0]: unknown_cc.update(res.name for res in chain if res.name not in monlib.monomers)
    if unknown_cc:
        if stop_for_unknowns:
            raise RuntimeError("Provide restraint cif file(s) for {}".format(",".join(unknown_cc)))
        else:
            logger.writeln("WARNING: ad-hoc restraints will be generated for {}".format(",".join(unknown_cc)))
            logger.writeln("         it is strongly recommended to generate them using AceDRG.")
    
    st = st.clone()
    sio = io.StringIO()
    topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.NoChange, warnings=sio, reorder=True,
                                  ignore_unknown_links=True)

    # possible warnings:
    # Warning: no atom X expected in XXX
    # and others?
    unknown_atoms_cc = set()
    link_related = set()
    for l in sio.getvalue().splitlines():
        r = re.search("Warning: definition not found for [^/]*/([^/ ]*) [^/]*/([^\./]*)", l) # chain/resn seqid/atom.alt ; ignore alt
        if r:
            unk = r.group(2), r.group(1)

            cc = monlib.monomers[unk[1]] if unk[1] in monlib.monomers else None
            cc_atom = [x for x in cc.atoms if x.id == unk[0]] if cc else None
            if cc and cc_atom: # if atom is found in chemcomp, it must be an atom that should be removed.
                logger.writeln(l + " - this atom should have been removed when linking")
                if check_hydrogen or not cc_atom[0].is_hydrogen():
                    link_related.add(unk[1])
            elif unk not in unknown_atoms_cc:
                logger.writeln(l)
                unknown_atoms_cc.add(unk)
            continue

        # something else
        logger.writeln(l)

    if not check_hydrogen:
        todel = []
        for unk in unknown_atoms_cc:
            elements = [cra.atom.element for cra in st[0].all() if cra.residue.name==unk[1] and cra.atom.name==unk[0]]
            if elements and elements[0].is_hydrogen:
                todel.append(unk)
        unknown_atoms_cc = unknown_atoms_cc - set(todel)
        
    unknown_cc = set([cc for at, cc in unknown_atoms_cc])
        
    if stop_for_unknowns and (unknown_cc or link_related):
        msgs = []
        if unknown_cc: msgs.append("restraint cif file(s) for {}".format(",".join(unknown_cc)))
        if link_related: msgs.append("proper link cif file(s) for {} or check your model".format(",".join(link_related)))
        raise RuntimeError("Provide {}".format(" and ".join(msgs)))
    
    return monlib
# load_monomer_library()

def check_monlib_support_nucleus_distances(monlib, resnames):
    good = True
    nucl_not_found = []
    for resn in resnames:
        if resn not in monlib.monomers:
            logger.error("ERROR: monomer information of {} not loaded".format(resn))
            good = False
        else:
            mon = monlib.monomers[resn]
            no_nuc = False
            for bond in mon.rt.bonds:
                is_h = (mon.get_atom(bond.id1.atom).is_hydrogen(), mon.get_atom(bond.id2.atom).is_hydrogen())
                if any(is_h) and bond.value_nucleus != bond.value_nucleus:
                    no_nuc = True
                    break
            if no_nuc:
                nucl_not_found.append(resn)
                good = False

    if nucl_not_found:
        logger.writeln("WARNING: nucleus distance is not found for: {}".format(" ".join(nucl_not_found)))
        logger.writeln("         default scale ({}) is used for nucleus distances.".format(default_proton_scale))
    return good
# check_monlib_support_nucleus_distances()

def load_ener_lib(path=None):
    if path is None:
        if "CLIBD_MON" not in os.environ:
            logger.error("ERROR: CLIBD_MON is not set")
            return {}
        else:
            path = os.path.join(os.environ["CLIBD_MON"], "ener_lib.cif")

    if not os.path.exists(path):
        logger.error("ERROR: ener_lib file not found: {}".format(path))
        return {}

    enelib = gemmi.cif.read(path).sole_block()
    """
loop_
_lib_atom.type
_lib_atom.weight
_lib_atom.hb_type
_lib_atom.vdw_radius
_lib_atom.vdwh_radius
_lib_atom.ion_radius
_lib_atom.element
_lib_atom.valency
_lib_atom.sp
    """
    loop = enelib.find_loop("_lib_atom.type").get_loop()
    tags = [x.replace("_lib_atom.","") for x in loop.tags]
    ret = {}
    for i in range(loop.length()):
        key = loop.val(i, 0) # type
        #ret[key] = {}
        d = {}
        for j in range(1, loop.width()):
            val = loop.val(i, j)
            if tags[j] in ("weight", "vdw_radius", "vdwh_radius", "ion_radius"):
                d[tags[j]] = float(val) if val != "." else None
            elif tags[j] in ("valency", "sp"):
                d[tags[j]] = int(val) if val != "." else None
            else:
                d[tags[j]] = val

        ret[key] = d
    return ret
# load_ener_lib()

def find_and_fix_links(st, monlib, bond_margin=1.1, remove_unknown=False, add_found=True):
    """
    Find links not registered in st.connections
    This is required for correctly recognizing link in gemmi.prepare_topology
    if remove_unknown=True, undefined links and unmatched links are removed.
    """
    from servalcat.utils import model

    logger.writeln("Checking links in model")
    hunt = gemmi.LinkHunt()
    hunt.index_chem_links(monlib)
    matches = hunt.find_possible_links(st, bond_margin, 0)
    known_links = ("TRANS", "PTRANS", "NMTRANS", "CIS", "PCIS", "NMCIS", "p", "gap")
    conns = [x for x in st.connections] # to check later
    new_connections = []

    for m in matches:
        if m.conn:
            logger.writeln(" Link confirmed: {} atom1= {} atom2= {} dist= {:.2f} ideal= {:.2f}".format(m.chem_link.id,
                                                                                                    m.cra1, m.cra2,
                                                                                                    m.bond_length,
                                                                                                    m.chem_link.rt.bonds[0].value))
            if not m.cra1.atom_matches(m.conn.partner1): # need to swap
                assert m.cra1.atom_matches(m.conn.partner2)
                m.conn.partner1 = model.cra_to_atomaddress(m.cra1)
                m.conn.partner2 = model.cra_to_atomaddress(m.cra2)

            m.conn.link_id = m.chem_link.id
            if m.conn in conns: # may not be found if id duplicated
                conns.pop(conns.index(m.conn))
        elif add_found:
            # Known link is only accepted when in LINK record
            if not m.chem_link or m.chem_link.id in known_links:
                continue

            logger.writeln(" Link detected:  {} atom1= {} atom2= {} dist= {:.2f} ideal= {:.2f}".format(m.chem_link.id,
                                                                                                    m.cra1, m.cra2,
                                                                                                    m.bond_length,
                                                                                                    m.chem_link.rt.bonds[0].value))
            con = gemmi.Connection()
            con.type = gemmi.ConnectionType.Covale # XXX may be others
            con.link_id = m.chem_link.id
            con.partner1 = model.cra_to_atomaddress(m.cra1)
            con.partner2 = model.cra_to_atomaddress(m.cra2)
            new_connections.append(con)

    rm_idxes = []
    con_idxes = dict((c,i) for i,c in enumerate(st.connections))
    for con in conns:
        if con.link_id in known_links: continue
        cra1, cra2 = st[0].find_cra(con.partner1), st[0].find_cra(con.partner2)
        if None in (cra1.atom, cra2.atom):
            logger.writeln(" WARNING: atom(s) not found for link: atom1= {} atom2= {} id= {}".format(con.partner1, con.partner2, con.link_id))
            continue
        dist = cra1.atom.pos.dist(cra2.atom.pos)
        m, swap = monlib.match_link(cra1.residue, cra1.atom.name, cra2.residue, cra2.atom.name,
                                    cra1.atom.altloc if cra1.atom.altloc!="\0" else cra2.atom.altloc)
        if m:
            if swap:
                con.partner1 = model.cra_to_atomaddress(cra2)
                con.partner2 = model.cra_to_atomaddress(cra1)
            con.link_id = m.id
            logger.writeln(" Link confirmed: {} atom1= {} atom2= {} dist= {:.2f} ideal= {:.2f}".format(m.id,
                                                                                                       cra1, cra2,
                                                                                                       dist,
                                                                                                       m.rt.bonds[0].value))
        else:
            logger.writeln(" WARNING: unidentified link: atom1= {} atom2= {} dist= {:.2f} id= {}".format(con.partner1, con.partner2, dist, con.link_id))
            if remove_unknown: # should we just remove id?
                i = con_idxes.get(con)
                if i is not None: rm_idxes.append(i)

    if remove_unknown:
        for i in sorted(rm_idxes, reverse=True):
            st.connections.pop(i)

    if add_found:
        for con in new_connections: # st.connections should have not been modified earlier because referenced in the loop above
            st.connections.append(con)

# find_and_fix_links()

def add_hydrogens(st, monlib, pos="elec"):
    assert pos in ("elec", "nucl")
    # perhaps this should be done outside..
    st.entities.clear()
    st.setup_entities()

    topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAddButWater, warnings=logger, ignore_unknown_links=True)
    if pos == "nucl":
        logger.writeln("Generating hydrogens at nucleus positions")
        resnames = st[0].get_all_residue_names()
        check_monlib_support_nucleus_distances(monlib, resnames)
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=default_proton_scale)
    else:
        logger.writeln("Generating hydrogens at electron positions")
# add_hydrogens()

def plane_deviations(atoms):
    pp = gemmi.find_best_plane(atoms)
    return [gemmi.get_distance_from_plane(a.pos, pp) for a in atoms]
# plane_deviations()

def nbc_z(nbc, nbc_ops):
    cra1,cra2,imgidx,_,mindist,sigma,_,_ = nbc
    
    if imgidx == 0:
        pos2 = cra2.atom.pos
    else:
        pos2 = gemmi.Position(nbc_ops[imgidx].apply(cra2.atom.pos))
        
    b = cra1.atom.pos.dist(pos2)
    db = b - mindist
    if db < 0:
        return db/sigma
    return 0.
# nbc_z()

class Restraints:
    #import line_profiler
    #profile = line_profiler.LineProfiler()
    #import atexit
    #atexit.register(profile.print_stats)
    #@profile
    def __init__(self, st, monlib, enerlib=None):
        self.st = st # clone?
        st.setup_cell_images()
        self.monlib = monlib
        self.enerlib = enerlib if enerlib is not None else load_ener_lib()
        find_and_fix_links(self.st, self.monlib)
        self.st.entities.clear()
        self.st.setup_entities() # entity information is needed for links
        self.topo = gemmi.prepare_topology(self.st, monlib, warnings=logger,
                                           ignore_unknown_links=True, reorder=True) # updates atom serial
        # lookup tables
        self.lookup = dict((x.atom, (i, x)) for i, x in enumerate(self.st[0].all()))
        self.nbc = []
        self.nbc_transforms = []
        self.outlier_sigmas = dict(bond=5, angle=5, torsion=5, nonbonded=5, chiral=5, plane=5)

        self.set_atom_types()
        self.construct_graph()
        self.find_nonbonded()
        self.set_ideal_chiral_volumes()
    # __init__()
    
    #@profile
    def construct_graph(self):
        import networkx as nx
        n_atoms = len(self.lookup)
        g = nx.Graph()
        for b in self.topo.bonds:
            idx1 = self.lookup[b.atoms[0]][0]
            idx2 = self.lookup[b.atoms[1]][0]
            g.add_edge(idx1, idx2)
            
        self.g = g
    # construct_graph()

    def set_atom_types(self):
        self.atom_types = {} # {resname: {atom_name: chem_type}}
        for resname, mon in self.monlib.monomers.items():
            self.atom_types[resname] = dict((a.id, a.chem_type) for a in mon.atoms)
    # set_atom_types()

    def get_chem_type(self, resname, atomname):
        r = self.atom_types.get(resname)
        if r: return r.get(atomname)
        return None
    # get_chem_type()

    #@profile
    def max_vdwr(self):
        maxr = 0
        for res in self.atom_types.values():
            for at in res.values():
                e = self.enerlib.get(at)
                if e:
                    maxr = max(maxr, e["vdw_radius"])
        logger.writeln("maximum vdw_radius in this model= {}".format(maxr))
        return maxr
    # max_vdwr()

    #@profile
    def get_nbc_mindist(self, type1, type2, d_12):
        default_min_dist = 3.4
        ene1 = self.enerlib.get(type1)
        ene2 = self.enerlib.get(type2)
        vdwtype = 1
        sigma = 0.2
        if None in (ene1, ene2):
            for t, e in ((type1, ene1), (type2, ene2)):
                if e is None: logger.writeln("WARNING: unknwon energy type: {}".format(t))
            return default_min_dist, sigma, vdwtype

        # Test hbond
        ht1, ht2 = ene1["hb_type"], ene2["hb_type"]
        vdwr1, vdwr2 = ene1["vdw_radius"], ene2["vdw_radius"]
        mindist = vdwr1 + vdwr2
        
        if ((ht1 in "AB" and ht2 in "DB") or
            (ht2 in "AB" and ht1 in "DB")):
            mindist -= 0.3 # ADHB
            sigma = 0.2
            vdwtype = 3

        if d_12 == 3: # 1_4 related atoms
            vdwtype = 2
            sigma = 0.2
            mindist -= 0.3

        # TODO hydrogen ignored!!
        # TODO in the same ring?
        return mindist, sigma, vdwtype

    # get_nbc_mindist()        

    #@profile
    def find_nonbonded(self):
        def check_bonds(s, e): # return shortest path between (s,e). checks up to 3 bonds
            if s not in self.g: return -1

            checked = {s: 0}
            for d in range(3):
                for n1 in [x for x in checked if checked[x]==d]:
                    for n2 in self.g.neighbors(n1):
                        if e == n2: return d+1
                        checked[n2] = d+1
            return -1
        # check_bonds()

        self.nbc = []
        self.nbc_transforms = []
        ns = gemmi.NeighborSearch(self.st[0], self.st.cell, 4).populate()
        cs = gemmi.ContactSearch(self.max_vdwr()*2)        
        results = cs.find_contacts(ns)
        for r in results: # TODO check alt loc!
            idx1 = self.lookup[r.partner1.atom][0]
            idx2 = self.lookup[r.partner2.atom][0]
            idx1_2_d = check_bonds(idx1, idx2)
            if idx1_2_d == 3 or idx1_2_d < 0:
                type1 = self.get_chem_type(r.partner1.residue.name, r.partner1.atom.name)
                type2 = self.get_chem_type(r.partner2.residue.name, r.partner2.atom.name)
                
                mindist, sigma, vdwtype = self.get_nbc_mindist(type1, type2, idx1_2_d)
                if r.dist < mindist:
                    #print('atom1="{}" atom2="{}" image={} dist={:.2f} id={:.2f} type={}'.format(r.partner1, r.partner2, r.image_idx, r.dist, mindist, vdwtype))
                    z = (mindist - r.dist)/sigma
                    self.nbc.append((r.partner1, r.partner2, r.image_idx, r.dist, mindist, sigma, z, vdwtype))

        # convert for orthogonal coordiinates
        transform = lambda x: self.st.cell.orth.combine(x).combine(self.st.cell.frac)
        if self.nbc:
            self.nbc_transforms = [transform(ns.get_image_transformation(i)) for i in range(max(x[2] for x in self.nbc)+1)]
    # find_nonbonded()

    #@profile
    def set_ideal_chiral_volumes(self):
        tmp = []
        for chir in self.topo.chirs:
            tmp.append(self.topo.ideal_chiral_abs_volume(chir))
            
        self.chir_vols = numpy.array(tmp)
    # set_ideal_chiral_volumes()

    #@profile
    def show_all(self):
        topo_d = dict(bond=self.topo.bonds, angle=self.topo.angles, torsion=self.topo.torsions,
                      chiral=self.topo.chirs, plane=self.topo.planes, nonbonded=self.nbc)
        dfs = {}
        for k in topo_d:
            topo_k = topo_d[k]

            if k == "torsion":
                atom_labels = ["atom{}".format(i+1) for i in range(4)]
                df = pandas.DataFrame([[str(self.lookup[y][1]) for y in x.atoms] + [x.calculate(), x.restr.value,
                                                                            x.restr.period, x.restr.esd, x.calculate_z(), x.restr.label]
                                       for x in topo_k],
                                      columns= atom_labels + ["model", "ideal", "period", "esd", "z", "label"])
                df = df.loc[df.esd > 0]
            elif k == "chiral":
                idealstr = {gemmi.ChiralityType.Positive:"positive",
                            gemmi.ChiralityType.Negative:"negative",
                            gemmi.ChiralityType.Both:"both"}
                df = pandas.DataFrame([[str(self.lookup[y][1]) for y in x.atoms] + [x.calculate(), self.topo.ideal_chiral_abs_volume(x),
                                                                            idealstr[x.restr.sign], 0.2, x.check()] for x in topo_k],
                                      columns=["atomc", "atom1", "atom2", "atom3", "model", "ideal_abs", "sign", "esd", "check"])
                df["z"] = numpy.abs([b.calculate_z(iv, 0.2) for b, iv in zip(topo_k, df["ideal_abs"])])
            elif k == "plane":
                index = pandas.MultiIndex.from_arrays([["{}_{}".format(i, x.restr.label) for i, x in enumerate(topo_k) for y in x.atoms],
                                                       [str(self.lookup[y][1]) for x in topo_k for y in x.atoms]],
                                                       names=["label", "atom"])
                df = pandas.DataFrame([(d, x.restr.esd) for i,x in enumerate(topo_k)
                                       for y,d in zip(x.atoms, plane_deviations(x.atoms))],
                                      index=index,
                                      columns=["dev", "esd"])
                df["z"] = numpy.abs(df.dev/df.esd)
            elif k == "nonbonded":
                df = pandas.DataFrame([(str(partner1), str(partner2), image_idx, dist, mindist, vdwtype, sigma, z)
                                       for partner1, partner2, image_idx, dist, mindist, sigma, z, vdwtype in topo_k],
                                      columns=["atom1", "atom2", "image", "model", "mindist", "vdwtype", "esd", "z"])
            else:
                n_atoms = dict(bond=2, angle=3, torsion=4)
                atom_labels = ["atom{}".format(i+1) for i in range(n_atoms[k])]
                df = pandas.DataFrame([[str(self.lookup[y][1]) for y in x.atoms] + [x.calculate(), x.restr.value, x.restr.esd, x.calculate_z()]
                                       for x in topo_k],
                                      columns= atom_labels + ["model", "ideal", "esd", "z"])

            if k in ("angle", "torsion"): df["model"] = numpy.rad2deg(df["model"])
            dfs[k] = df

            if k == "chiral":
                df = df.loc[(~df.check) | (df.z > self.outlier_sigmas[k])].sort_values("z", ascending=False)
            elif k == "plane":
                # sort by z within label (same plane)
                df = df.reset_index().sort_values(["label", "z"], ascending=[True, False]).set_index(["label", "atom"])
                # sort by max z
                tmp = df.groupby(level=0)["z"].max()
                idx=tmp[tmp > self.outlier_sigmas[k]].sort_values(ascending=False).index
                df = df.loc[idx]
            else:
                df = df.loc[(df.z > self.outlier_sigmas[k])].sort_values("z", ascending=False)

            n_outl = len(df.index.unique(0)) if k == "plane" else len(df.index)
            logger.writeln("\n{} {} outliers (> {} sigma)".format(n_outl, k, self.outlier_sigmas[k]))
            if n_outl > 0:
                logger.writeln(df.to_string(index=(k=="plane")))

        logger.writeln("\nSummary:")
        tmp = []
        for k in dfs:
            df = dfs[k]
            if k == "nonbonded": df = df.loc[df.model < df.mindist]
            tmp.append([k, len(df.index), numpy.sqrt(numpy.mean((df.z*df.esd)**2)), numpy.mean(df.esd), numpy.sqrt(numpy.mean(df.z**2))])
            if k == "torsion":
                for p, g in df.groupby("period", sort=True):
                    tmp.append(["{} (period {})".format(k, p), len(g.index), numpy.sqrt(numpy.mean((g.z*g.esd)**2)),
                                numpy.mean(g.esd), numpy.sqrt(numpy.mean(g.z**2))])
            elif k == "chiral":
                df = df.loc[df.sign!="both"]
                tmp.append(["{} (non-both)".format(k), len(df.index), numpy.sqrt(numpy.mean((df.z*df.esd)**2)),
                            numpy.mean(df.esd), numpy.sqrt(numpy.mean(df.z**2))])

            
        logger.writeln(pandas.DataFrame(tmp,
                                      columns=["Restraint type", "number", "rmsd", "mean(esd)", "rmsz"],
                                      ).to_string(index=False, float_format="%.3f"))

        logger.writeln("\n!WARNING!: this function has problems at the moment. Will be sorted out later.")
        logger.writeln("""\
TODO
- nbc distance for metals (e.g. Mg-O) and hydrogen atoms are not correct
- redundant restraint definitions should be sorted out (e.g. A has C2e-nyu0 and C3e-nyu0 for the same atoms)
- separate refined and non-refined atoms (hydrogen, occ 0?)""")

        return dfs
    # show_all()
# class Restraints
