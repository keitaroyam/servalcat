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
import gemmi
import string
import random
import numpy
import json

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
            logger.writeln("WARNING:: inconsistent mod description for {}".format(mod_id))
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

def load_monomer_library(st, monomer_dir=None, cif_files=None, stop_for_unknowns=False,
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
            atom_id_list = b.find_values("_chem_comp_atom.atom_id")
            if atom_id_list:
                name = b.name.replace("comp_", "")
                if name in monlib.monomers:
                    logger.writeln("WARNING:: updating monomer {} using {}".format(name, f))
                    del monlib.monomers[name]

                # Check if bond length values are included
                # This is to fail if cif file is e.g. from PDB website
                if len(atom_id_list) > 1 and not b.find_values("_chem_comp_bond.value_dist"):
                    raise RuntimeError("{} does not contain bond length value for {}. You need to generate restraints (e.g. using acedrg).".format(f, name))
                    
            for row in b.find("_chem_link.", ["id"]):
                link_id = row.str(0)
                if link_id in monlib.links:
                    logger.writeln("WARNING:: updating link {} using {}".format(link_id, f))
                    del monlib.links[link_id]

        # If modification id is duplicated, need to rename
        rename_cif_modification_if_necessary(doc, monlib.modifications)
        monlib.read_monomer_doc(doc)
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
    logger.writeln("       loaded monomers: {}".format(" ".join([x for x in monlib.monomers])))
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
    
    return monlib
# load_monomer_library()

def prepare_topology(st, monlib, h_change, ignore_unknown_links=False, raise_error=True, check_hydrogen=False,
                     use_cispeps=False, add_metal_restraints=True):
    if add_metal_restraints:
        metalc = MetalCoordination(monlib)
        keywords, todel = metalc.setup_restraints(st)
        con_bak = []
        for i in sorted(todel, reverse=True):
            # temporarily remove connection not to put a bond restraint
            con = st.connections.pop(i)
            con_bak.append((i, con))
            # flag non-hydrogen
            cra2 = st[0].find_cra(con.partner2, ignore_segment=True)
            cra2.atom.calc_flag = gemmi.CalcFlag.NoHydrogen
    else:
        keywords = []
    # these checks can be done after sorting links
    logger.writeln("Creating restraints..")
    sio = io.StringIO()
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=sio, reorder=False,
                                  ignore_unknown_links=ignore_unknown_links, use_cispeps=use_cispeps)
    for l in sio.getvalue().splitlines(): logger.writeln(" " + l)
    unknown_cc = set()
    link_related = set()
    nan_hydr = set()

    # collect info
    info = {}
    for cinfo in topo.chain_infos:
        toadd = info.setdefault(cinfo.chain_ref.name, {})
        if cinfo.polymer:
            toadd["polymer"] = (str(cinfo.polymer_type).replace("PolymerType.", ""),
                                cinfo.res_infos[0].res.seqid,
                                cinfo.res_infos[-1].res.seqid,
                                len(cinfo.res_infos))
        else:
            l = toadd.setdefault("nonpolymer", [])
            for ri in cinfo.res_infos:
                l.append(ri.res.name)
    logger.writeln("\nChain info:")
    for chain in info:
        logger.writeln(" chain {}".format(chain))
        if "polymer" in info[chain]:
            logger.writeln("  {}: {}..{} ({} residues)".format(*info[chain]["polymer"]))
        if "nonpolymer" in info[chain]:
            n_res = len(info[chain]["nonpolymer"])
            uniq = set(info[chain]["nonpolymer"])
            logger.writeln("  ligands: {} ({} residues)".format(" ".join(uniq), n_res))
    logger.writeln("")
    
    for cinfo in topo.chain_infos:
        for rinfo in cinfo.res_infos:
            cc_org = monlib.monomers[rinfo.res.name] if rinfo.res.name in monlib.monomers else None
            for ia in reversed(range(len(rinfo.res))):
                atom = rinfo.res[ia]
                atom_str = "{}/{} {}/{}".format(cinfo.chain_ref.name, rinfo.res.name, rinfo.res.seqid, atom.name)
                cc = rinfo.get_final_chemcomp(atom.altloc)
                if not cc.find_atom(atom.name):
                    # warning message should have already been given by gemmi
                    if cc_org and cc_org.find_atom(atom.name):
                        if check_hydrogen or not atom.is_hydrogen():
                            link_related.add(rinfo.res.name)
                    else:
                        if check_hydrogen or not atom.is_hydrogen():
                            unknown_cc.add(rinfo.res.name)
                
                if atom.is_hydrogen() and atom.calc_flag == gemmi.CalcFlag.Dummy:
                    logger.writeln(" Warning: hydrogen {} could not be added - Check dictionary".format(atom_str))
                    unknown_cc.add(rinfo.res.name)
                elif any(numpy.isnan(atom.pos.tolist())): # TODO add NaN test before prepare_toplogy
                    logger.writeln(" Warning: {} position NaN!".format(atom_str))
                    nan_hydr.add(rinfo.res.name)

    if raise_error and (unknown_cc or link_related):
        msgs = []
        if unknown_cc: msgs.append("restraint cif file(s) for {}".format(",".join(unknown_cc)))
        if link_related: msgs.append("proper link cif file(s) for {} or check your model".format(",".join(link_related)))
        raise RuntimeError("Provide {}".format(" and ".join(msgs)))
    if raise_error and nan_hydr:
        raise RuntimeError("Some hydrogen positions became NaN. The geometry of your model may be of low quality. Consider not adding hydrogen")
    if not use_cispeps:
        topo.set_cispeps_in_structure(st)
    if add_metal_restraints:
        for i, con in sorted(con_bak):
            st.connections.insert(i, con)
    return topo, keywords
# prepare_topology()

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

def find_and_fix_links(st, monlib, bond_margin=1.3, find_metal_links=True, add_found=True, find_symmetry_related=True):
    metalc = MetalCoordination(monlib)
    """
    Identify link ids for st.connections and find new links
    This is required for correctly recognizing link in gemmi.prepare_topology
    Note that it ignores segment IDs
    FIXME it assumes only one bond exists in a link. It may not be the case in future.
    """
    from servalcat.utils import model

    logger.writeln("Checking links defined in the model")
    for con in st.connections:
        if con.type == gemmi.ConnectionType.Hydrog: continue
        cra1, cra2 = st[0].find_cra(con.partner1, ignore_segment=True), st[0].find_cra(con.partner2, ignore_segment=True)
        if cra1.atom.element.is_metal or cra2.atom.element.is_metal:
            con.type = gemmi.ConnectionType.MetalC
        if None in (cra1.atom, cra2.atom):
            logger.writeln(" WARNING: atom(s) not found for link: id= {} atom1= {} atom2= {}".format(con.link_id, con.partner1, con.partner2))
            continue
        if con.asu == gemmi.Asu.Different: # XXX info from metadata may be wrong
            nimage = st.cell.find_nearest_image(cra1.atom.pos, cra2.atom.pos, con.asu)
            image_idx = nimage.sym_idx
            dist = nimage.dist()
        else:
            image_idx = 0
            dist = cra1.atom.pos.dist(cra2.atom.pos)
        con.reported_distance = dist
        atoms_str = "atom1= {} atom2= {} image= {}".format(cra1, cra2, image_idx)
        if con.link_id:
            link = monlib.get_link(con.link_id)
            inv = False
            if link is None:
                logger.writeln(" WARNING: link {} not found in the library. Please provide link dictionary.".format(con.link_id))
                con.link_id = "" # let gemmi find proper link in prepare_topology()
                continue
            else:
                match, _, _ = monlib.test_link(link, cra1.residue.name, cra1.atom.name, cra2.residue.name, cra2.atom.name)
                if not match and monlib.test_link(link, cra2.residue.name, cra2.atom.name, cra1.residue.name, cra1.atom.name)[0]:
                    match = True
                    inv = True
                if not match:
                    logger.writeln(" WARNING: link id and atoms mismatch: id= {} {}".format(link.id, atoms_str))
                    continue
        else:
            link, inv, _, _ = monlib.match_link(cra1.residue, cra1.atom.name, cra1.atom.altloc,
                                                cra2.residue, cra2.atom.name, cra2.atom.altloc)
            if link:
                con.link_id = link.id
            elif find_metal_links and con.type == gemmi.ConnectionType.MetalC:
                logger.writeln(" Metal link will be added: {} dist= {:.2f}".format(atoms_str, dist))
                if cra2.atom.element.is_metal:
                    inv = True # make metal first
            else:
                ideal_dist = monlib.find_ideal_distance(cra1, cra2)
                logger.writeln(" Link unidentified (simple bond will be used): {} dist= {:.2f} ideal= {:.2f}".format(atoms_str,
                                                                                                                     dist,
                                                                                                                     ideal_dist))
                continue
        if link:
            logger.writeln(" Link confirmed: id= {} {} dist= {:.2f} ideal= {:.2f}".format(link.id,
                                                                                          atoms_str,
                                                                                          dist,
                                                                                          link.rt.bonds[0].value))
            if con.link_id == "disulf":
                con.type = gemmi.ConnectionType.Disulf
        if inv:
            con.partner1 = model.cra_to_atomaddress(cra2)
            con.partner2 = model.cra_to_atomaddress(cra1)
    if len(st.connections) == 0:
        logger.writeln(" no links defined in the model")

    logger.writeln("Finding new links (will {} added)".format("be" if add_found else "not be"))
    ns = gemmi.NeighborSearch(st[0], st.cell, 5.).populate()
    cs = gemmi.ContactSearch(4.)
    cs.ignore = gemmi.ContactSearch.Ignore.AdjacentResidues # may miss polymer links not contiguous in a chain?
    results = cs.find_contacts(ns)
    onsb = set(gemmi.Element(x) for x in "ONSB")
    n_found = 0
    for r in results:
        if st.find_connection_by_cra(r.partner1, r.partner2): continue
        link, inv, _, _ = monlib.match_link(r.partner1.residue, r.partner1.atom.name, r.partner1.atom.altloc,
                                            r.partner2.residue, r.partner2.atom.name, r.partner2.atom.altloc,
                                            (r.dist / 1.4)**2)
        if link is None and r.partner2.atom.element.is_metal:
            inv = True # make metal first
        if inv:
            cra1, cra2 = r.partner2, r.partner1
        else:
            cra1, cra2 = r.partner1, r.partner2
        im = st.cell.find_nearest_pbc_image(cra1.atom.pos, cra2.atom.pos, r.image_idx)
        #assert r.image_idx == im.sym_idx # should we check this?
        if not find_symmetry_related and not im.same_asu():
            continue
        atoms_str = "atom1= {} atom2= {} image= {}".format(cra1, cra2, r.image_idx)
        if im.pbc_shift != (0,0,0):
            atoms_str += " ({},{},{})".format(*im.pbc_shift)
        if link:
            if r.dist > link.rt.bonds[0].value * bond_margin: continue
            logger.writeln(" New link found: id= {} {} dist= {:.2f} ideal= {:.2f}".format(link.id,
                                                                                          atoms_str,
                                                                                          r.dist,
                                                                                          link.rt.bonds[0].value))
        elif find_metal_links:
            # link only metal - O/N/S/B
            if r.partner1.atom.element.is_metal == r.partner2.atom.element.is_metal: continue
            if not cra2.atom.element in onsb: continue
            max_ideal = metalc.find_max_dist(cra1, cra2)
            if r.dist > max_ideal * 1.1: continue # tolerance should be smaller than that for other links
            logger.writeln(" Metal link found: {} dist= {:.2f} max_ideal= {:.2f}".format(atoms_str,
                                                                                         r.dist, max_ideal))
        n_found += 1
        if not add_found: continue
        con = gemmi.Connection()
        con.name = "added{}".format(n_found)
        if link:
            con.link_id = link.id
            con.type = gemmi.ConnectionType.Disulf if link.id == "disulf" else gemmi.ConnectionType.Covale
        else:
            con.type = gemmi.ConnectionType.MetalC
        con.asu = gemmi.Asu.Same if im.same_asu() else gemmi.Asu.Different
        con.partner1 = model.cra_to_atomaddress(cra1)
        con.partner2 = model.cra_to_atomaddress(cra2)
        con.reported_distance = r.dist
        st.connections.append(con)
    if n_found == 0:
        logger.writeln(" no links found")
# find_and_fix_links()

def add_hydrogens(st, monlib, pos="elec"):
    assert pos in ("elec", "nucl")
    topo = prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAddButWater, ignore_unknown_links=True)
    
    if pos == "nucl":
        logger.writeln("Generating hydrogens at nucleus positions")
        resnames = st[0].get_all_residue_names()
        check_monlib_support_nucleus_distances(monlib, resnames)
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=default_proton_scale)
    else:
        logger.writeln("Generating hydrogens at electron positions")
# add_hydrogens()

def make_atom_spec(cra):
    chain = cra.chain.name
    resi = cra.residue.seqid.num
    ins = cra.residue.seqid.icode
    atom = cra.atom.name
    s = "chain {} resi {} ins {} atom {}".format(chain, resi, ins if ins.strip() else ".", atom)
    if cra.atom.altloc != "\0":
        s += " alt {}".format(cra.atom.altloc)
    return s
# make_atom_spec()        

class MetalCoordination:
    def __init__(self, monlib, dbfile=None):
        self.monlib = monlib
        if dbfile is None:
            dbfile = os.path.join(monlib.path(), "metals.json")
        if os.path.exists(dbfile):
            self.metals = json.load(open(dbfile))["metal_coordination"]
        else:
            self.metals = {}
            logger.writeln("WARNING: {} not found".format(dbfile))
    # __init__()

    def find_max_dist(self, cra_metal, cra_ligand):
        vals = self.find_ideal_distances(cra_metal.atom.element, cra_ligand.atom.element)
        if len(vals) == 0:
            # if not found
            return self.monlib.find_ideal_distance(cra_metal, cra_ligand)
        return max(x["median"] for x in vals)
    # find_max_dist()

    def find_ideal_distances(self, el_metal, el_ligand):
        ideals = {}
        if el_metal.name not in self.metals or el_ligand.name not in self.metals[el_metal.name]:
            return []
        return self.metals[el_metal.name][el_ligand.name]
    # find_ideal_distances
    
    def setup_restraints(self, st):
        ret = [] # returns Refmac keywords
        lookup = {x.atom: x for x in st[0].all()}
        coords = {}
        todel = []
        for i, con in enumerate(st.connections):
            if con.link_id == "" and con.type == gemmi.ConnectionType.MetalC:
                cra1 = st[0].find_cra(con.partner1, ignore_segment=True)
                cra2 = st[0].find_cra(con.partner2, ignore_segment=True)
                if None in (cra1.atom, cra2.atom): continue
                coords.setdefault(cra1.atom.element, {}).setdefault(cra1.atom, []).append((cra2.atom, i))
        if coords:
            logger.writeln("Metal coordinations detected")
        for metal in coords:
            logger.writeln(" Metal: {}".format(metal.name))
            ligand_els = {x[0].element for m in coords[metal] for x in coords[metal][m]}
            logger.writeln("  ideal distances")
            ideals = {}
            for el in ligand_els:
                logger.write("   {}: ".format(el.name))
                vals = self.find_ideal_distances(metal, el)
                if len(vals) == 0:
                    logger.writeln(" uknown (values from ener_lib will be used)")
                else:
                    logger.writeln(" ".join("{:.4f} ({} coord)".format(x["median"], x["coord"]) for x in vals))
                    ideals[el] = [(x["median"], x["mad"]) for x in vals if x["mad"] > 0]
            logger.writeln("")
            for i, am in enumerate(coords[metal]):
                logger.writeln("  site {}: {}".format(i+1, lookup[am]))
                for j, (lig, con_idx) in enumerate(coords[metal][am]):
                    con = st.connections[con_idx]
                    logger.writeln("    ligand {}: {} dist= {:.2f}".format(j+1, lookup[lig],
                                                                           con.reported_distance))
                    specs = [make_atom_spec(x) for x in (lookup[am], lookup[lig])]
                    if lig.element not in ideals:
                        continue
                    todel.append(con_idx)
                    for k, (ideal, sigma) in enumerate(ideals[lig.element]):
                        exte_str  = "exte dist first {} seco {} ".format(*specs)
                        exte_str += "valu {:.4f} sigm {:.4f} type 1 ".format(ideal, sigma)
                        if con.asu == gemmi.Asu.Different:
                            exte_str += "symm y"
                        ret.append(exte_str)
                        #b = ext.Geometry.Bond(am, lig)
                        #b.values.append(ext.Geometry.Bond.Value(ideal, sigma, ideal, sigma))
                        #b.type = 0 if k == 0 else 1
                        #ret.append(b)
                logger.writeln("")
        return ret, list(set(todel))
    # setup_restraints()
