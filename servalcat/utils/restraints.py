"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.refmac import refmac_keywords
from servalcat import ext
import os
import gemmi
import string
import random
import numpy
import pandas
import json
import fnmatch

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
                         ignore_monomer_dir=False, update_old_atom_names=True,
                         params=None):
    resnames = st[0].get_all_residue_names()

    if monomer_dir is None and not ignore_monomer_dir:
        if "CLIBD_MON" not in os.environ:
            logger.error("WARNING: CLIBD_MON is not set")
        else:
            monomer_dir = os.environ["CLIBD_MON"]

    if cif_files is None:
        cif_files = []
        
    monlib = gemmi.MonLib()
    if monomer_dir and not ignore_monomer_dir:
        if not os.path.isdir(monomer_dir):
            raise RuntimeError("not a directory: {}".format(monomer_dir))

        logger.writeln("Reading monomers from {}".format(monomer_dir))
        monlib.read_monomer_lib(monomer_dir, resnames, logger)

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
                if b.find_values("_chem_comp_bond.comp_id") and not b.find_values("_chem_comp_bond.value_dist"):
                    raise RuntimeError(f"Bond length information for {name} is missing from {f}. Please generate restraints using a tool like acedrg.")
                    
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

    if update_old_atom_names:
        monlib.update_old_atom_names(st, logger)

    if params:
        update_torsions(monlib, params.get("restr", {}).get("torsion_include", {}))
    
    return monlib
# load_monomer_library()

def fix_elements_in_model(monlib, st):
    monlib_els = {m: {a.id: a.el for a in monlib.monomers[m].atoms} for m in monlib.monomers}
    lookup = {x.atom: x for x in st[0].all()}
    for chain in st[0]:
        for res in chain:
            d = monlib_els.get(res.name)
            if not d: continue # should not happen
            for at in res:
                if at.name not in d: # for example atom names of element D may be different, which will be sorted later
                    continue
                el = d[at.name]
                if at.element != el:
                    logger.writeln(f"WARNING: correcting element of {lookup[at]} to {el.name}")
                    at.element = el
# correct_elements_in_model()

def update_torsions(monlib, params):
    # take subset
    params = [p for p in params
              if any(x in p for x in ("tors_value", "tors_sigma", "tors_period"))]
    if not params:
        return
    logger.writeln("Updating torsion targets in dictionaries")
    for p in params:
        if "residue" in p:
            tors = [cc.rt.torsions for cc in monlib.monomers.values()
                    if fnmatch.fnmatch(cc.name, p["residue"])]
        elif "group" in p:
            g = gemmi.ChemComp.read_group(p["group"])
            # should warn if g is Null
            tors = [cc.rt.torsions for cc in monlib.monomers.values()
                    if cc.group == g]
        elif "link" in p:
            tors = [ln.rt.torsions for ln in monlib.links.values()
                    if fnmatch.fnmatch(ln.id, p["link"])]
        else:
            tors = []
        if not tors:
            continue
        logger.writeln(f" rule = {p}")
        for tt in tors:
            for t in tt:
                if fnmatch.fnmatch(t.label, p["tors_name"]):
                    if "tors_value" in p:
                        t.value = p["tors_value"]
                    if "tors_sigma" in p:
                        t.esd = p["tors_sigma"]
                    if "tors_period" in p:
                        t.period = p["tors_period"]
# update_torsions()

def make_torsion_rules(restr_params):
    # Defaults
    include_rules = [{"group": "peptide", "tors_name": "chi*"},
                     {"link": "*", "tors_name": "omega"},
                     {"residue": "*", "tors_name": "sp2_sp2*"},
                     {"link": "*", "tors_name": "sp2_sp2*"},
                     ]
    exclude_rules = []

    # Override include/exclude rules
    for i, name in enumerate(("torsion_include", "torsion_exclude")):
        rules = (include_rules, exclude_rules)[i]
        for p in restr_params.get(name, []):
            r = {}
            if p["flag"]:
                for k in "residue", "group", "link":
                    if k in p:
                        r[k] = p[k]
                if r and "tors_name" in p:
                    r["tors_name"] = p["tors_name"]
                    rules.append(r)
            else:
                rules.clear()

    # How to tell about hydrogen?
    logger.writeln("Torsion angle rules:")
    for l, rr in (("include", include_rules), ("exclude", exclude_rules)):
        logger.writeln(f" {l}:")
        if not rr:
            logger.writeln(f"  none")
        for r in rr:
            logger.writeln(f"  {r}")

    return include_rules, exclude_rules
# make_torsion_rules())

def select_restrained_torsions(monlib, include_rules, exclude_rules):
    ret = {"monomer": {}, "link": {}}
    
    # Collect monomer/link related torsions
    all_tors = {"mon": {}, "link": {}}
    groups = {}
    for mon_id in monlib.monomers:
        mon = monlib.monomers[mon_id]
        groups.setdefault(mon.group, []).append(mon_id)
        all_tors["mon"][mon_id] = [x.label for x in mon.rt.torsions]
    for mod_id in monlib.modifications:
        mod = monlib.modifications[mod_id]
        tors = [x.label for x in mod.rt.torsions if chr(x.id1.comp) in ("a", "c")] # don't need delete
        if not tors: continue
        gr = gemmi.ChemComp.read_group(mod.group_id)
        if mod.comp_id and mod.comp_id in all_tors["mon"]:
            all_tors["mon"][mod.comp_id].extend(tors)
        elif not mod.comp_id and gr in groups:
            for mon_id in groups[gr]:
                all_tors["mon"][mon_id].extend(tors)
    for lnk_id in monlib.links:
        lnk = monlib.links[lnk_id]
        if lnk.rt.torsions:
            all_tors["link"][lnk_id] = [x.label for x in lnk.rt.torsions]
    for k in all_tors:
        for kk in all_tors[k]:
            all_tors[k][kk] = set(all_tors[k][kk])
            
    # Apply include/exclude rule
    for mon in all_tors["mon"]:
        match_f = lambda r: ("tors_name" in r and
                             ("residue" in r and fnmatch.fnmatch(mon, r["residue"]) or
                              mon in groups.get(gemmi.ChemComp.read_group(r.get("group", "")), [])))
        use_tors = []
        for r in include_rules:
            if match_f(r):
                use_tors.extend(x for x in all_tors["mon"][mon] if fnmatch.fnmatch(x, r["tors_name"]))
        for r in exclude_rules:
            if match_f(r):
                use_tors = [x for x in use_tors if not fnmatch.fnmatch(x, r["tors_name"])]
        if use_tors:
            ret["monomer"][mon] = sorted(use_tors)
    for lnk in all_tors["link"]:
        match_f = lambda r: ("tors_name" in r and
                             "link" in r and fnmatch.fnmatch(lnk, r["link"]))
        use_tors = []
        for r in include_rules:
            if match_f(r):
                use_tors.extend(x for x in all_tors["link"][lnk] if fnmatch.fnmatch(x, r["tors_name"]))
        for r in exclude_rules:
            if match_f(r):
                use_tors = [x for x in use_tors if not fnmatch.fnmatch(x, r["tors_name"])]
        if use_tors:
            ret["link"][lnk] = sorted(use_tors)

    return ret
# select_restrained_torsions()

def prepare_topology(st, monlib, h_change, ignore_unknown_links=False, raise_error=True, check_hydrogen=False,
                     use_cispeps=False, add_metal_restraints=True, params=None):
    # Check duplicated atoms
    bad = []
    for chain in st[0]:
        bad_res = []
        for res  in chain:
            n_uniq = len({(a.name, a.altloc) for a in res})
            if n_uniq != len(res):
                bad_res.append(str(res.seqid))
        if bad_res:
            bad.append(" chain {}: {}".format(chain.name, " ".join(bad_res)))
    if bad:
        raise RuntimeError("Following residues have duplicated atoms. Check your model.\n{}".format("\n".join(bad)))

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
        if params:
            parsed = refmac_keywords.parse_keywords(keywords).get("exte")
            if parsed:
                params["exte"] = params.get("exte", []) + parsed
    else:
        keywords = []
    # these checks can be done after sorting links
    logger.writeln("Creating restraints..")
    with logger.with_prefix("  "):
        topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=logger, reorder=False,
                                      ignore_unknown_links=ignore_unknown_links, use_cispeps=use_cispeps)
    unknown_cc = set()
    link_related = set()
    nan_hydr = set()

    def extra_defined(res1, res2): # TODO should check alt
        for link in topo.extras:
            res12 = (link.res1, link.res2)
            if link.link_id and (res12 == (res1, res2) or res12 == (res2, res1)):
                return True
        return False

    # collect info
    info = {}
    for cinfo in topo.chain_infos:
        toadd = info.setdefault(cinfo.chain_ref.name, {})
        if cinfo.polymer:
            gaps = []
            for rinfo in cinfo.res_infos:
                if (rinfo.prev and rinfo.prev[0].link_id in ("gap", "") and
                    not extra_defined(rinfo.prev[0].res1, rinfo.prev[0].res2)):
                    gaps.append((rinfo.prev[0].res1, rinfo.prev[0].res2))
            toadd["polymer"] = (str(cinfo.polymer_type).replace("PolymerType.", ""),
                                cinfo.res_infos[0].res.seqid,
                                cinfo.res_infos[-1].res.seqid,
                                len(cinfo.res_infos), gaps)
        else:
            l = toadd.setdefault("nonpolymer", [])
            for ri in cinfo.res_infos:
                l.append(ri.res.name)
    logger.writeln("\nChain info:")
    for chain in info:
        logger.writeln(" chain {}".format(chain))
        if "polymer" in info[chain]:
            logger.writeln("  {}: {}..{} ({} residues)".format(*info[chain]["polymer"][:-1]))
            for gap in info[chain]["polymer"][-1]:
                logger.writeln("    gap between {} and {}".format(*gap))
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

def remove_duplicated_links(connections):
    # ignore p.res_id.name?
    totuple = lambda p: (p.chain_name, p.res_id.seqid.num, p.res_id.seqid.icode, p.atom_name, p.altloc)
    dic = {}
    for i, con in enumerate(connections):
        dic.setdefault(tuple(sorted([totuple(con.partner1), totuple(con.partner2)])), []).append(i)
    todel = []
    for k in dic:
        if len(dic[k]) > 1:
            ids = set(connections[c].link_id for c in dic[k] if connections[c].link_id.strip())
            if len(ids) > 1:
                logger.writeln(" WARNING: duplicated links are found with different link_id")
            tokeep = dic[k][0]
            if ids:
                for c in dic[k]:
                    if connections[c].link_id.strip():
                        tokeep = c
                        break
            todel.extend(c for c in dic[k] if c != tokeep)

    for i in sorted(todel, reverse=True):
        del connections[i]
    if todel:
        logger.writeln(" {} duplicated links were removed.".format(len(todel)))
# remove_duplicated_links()

def find_and_fix_links(st, monlib, bond_margin=1.3, find_metal_links=True, add_found=True, find_symmetry_related=True,
                       metal_margin=1.1, add_only_from=None):
    metalc = MetalCoordination(monlib)
    """
    Identify link ids for st.connections and find new links
    This is required for correctly recognizing link in gemmi.prepare_topology
    Note that it ignores segment IDs
    FIXME it assumes only one bond exists in a link. It may not be the case in future.
    """
    from servalcat.utils import model

    logger.writeln("Checking links defined in the model")
    remove_duplicated_links(st.connections)
    for con in st.connections:
        if con.type == gemmi.ConnectionType.Hydrog: continue
        if con.link_id == "gap": continue # TODO check residues?
        cra1, cra2 = st[0].find_cra(con.partner1, ignore_segment=True), st[0].find_cra(con.partner2, ignore_segment=True)
        if None in (cra1.atom, cra2.atom):
            logger.writeln(" WARNING: atom(s) not found for link: id= {} atom1= {} atom2= {}".format(con.link_id, con.partner1, con.partner2))
            continue
        if cra1.atom.element.is_metal or cra2.atom.element.is_metal:
            con.type = gemmi.ConnectionType.MetalC
        if con.asu != gemmi.Asu.Same: # XXX info from metadata may be wrong
            im = st.cell.find_nearest_image(cra1.atom.pos, cra2.atom.pos, con.asu)
            image_idx = im.sym_idx
            con.asu = gemmi.Asu.Same if im.same_asu() else gemmi.Asu.Different
            dist = im.dist()
        else:
            image_idx = 0
            con.asu = gemmi.Asu.Same
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
            elif con.type == gemmi.ConnectionType.MetalC:
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

    logger.writeln("Finding new links (will be added if marked by *)")
    ns = gemmi.NeighborSearch(st[0], st.cell, 5.).populate()
    cs = gemmi.ContactSearch(4.)
    cs.ignore = gemmi.ContactSearch.Ignore.SameResidue
    results = cs.find_contacts(ns)
    onsb = set(gemmi.Element(x) for x in "ONSB")
    n_found = 0

    # st.find_connection_by_cra is quite slow (spent ~12 sec for 7k00, 6301 connections)
    # now it's ~6 times faster
    connections = {tuple((p.chain_name, p.res_id.seqid, p.res_id.name, p.atom_name, p.altloc) for p in (c.partner1, c.partner2))
                   for c in st.connections if c.type != gemmi.ConnectionType.Hydrog}
    def find_connection(cra1, cra2):
        key = lambda cra: (cra.chain.name, cra.residue.seqid, cra.residue.name, cra.atom.name, cra.atom.altloc)
        return (key(cra1), key(cra2)) in connections or (key(cra2), key(cra1)) in connections
    
    for r in results:
        # skip adjacent residues in a polymer entity
        if (r.partner1.chain == r.partner2.chain and
            r.partner1.residue.entity_type == r.partner2.residue.entity_type == gemmi.EntityType.Polymer and
            r.partner1.residue.entity_id == r.partner2.residue.entity_id):
            if r.partner1.chain.next_residue(r.partner1.residue) == r.partner2.residue:
                atom1, atom2 = r.partner1.atom.name, r.partner2.atom.name
            elif r.partner1.chain.next_residue(r.partner2.residue) == r.partner1.residue:
                atom1, atom2 = r.partner2.atom.name, r.partner1.atom.name
            else:
                atom1, atom2 = None, None
            if atom1 is not None:
                ent = st.get_entity(r.partner1.residue.entity_id)
                if (ent.polymer_type in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD) and
                    atom1 == "C" and atom2 == "N"):
                    continue
                if (ent.polymer_type in (gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid) and
                    atom1 == "O3'" and atom2 == "P"):
                    continue
        if find_connection(r.partner1, r.partner2): continue
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
            will_be_added = add_found and (not add_only_from or link.id in add_only_from)
            logger.writeln(" {}New link found: id= {} {} dist= {:.2f} ideal= {:.2f}".format("*" if will_be_added else " ",
                                                                                            link.id,
                                                                                            atoms_str,
                                                                                            r.dist,
                                                                                            link.rt.bonds[0].value))
        elif find_metal_links:
            # link only metal - O/N/S/B
            if r.partner1.atom.element.is_metal == r.partner2.atom.element.is_metal: continue
            if not cra2.atom.element in onsb: continue
            max_ideal = metalc.find_max_dist(cra1, cra2)
            if r.dist > max_ideal * metal_margin: continue # tolerance should be smaller than that for other links
            will_be_added = add_found
            logger.writeln(" {}Metal link found: {} dist= {:.2f} max_ideal= {:.2f}".format("*" if will_be_added else " ",
                                                                                           atoms_str,
                                                                                           r.dist, max_ideal))
        else:
            continue
        n_found += 1
        if not will_be_added: continue
        con = gemmi.Connection()
        con.name = "added{}".format(n_found)
        if link:
            con.link_id = link.id
            con.type = gemmi.ConnectionType.Disulf if link.id == "disulf" else gemmi.ConnectionType.Covale
        if cra1.atom.element.is_metal or cra2.atom.element.is_metal:
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
    topo = prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAddButWater, ignore_unknown_links=False)
    
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

def dictionary_block_names(monlib, topo):
    used = {x.lower() for x in monlib.monomers}
    for chain_info in topo.chain_infos:
        for res_info in chain_info.res_infos:
            for link in res_info.prev:
                # won't be included if the name starts with "auto-", but don't do such checks here
                used.add("link_" + link.link_id.lower())
        for mod in res_info.mods:
            used.add("mod_" + mod.id.lower())
    for extra in topo.extras:
        used.add("link_" + extra.link_id.lower())
    return used
# dictionary_block_names()

def prepare_ncs_restraints(st, rms_loc_nlen=5, min_nalign=10, max_rms_loc=2.0):
    logger.writeln("Finding NCS..")
    polymers = {}
    for chain in st[0]:
        rs = chain.get_polymer()
        p_type = rs.check_polymer_type()
        if p_type in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD,
                      gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid):
            polymers.setdefault(p_type, []).append((chain, rs))

    scoring = gemmi.AlignmentScoring("p") # AlignmentScoring::partial_model
    al_res = []
    ncslist = ext.NcsList()
    for pt in polymers:
        #print(pt, [x[0].name for x in polymers[pt]])
        pols = polymers[pt]
        for i in range(len(pols)-1):
            q = [x.name for x in pols[i][1]]
            for j in range(i+1, len(pols)):
                al = gemmi.align_sequence_to_polymer(q, pols[j][1], pt, scoring)
                if al.match_count < min_nalign: continue
                su = gemmi.calculate_superposition(pols[i][1], pols[j][1], pt, gemmi.SupSelect.All)
                obj = ext.NcsList.Ncs(al, pols[i][1], pols[j][1], pols[i][0].name, pols[j][0].name)
                obj.calculate_local_rms(rms_loc_nlen)
                if len(obj.local_rms) == 0 or numpy.all(numpy.isnan(obj.local_rms)):
                    continue
                ave_local_rms = numpy.nanmean(obj.local_rms)
                if ave_local_rms > max_rms_loc: continue
                ncslist.ncss.append(obj)
                al_res.append({"chain_1": "{} ({}..{})".format(obj.chains[0], obj.seqids[0][0], obj.seqids[-1][0]),
                               "chain_2": "{} ({}..{})".format(obj.chains[1], obj.seqids[0][1], obj.seqids[-1][1]),
                               "aligned": al.match_count,
                               "identity": al.calculate_identity(1),
                               "rms": su.rmsd,
                               "ave(rmsloc)": ave_local_rms,
                               })
                if al_res[-1]["identity"] < 100:
                    wrap_width = 100
                    logger.writeln(f"seq1: {pols[i][0].name} {pols[i][1][0].seqid}..{pols[i][1][-1].seqid}")
                    logger.writeln(f"seq2: {pols[j][0].name} {pols[j][1][0].seqid}..{pols[j][1][-1].seqid}")
                    logger.writeln(f"match_count: {al.match_count} (identity: {al_res[-1]['identity']:.2f})")
                    s1 = gemmi.one_letter_code(q)
                    p_seq = gemmi.one_letter_code(pols[j][1].extract_sequence())
                    p1, p2 = al.add_gaps(s1, 1), al.add_gaps(p_seq, 2)
                    for k in range(0, len(p1), wrap_width):
                        logger.writeln(" seq1 {}".format(p1[k:k+wrap_width]))
                        logger.writeln("      {}".format(al.match_string[k:k+wrap_width]))
                        logger.writeln(" seq2 {}\n".format(p2[k:k+wrap_width]))

    ncslist.set_pairs()
    df = pandas.DataFrame(al_res)
    df.index += 1
    logger.writeln(df.to_string(float_format="%.2f"))
    return ncslist
# prepare_ncs_restraints()

class MetalCoordination:
    def __init__(self, monlib, dbfile=None):
        self.monlib = monlib
        if dbfile is None:
            dbfile = os.path.join(monlib.path(), "metals.json")
        if os.path.exists(dbfile):
            with open(dbfile) as f:
                self.metals = json.load(f)["metal_coordination"]
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
                ener_ideal = self.monlib.find_ideal_distance(cra1, cra2)
                coords.setdefault(cra1.atom.element, {}).setdefault(cra1.atom, []).append((cra2.atom, i, ener_ideal))
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
                    ener_ideals = {x[2] for m in coords[metal] for x in coords[metal][m] if x[0].element == el}
                    logger.write(" ".join("{:.2f}".format(x) for x in ener_ideals))
                    logger.writeln(" (from ener_lib)")
                else:
                    logger.writeln(" ".join("{:.4f} ({} coord)".format(x["median"], x["coord"]) for x in vals))
                    ideals[el] = [(x["median"], max(0.02, x["mad"]*1.5)) for x in vals if x["mad"] > 0]
            logger.writeln("")
            for i, am in enumerate(coords[metal]):
                logger.writeln("  site {}: {}".format(i+1, lookup[am]))
                for j, (lig, con_idx, _) in enumerate(coords[metal][am]):
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
