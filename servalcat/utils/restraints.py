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

def prepare_topology(st, monlib, h_change, ignore_unknown_links=False, raise_error=True, check_hydrogen=False):
    # these checks can be done after sorting links
    logger.writeln("Creating restraints..")
    sio = io.StringIO()
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=sio, reorder=False,
                                  ignore_unknown_links=ignore_unknown_links)

    unknown_cc = set()
    link_related = set()
    nan_hydr = set()
    for cinfo in topo.chain_infos:
        for rinfo in cinfo.res_infos:
            cc_org = monlib.monomers[rinfo.res.name] if rinfo.res.name in monlib.monomers else None
            for ia in reversed(range(len(rinfo.res))):
                atom = rinfo.res[ia]
                atom_str = "{}/{} {}/{}".format(cinfo.chain_ref.name, rinfo.res.name, rinfo.res.seqid, atom.name)
                cc = rinfo.get_final_chemcomp(atom.altloc)
                if not cc.find_atom(atom.name):
                    msg = " Warning: definition not found for " + atom_str
                    if cc_org and cc_org.find_atom(atom.name):
                        logger.writeln(msg + " - this atom should have been removed when linking")
                        if check_hydrogen or not atom.is_hydrogen():
                            link_related.add(rinfo.res.name)
                    else:
                        logger.writeln(msg)
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
    return topo
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

def find_and_fix_links(st, monlib, bond_margin=1.1, remove_unknown=False, add_found=True):
    """
    Find links not registered in st.connections
    This is required for correctly recognizing link in gemmi.prepare_topology
    if remove_unknown=True, undefined links and unmatched links are removed.
    Note that it ignores segment IDs
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
        if con.type == gemmi.ConnectionType.Hydrog: continue
        cra1, cra2 = st[0].find_cra(con.partner1, ignore_segment=True), st[0].find_cra(con.partner2, ignore_segment=True)
        if None in (cra1.atom, cra2.atom):
            logger.writeln(" WARNING: atom(s) not found for link: atom1= {} atom2= {} id= {}".format(con.partner1, con.partner2, con.link_id))
            continue
        
        dist = cra1.atom.pos.dist(cra2.atom.pos)
        m, swap, _, _ = monlib.match_link(cra1.residue, cra1.atom.name, cra1.atom.altloc,
                                          cra2.residue, cra2.atom.name, cra2.atom.altloc)

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

    topo = prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAddButWater, ignore_unknown_links=True)
    
    if pos == "nucl":
        logger.writeln("Generating hydrogens at nucleus positions")
        resnames = st[0].get_all_residue_names()
        check_monlib_support_nucleus_distances(monlib, resnames)
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=default_proton_scale)
    else:
        logger.writeln("Generating hydrogens at electron positions")
# add_hydrogens()
