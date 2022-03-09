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

default_proton_scale = 1.13 # scale of X-proton distance to X-H(e) distance

def decide_new_mod_id(mod_id, mods):
    i = 1
    while True:
        i += 1
        new_id = "{}-{}".format(mod_id, i)
        if new_id not in mods:
            return new_id
# decide_new_mod_id()

def rename_cif_modification_if_necessary(doc, known_ids):
    # FIXME Problematic if other file refers to modification that is renamed in this function - but how can we know?
    
    mod_list = doc.find_block("mod_list")
    if not mod_list: return {}
    
    trans = {}
    for row in mod_list.find("_chem_mod.", ["id"]):
        mod_id = row.str(0)
        if mod_id in known_ids:
            new_id = decide_new_mod_id(mod_id, known_ids)
            trans[mod_id] = new_id
            row[0] = new_id # modify id
            logger.write("INFO:: renaming modification id {} to {}".format(mod_id, new_id))

    # modify ids in mod_* blocks
    for mod_id in trans:
        b = doc.find_block("mod_{}".format(mod_id))
        if not b: # should raise error?
            logger.write("WARNING:: inconsistent data_mod_list for {} in {}".format(mod_id, f))
            continue
        b.name = "mod_{}".format(trans[mod_id]) # modify name
        for item in b:
            for tag in item.loop.tags:
                if tag.endswith(".mod_id"):
                    for row in b.find(tag[:tag.rindex(".")+1], ["mod_id"]):
                        row[0] = trans[mod_id]

    # Update mod id in links
    link_list = doc.find_block("link_list")
    if trans and link_list:
        for row in link_list.find("_chem_link.", ["mod_id_1", "mod_id_2"]):
            for i in range(2):
                if row.str(i) in trans:
                    row[i] = trans[row.str(i)]

    return trans
# rename_cif_modification_if_necessary()

def load_monomer_library(st, monomer_dir=None, cif_files=None, stop_for_unknowns=False, check_hydrogen=False):
    resnames = st[0].get_all_residue_names()

    if monomer_dir is None:
        if "CLIBD_MON" not in os.environ:
            logger.error("ERROR: CLIBD_MON is not set")
        else:
            monomer_dir = os.environ["CLIBD_MON"]

    if cif_files is None:
        cif_files = []
        
    if not os.path.isdir(monomer_dir):
        logger.error("ERROR: not a directory: {}".format(monomer_dir))
        return

    if monomer_dir:
        logger.write("Reading monomers from {}".format(monomer_dir))
        monlib = gemmi.read_monomer_lib(monomer_dir, resnames, ignore_missing=True)
    else:
        monlib = gemmi.MonLib()

    for f in cif_files:
        logger.write("Reading monomer: {}".format(f))
        doc = gemmi.cif.read(f)
        for b in doc:
            if b.find_values("_chem_comp_atom.atom_id"):
                name = b.name.replace("comp_", "")
                if name in monlib.monomers:
                    logger.write("WARNING:: updating monomer {} using {}".format(name, f))
                    del monlib.monomers[name]
                monlib.add_monomer_if_present(b)

                # Check if bond length values are included
                # This is to fail if cif file is e.g. from PDB website
                for b in monlib.monomers[name].rt.bonds:
                    if b.value != b.value:
                        raise RuntimeError("{} does not contain bond length value for {}. You need to generate restraints (e.g. using acedrg).".format(f, name))
                    
        link_list = doc.find_block("link_list")
        if link_list:
            for row in link_list.find("_chem_link.", ["id"]):
                link_id = row.str(0)
                if link_id in monlib.links:
                    logger.write("WARNING:: updating link {} using {}".format(link_id, f))
                    del monlib.links[link_id]

        # If modification id is duplicated, need to rename
        rename_cif_modification_if_necessary(doc, monlib.modifications)
        
        monlib.insert_chemlinks(doc)
        monlib.insert_chemmods(doc)
        monlib.insert_comp_list(doc)

    not_loaded = set(resnames).difference(monlib.monomers)
    if not_loaded:
        logger.write("WARNING: monomers not loaded: {}".format(" ".join(not_loaded)))
        
    logger.write("Monomer library loaded: {} monomers, {} links, {} modifications".format(len(monlib.monomers),
                                                                                          len(monlib.links),
                                                                                          len(monlib.modifications)))
    logger.write("       Monomers: {}".format(" ".join([x for x in monlib.monomers])))
    logger.write("          Links: {}".format(" ".join([x for x in monlib.links])))
    logger.write("  Modifications: {}".format(" ".join([x for x in monlib.modifications])))
    logger.write("")

    logger.write("Checking if unknown atoms exist..")
    st = st.clone()
    sio = io.StringIO()
    topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.NoChange, warnings=sio, reorder=True,
                                  ignore_unknown_links=True)

    # possible warnings:
    # Warning: unknown chemical component XXX in chain X
    # Warning: no atom X expected in XXX
    # and others?
    unknown_cc = set()
    unknown_atoms_cc = set()
    for l in sio.getvalue().splitlines():
        r = re.search("Warning: unknown chemical component (.*) in chain", l)
        if r:
            unk = r.group(1)
            if unk not in unknown_cc:
                logger.write(l)
                unknown_cc.add(unk)
            continue
        r1 = re.search("Warning: no atom (.*) expected in (.*)$", l)
        r2 = re.search("Warning: definition not found for [^/]*/([^/ ]*) [^/]*/([^/]*)$", l) # chain/resn seqid/atom.alt
        if r1 or r2:
            if r1:
                unk = r1.groups() # (atom, cc)
            else:
                unk = r2.groups()[::-1] # (atom, cc)
                
            if unk[1] not in unknown_cc and unk not in unknown_atoms_cc:
                logger.write(l)
                unknown_atoms_cc.add(unk)
            continue

        # something else
        logger.write(l)

    if not check_hydrogen:
        todel = []
        for unk in unknown_atoms_cc:
            elements = [cra.atom.element for cra in st[0].all() if cra.residue.name==unk[1] and cra.atom.name==unk[0]]
            if elements and elements[0].is_hydrogen:
                todel.append(unk)
        unknown_atoms_cc = unknown_atoms_cc - set(todel)
        
    unknown_cc.update(cc for at, cc in unknown_atoms_cc)
        
    if stop_for_unknowns and unknown_cc:
        raise RuntimeError("Provide restraint cif file(s) for {}".format(",".join(unknown_cc)))
   
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
        logger.write("WARNING: nucleus distance is not found for: {}".format(" ".join(nucl_not_found)))
        logger.write("         default scale ({}) is used for nucleus distances.".format(default_proton_scale))
    return good
# check_monlib_support_nucleus_distances()

def find_and_fix_links(st, monlib, bond_margin=1.1, remove_unknown=False, add_found=True):
    """
    Find links not registered in st.connections
    This is required for correctly recognizing link in gemmi.prepare_topology
    if remove_unknown=True, undefined links and unmatched links are removed.
    """
    from servalcat.utils import model

    logger.write("Checking links in model")
    hunt = gemmi.LinkHunt()
    hunt.index_chem_links(monlib)
    matches = hunt.find_possible_links(st, bond_margin, 0)
    known_links = ("TRANS", "PTRANS", "NMTRANS", "CIS", "PCIS", "NMCIS", "p", "gap")
    conns = [x for x in st.connections] # to check later
    new_connections = []

    for m in matches:
        if m.conn:
            logger.write(" Link confirmed: {} atom1= {} atom2= {} dist= {:.2f} ideal= {:.2f}".format(m.chem_link.id,
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
        else:
            # Known link is only accepted when in LINK record
            if not m.chem_link or m.chem_link.id in known_links:
                continue

            logger.write(" Link detected:  {} atom1= {} atom2= {} dist= {:.2f} ideal= {:.2f}".format(m.chem_link.id,
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
        if remove_unknown:
            i = con_idxes.get(con)
            if i is not None: rm_idxes.append(i)
            
        at1, at2 = st[0].find_cra(con.partner1).atom, st[0].find_cra(con.partner2).atom
        if None in (at1, at2):
            logger.write(" WARNING: atom(s) not found for link: atom1= {} atom2= {} id= {}".format(con.partner1, con.partner2, con.link_id))
            continue
        dist = at1.pos.dist(at2.pos)
        logger.write(" WARNING: unidentified link: atom1= {} atom2= {} dist= {:.2f} id= {}".format(con.partner1, con.partner2, dist, con.link_id))

    if remove_unknown:
        for i in sorted(rm_idxes, reverse=True):
            st.connections.pop(i)

    if add_found:
        for con in new_connections: # st.connections should have not been modified earlier because referenced in the loop above
            st.connections.append(con)

# find_and_fix_links()

def add_hydrogens(st, monlib, pos="elec"):
    assert pos in ("elec", "nucl")
    st.setup_entities()

    # Check links. XXX Is it ok to update st?
    find_and_fix_links(st, monlib, remove_unknown=True)
    
    topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAddButWater, warnings=logger)
    if pos == "nucl":
        logger.write("Generating hydrogens at nucleus positions")
        resnames = st[0].get_all_residue_names()
        check_monlib_support_nucleus_distances(monlib, resnames)
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=default_proton_scale)
    else:
        logger.write("Generating hydrogens at electron positions")
# add_hydrogens()

def plane_deviations(atoms):
    pp = gemmi.find_best_plane(atoms)
    return [gemmi.get_distance_from_plane(a.pos, pp) for a in atoms]
# plane_deviations()

def show_all(st, monlib):
    st.setup_entities() # entity information is needed for links
    topo = gemmi.prepare_topology(st, monlib, warnings=logger)
    
    lookup = dict([(x.atom, (x.chain, x.residue, x.atom)) for x in st[0].all()])
    label = lambda c, r, a: "{:4s}{:1s}{:3s}{:>2s}{:4d}{:1s}".format(a.name, a.altloc if a.altloc!="\x00" else " ",
                                                                     c.name,
                                                                     r.name, r.seqid.num, r.seqid.icode)

    topo_d = dict(bond=topo.bonds, angle=topo.angles, torsion=topo.torsions,)#, chir=topo.chirs)
    for k in topo_d:
        topo_k = topo_d[k]
        all_z = [b.calculate_z() for b in topo_k]
        all_d = [b.calculate() for b in topo_k]
        if k in ("angle", "torsion"): all_d = numpy.rad2deg(all_d)
        all_i = [b.restr.value for b in topo_k]
        perm = range(len(all_z))
        perm = numpy.argsort(all_z)
        if k == "torsion":
            tmp = [all_z[i] * topo_k[i].restr.esd for i in range(len(topo_k))]
            #print("TORSION_DEV=", tmp)
            rmsd = numpy.sqrt(numpy.average(numpy.array(tmp)**2))
        else:
            rmsd = numpy.sqrt(numpy.average((numpy.array(all_d)-all_i)**2))
        print("# {}_rmsd= {:.5f}".format(k, rmsd))
        for i in reversed(perm):
            t = topo_k[i]
            labs = " ".join([label(*lookup[x]) for x in t.atoms])
            b_ave = sum([x.b_iso for x in t.atoms])/len(t.atoms)
            print("{} {:.4f} {:.4f} {:.2f}".format(labs, all_d[i], all_z[i], b_ave))
            #if k=="torsion": print(t.restr.value, t.restr.esd, t.restr.period)
            #if all_z[i] < 10: break


    # Plane
    topo_k = topo.planes
    all_d = [numpy.sqrt(numpy.average(numpy.square(plane_deviations(t.atoms)))) for t in topo_k]
    perm = numpy.argsort(all_d)
    print("# planes_meandev= {:.4f}".format(numpy.average(all_d)))
    for i in reversed(perm):
        t = topo_k[i]
        labs = " ".join([label(*lookup[x]) for x in t.atoms])
        print("{} {:.4f}".format(labs, all_d[i]))

    # Chir
    topo_k = topo.chirs
    all_d = [b.calculate() for b in topo_k]
    all_i = [b.restr.sign for b in topo_k]
    all_c = [b.check() for b in topo_k]
    perm = numpy.argsort(all_c)
    print("# chirs")
    for i in reversed(perm):
        t = topo_k[i]
        labs = " ".join([label(*lookup[x]) for x in t.atoms])
        if not all_c[i]:
            print("{} {:.4f} {} {}".format(labs, all_d[i], all_i[i], all_c[i]))

if __name__ == "__main__":
    import sys
    model_in = sys.argv[1]
    st = gemmi.read_structure(model_in)
    monlib = load_monomer_library(st,
                                  cif_files=sys.argv[2:] if len(sys.argv)>2 else None)
    find_and_fix_links(st, monlib, remove_unknown=True)
    show_all(st, monlib)
