"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
import os
import gemmi
import numpy

default_proton_scale = 1.13 # scale of X-proton distance to X-H(e) distance

def filename_in_monlib(monomer_dir, name):
    return os.path.join(monomer_dir, name[0].lower(), name+".cif")

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
        resinlib = list(filter(lambda x: os.path.exists(filename_in_monlib(monomer_dir, x)), resnames))
        monlib = gemmi.read_monomer_lib(monomer_dir, resinlib)
    else:
        monlib = gemmi.MonLib()

    for f in cif_files:
        logger.write("Reading monomer: {}".format(f))
        doc = gemmi.cif.read(f)
        for b in doc:
            if b.find_values("_chem_comp_atom.atom_id"):
                name = b.name.replace("comp_", "")
                if name in monlib.monomers:
                    logger.write("WARNING:: updating {} using {}".format(name, f))
                    del monlib.monomers[name]
                monlib.add_monomer_if_present(b)
            
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

    if stop_for_unknowns:
        logger.write("Checking if unknown atoms exist..")
        st = st.clone()
        # XXX this creates non-sense ChemLink if not match (see topo.hpp line 443)
        topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange(0)) # .None is not allowed..
        unknowns = set()
        for cinfo in topo.chain_infos:
            for rinfo in cinfo.res_infos:
                if rinfo.chemcomp.name == "": # safer check?
                    logger.write(" Unknown residue found: {}/{} {}".format(cinfo.name, rinfo.res.name,
                                                                           rinfo.res.seqid))
                    unknowns.add(rinfo.res.name)
                    continue

                cc_atoms = set((a.id for a in rinfo.chemcomp.atoms)) # TODO keep it for speedup?
                if check_hydrogen:
                    atoms = set((a.name for a in rinfo.res))
                else:
                    atoms = set((a.name for a in rinfo.res if not a.is_hydrogen()))
                    
                if not cc_atoms.issuperset(atoms):
                    logger.write(" Unknown atom(s) found: {}/{} {}/{}".format(cinfo.name, rinfo.res.name,
                                                                              rinfo.res.seqid,
                                                                              ",".join(atoms.difference(cc_atoms))))
                    unknowns.add(rinfo.res.name)
        if unknowns:
            raise RuntimeError("Provide restraint cif file(s) for {}".format(",".join(unknowns)))

   
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

def find_links(st, monlib):
    from servalcat.utils import model
    # Find links not registered in st.connections

    hunt = gemmi.LinkHunt()
    hunt.index_chem_links(monlib)
    matches = hunt.find_possible_links(st, 1.5, 0)
    known_links = ("TRANS", "PTRANS", "NMTRANS", "CIS", "PCIS", "NMCIS", "p")
    matches = [x for x in matches if x.chem_link and not x.conn and x.chem_link.id not in known_links]
    connections = []
    for m in matches:
        logger.write("New link detected: {} atom1= {} atom2= {} dist= {:.2f} ideal= {:.2f}".format(m.chem_link.id,
                                                                                                   m.cra1, m.cra2,
                                                                                                   m.bond_length,
                                                                                                   m.chem_link.rt.bonds[0].value))
        con = gemmi.Connection()
        con.type = gemmi.ConnectionType.Covale # XXX may be others
        con.link_id = m.chem_link.id
        con.partner1 = model.cra_to_atomaddress(m.cra1)
        con.partner2 = model.cra_to_atomaddress(m.cra2)
        connections.append(con)

    return connections
# find_links()

def add_hydrogens(st, monlib, pos="elec"):
    assert pos in ("elec", "nucl")
    st.setup_entities()

    # Check links. XXX Is it ok to update st?
    connections = find_links(st, monlib)
    st.connections.extend(connections)
    
    topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAddButWater)
    if pos == "nucl":
        logger.write("Generating hydrogens at nucleus positions")
        restraints.check_monlib_support_nucleus_distances(monlib, resnames)
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
    topo = gemmi.prepare_topology(st, monlib)
    
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
    show_all(st, monlib)
