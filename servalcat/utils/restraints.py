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

def load_monomer_library(resnames, monomer_dir=None, cif_files=None):
    if monomer_dir is None:
        if "CLIBD_MON" not in os.environ:
            logger.write("ERROR: CLIBD_MON is not set")
        else:
            monomer_dir = os.environ["CLIBD_MON"]

    if cif_files is None:
        cif_files = []
        
    if not os.path.isdir(monomer_dir):
        logger.write("ERROR: not a directory: {}".format(monomer_dir))
        return

    if monomer_dir:
        monlib = gemmi.read_monomer_lib(monomer_dir, resnames)
    else:
        monlib = gemmi.MonLib()

    for f in cif_files: # Support link!!
        logger.write("Reading monomer: {}".format(f))
        doc = gemmi.cif.read(f)
        for b in doc:
            if b.find_values("_chem_comp_atom.atom_id"):
                name = b.name.replace("comp_", "")
                if name in monlib.monomers:
                    logger.write("WARNING:: updating {} using {}".format(name, f))
                    del monlib.monomers[name]
                monlib.add_monomer_if_present(b)

    logger.write("Monomer library loaded: {} monomers, {} links, {} modifications".format(len(monlib.monomers),
                                                                                          len(monlib.links),
                                                                                          len(monlib.modifications)))
    logger.write("       Monomers: {}".format(" ".join([x for x in monlib.monomers])))
    logger.write("          Links: {}".format(" ".join([x for x in monlib.links])))
    logger.write("  Modifications: {}".format(" ".join([x for x in monlib.modifications])))
        
    return monlib
# load_monomer_library()

def check_monlib_support_nucleus_distances(monlib, resnames):
    good = True
    for resn in resnames:
        if resn not in monlib.monomers:
            logger.write("ERROR: monomer information of {} not loaded".format(resn))
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
                logger.write("ERROR: nucleus distance is not found for {}".format(resn))
                good = False

    return good
# check_monlib_support_nucleus_distances()

