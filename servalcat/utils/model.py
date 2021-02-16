"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
import gemmi
import numpy
import os

def load_monomer_library(resnames, monomer_dir=None): #TODO add cif_paths
    if monomer_dir is None:
        if "CLIBD_MON" not in os.environ:
            logger.write("ERROR: CLIBD_MON is not set")
            return
        monomer_dir = os.environ["CLIBD_MON"]
        
    if not os.path.isdir(monomer_dir):
        logger.write("ERROR: not a directory: {}".format(monomer_dir))
        return
        
    monlib = gemmi.read_monomer_lib(monomer_dir, resnames)
    return monlib
# load_monomer_library()

def shake_structure(st, sigma):
    print("Randomizing structure with rmsd of {}".format(sigma))
    st2 = st.clone()
    for model in st2:
        for cra in model.all():
            r = numpy.random.normal(0, sigma, 3)
            cra.atom.pos += gemmi.Position(*r)

    return st2
# shake_structure()

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

def calc_fc_em(st, d_min, mott_bethe=True, monlib=None, blur=0, r_cut=1e-5, rate=1.5):
    # XXX check minimum B and set appropriate blur

    if monlib is not None and st[0].count_hydrogen_sites() > 0:
        st = st.clone()
        topo = gemmi.prepare_topology(st, monlib)
        resnames = st[0].get_all_residue_names()
        check_monlib_support_nucleus_distances(monlib, resnames)
    else:
        topo = None
        
    if mott_bethe:
        dc = gemmi.DensityCalculatorX()
        dc.d_min = d_min
        dc.blur = blur
        dc.r_cut = r_cut
        dc.rate = rate
        dc.set_grid_cell_and_spacegroup(st)

        if topo is None:
            dc.addends.subtract_z()
            dc.put_model_density_on_grid(st[0])
        else:
            # 
            dc.initialize_grid()
            dc.addends.subtract_z(except_hydrogen=True)
            dc.add_model_density_to_grid(st[0])
            # 
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus)
            for cra in st[0].all():
                if cra.atom.is_hydrogen():
                    dc.add_c_contribution_to_grid(cra.atom, -1)
                    
            dc.sum_symmetry_equivalent_grid_points()

        grid = gemmi.transform_map_to_f_phi(dc.grid)
        asu_data = grid.prepare_asu_data(dmin=d_min, mott_bethe=True, unblur=dc.blur)
    else:
        dc = gemmi.DensityCalculatorE()
        dc.d_min = d_min
        dc.blur = blur
        d.r_cut = r_cut
        dc.rate = rate
        dc.addends.subtract_z()
        dc.set_grid_cell_and_spacegroup(st)
        dc.put_model_density_on_grid(st[0])
        grid = gemmi.transform_map_to_f_phi(dc.grid)
        asu_data = grid.prepare_asu_data(dmin=d_min)
        
    return asu_data

# calc_fc_em()

def all_B(st):
    ret = []
    for mol in st:
        for chain in mol: # TODO use all()!
            for res in chain:
                for at in res:
                    ret.append(at.b_iso)

    return ret
# all_B()
