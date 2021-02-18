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

def determine_blur_for_dencalc(st, grid):
    b_min = min((cra.atom.b_iso for cra in st[0].all()))
    b_need = grid**2*8*numpy.pi**2/1.1 # Refmac's way
    b_add = b_need - b_min
    return b_add
# determine_blur_for_dencalc()

def calc_fc_fft(st, d_min, source, mott_bethe=True, monlib=None, blur=None, r_cut=1e-5, rate=1.5,
                omit_proton=False):
    assert source in ("xray", "electron")
    if source != "electron": assert not mott_bethe
    if omit_proton:
        assert mott_bethe
        if st[0].count_hydrogen_sites() == 0:
            logger.write("WARNING: omit_proton requested, but no hydrogen exists!")
    
    if blur is None:
        blur = determine_blur_for_dencalc(st, d_min/2/rate)
        logger.write("Setting blur= {:.2f} in density calculation".format(blur))
        
    if not omit_proton and monlib is not None and st[0].count_hydrogen_sites() > 0:
        st = st.clone()
        topo = gemmi.prepare_topology(st, monlib)
        resnames = st[0].get_all_residue_names()
        check_monlib_support_nucleus_distances(monlib, resnames)
    else:
        topo = None
        
    if source == "xray" or mott_bethe:
        dc = gemmi.DensityCalculatorX()
    else:
        dc = gemmi.DensityCalculatorE()

    dc.d_min = d_min
    dc.blur = blur
    dc.r_cut = r_cut
    dc.rate = rate
    dc.set_grid_cell_and_spacegroup(st)

    if mott_bethe:
        if omit_proton:
            logger.write("Calculating proton-omit Fc using Mott-Bethe formula")
            dc.initialize_grid()
            dc.addends.subtract_z(except_hydrogen=True)
            dc.add_model_density_to_grid(st[0])
            dc.grid.symmetrize_sum()
        elif topo is None:
            logger.write("Calculating Fc using Mott-Bethe formula")
            dc.addends.subtract_z()
            dc.put_model_density_on_grid(st[0])
        else:
            logger.write("Calculating proton-shifted Fc using Mott-Bethe formula")
            # Z-fx but for hydrogen -fx only
            dc.initialize_grid()
            dc.addends.subtract_z(except_hydrogen=True)
            dc.add_model_density_to_grid(st[0])
            # Shift proton positions and add Z components
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus)
            for cra in st[0].all():
                if cra.atom.is_hydrogen():
                    dc.add_c_contribution_to_grid(cra.atom, -1)
                    
            dc.grid.symmetrize_sum()
    else:
        logger.write("Calculating Fc")
        dc.put_model_density_on_grid(st[0])

    grid = gemmi.transform_map_to_f_phi(dc.grid)
    asu_data = grid.prepare_asu_data(dmin=d_min, mott_bethe=mott_bethe, unblur=dc.blur)
        
    return asu_data

# calc_fc_em()

def calc_fc_direct(st, d_min, source, mott_bethe, monlib=None):
    assert source in ("xray", "electron")
    if source != "electron": assert not mott_bethe
    
    unit_cell = st.cell
    spacegroup = gemmi.SpaceGroup(st.spacegroup_hm)
    miller_array = gemmi.make_miller_array(unit_cell, spacegroup, d_min)
    topo = None

    if source == "xray" or mott_bethe:
        calc = gemmi.StructureFactorCalculatorX(st.cell)
    else:
        calc = gemmi.StructureFactorCalculatorE(st.cell)
        
    
    if source == "electron" and mott_bethe:
        if monlib is not None and st[0].count_hydrogen_sites() > 0:
            st = st.clone()
            topo = gemmi.prepare_topology(st, monlib)
            resnames = st[0].get_all_residue_names()
            check_monlib_support_nucleus_distances(monlib, resnames)

        calc.addends.clear()
        calc.addends.subtract_z(except_hydrogen=True)

    vals = []
    for hkl in miller_array:
        sf = calc.calculate_sf_from_model(st[0], hkl)
        if mott_bethe: sf *= calc.mott_bethe_factor()
        vals.append(sf)

    if topo is not None:
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus)

        for i, hkl in enumerate(miller_array):
            sf = calc.calculate_mb_z(st[0], hkl, only_h=True)
            if mott_bethe: sf *= calc.mott_bethe_factor()
            vals[i] += sf
    
    asu = gemmi.ComplexAsuData(unit_cell, spacegroup,
                               miller_array, vals)
    return asu
# calc_fc_direct()

def all_B(st):
    ret = []
    for mol in st:
        for chain in mol: # TODO use all()!
            for res in chain:
                for at in res:
                    ret.append(at.b_iso)

    return ret
# all_B()
