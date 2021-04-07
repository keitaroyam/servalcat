"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import restraints
import gemmi
import numpy
import os

def shake_structure(st, sigma):
    print("Randomizing structure with rmsd of {}".format(sigma))
    st2 = st.clone()
    sigma /= numpy.sqrt(3)
    for model in st2:
        for cra in model.all():
            r = numpy.random.normal(0, sigma, 3)
            cra.atom.pos += gemmi.Position(*r)

    return st2
# shake_structure()

def normalize_it92(st=None, all_elements=False, quiet=False):
    elements = set()
        
    if all_elements:
        elements.add("H")
        elements.add("D")
        for i in range(2, 119):
            elements.add(gemmi.Element(i).name)
    elif st is not None:
        for cra in st[0].all():
            elements.add(cra.atom.element.name)

    for el in elements:
        elem = gemmi.Element(el)
        z = elem.atomic_number
        coef = elem.it92
        norm = z/(sum(coef.a)+coef.c)
        if not quiet: logger.write("Normalizing atomic scattering factor of {} by {}".format(el, norm))
        new_coef = [x*norm for x in coef.a] + coef.b + [coef.c*norm]
        coef.set_coefs(new_coef)
# normalize_it92()

def determine_blur_for_dencalc(st, grid):
    b_min = min((cra.atom.b_iso for cra in st[0].all()))
    b_need = grid**2*8*numpy.pi**2/1.1 # Refmac's way
    b_add = b_need - b_min
    return b_add
# determine_blur_for_dencalc()

def calc_fc_fft(st, d_min, source, mott_bethe=True, monlib=None, blur=None, r_cut=1e-5, rate=1.5,
                omit_proton=False, omit_h_electron=False):
    assert source in ("xray", "electron")
    if source != "electron": assert not mott_bethe
    if omit_proton or omit_h_electron:
        assert mott_bethe
        if st[0].count_hydrogen_sites() == 0:
            logger.write("WARNING: omit_proton/h_electron requested, but no hydrogen exists!")
            omit_proton = omit_h_electron = False
        elif omit_proton and omit_h_electron:
            logger.write("omit_proton and omit_h_electron requested. removing hydrogens")
            st = st.clone()
            st.remove_hydrogens()
            omit_proton = omit_h_electron = False
    
    if blur is None:
        blur = determine_blur_for_dencalc(st, d_min/2/rate)
        logger.write("Setting blur= {:.2f} in density calculation".format(blur))
        
    if mott_bethe and not omit_proton and monlib is not None and st[0].count_hydrogen_sites() > 0:
        st = st.clone()
        topo = gemmi.prepare_topology(st, monlib)
        resnames = st[0].get_all_residue_names()
        restraints.check_monlib_support_nucleus_distances(monlib, resnames)
        # Shift electron positions
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)
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
            method_str = "proton-omit Fc"
        elif omit_h_electron:
            if topo is None:
                method_str = "hydrogen electron-omit Fc"
            else:
                method_str = "hydrogen electron-omit, proton-shifted Fc"
        elif topo is not None:
            method_str = "proton-shifted Fc"
        else:
            method_str = "Fc"

        logger.write("Calculating {} using Mott-Bethe formula".format(method_str))
            
        dc.initialize_grid()
        dc.addends.subtract_z(except_hydrogen=True)

        if omit_h_electron:
            st2 = st.clone()
            st2.remove_hydrogens()
            dc.add_model_density_to_grid(st2[0])
        else:
            dc.add_model_density_to_grid(st[0])

        # Subtract hydrogen Z
        if not omit_proton and st[0].count_hydrogen_sites() > 0:
            if topo is not None:
                # Shift proton positions
                topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus,
                                               default_scale=restraints.default_proton_scale)
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

# calc_fc_fft()

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
            restraints.check_monlib_support_nucleus_distances(monlib, resnames)

        calc.addends.clear()
        calc.addends.subtract_z(except_hydrogen=True)

    vals = []
    for hkl in miller_array:
        sf = calc.calculate_sf_from_model(st[0], hkl)
        if mott_bethe: sf *= calc.mott_bethe_factor()
        vals.append(sf)

    if topo is not None:
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus,
                                       default_scale=restraints.default_proton_scale)

        for i, hkl in enumerate(miller_array):
            sf = calc.calculate_mb_z(st[0], hkl, only_h=True)
            if mott_bethe: sf *= calc.mott_bethe_factor()
            vals[i] += sf
    
    asu = gemmi.ComplexAsuData(unit_cell, spacegroup,
                               miller_array, vals)
    return asu
# calc_fc_direct()

def expand_ncs(st, how=gemmi.HowToNameCopiedChain.Short, special_pos_threshold=0.01):
    if len(st.ncs) == 0: return

    if special_pos_threshold >= 0:
        st.setup_cell_images()
        ns = gemmi.NeighborSearch(st[0], st.cell, 3).populate()
        cs = gemmi.ContactSearch(special_pos_threshold)
        cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
        cs.special_pos_cutoff_sq = 0.0
        results = cs.find_contacts(ns)
        for r in results:
            print(r.partner1, r.partner2, r.partner1.atom.pos==r.partner2.atom.pos, r.image_idx, r.dist)

    
    st.expand_ncs(how)
    
    
# expand_ncs()

def adp_analysis(st):
    logger.write("= ADP analysis =")
    all_B = []
    for i, mol in enumerate(st):
        logger.write("Model {}:".format(i))
        logger.write("            min    Q1   med    Q3   max")
        bs = []
        for chain in mol:
            for res in chain:
                for atom in res:
                    bs.append(atom.b_iso)
            qs = numpy.quantile(bs, [0,0.25,0.5,0.75,1])
            logger.write("Chain {:3s}".format(chain.name) + " {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}".format(*qs))
            all_B.extend(bs)

    qs = numpy.quantile(all_B, [0,0.25,0.5,0.75,1])
    logger.write("All       {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}".format(*qs))
    logger.write("")
# adp_analysis()        

def all_B(st):
    ret = []
    for mol in st:
        for cra in mol.all():
            ret.append(cra.atom.b_iso)

    return ret
# all_B()
