"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import pandas
import scipy.sparse
import os
import time
import itertools
import string
from servalcat import ext

gemmi.IT92_normalize()
gemmi.IT92_set_ignore_charge(False)
gemmi.Element("X").it92.set_coefs(gemmi.Element("O").it92.get_coefs()) # treat X (unknown) as O
ext.IT92_normalize_etc(gemmi.Element("O")) # the same changes to gemmi in servalcat c++ code
u_to_b = 8 * numpy.pi**2
b_to_u = 1. / u_to_b

from servalcat.utils import logger
from servalcat.utils import restraints
from servalcat.utils import maps

def shake_structure(st, sigma, copy=True):
    print("Randomizing structure with rmsd of {}".format(sigma))
    if copy:
        st2 = st.clone()
    else:
        st2 = st
        
    sigma /= numpy.sqrt(3)
    for model in st2:
        for cra in model.all():
            r = numpy.random.normal(0, sigma, 3)
            cra.atom.pos += gemmi.Position(*r)

    return st2
# shake_structure()

def setup_entities(st, clear=False, overwrite_entity_type=False, force_subchain_names=False):
    if clear:
        st.entities.clear()
        st.add_entity_ids(overwrite=True) # clear entity_id so that ensure_entities() will work properly
    st.add_entity_types(overwrite_entity_type)
    st.assign_subchains(force_subchain_names)
    st.ensure_entities()
    st.add_entity_ids()
    st.deduplicate_entities()
# setup_entities()

def determine_blur_for_dencalc(st, grid):
    b_min = st[0].calculate_b_aniso_range()[0]
    b_need = grid**2*8*numpy.pi**2/1.1 # Refmac's way
    b_add = b_need - b_min
    return b_add
# determine_blur_for_dencalc()

def remove_charge(sts):
    nonzero = False
    for st in sts:
        for cra in st[0].all():
            if cra.atom.charge != 0: nonzero = True
            cra.atom.charge = 0
    if nonzero:
        logger.writeln("Warning: all atomic charges were set to zero.")
# remove_charge()

class CustomCoefUtil:
    def __init__(self):
        self.scat_lookup = {} # atom_key: scat_id
        self.elem_lookup = {} # scat_id: set of element_name; there should be a single element for each id though
        self.coeffs = {} # scat_id: coeffs

    def cra2key(self, cra):
        return (cra.chain.name, cra.residue.seqid,
                cra.atom.name, cra.atom.altloc)
        
    def read_from_cif(self, st, cif_in):
        doc = gemmi.cif.read(cif_in)
        # gemmi reads structure from the first block
        block = doc[0]
        
        # _atom_site.id is read as atom.serial
        # it is mmcif writer's responsibility to ensure unique serials
        serial2cra = {x.atom.serial: x for x in st[0].all()}
        self.scat_lookup = {}
        for r in block.find("_atom_site.", ["id", "scat_id"]):
            atom_id = gemmi.cif.as_int(r[0])
            scat_id = gemmi.cif.as_int(r[1])
            cra = serial2cra[atom_id]
            self.scat_lookup[self.cra2key(cra)] = scat_id
            self.elem_lookup.setdefault(scat_id, set()).add(cra.atom.element.name)

        # read coeffs
        self.coeffs = {gemmi.cif.as_int(r[0]): [gemmi.cif.as_number(r[i]) for i in range(1, 11)]
                       for r in block.find("_lmb_scat_coef.", ["scat_id",
                                                               "coef_a1", "coef_a2", "coef_a3", "coef_a4", "coef_a5",
                                                               "coef_b1", "coef_b2", "coef_b3", "coef_b4", "coef_b5"])}
    # read_from_cif()

    def set_coeffs(self, st):
        #logger.writeln("debug: using c4322")
        max_serial = max(cra.atom.serial for cra in st[0].all())
        pp = [[0.]*10 for _ in range(max_serial+1)]
        for cra in st[0].all():
            scat_id = self.scat_lookup.get(self.cra2key(cra))
            if scat_id is None:
                raise RuntimeError(f"scat_id unknown {cra}")
            pp[cra.atom.serial] = self.coeffs[scat_id]
            #pp[cra.atom.serial] = cra.atom.element.c4322.get_coefs() # test
        gemmi.set_custom_form_factors(pp)
        ext.set_custom_form_factors(pp)
    # set_coeffs()

    def show_info(self):
        logger.writeln("Custom atomic scattering factors will be used")
        df = pandas.DataFrame([[k]+v for k, v in self.coeffs.items()],
                              columns=["scat_id"] +[f"{k}{i+1}" for k in ("a", "b") for i in range(5)])
        df["count"] = [list(self.scat_lookup.values()).count(i) for i in df["scat_id"]]
        df["elem"] = [" ".join(self.elem_lookup[i]) for i in df["scat_id"]]
        logger.writeln(df.to_string(index=False))
        logger.writeln("")
    # show_info()
# class CustomCoefUtil

def check_atomsf(sts, source, mott_bethe=True):
    assert source in ("xray", "electron", "neutron")
    if source != "electron": mott_bethe = False
    logger.writeln("Atomic scattering factors for {}".format("xray (use Mott-Bethe to convert to electrons)" if mott_bethe else source))
    if source != "xray" and not mott_bethe:
        logger.writeln("  Note that charges will be ignored")
    el_charges = {(cra.atom.element, cra.atom.charge) for st in sts for cra in st[0].all()}
    elems = {x[0] for x in el_charges}
    tmp = {}
    if source == "xray" or mott_bethe:
        shown = set()
        for el, charge in sorted(el_charges, key=lambda x: (x[0].atomic_number, x[1])):
            sf = gemmi.IT92_get_exact(el, charge)
            if not sf:
                logger.writeln("  Warning: no scattering factor found for {}{:+}".format(el.name, charge))
                sf = el.it92
                charge = 0
            if (el, charge) in shown: continue
            label = el.name if charge == 0 else "{}{:+}".format(el.name, charge)
            shown.add((el, charge))
            tmp[label] = {**{f"{k}{i+1}": x for k in ("a", "b") for i, x in enumerate(getattr(sf, k))}, "c": sf.c}
    else:
        for el in sorted(elems, key=lambda x: x.atomic_number):
            if source == "electron":
                tmp[el.name] = {f"{k}{i+1}": x for k in ("a", "b") for i, x in enumerate(getattr(el.c4322, k))}
            else:
                tmp[el.name] = {"a": el.neutron92.get_coefs()[0]}
    with logger.with_prefix("  "):
        logger.writeln(pandas.DataFrame(tmp).T.to_string())
    logger.writeln("")
# check_atomsf()

def calc_sum_ab(st):
    sum_ab = dict()
    ret = 0.
    for cra in st[0].all():
        if cra.atom.element not in sum_ab:
            it92 = cra.atom.element.it92
            sum_ab[cra.atom.element] = sum(x*y for x,y in zip(it92.a, it92.b))
        ret += sum_ab[cra.atom.element] * cra.atom.occ
    return ret
# calc_sum_ab()

def calc_fc_fft(st, d_min, source, mott_bethe=True, monlib=None, blur=None, cutoff=1e-5, rate=1.5,
                omit_proton=False, omit_h_electron=False, miller_array=None):
    assert source in ("xray", "electron", "neutron", "custom")
    if source != "electron": mott_bethe = False
    topo = None
    if st[0].has_hydrogen():
        st = st.clone()
        if source == "neutron":
            # nothing happens if not st.has_d_fraction
            st.store_deuterium_as_fraction(False)
        if omit_proton or omit_h_electron:
            assert mott_bethe
            if omit_proton and omit_h_electron:
                logger.writeln("omit_proton and omit_h_electron requested. removing hydrogens")
                st.remove_hydrogens()
                omit_proton = omit_h_electron = False
        if mott_bethe and not omit_proton and monlib is not None:
            topo = gemmi.prepare_topology(st, monlib, warnings=logger, ignore_unknown_links=True)
            resnames = st[0].get_all_residue_names()
            restraints.check_monlib_support_nucleus_distances(monlib, resnames)
            # Shift electron positions
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)
    elif omit_proton or omit_h_electron:
        logger.writeln("WARNING: omit_proton/h_electron requested, but no hydrogen exists!")
        omit_proton = omit_h_electron = False
        
    # for printing
    method_str = ""
    if mott_bethe:
        if omit_proton:
            method_str += "proton-omit "
        elif omit_h_electron:
            if topo is None:
                method_str += "hydrogen electron-omit "
            else:
                method_str += "hydrogen electron-omit, proton-shifted "
        elif topo is not None:
            method_str += "proton-shifted "
    method_str += f"Fc with {source} scattering factors"
    if mott_bethe:
        method_str += " through Mott-Bethe formula from X-ray sf"
    logger.writeln(f"Calculating {method_str}..")
    
    if blur is None: blur = determine_blur_for_dencalc(st, d_min/2/rate)
    #blur = max(0, blur) # negative blur may cause non-positive definite in case of anisotropic Bs
    logger.writeln(" Setting blur= {:.2f} in density calculation (unblurred later)".format(blur))
    
    if source == "xray" or mott_bethe:
        dc = gemmi.DensityCalculatorX()
    elif source == "electron":
        dc = gemmi.DensityCalculatorE()
    elif source == "neutron":
        dc = gemmi.DensityCalculatorN()
    elif source == "custom":
        dc = gemmi.DensityCalculatorC()        
    else:
        raise RuntimeError("unknown source")

    dc.d_min = d_min
    dc.blur = blur
    dc.cutoff = cutoff
    dc.rate = rate
    dc.grid.setup_from(st)

    t_start = time.time()
    if mott_bethe:
        dc.initialize_grid()
        dc.addends.subtract_z(except_hydrogen=True)

        if omit_h_electron:
            st2 = st.clone()
            st2.remove_hydrogens()
            dc.add_model_density_to_grid(st2[0])
        else:
            dc.add_model_density_to_grid(st[0])

        # Subtract hydrogen Z
        if not omit_proton and st[0].has_hydrogen():
            if topo is not None:
                # Shift proton positions
                topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus,
                                               default_scale=restraints.default_proton_scale)
            for cra in st[0].all():
                if cra.atom.is_hydrogen():
                    dc.add_c_contribution_to_grid(cra.atom, -1)

        dc.grid.symmetrize_sum()
        sum_ab = calc_sum_ab(st) * len(st.find_spacegroup().operations())
        mb_000 = sum_ab * gemmi.mott_bethe_const() / 4
    else:
        dc.put_model_density_on_grid(st[0])
        mb_000 = 0

    logger.writeln(" done. Fc calculation time: {:.1f} s".format(time.time() - t_start))
    grid = gemmi.transform_map_to_f_phi(dc.grid)
    
    if miller_array is None:
        return grid.prepare_asu_data(dmin=d_min, mott_bethe=mott_bethe, unblur=dc.blur)
    else:
        return grid.get_value_by_hkl(miller_array, mott_bethe=mott_bethe, unblur=dc.blur,
                                     mott_bethe_000=mb_000)
# calc_fc_fft()

def calc_fc_direct(st, d_min, source, mott_bethe, monlib=None, miller_array=None):
    assert source in ("xray", "electron")
    if source != "electron": mott_bethe = False

    miller_array_given = miller_array is not None
    unit_cell = st.cell
    spacegroup = gemmi.SpaceGroup(st.spacegroup_hm)
    if not miller_array_given: miller_array = gemmi.make_miller_array(unit_cell, spacegroup, d_min)
    topo = None

    if source == "xray" or mott_bethe:
        calc = gemmi.StructureFactorCalculatorX(st.cell)
    else:
        calc = gemmi.StructureFactorCalculatorE(st.cell)
        
    
    if source == "electron" and mott_bethe:
        if monlib is not None and st[0].has_hydrogen():
            st = st.clone()
            topo = gemmi.prepare_topology(st, monlib, warnings=logger, ignore_unknown_links=True)
            resnames = st[0].get_all_residue_names()
            restraints.check_monlib_support_nucleus_distances(monlib, resnames)

        calc.addends.clear()
        calc.addends.subtract_z(except_hydrogen=True)

    vals = []
    for hkl in miller_array:
        sf = calc.calculate_sf_from_model(st[0], hkl) # attention: traverse cell.images
        if mott_bethe: sf *= calc.mott_bethe_factor()
        vals.append(sf)

    if topo is not None:
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus,
                                       default_scale=restraints.default_proton_scale)

        for i, hkl in enumerate(miller_array):
            sf = calc.calculate_mb_z(st[0], hkl, only_h=True)
            if mott_bethe: sf *= calc.mott_bethe_factor()
            vals[i] += sf

    if miller_array_given:
        return numpy.array(vals)
    else:
        asu = gemmi.ComplexAsuData(unit_cell, spacegroup,
                                   miller_array, vals)
        return asu
# calc_fc_direct()

def get_em_expected_hydrogen(st, d_min, monlib, weights=None, blur=None, cutoff=1e-5, rate=1.5, optimize=False):
    # Very crude implementation to find peak from calculated map
    assert st[0].has_hydrogen()
    if blur is None: blur = determine_blur_for_dencalc(st, d_min/2/rate)
    blur = max(0, blur)
    logger.writeln("Setting blur= {:.2f} in density calculation".format(blur))

    st = st.clone()
    topo = gemmi.prepare_topology(st, monlib, warnings=logger, ignore_unknown_links=True)
    resnames = st[0].get_all_residue_names()
    restraints.check_monlib_support_nucleus_distances(monlib, resnames)

    topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)
    st_e = st.clone()
    topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus)
    st_n = st.clone()
    
    dc = gemmi.DensityCalculatorX()
    dc.d_min = d_min
    dc.blur = blur
    dc.cutoff = cutoff
    dc.rate = rate

    # Decide box_size
    max_r = max([dc.estimate_radius(cra.atom) for cra in st[0].all()])
    logger.writeln("max_r= {:.2f}".format(max_r))
    box_size = max_r*2 + 1 # padding
    logger.writeln("box_size= {:.2f}".format(box_size))
    mode_all = False #True
    if mode_all:
        dc.grid.setup_from(st)
    else:
        dc.grid.unit_cell = gemmi.UnitCell(box_size, box_size, box_size, 90, 90, 90)
        dc.grid.spacegroup = gemmi.SpaceGroup("P1")
        cbox = gemmi.Position(box_size/2, box_size/2, box_size/2)
    
    if mode_all: dc.initialize_grid()

    if weights is not None:
        w_s, w_w = weights # s_list and w_list
    else:
        w_s, w_w = None, None
        
    for ichain in range(len(st[0])):
        chain = st[0][ichain]
        for ires in range(len(chain)):
            residue = chain[ires]
            for iatom in range(len(residue)):
                atom = residue[iatom]
                if not atom.is_hydrogen(): continue
                h_n = st_n[0][ichain][ires][iatom]
                h_e = st_e[0][ichain][ires][iatom]
                if not mode_all:
                    dc.initialize_grid()
                    h_n.occ = 1.
                    h_e.occ = 1.
                    n_pos = gemmi.Position(h_n.pos)
                    h_n.pos = cbox
                    h_e.pos = cbox + h_e.pos - n_pos
                dc.add_atom_density_to_grid(h_e)
                dc.add_c_contribution_to_grid(h_n, -1)
                if not mode_all:
                    grid = gemmi.transform_map_to_f_phi(dc.grid)
                    asu_data = grid.prepare_asu_data(dmin=d_min, mott_bethe=True, unblur=dc.blur)
                    if w_s is not None:
                        asu_data.value_array[:] *= numpy.interp(1./asu_data.make_d_array(), w_s, w_w)
                        
                    denmap = asu_data.transform_f_phi_to_map(exact_size=(int(box_size*10), int(box_size*10), int(box_size*10)))
                    m = numpy.unravel_index(numpy.argmax(denmap), denmap.shape)
                    peakpos = denmap.get_position(m[0], m[1], m[2])
                    if optimize: peakpos = maps.optimize_peak(denmap, peakpos)
                    atom.pos = peakpos - cbox + n_pos

    if mode_all:
        grid = gemmi.transform_map_to_f_phi(dc.grid)
        asu_data = grid.prepare_asu_data(dmin=d_min, mott_bethe=True, unblur=dc.blur)
        if w_s is not None:
            asu_data.value_array[:] *= numpy.interp(1./asu_data.make_d_array(), w_s, w_w)
        denmap = asu_data.transform_f_phi_to_map(sample_rate=3)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = denmap
        ccp4.update_ccp4_header(2, True) # float, update stats
        ccp4.write_ccp4_map("debug.ccp4")

    return st

# get_em_expected_hydrogen()

def translate_into_box(st, origin=None, apply_shift=True):
    if origin is None: origin = gemmi.Position(0,0,0)
    
    # apply unit cell translations to put model into a box (unit cell)
    omat = st.cell.orth.mat.array
    fmat = st.cell.frac.mat.array.transpose()
    com = numpy.array((st[0].calculate_center_of_mass() - origin).tolist())
    shift = sum([omat[:,i]*numpy.floor(1-numpy.dot(com, fmat[:,i])) for i in range(3)])
    tr = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*shift))
    if apply_shift:
        for m in st: m.transform_pos_and_adp(tr)
    return shift
# translate_into_box()

def box_from_model(model, padding):
    allpos = numpy.array([cra.atom.pos.tolist() for cra in model.all()])
    ext = numpy.max(allpos, axis=0) - numpy.min(allpos, axis=0) + padding
    cell = gemmi.UnitCell(ext[0], ext[1], ext[2], 90, 90, 90)
    return cell
# box_from_model()

def cra_to_indices(cra, model):
    ret = [None, None, None]
    for ic in range(len(model)):
        chain = model[ic]
        if cra.chain != chain: continue
        ret[0] = ic
        for ir in range(len(chain)):
            res = chain[ir]
            if cra.residue != res: continue
            ret[1] = ir
            for ia in range(len(res)):
                if cra.atom == res[ia]:
                    ret[2] = ia

    return tuple(ret)
# cra_to_indices()

def cra_to_atomaddress(cra):
    aa = gemmi.AtomAddress(cra.chain.name,
                           cra.residue.seqid, cra.residue.name,
                           cra.atom.name, cra.atom.altloc)
    aa.res_id.segment = cra.residue.segment
    return aa
# cra_to_atomaddress()

def check_occupancies(st, raise_error=False):
    bad = []
    for cra in st[0].all():
        if not 0 <= cra.atom.occ <= 1 + 1e-6:
            bad.append(cra)
    if bad:
        logger.writeln("Bad occupancies:")
        for cra in bad:
            logger.writeln(f" {cra} occ= {cra.atom.occ:.4f}")
        if raise_error:
            raise RuntimeError("Please check your model and fix bad occupancies")
# check_occupancies()

def find_special_positions(st, special_pos_threshold=0.2, fix_occ=True, fix_pos=True, fix_adp=True):
    ns = gemmi.NeighborSearch(st[0], st.cell, 3).populate()
    cs = gemmi.ContactSearch(special_pos_threshold * 2)
    cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
    cs.special_pos_cutoff_sq = 0
    results = cs.find_contacts(ns)
    found = {}
    cra = {}
    for r in results:
        if r.partner1.atom != r.partner2.atom: continue
        found.setdefault(r.partner1.atom, []).append(r.image_idx)
        cra[r.partner1.atom] = r.partner1

    if found: logger.writeln("Atoms on special position detected.")
    tostr = lambda x: ", ".join("{:.3e}".format(v) for v in x)
    ret = []
    for atom in found:
        images = found[atom]
        n_images = len(images) + 1
        sum_occ = atom.occ * n_images
        logger.writeln(" {} multiplicity= {} images= {} occupancies_total= {:.2f}".format(cra[atom], n_images, images, sum_occ))
        if sum_occ > 1.001 and fix_occ:
            new_occ = atom.occ / n_images
            logger.writeln("  correcting occupancy= {:.2f}".format(new_occ))
            atom.occ = new_occ
        if fix_pos:
            fpos = gemmi.Fractional(st.cell.frac.apply(atom.pos))
            fdiff = sum([(st.cell.images[i-1].apply(fpos) - fpos).wrap_to_zero() for i in images], gemmi.Fractional(0,0,0)) / n_images
            diff = st.cell.orth.apply(fdiff)
            atom.pos += gemmi.Position(diff)
            logger.writeln("  correcting position= {}".format(tostr(atom.pos.tolist())))
            logger.writeln("             pos_viol= {}".format(tostr(diff.tolist())))
        if fix_adp and atom.aniso.nonzero():
            aniso_bak = atom.aniso.elements_pdb()
            fani = atom.aniso.transformed_by(st.cell.frac.mat)
            fani_avg = sum([fani.transformed_by(st.cell.images[i-1].mat) for i in images], fani).scaled(1/n_images)
            atom.aniso = fani_avg.transformed_by(st.cell.orth.mat)
            diff = numpy.array(atom.aniso.elements_pdb()) - aniso_bak
            logger.writeln("  correcting aniso= {}".format(tostr(atom.aniso.elements_pdb())))
            logger.writeln("        aniso_viol= {}".format(tostr(diff)))

        mats = [st.cell.orth.combine(st.cell.images[i-1]).combine(st.cell.frac).mat.array for i in images]
        mat_total = (numpy.identity(3) + sum(numpy.array(m) for m in mats)) / n_images
        mat_total_aniso = (numpy.identity(6) + sum(mat33_as66(m.tolist()) for m in mats)) / n_images
        mat_total_aniso = numpy.linalg.pinv(mat_total_aniso)
        ret.append((atom, images, mat_total, mat_total_aniso))

    return ret
# find_special_positions()    

def expand_ncs(st, special_pos_threshold=0.01, howtoname=gemmi.HowToNameCopiedChain.Short):
    # TODO modify st.connections for atoms at special positions
    if len(st.ncs) == 0: return
    find_special_positions(st, special_pos_threshold) # just to show info, a bit waste of cpu time..
    logger.writeln("Expanding symmetry..")
    st.expand_ncs(howtoname, merge_dist=1e-4)
# expand_ncs()

def prepare_assembly(name, chains, ops, is_helical=False):
    a = gemmi.Assembly(name)
    g = gemmi.Assembly.Gen()
    if sum(map(lambda x: x.tr.is_identity(), ops)) == 0:
        g.operators.append(gemmi.Assembly.Operator()) # add identity
    for i, nop in enumerate(ops):
        op = gemmi.Assembly.Operator()
        op.transform = nop.tr
        if not nop.tr.is_identity():
            if is_helical:
                op.type = "helical symmetry operation"
            else:
                op.type = "point symmetry operation"
        g.operators.append(op)
    g.chains = chains
    a.generators.append(g)
    if is_helical:
        a.special_kind = gemmi.AssemblySpecialKind.RepresentativeHelical
    else:
        a.special_kind = gemmi.AssemblySpecialKind.CompletePoint
    return a
# prepare_assembly()

def filter_contacting_ncs(st, cutoff=5.):
    if len(st.ncs) == 0: return
    logger.writeln("Filtering out non-contacting NCS copies with cutoff={:.2f} A".format(cutoff))
    st.setup_cell_images()
    ns = gemmi.NeighborSearch(st[0], st.cell, cutoff*2).populate() # This is considered crystallographic cell if not 1 1 1. Undesirable result may be seen.
    cs = gemmi.ContactSearch(cutoff)
    cs.twice = True # since we need all image_idx
    cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
    results = cs.find_contacts(ns)
    indices = {r.image_idx for r in results}
    logger.writeln(" contacting copies: {}".format(indices))
    ops = [st.ncs[i-1] for i in indices] # XXX is this correct? maybe yes as long as identity operator is not there
    st.ncs.clear()
    st.ncs.extend(ops)
# filter_contacting_ncs()

def check_symmetry_related_model_duplication(st, distance_cutoff=0.5, max_allowed_ratio=0.5):
    logger.writeln("Checking if model in asu is given.")
    n_atoms = st[0].count_atom_sites()
    st.setup_cell_images()
    ns = gemmi.NeighborSearch(st[0], st.cell, 3).populate()
    cs = gemmi.ContactSearch(distance_cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
    results = cs.find_contacts(ns)
    n_contacting_atoms = len(set([a for r in results for a in (r.partner1.atom, r.partner2.atom)]))
    logger.writeln(" N_atoms= {} N_contacting_atoms= {}".format(n_atoms, n_contacting_atoms))
    return n_contacting_atoms / n_atoms > max_allowed_ratio # return True if too many contacts
# check_symmetry_related_model_duplication()

def adp_analysis(st, ignore_zero_occ=True):
    logger.writeln("= ADP analysis =")
    if ignore_zero_occ:
        logger.writeln("(zero-occupancy atoms are ignored)")

    all_B = []
    for i, mol in enumerate(st):
        if len(st) > 1: logger.writeln("Model {}:".format(i))
        logger.writeln("            min    Q1   med    Q3   max")
        stats = adp_stats_per_chain(mol, ignore_zero_occ)
        for chain, natoms, qs in stats:
            logger.writeln(("Chain {:3s}".format(chain) if chain!="*" else "All      ") + " {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}".format(*qs))
    logger.writeln("")
# adp_analysis()

def adp_stats_per_chain(model, ignore_zero_occ=True):
    bs = {}
    for cra in model.all():
        if not ignore_zero_occ or cra.atom.occ > 0:
            bs.setdefault(cra.chain.name, []).append(cra.atom.b_iso)

    ret = []
    for chain in model:
        if chain.name in [x[0] for x in ret]: continue
        qs = numpy.quantile(bs[chain.name], [0,0.25,0.5,0.75,1])
        ret.append((chain.name, len(bs[chain.name]), qs))

    if len(bs) > 1:
        all_bs = sum(bs.values(), [])
        qs = numpy.quantile(all_bs, [0,0.25,0.5,0.75,1])
        ret.append(("*", len(all_bs), qs))
        
    return ret
# adp_stats_per_chain()

def reset_adp(model, bfactor=None, adp_mode="iso"):
    for cra in model.all():
        if bfactor is not None:
            cra.atom.b_iso = bfactor
        if adp_mode == "iso" or (adp_mode == "fix" and bfactor is not None):
            cra.atom.aniso = gemmi.SMat33f(0,0,0,0,0,0)
        elif adp_mode == "aniso":
            if cra.atom.aniso.nonzero() and bfactor is None: # just in case
                b_iso = cra.atom.aniso.trace() / 3 * u_to_b
                if abs(cra.atom.b_iso - b_iso) > 1e-2:
                    logger.writeln(f"WARNING: {cra} B_iso={cra.atom.b_iso:.3f} and tr(B_aniso)/3={b_iso:.3f} are different. Resetting B_iso from B_aniso")
                cra.atom.b_iso = b_iso
            else:
                u = cra.atom.b_iso * b_to_u
                cra.atom.aniso = gemmi.SMat33f(u, u, u, 0, 0, 0)
# reset_adp()

def shift_b(model, delta_b, min_b=0.01):
    delta_u = delta_b * b_to_u
    min_u = min_b * b_to_u
    for cra in model.all():
        cra.atom.b_iso = max(cra.atom.b_iso + delta_b, min_b)
        if cra.atom.aniso.nonzero():
            M = cra.atom.aniso.as_mat33().array
            v, Q = numpy.linalg.eigh(M)
            v = numpy.maximum(v + delta_u, min_u)
            M2 = Q.dot(numpy.diag(v)).dot(Q.T)
            cra.atom.aniso = gemmi.SMat33f(M2[0,0], M2[1,1], M2[2,2], M2[0,1], M2[0,2], M2[1,2])
            cra.atom.b_iso = cra.atom.aniso.trace() / 3 * u_to_b
# shift_b()

def initialize_values(model, params):
    for k in params:
        if k not in ("adp", "occ", "dfrac"):
            continue
        for selstr, value in params[k].items():
            sel = gemmi.Selection(selstr)
            for chain in sel.chains(model):
                for residue in sel.residues(chain):
                    for atom in sel.atoms(residue):
                        setattr(atom, {"adp":"b_iso", "occ": "occ", "dfrac": "fraction"}[k], value)
                        if k == "adp" and atom.aniso.nonzero():
                            u = atom.b_iso * b_to_u
                            atom.aniso = gemmi.SMat33f(u, u, u, 0, 0, 0)
# initialize_values()

def all_chain_ids(st):
    return [chain.name for model in st for chain in model]
# all_chain_ids()

def all_B(st, ignore_zero_occ=True):
    ret = []
    for mol in st:
        for cra in mol.all():
            if not ignore_zero_occ or cra.atom.occ > 0:
                ret.append(cra.atom.b_iso)

    return ret
# all_B()

def mat33_as66(m):
    # suppose R is a transformation matrix that is applied to 3x3 symmetric matrix U: R U R^T
    # this function constructs equivalent transformation for 6-element vector: R' u
    r = numpy.zeros((6,6))
    for k, (i, j) in enumerate(((0,0), (1,1), (2,2), (0,1), (0,2), (1,2))):
        r[k,:] = (m[i][0] * m[j][0],
                  m[i][1] * m[j][1],
                  m[i][2] * m[j][2],
                  m[i][0] * m[j][1] + m[i][1] * m[j][0],
                  m[i][0] * m[j][2] + m[i][2] * m[j][0],
                  m[i][1] * m[j][2] + m[i][2] * m[j][1])
    return r
def adp_constraints(ops, cell, tr0=True):
    # think about f = (b-Rb)^T (b-Rb) = b^T b - b^T R b -b^T R^T b + b^T R^T R b
    # d^2f/db db^T = 2I - 2(R+R^T) + 2(R^T R)
    # eigenvectors of this second derivative matrix corresponding to 0-valeud eigenvalues are directions to refine
    x = numpy.zeros((6,6))
    if tr0:
        x[:3,:3] += numpy.ones((3,3)) * 2
    for op in ops:
        r = mat33_as66(cell.op_as_transform(op).mat.tolist())
        x += 2 * numpy.identity(6) - 2 * (r + r.T) + 2 * numpy.dot(r.T, r)

    evals, evecs = numpy.linalg.eigh(x)
    ret = []
    for i in range(6):
        if numpy.isclose(evals[i], 0):
            ret.append(evecs[:, i])

    if len(ret) > 0:
        ret = numpy.vstack(ret)
        ret = numpy.where(numpy.abs(ret) < 1e-9, 0, ret)
        return ret
    return numpy.empty((0, 6))
# adp_constraints()

def to_dataframe(st):
    keys = ("model", "chain", "resn", "subchain", "segment", "seqnum", "icode", "altloc",
            "u11", "u22", "u33", "u12", "u13", "u23",
            "b_iso", "charge", "elem", "atom", "occ",
            "x", "y", "z", "tlsgroup")
    d = dict([(x,[]) for x in keys])
    app = lambda k, v: d[k].append(v)
    
    for m in st:
        for cra in m.all():
            c,r,a = cra.chain, cra.residue, cra.atom
            # TODO need support r.het_flag, r.flag, a.calc_flag, a.flag, a.serial?
            app("model", m.num)
            app("chain", c.name)
            app("resn", r.name)
            app("subchain", r.subchain)
            app("segment", r.segment)
            app("seqnum", r.seqid.num)
            app("icode", r.seqid.icode)
            app("altloc", a.altloc)
            app("u11", a.aniso.u11)
            app("u22", a.aniso.u22)
            app("u33", a.aniso.u33)
            app("u12", a.aniso.u12)
            app("u13", a.aniso.u13)
            app("u23", a.aniso.u23)
            app("b_iso", a.b_iso)
            app("charge", a.charge)
            app("elem", a.element.name)
            app("atom", a.name)
            app("occ", a.occ)
            app("x", a.pos.x)
            app("y", a.pos.y)
            app("z", a.pos.z)
            app("tlsgroup", a.tls_group_id)

    return pandas.DataFrame(data=d)
# to_dataframe()

def from_dataframe(df, st=None): # Slow!
    if st is None:
        st = gemmi.Structure()
    else:
        st = st.clone()
        for i in range(len(st)):
            del st[0]
        
    for m_num, dm in df.groupby("model"):
        st.add_model(gemmi.Model(m_num))
        m = st[-1]
        for c_name, dc in dm.groupby("chain"):
            m.add_chain(gemmi.Chain(c_name))
            c = m[-1]
            for rkey, dr in dc.groupby(["seqnum","icode","resn","segment","subchain"]):
                c.add_residue(gemmi.Residue())
                r = c[-1]
                r.seqid.num = rkey[0]
                r.seqid.icode = rkey[1]
                r.name = rkey[2]
                r.segment = rkey[3]
                r.subchain = rkey[4]
                for _, row in dr.iterrows():
                    r.add_atom(gemmi.Atom())
                    a = r[-1]
                    a.altloc = row["altloc"]
                    a.name = row["atom"]
                    a.aniso.u11 = row["u11"]
                    a.aniso.u22 = row["u22"]
                    a.aniso.u33 = row["u33"]
                    a.aniso.u12 = row["u12"]
                    a.aniso.u13 = row["u13"]
                    a.aniso.u23 = row["u23"]
                    a.b_iso = row["b_iso"]
                    a.charge = row["charge"]
                    a.element = gemmi.Element(row["elem"])
                    a.occ = row["occ"]
                    a.pos.x = row["x"]
                    a.pos.y = row["y"]
                    a.pos.z = row["z"]
                    a.tls_group_id = row["tlsgroup"]
                    
    return st
# from_dataframe()

def st_from_positions(positions, bs=None, qs=None):
    st = gemmi.Structure()
    st.add_model(gemmi.Model(1))
    st[0].add_chain(gemmi.Chain("A"))
    c = st[0][0]
    if bs is None: bs = (0. for _ in range(len(positions)))
    if qs is None: qs = (1. for _ in range(len(positions)))
    for i, (pos, b, q) in enumerate(zip(positions, bs, qs)):
        c.add_residue(gemmi.Residue())
        r = c[-1]
        r.seqid.num = i
        r.name = "HOH"
        r.add_atom(gemmi.Atom())
        a = r[-1]
        a.name = "O"
        a.element = gemmi.Element("O")
        a.pos = pos
        a.b_iso = b
        a.occ = q
                    
    return st
# st_from_positions()
            
def invert_model(st):
    # invert x-axis
    A = st.cell.orth.mat.array
    center = numpy.sum(A,axis=1) / 2
    center = gemmi.Vec3(*center)
    mat = gemmi.Mat33([[-1,0,0],[0,1,0],[0,0,1]]) 
    vec = mat.multiply(-center) + center
    tr = gemmi.Transform(mat, vec)
    st[0].transform_pos_and_adp(tr)

    # invert peptides
# invert_model()

def cif2cart_matrix(cell):
    # transformation matrix from U_cif to U_cart
    ruc = cell.reciprocal()
    ret = cell.orth.mat.multiply_by_diagonal(gemmi.Vec3(ruc.a, ruc.b, ruc.c))
    return ret
# cif2cart_matrix()

def cx_to_mx(ss): #SmallStructure to Structure
    st = gemmi.Structure()
    st.spacegroup_hm = ss.spacegroup.xhm()
    st.cell = ss.cell
    st.add_model(gemmi.Model(1))
    st[-1].add_chain(gemmi.Chain("A"))
    st[-1][-1].add_residue(gemmi.Residue())
    st[-1][-1][-1].seqid.num = 1
    st[-1][-1][-1].name = "00"
    cif2cart = cif2cart_matrix(ss.cell)
    
    for site in ss.sites:
        st[-1][-1][-1].add_atom(gemmi.Atom())
        a = st[-1][-1][-1][-1]
        a.name = site.label
        a.aniso = gemmi.SMat33f(*site.aniso.transformed_by(cif2cart).elements_pdb())
        a.b_iso = site.u_iso * u_to_b
        #a.charge = ?
        a.element = site.element
        a.occ = site.occ
        a.pos = site.orth(ss.cell)
        
    return st
# cx_to_mx()

def fix_deuterium_residues(st):
    # we do not have DOD. will not change ND4->NH4 and SPW->SPK, as hydrogen atom names are different
    n_changed = 0
    for chain in st[0]:
        for res in chain:
            if res.name == "DOD":
                res.name = "HOH"
                n_changed += 1
    for con in st.connections:
        for p in (con.partner1, con.partner2):
            if p.res_id.name == "DOD":
                p.res_id.name = "HOH"
    if n_changed > 0:
        logger.writeln("Warning: {} DOD residues have been renamed to HOH".format(n_changed))
