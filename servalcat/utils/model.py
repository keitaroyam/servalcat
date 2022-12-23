"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import restraints
from servalcat.utils import maps
import gemmi
import numpy
import pandas
import scipy.sparse
import os
import time
import itertools
import string

gemmi.IT92_normalize()
gemmi.Element("X").it92.set_coefs(gemmi.Element("O").it92.get_coefs()) # treat X (unknown) as O

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

def setup_entities(st, clear=False, clear_entity_type=False, overwrite_entity_type=False, force_subchain_names=False):
    if clear: st.entities.clear()
    if clear_entity_type:
        for m in st:
            for c in m:
                for r in c:
                    r.entity_type = gemmi.EntityType.Unknown

    st.add_entity_types(overwrite_entity_type)
    st.assign_subchains(force_subchain_names)
    st.ensure_entities()
    st.deduplicate_entities()
# setup_entities()

def determine_blur_for_dencalc(st, grid):
    b_min = min((cra.atom.b_iso for cra in st[0].all()))
    eig_mins = [min(cra.atom.aniso.calculate_eigenvalues()) for cra in st[0].all() if cra.atom.aniso.nonzero()]
    if len(eig_mins) > 0: b_min = min(b_min, min(eig_mins) * 8*numpy.pi**2)

    b_need = grid**2*8*numpy.pi**2/1.1 # Refmac's way
    b_add = b_need - b_min
    return b_add
# determine_blur_for_dencalc()

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

def calc_fc_fft(st, d_min, source, mott_bethe=True, monlib=None, blur=None, cutoff=1e-7, rate=1.5,
                omit_proton=False, omit_h_electron=False, miller_array=None):
    assert source in ("xray", "electron", "neutron")
    if source != "electron": assert not mott_bethe
    if omit_proton or omit_h_electron:
        assert mott_bethe
        if not st[0].has_hydrogen():
            logger.writeln("WARNING: omit_proton/h_electron requested, but no hydrogen exists!")
            omit_proton = omit_h_electron = False
        elif omit_proton and omit_h_electron:
            logger.writeln("omit_proton and omit_h_electron requested. removing hydrogens")
            st = st.clone()
            st.remove_hydrogens()
            omit_proton = omit_h_electron = False
    
    if blur is None: blur = determine_blur_for_dencalc(st, d_min/2/rate)
    blur = max(0, blur) # negative blur may cause non-positive definite in case of anisotropic Bs
    logger.writeln("Setting blur= {:.2f} in density calculation (unblurred later)".format(blur))
        
    if mott_bethe and not omit_proton and monlib is not None and st[0].has_hydrogen():
        st = st.clone()
        topo = gemmi.prepare_topology(st, monlib, warnings=logger, ignore_unknown_links=True)
        resnames = st[0].get_all_residue_names()
        restraints.check_monlib_support_nucleus_distances(monlib, resnames)
        # Shift electron positions
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)
    else:
        topo = None
        
    if source == "xray" or mott_bethe:
        dc = gemmi.DensityCalculatorX()
    elif source == "electron":
        dc = gemmi.DensityCalculatorE()
    elif source == "neutron":
        dc = gemmi.DensityCalculatorN()
    else:
        raise RuntimeError("unknown source")

    dc.d_min = d_min
    dc.blur = blur
    dc.cutoff = cutoff
    dc.rate = rate
    dc.set_grid_cell_and_spacegroup(st)

    t_start = time.time()
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

        logger.writeln("Calculating {} using Mott-Bethe formula".format(method_str))
        
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
        mb_000 = sum_ab * 0.023933660963366372 # 1 / (8 * pi() * pi() * bohrradius()
    else:
        logger.writeln("Calculating Fc")
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
    if source != "electron": assert not mott_bethe

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
        dc.set_grid_cell_and_spacegroup(st)
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

def translate_into_box(st, origin=None):
    if origin is None: origin = gemmi.Position(0,0,0)
    
    # apply unit cell translations to put model into a box (unit cell)
    omat = numpy.array(st.cell.orthogonalization_matrix)
    fmat = numpy.array(st.cell.fractionalization_matrix).transpose()
    shifts = []
    for m in st:
        com = numpy.array((m.calculate_center_of_mass() - origin).tolist())
        shift = sum([omat[:,i]*numpy.floor(1-numpy.dot(com, fmat[:,i])) for i in range(3)])
        tr = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*shift))
        shifts.append(shift)
        m.transform_pos_and_adp(tr)
    return shifts
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

def expand_ncs(st, special_pos_threshold=0.01, howtoname=gemmi.HowToNameCopiedChain.Short):
    if len(st.ncs) == 0: return
    
    logger.writeln("Expanding symmetry..")
    # Take care of special positions
    if special_pos_threshold >= 0:
        # First expand ncs with Dup regardless of the choice
        st.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
        cra2key = lambda x: (x.chain.name, x.residue.seqid.num, x.residue.seqid.icode,
                             x.atom.name, x.atom.element.name, x.atom.altloc.replace("\0"," "))
        ns = gemmi.NeighborSearch(st[0], st.cell, 3).populate()
        cs = gemmi.ContactSearch(special_pos_threshold)
        #cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
        #cs.special_pos_cutoff_sq = special_pos_threshold
        results = cs.find_contacts(ns)

        # find overlaps between different segments
        pairs = {}
        cra_dict = {}
        for r in results:
            if r.partner1.residue.segment == r.partner2.residue.segment: continue
            key1, key2 = cra2key(r.partner1), cra2key(r.partner2)
            if key1 == key2:
                segi1, segi2 = int(r.partner1.residue.segment), int(r.partner2.residue.segment)
                pairs.setdefault(key1, []).append([segi1, segi2])
                cra_dict[key1+(segi1,)] = cra_to_atomaddress(r.partner1)
                cra_dict[key1+(segi2,)] = cra_to_atomaddress(r.partner2)

        if pairs: logger.writeln("Atoms on special position detected.")
        res_to_be_removed = []
        for key in sorted(pairs):
            logger.writeln(" Site: chain='{}' seq='{}{}' atom='{}' elem='{}' altloc='{}'".format(*key))
            # use graph to find connected components
            segs = sorted(set(sum(pairs[key], []))) # index->segid
            segd = dict([(s,i) for i,s in enumerate(segs)]) # reverse lookup
            g = numpy.zeros((len(segs),len(segs)), dtype=int)
            for p in pairs[key]:
                i, j = segd[p[0]], segd[p[1]]
                g[i,j] = g[j,i] = 1
            nc, labs = scipy.sparse.csgraph.connected_components(g, directed=False)
            groups = [[] for i in range(nc)] # list of segids
            for i, l in enumerate(labs): groups[l].append(segs[i])
            for group in groups:
                group.sort() # first segid will be kept
                sum_occ = sum([st[0].find_cra(cra_dict[key+(i,)]).atom.occ for i in group])
                logger.writeln("  multiplicity= {} occupancies_total= {:.2f} segids= {}".format(len(group), sum_occ, group))
                sum_pos = sum([st[0].find_cra(cra_dict[key+(i,)]).atom.pos for i in group], gemmi.Position(0,0,0))
                if len(group) < 2: continue # should never happen
                # modify first atom
                cra0 = st[0].find_cra(cra_dict[key+(group[0],)])
                cra0.atom.occ = max(1, sum_occ)
                cra0.atom.pos = sum_pos/len(group)
                # remove remaining atoms
                for g in group[1:]:
                    cra = st[0].find_cra(cra_dict[key+(g,)])
                    cra.residue.remove_atom(cra.atom.name, cra.atom.altloc, cra.atom.element)
                    if len(cra.residue) == 0: # empty residue needs to be removed
                        r_idx = [i for i, r in enumerate(cra.chain) if r==cra.residue]
                        res_to_be_removed.append((cra.chain, r_idx[0]))
                        
        chain_to_be_removed = []
        res_to_be_removed.sort(key=lambda x:([x[0].name, x[1]]))
        for chain, idx in reversed(res_to_be_removed):
            del chain[idx]
            if len(chain) == 0: # empty chain needs to be removed..
                c_idx = [i for i, c in enumerate(st[0]) if c==chain]
                chain_to_be_removed.append(c_idx[0])
        chain_to_be_removed.sort()
        for idx in reversed(chain_to_be_removed): #
            del st[0][idx] # we cannot use remove_chain() because ID may be duplicated

        # copy segment to subchain, as segment is not written to mmCIF file
        for chain in st[0]:
            for res in chain:
                res.subchain = res.segment
                
        # rename chain IDs
        if howtoname != gemmi.HowToNameCopiedChain.Dup:
            # want to keep original chain IDs
            for chain in st[0]:
                for res in chain:
                    if res.segment == "0": res.segment = ""

            # new id = original id + ncs id
            st[0].split_chains_by_segments(gemmi.HowToNameCopiedChain.Dup)
            if howtoname == gemmi.HowToNameCopiedChain.Short:
                st.shorten_chain_names()
    else:
        st.expand_ncs(howtoname)

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
    cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
    results = cs.find_contacts(ns)
    indices = set([r.image_idx for r in results])
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
            app("model", m.name)
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
        
    for m_name, dm in df.groupby("model"):
        st.add_model(gemmi.Model(m_name))
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
    st.add_model(gemmi.Model("1"))
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
    A = numpy.array(st.cell.orthogonalization_matrix.tolist())
    center = numpy.sum(A,axis=1) / 2
    center = gemmi.Vec3(*center)
    mat = gemmi.Mat33([[-1,0,0],[0,1,0],[0,0,1]]) 
    vec = mat.multiply(-center) + center
    tr = gemmi.Transform(mat, vec)
    st[0].transform_pos_and_adp(tr)

    # invert peptides
# invert_model()

def cx_to_mx(ss): #SmallStructure to Structure
    st = gemmi.Structure()
    st.spacegroup_hm = ss.spacegroup_hm
    st.cell = ss.cell
    st.add_model(gemmi.Model("1"))
    st[-1].add_chain(gemmi.Chain("A"))
    st[-1][-1].add_residue(gemmi.Residue())
    st[-1][-1][-1].seqid.num = 1
    st[-1][-1][-1].name = "00"

    ruc = ss.cell.reciprocal()
    cif2cart = ss.cell.orthogonalization_matrix.multiply_by_diagonal(gemmi.Vec3(ruc.a, ruc.b, ruc.c))
    as_smat33f = lambda x: gemmi.SMat33f(x.u11, x.u22, x.u33, x.u12, x.u13, x.u23)
    
    for site in ss.sites:
        st[-1][-1][-1].add_atom(gemmi.Atom())
        a = st[-1][-1][-1][-1]
        a.name = site.label
        a.aniso = as_smat33f(site.aniso.transformed_by(cif2cart))
        a.b_iso = site.u_iso * 8*numpy.pi**2
        #a.charge = ?
        a.element = site.element
        a.occ = site.occ
        a.pos = site.orth(ss.cell)
        
    return st
# cx_to_mx()
