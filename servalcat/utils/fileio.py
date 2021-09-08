"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import model
import os
import subprocess
import gemmi
import numpy
import gzip

def splitext(path):
    if path.endswith((".bz2",".gz")):
        return os.path.splitext(path[:path.rindex(".")])
    else:
        return os.path.splitext(path)
# splitext()

def check_model_format(xyzin):
    # TODO check format actually
    # TODO mmjson is possible?
    ext = splitext(xyzin)[1]
    if ext.endswith("cif"):
        return ".mmcif"
    else:
        return ".pdb"
# check_model_format()

def write_mmcif(st, cif_out, cif_ref=None):
    """
    Refmac fails if _entry.id is longer than 80 chars including quotations
    """
    st_new = st.clone()
    print("Writing mmCIF file:", cif_out)
    if cif_ref:
        print("  using mmCIF metadata from:", cif_ref)
        groups = gemmi.MmcifOutputGroups(False)
        groups.ncs = True
        groups.atoms = True
        groups.cell = True
        groups.scale = True
        groups.assembly = True
        # FIXME is this all? 
        try:
            doc = read_cif_safe(cif_ref)
        except Exception as e:
            # Sometimes refmac writes a broken mmcif file..
            logger.error("Error in mmCIF reading: {}".format(e))
            logger.error("  Give up using cif reference.")
            return write_mmcif(st, cif_out)
            
        blocks = list(filter(lambda b: b.find_loop("_atom_site.id"), doc))
        if len(blocks) == 0:
            logger.write("No _atom_site found in {}".format(cif_ref))
            logger.write("  Give up using cif reference.")
            return write_mmcif(st, cif_out)
        block = blocks[0]
        # to remove fract_transf_matrix. maybe we should keep some (like _atom_sites.solution_hydrogens)?
        # we do not want this because cell may be updated
        block.find_mmcif_category("_atom_sites.").erase()
        st_new.update_mmcif_block(block, groups)
        st_new.info["_entry.id"] = st_new.info["_entry.id"][:78]
        doc.write_file(cif_out)
    else:
        st_new.name = st_new.name[:78] # this will become _entry.id
        if "_entry.id" in st_new.info: st_new.info["_entry.id"] = st_new.info["_entry.id"][:78]
        st_new.make_mmcif_document().write_file(cif_out)
# write_mmcif()

def write_pdb(st, pdb_out):
    logger.write("Writing PDB file: {}".format(pdb_out))
    chain_id_lens = [len(x) for x in model.all_chain_ids(st)]
    if chain_id_lens and max(chain_id_lens) > 2:
        st = st.clone()
        st.shorten_chain_names()
    st.write_pdb(pdb_out, use_linkr=True)
# write_pdb()

def write_model(st, prefix=None, file_name=None, pdb=False, cif=False, cif_ref=None):
    if file_name:
        if file_name.endswith("cif"):
            write_mmcif(st, file_name, cif_ref)
        elif file_name.endswith(".pdb"):
            write_pdb(st, file_name)
        else:
            raise Exception("Cannot determine file format from file name: {}".format(file_name))
    else:            
        if cif:
            write_mmcif(st, prefix+".mmcif", cif_ref)
        if pdb:
            write_pdb(st, prefix+".pdb")
# write_model()

def read_shifts_txt(shifts_txt):
    ret = {}
    s = open(shifts_txt).read()
    s = s.replace("\n"," ").split()
    for i in range(len(s)-3):
        if s[i] in ("pdbin", "pdbout") and s[i+1] in ("cell", "shifts"):
            n = 6 if s[i+1] == "cell" else 3
            ret["{} {}".format(s[i], s[i+1])] = [float(x) for x in s[i+2:i+2+n]]

    return ret
# read_shifts_txt()

def read_ccp4_map(filename, setup=True, default_value=0., pixel_size=None):
    m = gemmi.read_ccp4_map(filename)
    g = m.grid
    grid_cell = [m.header_i32(x) for x in (8,9,10)]
    grid_start = [m.header_i32(x) for x in (5,6,7)]
    grid_shape = [m.header_i32(x) for x in (1,2,3)]
    axis_pos = m.axis_positions()
    axis_letters = ["","",""]
    for i, l in zip(axis_pos, "XYZ"): axis_letters[i] = l
    spacings = [1./g.unit_cell.reciprocal().parameters[i]/grid_cell[i] for i in (0,1,2)]
    voxel_size = [g.unit_cell.parameters[i]/grid_cell[i] for i in (0,1,2)]
    label = m.header_str(57, 80)
    label = label[:label.find("\0")]
    logger.write("Reading CCP4/MRC map file {}".format(filename))
    logger.write("   Cell Grid: {:4d} {:4d} {:4d}".format(*grid_cell))
    logger.write("    Map mode: {}".format(m.header_i32(4)))
    logger.write("       Start: {:4d} {:4d} {:4d}".format(*grid_start))
    logger.write("       Shape: {:4d} {:4d} {:4d}".format(*grid_shape))
    logger.write("        Cell: {} {} {} {} {} {}".format(*g.unit_cell.parameters))
    logger.write("  Axis order: {}".format(" ".join(axis_letters)))
    logger.write(" Space group: {}".format(m.header_i32(23)))
    logger.write("     Spacing: {:.6f} {:.6f} {:.6f}".format(*spacings))
    logger.write("  Voxel size: {:.6f} {:.6f} {:.6f}".format(*voxel_size))
    logger.write("       Label: {}".format(label))
    logger.write("")

    if setup:
        if default_value is None: default_value = float("nan")
        m.setup(default_value)
        grid_start = [grid_start[i] for i in axis_pos]
        
    if pixel_size is not None:
        try:
            len(pixel_size)
        except TypeError:
            pixel_size = [pixel_size, pixel_size, pixel_size]
            
        logger.write("Overriding pixel size with {:.6f} {:.6f} {:.6f}".format(*pixel_size))
        orgc = m.grid.unit_cell.parameters
        new_abc = [orgc[i]*pixel_size[i]/voxel_size[i] for i in (0,1,2)]
        m.grid.unit_cell = gemmi.UnitCell(new_abc[0], new_abc[1], new_abc[2],
                                          orgc[3], orgc[4], orgc[5])
        logger.write(" New cell= {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(*m.grid.unit_cell.parameters))

    return [m.grid, grid_start] # TODO should return grid_shape so that the same region can be written
# read_ccp4_map()

def read_map_from_mtz(mtz_in, cols, grid_size=None, sample_rate=3):
    mtz = gemmi.read_mtz_file(mtz_in)
    d_min = mtz.resolution_high() # TODO get resolution for column?
    if grid_size is None:
        grid_size = mtz.get_size_for_hkl(sample_rate=sample_rate)
    F = mtz.get_f_phi_on_grid(cols[0], cols[1], grid_size)
    m = gemmi.transform_f_phi_grid_to_map(F)
    return d_min, m
# read_map_from_mtz()

def read_asu_data_from_mtz(mtz_in, cols):
    assert 0 < len(cols) < 3
    mtz = gemmi.read_mtz_file(mtz_in)
    sg = mtz.spacegroup
    miller = mtz.make_miller_array()
    f = mtz.column_with_label(cols[0])
    cell = mtz.get_cell(f.dataset_id)
    if len(cols) == 2:
        phi = mtz.column_with_label(cols[1])
        assert f.type == "F"
        assert phi.type == "P"
        phi = numpy.deg2rad(phi)
        f_comp = f * (numpy.cos(phi) + 1j * numpy.sin(phi))
        asu = gemmi.ComplexAsuData(cell, sg, miller, f_comp)
        return asu
    else:
        if f.is_integer():
            gr_t = gemmi.IntAsuData
        else:
            gr_t = gemmi.FloatAsuData
        
        asu = gr_t(cell, sg, miller, f)
        return asu
# read_asu_data_from_mtz()

def read_cif_safe(cif_in):
    ifs = gzip.open(cif_in, "rt") if cif_in.endswith(".gz") else open(cif_in)
    s = ifs.read()
    if "\0" in s: # Refmac occasionally writes \0 in some fileds..
        logger.write(" WARNING: null character detected. Replacing with '.'")
        s = s.replace("\0", ".")
    doc = gemmi.cif.read_string(s)
    return doc
# read_cif_safe()

def read_structure(xyz_in):
    spext = splitext(xyz_in)
    if spext[1].lower() in (".pdb", ".ent"):
        logger.write("Reading PDB file: {}".format(xyz_in))
        return gemmi.read_pdb(xyz_in)
    elif spext[1].lower() in (".cif", ".mmcif"):
        logger.write("Reading mmCIF file: {}".format(xyz_in))
        doc = read_cif_safe(xyz_in)
        blocks = list(filter(lambda b: b.find_loop("_atom_site.id"), doc))
        if len(blocks) == 0:
            raise RuntimeError("No block having _atom_site found")
        if len(blocks) > 1:
            logger.write(" WARNING: more than one block having _atom_site found. Will use first one.")
        return gemmi.make_structure_from_block(blocks[0])
    else:
        raise RuntimeError("Unsupported file type: {}".format(spext[1]))
# read_structure()

def read_structure_from_pdb_and_mmcif(xyz_in):
    st = read_structure(xyz_in)
    cif_ref = None
    spext = splitext(xyz_in)
    if spext[1] in (".pdb", ".ent"):
        cif_in = spext[0] + ".mmcif"
        if os.path.isfile(cif_in):
            print(" Will use mmcif metadata from {}".format(cif_in))
            cif_ref = cif_in
    elif spext[1] in (".cif", ".mmcif"):
        cif_ref = xyz_in
        pdb_in = spext[0] + ".pdb"
        if os.path.isfile(pdb_in):
            print(" Reading PDB REMARKS from {}".format(pdb_in))
            tmp = gemmi.read_structure(pdb_in)
            st.raw_remarks = tmp.raw_remarks

    if cif_ref is None and xyz_in.endswith("cif"):
        cif_ref = xyz_in
            
    return st, cif_ref
# read_structure_from_pdb_and_mmcif()

def merge_ligand_cif(cifs_in, cif_out):
    # TODO Check duplication?
    
    docs = [gemmi.cif.read(x) for x in cifs_in]
    tags = dict(comp=["_chem_comp.id"],
                link=["_chem_link.id"],
                mod=["_chem_mod.id"])
    found = dict(comp=0, link=0, mod=0)
    for d in docs:
        for k in tags:
            b = d.find_block("{}_list".format(k))
            if not b: continue
            found[k] += 1
            l = b.find_loop(tags[k][0]).get_loop()
            for t in l.tags:
                if t not in tags[k]: tags[k].append(t)
  
    doc = gemmi.cif.Document()
    for k in tags:
        if not found[k]: continue
        lst = doc.add_new_block("{}_list".format(k))
        loop = lst.init_loop("", tags[k])
        tags_for_find = [tags[k][0]] + ["?"+x for x in tags[k][1:]]
        
        for d in docs:
            b = d.find_block("{}_list".format(k))
            if not b: continue
            vals = b.find(tags_for_find)
            for v in vals:
                rl = [v.get(x) if v.has(x) else "." for x in range(len(tags[k]))]
                loop.add_row(rl)

    for d in docs:
        for b in d:
            if not b.name.endswith("_list"):
                doc.add_copied_block(b)

    doc.write_file(cif_out)
# merge_ligand_cif()
