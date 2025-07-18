"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import model
from servalcat.utils import hkl
from servalcat.utils import restraints
import os
import shutil
import glob
import re
import subprocess
import gemmi
import numpy
import gzip
import traceback

def splitext(path):
    if path.endswith((".bz2",".gz")):
        return os.path.splitext(path[:path.rindex(".")])
    else:
        return os.path.splitext(path)
# splitext()

def rotate_file(filename, copy=False):
    if not os.path.exists(filename): return

    # make list [ [filename, number], ... ]
    old_list = []
    dot_files = glob.glob(filename + ".*")
    for f in dot_files:
        suffix = f.replace(filename+".", "")
        try:
            i = int(suffix)
            if str(i) == suffix: # ignore if suffix was such as 003...
                old_list.append([f, i])
        except ValueError as e:
            continue

    old_list.sort(key=lambda x: x[1])

    # rotate files
    for f, i in reversed(old_list):
        logger.writeln("Rotating file: {}".format(f))
        os.rename(f, "%s.%d" % (f[:f.rfind(".")], i+1))

    if copy:
        shutil.copyfile(filename, filename + ".1")
    else:
        os.rename(filename, filename + ".1")

    return filename + ".1"
# rotate_file()

def check_model_format(xyzin):
    # TODO check format actually
    # TODO mmjson is possible?
    ext = splitext(xyzin)[1]
    if ext.endswith("cif"):
        return ".mmcif"
    else:
        return ".pdb"
# check_model_format()

def write_mmcif(st, cif_out, cif_ref=None, cif_ref_doc=None):
    """
    Refmac fails if _entry.id is longer than 80 chars including quotations
    """
    st_new = st.clone()
    logger.writeln("Writing mmCIF file: {}".format(cif_out))
    if cif_ref or cif_ref_doc:
        if cif_ref:
            logger.writeln("  using mmCIF metadata from: {}".format(cif_ref))
        groups = gemmi.MmcifOutputGroups(False)
        groups.group_pdb = True
        groups.ncs = True
        groups.atoms = True
        groups.cell = True
        groups.scale = True
        groups.assembly = True
        groups.entity = True
        groups.entity_poly = True
        groups.entity_poly_seq = True
        groups.cis = True
        groups.conn = True
        groups.software = True
        groups.auth_all = True
        # FIXME is this all?
        if cif_ref:
            try:
                cif_ref_doc = read_cif_safe(cif_ref)
            except Exception as e:
                # Sometimes refmac writes a broken mmcif file..
                logger.error("Error in mmCIF reading: {}".format(e))
                logger.error("  Give up using cif reference.")
                return write_mmcif(st, cif_out)
            
        blocks = list(filter(lambda b: b.find_loop("_atom_site.id"), cif_ref_doc))
        if len(blocks) == 0:
            logger.writeln("No _atom_site found in reference")
            logger.writeln("  Give up using cif reference.")
            return write_mmcif(st, cif_out)
        block = blocks[0]
        # to remove fract_transf_matrix. maybe we should keep some (like _atom_sites.solution_hydrogens)?
        # we do not want this because cell may be updated
        block.find_mmcif_category("_atom_sites.").erase()
        st_new.update_mmcif_block(block, groups)
        if "_entry.id" in st_new.info: st_new.info["_entry.id"] = st_new.info["_entry.id"][:78]
        cif_ref_doc.write_file(cif_out, options=gemmi.cif.Style.Aligned)
    else:
        st_new.name = st_new.name[:78] # this will become _entry.id
        if "_entry.id" in st_new.info: st_new.info["_entry.id"] = st_new.info["_entry.id"][:78]
        groups = gemmi.MmcifOutputGroups(True, auth_all=True)
        doc = gemmi.cif.Document()
        block = doc.add_new_block("new")
        st_new.update_mmcif_block(block, groups)
        doc.write_file(cif_out, options=gemmi.cif.Style.Aligned)
# write_mmcif()

def write_pdb(st, pdb_out):
    logger.writeln("Writing PDB file: {}".format(pdb_out))
    st = st.clone()
    chain_id_lens = [len(x) for x in model.all_chain_ids(st)]
    if chain_id_lens and max(chain_id_lens) > 2:
        st.shorten_chain_names()
    st.shorten_ccd_codes()
    if st.shortened_ccd_codes:
        msg = " ".join("{}->{}".format(o,n) for o,n in st.shortened_ccd_codes)
        logger.writeln(" Using shortened residue names in the output pdb file: " + msg)
    st.write_pdb(pdb_out, use_linkr=True)
# write_pdb()

def write_model(st, prefix=None, file_name=None, pdb=False, cif=False, cif_ref=None, hout=True):
    if not hout and st[0].has_hydrogen():
        st = st.clone()
        st.remove_hydrogens()
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
    with open(shifts_txt) as f:
        s = f.read()
    s = s.replace("\n"," ").split()
    for i in range(len(s)-3):
        if s[i] in ("pdbin", "pdbout") and s[i+1] in ("cell", "shifts"):
            n = 6 if s[i+1] == "cell" else 3
            ret["{} {}".format(s[i], s[i+1])] = [float(x) for x in s[i+2:i+2+n]]

    return ret
# read_shifts_txt()

def read_ccp4_map(filename, header_only=False, setup=True, default_value=0., pixel_size=None, ignore_origin=True):
    if header_only:
        m = gemmi.read_ccp4_header(filename)
    else:
        m = gemmi.read_ccp4_map(filename)
    grid_cell = [m.header_i32(x) for x in (8,9,10)]
    grid_start = [m.header_i32(x) for x in (5,6,7)]
    grid_shape = [m.header_i32(x) for x in (1,2,3)]
    axis_pos = m.axis_positions()
    axis_letters = ["","",""]
    for i, l in zip(axis_pos, "XYZ"): axis_letters[i] = l
    cell = gemmi.UnitCell(*(m.header_float(x) for x in range(11,17)))
    spacings = [1./cell.reciprocal().parameters[i]/grid_cell[i] for i in (0,1,2)]
    voxel_size = [cell.parameters[i]/grid_cell[i] for i in (0,1,2)]
    origin = [m.header_float(x) for x in (50,51,52)]
    label = m.header_str(57, 80)
    label = label[:label.find("\0")]
    logger.writeln("Reading CCP4/MRC map file {}".format(filename))
    logger.writeln("   Cell Grid: {:4d} {:4d} {:4d}".format(*grid_cell))
    logger.writeln("    Map mode: {}".format(m.header_i32(4)))
    logger.writeln("       Start: {:4d} {:4d} {:4d}".format(*grid_start))
    logger.writeln("       Shape: {:4d} {:4d} {:4d}".format(*grid_shape))
    logger.writeln("        Cell: {} {} {} {} {} {}".format(*cell.parameters))
    logger.writeln("  Axis order: {}".format(" ".join(axis_letters)))
    logger.writeln(" Space group: {}".format(m.header_i32(23)))
    logger.writeln("     Spacing: {:.6f} {:.6f} {:.6f}".format(*spacings))
    logger.writeln("  Voxel size: {:.6f} {:.6f} {:.6f}".format(*voxel_size))
    logger.writeln("      Origin: {:.6e} {:.6e} {:.6e}".format(*origin))
    if not numpy.all(numpy.asarray(origin) == 0.):
        logger.writeln("             ! WARNING: ORIGIN header is not supported.")
        if ignore_origin:
            logger.writeln("             ! WARNING: removing ORIGIN values. This might cause a misalignment between map and model.")
            for i in (50,51,52): m.set_header_float(i, 0.)
    logger.writeln("       Label: {}".format(label))
    logger.writeln("")

    if header_only:
        grid = gemmi.FloatGrid(*grid_cell if setup else grid_shape) # waste of memory, but unavoidable for now
        grid.set_unit_cell(cell)
        grid.spacegroup = gemmi.find_spacegroup_by_number(m.header_i32(23))
    else:
        grid = m.grid

    if setup:
        if not header_only:
            if default_value is None: default_value = float("nan")
            m.setup(default_value)
        grid_start = [grid_start[i] for i in axis_pos]
        
    if pixel_size is not None:
        try:
            len(pixel_size)
        except TypeError:
            pixel_size = [pixel_size, pixel_size, pixel_size]
            
        logger.writeln("Overriding pixel size with {:.6f} {:.6f} {:.6f}".format(*pixel_size))
        orgc = grid.unit_cell.parameters
        new_abc = [orgc[i]*pixel_size[i]/voxel_size[i] for i in (0,1,2)]
        new_cell = gemmi.UnitCell(new_abc[0], new_abc[1], new_abc[2], orgc[3], orgc[4], orgc[5])
        grid.set_unit_cell(new_cell)
        logger.writeln(" New cell= {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(*grid.unit_cell.parameters))

    return [grid, grid_start, grid_shape]
# read_ccp4_map()

def read_halfmaps(files, pixel_size=None, fail=True):
    if fail and len(files) != 2:
        raise SystemExit("Error: Give exactly two files for half maps")
    maps = [read_ccp4_map(f, pixel_size=pixel_size) for f in files]
    if numpy.array_equal(maps[0][0].array, maps[1][0].array):
        raise SystemExit("Error: Half maps have exactly the same values. Check your input.")
    
    assert maps[0][0].shape == maps[1][0].shape
    assert maps[0][0].unit_cell == maps[1][0].unit_cell
    assert maps[0][1] == maps[1][1]
    
    return maps
# read_halfmaps()

def read_mmhkl(hklin, cif_index=0): # mtz or mmcif
    spext = splitext(hklin)
    if spext[1].lower() == ".mtz":
        logger.writeln("Reading MTZ file: {}".format(hklin))
        mtz = gemmi.read_mtz_file(hklin)
    elif spext[1].lower() in (".cif", ".ent"):
        logger.writeln("Reading mmCIF file (hkl data): {} at index {}".format(hklin, cif_index+1))
        doc = gemmi.cif.read(hklin)
        blocks = gemmi.as_refln_blocks(doc)
        cif2mtz = gemmi.CifToMtz()
        mtz = cif2mtz.convert_block_to_mtz(blocks[cif_index])
    else:
        raise RuntimeError("Unsupported file type: {}".format(spext[1]))
    if mtz.spacegroup is None:
        raise RuntimeError("Missing space group information")
    logger.writeln("    Unit cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*mtz.cell.parameters))
    logger.writeln("  Space group: {}".format(mtz.spacegroup.xhm()))
    logger.writeln("      Columns: {}".format(" ".join(mtz.column_labels())))
    logger.writeln("")
    return mtz
# read_mmhkl()

def is_mmhkl_file(hklin):
    spext = splitext(hklin)
    if spext[1].lower() == ".mtz":
        return True
    if spext[1].lower() == ".hkl": # macromolecule files should not have .hkl extension
        return False
    if spext[1].lower() in (".cif", ".ent"):
        for b in gemmi.cif.read(hklin):
            if b.find_values("_refln.index_h"):
                return True
            if b.find_values("_refln_index_h"):
                return False
    # otherwise cannot decide
# is_smhkl()

def software_items_from_mtz(hklin):
    try:
        if type(hklin) is gemmi.Mtz:
            mtz = hklin
        elif splitext(hklin)[1].lower() != ".mtz":
            return []
        else:
            mtz = gemmi.read_mtz_file(hklin, with_data=False)
        return gemmi.get_software_from_mtz_history(mtz.history)
    except:
        logger.writeln(f"Failed to read software info from {hklin}")
        logger.writeln(traceback.format_exc())
        return []
# software_items_from_mtz()

def read_map_from_mtz(mtz_in, cols, grid_size=None, sample_rate=3):
    mtz = read_mmhkl(mtz_in)
    d_min = mtz.resolution_high() # TODO get resolution for column?
    if grid_size is None:
        grid_size = mtz.get_size_for_hkl(sample_rate=sample_rate)
    F = mtz.get_f_phi_on_grid(cols[0], cols[1], grid_size)
    m = gemmi.transform_f_phi_grid_to_map(F)
    return d_min, m
# read_map_from_mtz()

def read_asu_data_from_mtz(mtz_in, cols):
    assert 0 < len(cols) < 3
    mtz = read_mmhkl(mtz_in)
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
        asu = gemmi.ComplexAsuData(cell, sg, miller, f_comp) # ensure asu?
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
    with gzip.open(cif_in, "rt") if cif_in.endswith(".gz") else open(cif_in) as ifs:
        s = ifs.read()
    if "\0" in s: # Refmac occasionally writes \0 in some fields..
        logger.writeln(" WARNING: null character detected. Replacing with '.'")
        s = s.replace("\0", ".")
    doc = gemmi.cif.read_string(s)
    return doc
# read_cif_safe()

def read_structure(xyz_in, assign_het_flags=True, merge_chain_parts=True, ignore_ter=True):
    spext = splitext(xyz_in)
    st = None
    if spext[1].lower() in (".pdb", ".ent"):
        logger.writeln("Reading PDB file: {}".format(xyz_in))
        st = gemmi.read_pdb(xyz_in, ignore_ter=ignore_ter)
    elif spext[1].lower() in (".cif", ".mmcif"):
        doc = read_cif_safe(xyz_in)
        for block in doc:
            if block.find_loop("_atom_site.id"):
                if st is None:
                    logger.writeln("Reading mmCIF file: {}".format(xyz_in))
                    st =  gemmi.make_structure_from_block(block)
                else:
                    logger.writeln(" WARNING: more than one block having structure found. Will use first one.")
                    break
            elif block.find_loop("_atom_site_label"):
                if st is None:
                    logger.writeln("Reading smCIF file: {}".format(xyz_in))
                    ss = gemmi.read_small_structure(xyz_in)
                    if not ss.sites:
                        raise RuntimeError("No atoms found in cif file.")
                    st = model.cx_to_mx(ss)
                else:
                    logger.writeln(" WARNING: more than one block having structure found. Will use first one.")
                    break
            elif (block.find_loop("_chem_comp_atom.x") or
                  block.find_loop("_chem_comp_atom.model_Cartn_x") or
                  block.find_loop("_chem_comp_atom.pdbx_model_Cartn_x_ideal")):
                if st is None:
                    logger.writeln("Reading chemical component file: {}".format(xyz_in))
                    st = gemmi.make_structure_from_chemcomp_block(block)
                    for i in range(len(st)-1):
                        del st[1]
    elif spext[1].lower() in (".ins", ".res"):
        logger.writeln("Reading SHELX ins/res file: {}".format(xyz_in))
        st = model.cx_to_mx(read_shelx_ins(ins_in=xyz_in)[0])
        st.setup_cell_images()
    else:
        raise RuntimeError("Unsupported file type: {}".format(spext[1]))
    if st is not None:
        if st.cell.is_crystal():
            logger.writeln("    Unit cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*st.cell.parameters))
            logger.writeln("  Space group: {}".format(st.spacegroup_hm))
        if st.ncs:
            n_given = sum(1 for x in st.ncs if x.given)
            logger.writeln(" No. strict NCS: {} ({} already applied)".format(len(st.ncs), n_given))
        logger.writeln("")
    if assign_het_flags:
        st.assign_het_flags()
    if merge_chain_parts:
        st.merge_chain_parts()
    return st
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
    docs = [gemmi.cif.read(x) for x in cifs_in]
    tags = dict(comp=["_chem_comp.id"],
                link=["_chem_link.id"],
                mod=["_chem_mod.id"])
    list_names = [k+"_list" for k in tags]

    # Check duplicated block names
    names = {}
    for i, doc in enumerate(docs):
        for j, b in enumerate(doc):
            if b.name not in list_names and not b.name.startswith("mod_"):
                names.setdefault(b.name, []).append((i,j))

    # Keep only last one if duplicated
    todel = []
    for k in names:
        if len(names[k]) > 1:
            for i,j in reversed(names[k][:-1]):
                logger.writeln("WARNING: removing duplicated {} from {}".format(k, cifs_in[i]))
                todel.append((i,j))
                for t in "comp", "link":
                    if k.startswith("{}_".format(t)):
                        comp_list = docs[i].find_block("{}_list".format(t))
                        table = comp_list.find("_chem_{}.".format(t), ["id"])
                        for l in reversed([l for l, row in enumerate(table) if row.str(0) == k[5:]]):
                            table.remove_row(l)

    for i,j in sorted(todel, reverse=True):
        del docs[i][j]

    # Accumulate list
    found = dict(comp=0, link=0, mod=0)
    for d in docs:
        for k in tags:
            b = d.find_block("{}_list".format(k))
            if not b: continue
            found[k] += 1
            l = b.find_loop(tags[k][0]).get_loop()
            for t in l.tags:
                if t not in tags[k]: tags[k].append(t)

    # Check duplicated modifications
    known_mods = [] # need to check monomer library?
    for d in docs:
        restraints.rename_cif_modification_if_necessary(d, known_mods)
        mod_list = d.find_block("mod_list")
        if mod_list:
            for row in mod_list.find("_chem_mod.", ["id"]):
                known_mods.append(row.str(0))
        
    doc = gemmi.cif.Document()
    # Add lists
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

    # Add other items
    for d in docs:
        for b in d:
            if b.name not in list_names:
                doc.add_copied_block(b)

    doc.write_file(cif_out, options=gemmi.cif.Style.Aligned)
# merge_ligand_cif()

def read_shelx_ins(ins_in=None, lines_in=None, ignore_q_peaks=True): # TODO support gz?
    assert (ins_in, lines_in).count(None) == 1
    ss = gemmi.SmallStructure()

    keywords = """
    TITL CELL ZERR LATT SYMM SFAC NEUT DISP UNIT LAUE REM  MORE END  HKLF OMIT SHEL BASF TWIN TWST EXTI SWAT
    ABIN ANSC ANSR MERG LIST SPEC RESI MOVE ANIS AFIX HFIX FRAG FEND EXYZ EADP EQIV CONN PART BIND FREE DFIX
    DANG BUMP SAME SADI CHIV FLAT DELU SIMU RIGU PRIG DEFS ISOR XNPD NCSY SUMP L.S. CGLS BLOC DAMP STIR WGHT
    FVAR WIGL BOND CONF MPLA RTAB HTAB ACTA SIZE TEMP WPDB FMAP GRID PLAN TIME HOPE MOLE
    """.split()
    re_kwd = re.compile("^({})_?".format("|".join(keywords)))
    
    # remove comments/blanks and concatenate lines
    lines = []
    concat_flag = False
    if ins_in:
        with open(ins_in) as f:
            lines_in = f.readlines()
    for l in lines_in:
        l = l.rstrip()
        if l.startswith("REM"): continue
        if l.startswith(";"): continue
        if not l.strip(): continue

        if l.endswith("="):
            l = l[:l.rindex("=")]
            if concat_flag:
                lines[-1] += l
            else:
                lines.append(l)
            concat_flag = True
        elif concat_flag:
            lines[-1] += l
            concat_flag = False
        else:
            lines.append(l)

    # parse lines
    sfacs = []
    latt, symms = 1, []
    fvar = []
    prev_free_u_iso = -1
    info = dict(hklf=0)
    cif2cart = None
    for l in lines:
        sp = l.split()
        ins = sp[0].upper()
        if ins == "TITL":
            pass
        elif l.startswith(" "): # title continued? instructions after space is allowed??
            pass
        elif ins == "CELL":
            #ss.wavelength = float(sp[1]) # next gemmi ver.
            ss.cell.set(*map(float, sp[2:]))
            cif2cart = model.cif2cart_matrix(ss.cell)
        elif ins == "LATT":
            latt = int(sp[1])
        elif ins == "SYMM":
            symms.append(gemmi.Op("".join(sp[1:])).wrap())
        elif ins == "SFAC": # TODO check numbers?
            if len(sp) < 2: continue
            sfacs.append(gemmi.Element(sp[1]))
            if len(sp) > 2:
                try: float(sp[2])
                except ValueError:
                    sfacs.extend([gemmi.Element(x) for x in sp[2:]])
        elif ins == "HKLF":
            info["hklf"] = int(sp[1])
        elif ins == "FVAR":
            fvar = list(map(float, sp[1:]))
        elif not re_kwd.search(ins):
            if not 4 < len(sp) < 13:
                logger.writeln("cannot parse this line: {}".format(l))
                continue
            site = gemmi.SmallStructure.Site()
            site.label = sp[0]
            try:
                site.element = sfacs[int(sp[1])-1]
            except:
                logger.error("failed to parse: {}".format(l))
                continue

            if site.label.startswith("Q") and ignore_q_peaks:
                logger.writeln("skip Q peak: {}".format(l))
                continue
            
            site.fract.fromlist(list(map(float, sp[2:5])))
            if len(sp) > 5:
                q_code = float(sp[5])
                # decompose q_code = 10 * m + p, where -5 < p <= 5 and m is an integer.
                m, p = divmod(q_code, 10.0)
                m = int(m)
                if p > 5.0:
                    p -= 10.0
                    m += 1
                if abs(m) > 1: # reference to an FVAR
                    if abs(m) > len(fvar):
                        logger.error("this line references an undefined FVAR: {}".format(l))
                    if m < 0:
                        # Here the SHELXL manual contradicts itself.
                        # It says -20.25 is m = -2, p = -0.25 but interprets it as 0.25 * (1 - fv2).
                        occ = (1 - fvar[-m - 1]) * -p
                    else:
                        occ = fvar[m - 1] * p
                else:
                    occ = p

                site.occ = occ

            if len(sp) > 11:
                u = list(map(float, sp[6:12]))
                site.aniso = gemmi.SMat33d(u[0], u[1], u[2], u[5], u[4], u[3])
                if cif2cart is None:
                    logger.writeln("WARNING: cannot calculate u_eq")
                    site.u_iso = sum(u[:3]) / 3.
                else:
                    site.u_iso = site.aniso.transformed_by(cif2cart).trace() / 3

                prev_free_u_iso = site.u_iso
                logger.writeln(f"updated prev_free_u_iso to {site.u_iso} at {site.label}")
            else:
                u_iso_code = float(sp[6])
                if -5 < u_iso_code and u_iso_code < -0.5:
                    if prev_free_u_iso > 0:
                        site.u_iso = -u_iso_code * prev_free_u_iso
#                        print(f"{prev_free_u_iso} * {-u_iso_code} = {site.u_iso} at {site.label}")
                    else:
                        logger.writeln(f"WARNING: parent atom not found for {site.label}")
                elif u_iso_code > 0:
                    site.u_iso = u_iso_code
                    prev_free_u_iso = site.u_iso
#                    print(f"updated prev_free_u_iso at {site.label}")
                else:
                    logger.writeln(f"WARNING: negative Ueq outside the (-0.5, -5) range for {site.label}")

            ss.add_site(site)

    # Determine space group
    if gemmi.Op() not in symms: # identity operator may not be present in ins file
        symms.append(gemmi.Op())

    lops = {1: [], # P
            2: [gemmi.Op("x+1/2,y+1/2,z+1/2")], # I
            3: [gemmi.Op("x+2/3,y+1/3,z+1/3"),  # R
                gemmi.Op("x+1/3,y+2/3,z+2/3")],
            4: [gemmi.Op("x,y+1/2,z+1/2"),      # F
                gemmi.Op("x+1/2,y,z+1/2"),
                gemmi.Op("x+1/2,y+1/2,z")],
            5: [gemmi.Op("x,y+1/2,z+1/2")],     # A
            6: [gemmi.Op("x+1/2,y,z+1/2")],     # B
            7: [gemmi.Op("x+1/2,y+1/2,z")],     # C
            }
    for op in lops[abs(latt)]:
        symms.extend([x*op for x in symms])
    if latt > 0:
        symms.extend([x*gemmi.Op("-x,-y,-z") for x in symms])

    ss.symops = [op.triplet() for op in set(symms)]
    ss.determine_and_set_spacegroup("s")
    # in case of non-regular setting, gemmi.SpaceGroup cannot be constructed anyway.
    if ss.spacegroup is None:
        raise RuntimeError("Cannot construct space group from symbols: {}".format(ss.symops))
    return ss, info
# read_shelx_ins()

def read_shelx_hkl(cell, sg, hklf, file_in=None, lines_in=None):
    assert (file_in, lines_in).count(None) == 1
    hkls, vals, sigs = [], [], []
    if file_in:
        with open(file_in) as f:
            lines_in = f.readlines()
    for l in lines_in:
        if l.startswith(";"): continue
        if not l.strip() or len(l) < 25: continue
        try:
            hkl = int(l[:4]), int(l[4:8]), int(l[8:12])
        except ValueError:
            logger.writeln("Error while parsing HKL part: {}".format(l))
            break
            
        if hkl == (0,0,0): break
        hkls.append(hkl)
        vals.append(float(l[12:20]))
        sigs.append(float(l[20:28]))
        # batch = l[28:32]
        # wavelength = l[32:40]

    ints = gemmi.Intensities()
    ints.set_data(cell, sg, numpy.asarray(hkls), numpy.asarray(vals), numpy.asarray(sigs))
    ints.merge_in_place(gemmi.DataType.Anomalous)
    if not (ints.isign_array < 0).any(): ints.type = gemmi.DataType.Mean
    logger.writeln(" Multiplicity: max= {} mean= {:.1f} min= {}".format(numpy.max(ints.nobs_array),
                                                                     numpy.mean(ints.nobs_array),
                                                                     numpy.min(ints.nobs_array)))
    mtz = ints.prepare_merged_mtz(with_nobs=False)
    if hklf == 3:
        conv = {"IMEAN": ("FP", "F"),
                "SIGIMEAN": ("SIGFP", "Q"),
                "I(+)": ("F(+)", "G"),
                "SIGI(+)": ("SIGF(+)", "L"),
                "I(-)": ("F(-)", "G"),
                "SIGI(-)": ("SIGF(-)", "L"),
                }
        for col in mtz.columns:
            if col.label in conv:
                col.label, col.type = conv[col.label]
    return mtz
# read_shelx_hkl()

def read_smcif_hkl(cif_in, cell_if_absent=None, sg_if_absent=None):
    # Very crude support for smcif - just because I do not know other varieties.
    # TODO other possible data types? (amplitudes?)
    # TODO check _refln_observed_status?
    logger.writeln("Reading hkl data from smcif: {}".format(cif_in))
    b = gemmi.cif.read(cif_in).sole_block()
    try:
        cell_par = [float(b.find_value("_cell_length_{}".format(x))) for x in ("a", "b", "c")]
        cell_par += [float(b.find_value("_cell_angle_{}".format(x))) for x in ("alpha", "beta", "gamma")]
        cell = gemmi.UnitCell(*cell_par)
        logger.writeln("    Unit cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*cell.parameters))
    except:
        logger.writeln(" WARNING: no unit cell in this file")
        cell = cell_if_absent

    for optag in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
        ops = [gemmi.Op(gemmi.cif.as_string(x)) for x in b.find_loop(optag)]
        sg = gemmi.find_spacegroup_by_ops(gemmi.GroupOps(ops))
        if sg:
            logger.writeln("  Space group: {}".format(sg.xhm()))
            break
    else:
        sg = sg_if_absent

    if cell is None or sg is None:
        raise RuntimeError("Cell and/or symmetry operations not found in {}".format(cif_in))
        
    l = b.find_values("_refln_index_h").get_loop()
    i_hkl = [l.tags.index("_refln_index_{}".format(h)) for h in "hkl"]
    i_int = l.tags.index("_refln_F_squared_meas")
    i_sig = l.tags.index("_refln_F_squared_sigma")
    hkls, vals, sigs = [], [], []
    for i in range(l.length()):
        hkl = [gemmi.cif.as_int(l[i, j]) for j in i_hkl]
        hkls.append(hkl)
        vals.append(gemmi.cif.as_number(l[i, i_int]))
        sigs.append(gemmi.cif.as_number(l[i, i_sig]))
    
    ints = gemmi.Intensities()
    ints.set_data(cell, sg, numpy.asarray(hkls), numpy.asarray(vals), numpy.asarray(sigs))
    ints.merge_in_place(gemmi.DataType.Anomalous)
    if not (ints.isign_array < 0).any(): ints.type = gemmi.DataType.Mean
    logger.writeln(" Multiplicity: max= {} mean= {:.1f} min= {}".format(numpy.max(ints.nobs_array),
                                                                     numpy.mean(ints.nobs_array),
                                                                     numpy.min(ints.nobs_array)))
    logger.writeln("")
    return ints.prepare_merged_mtz(with_nobs=False)
# read_smcif_hkl()
    
def read_smcif_shelx(cif_in):
    logger.writeln("Reading small molecule cif: {}".format(cif_in))
    b = gemmi.cif.read(cif_in).sole_block()
    res_str = b.find_value("_shelx_res_file")
    hkl_str = b.find_value("_shelx_hkl_file")
    if not res_str: raise RuntimeError("_shelx_res_file not found in {}".format(cif_in))
    if not hkl_str: raise RuntimeError("_shelx_hkl_file not found in {}".format(cif_in))
    
    ss, info = read_shelx_ins(lines_in=res_str.splitlines())
    mtz = read_shelx_hkl(ss.cell, ss.spacegroup, info.get("hklf"), lines_in=hkl_str.splitlines())
    return mtz, ss, info
# read_smcif_shelx()

def read_small_molecule_files(files):
    st, mtz, hklf = None, None, None
    # first pass - find structure
    for filename in files:
        ext = splitext(filename)[1]
        if ext in (".cif", ".res", ".ins", ".pdb", ".ent", ".mmcif"):
            try:
                st = read_structure(filename)
            except:
                continue
            logger.writeln("Coordinates read from: {}".format(filename))
            if ext in (".cif", ".res", ".ins"):
                if ext == ".cif":
                    b = gemmi.cif.read(filename).sole_block()
                    res_str = b.find_value("_shelx_res_file")
                else:
                    with open(filename) as f:
                        res_str = f.read()
                if res_str:
                    _, info = read_shelx_ins(lines_in=res_str.splitlines())
                    hklf = info["hklf"]
    if st is None:
        logger.writeln("ERROR: coordinates not found.")
        return None, None
    
    # second pass - find hkl
    for filename in files:
        ext = splitext(filename)[1]
        try:
            b = gemmi.cif.read(filename).sole_block()
            hkl_str = b.find_value("_shelx_hkl_file")
            if hkl_str:
                mtz = read_shelx_hkl(st.cell, st.find_spacegroup(), hklf, lines_in=hkl_str.splitlines())
                logger.writeln("reflection data read from: {}".format(filename))
            elif b.find_loop("_refln_index_h"):
                mtz = read_smcif_hkl(filename, st.cell, st.find_spacegroup())
        except ValueError: # not a cif file
            if ext == ".hkl":
                mtz = read_shelx_hkl(st.cell, st.find_spacegroup(), hklf, file_in=filename)
                logger.writeln("reflection data read from: {}".format(filename))

    return st, mtz

def read_sequence_file(f):
    # TODO needs improvement
    # return a list of [name, sequence]
    ret = []
    with open(f) as ifs:
        for i, l in enumerate(ifs):
            l = l.strip()
            if l.startswith(">"):
                name = l[1:].strip()
                ret.append([name, ""])
            elif l:
                if not ret: ret.append(["", ""])
                tmp = l.replace("*", "").replace("-", "").upper()
                r = re.search("[^A-Z]", tmp)
                if r:
                    raise RuntimeError(f"Invalid character in the sequence file: {f}:{i+1}")
                ret[-1][1] += tmp
    return ret
