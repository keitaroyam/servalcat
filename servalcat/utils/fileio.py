"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import model
from servalcat.utils import restraints
import os
import re
import subprocess
import gemmi
import numpy
import numpy.lib.recfunctions
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
        groups.group_pdb = True
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
        groups = gemmi.MmcifOutputGroups(True)
        groups.group_pdb = True
        doc = gemmi.cif.Document()
        block = doc.add_new_block("new")
        st_new.update_mmcif_block(block, groups)
        doc.write_file(cif_out)
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
    origin = [m.header_float(x) for x in (50,51,52)]
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
    logger.write("      Origin: {:.6e} {:.6e} {:.6e}".format(*origin))
    if not numpy.allclose(origin, [0,0,0]):
        logger.write("             ! WARNNING: ORIGIN header is not supported.")
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
                logger.write("WARNING: removing duplicated {} from {}".format(k, cifs_in[i]))
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

    doc.write_file(cif_out, style=gemmi.cif.Style.Aligned)
# merge_ligand_cif()

def read_small_structure(xyz_in):
    spext = splitext(xyz_in)
    if spext[1].lower() in (".ins", ".res"):
        logger.write("Reading SHELX ins/res file: {}".format(xyz_in))
        return model.cx_to_mx(read_shelx_ins(ins_in=xyz_in)[0])
    elif spext[1].lower() in (".pdb", ".ent"):
        logger.write("Reading PDB file: {}".format(xyz_in))
        st = gemmi.read_pdb(xyz_in)
        for m in st:
            for chain in m:
                # Fix if they are blank TODO if more than one chain/residue?
                if chain.name == "": chain.name = "A"
                for res in chain:
                    if res.name == "": res.name = "00"
        return st
    elif spext[1].lower() in (".cif", ".mmcif"):
        doc = read_cif_safe(xyz_in)
        blocks = list(filter(lambda b: b.find_loop("_atom_site.id"), doc))
        if len(blocks) > 0:
            if len(blocks) > 1:
                logger.write(" WARNING: more than one block having _atom_site found. Will use first one.")
            return gemmi.make_structure_from_block(blocks[0])
        else:
            ss = gemmi.read_small_structure(xyz_in)
            if not ss.sites:
                raise RuntimeError("No atoms found in cif file.")
            return model.cx_to_mx(ss)
    else:
        raise RuntimeError("Unsupported file type: {}".format(spext[1]))
# read_small_structure()

def read_shelx_ins(ins_in=None, lines_in=None): # TODO support gz?
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
    for l in open(ins_in) if ins_in else lines_in:
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
    info = dict(hklf=0)
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
        elif ins == "LATT":
            latt = int(sp[1])
        elif ins == "SYMM":
            trp = re.sub("0*\.50*", "1/2", "".join(sp[1:]))
            trp = re.sub("0*\.250*", "1/4", trp)
            trp = re.sub("0*\.750*", "3/4", trp)
            trp = re.sub("0*\.33*", "1/3", trp)
            trp = re.sub("0*\.6[67]*", "2/3", trp)
            trp = re.sub("0*\.16[67]*", "1/6", trp) # never seen?
            trp = re.sub("0*\.833*", "5/6", trp) # never seen?
            symms.append(gemmi.Op(trp).wrap())
        elif ins == "SFAC": # TODO check numbers?
            if len(sp) < 2: continue
            sfacs.append(gemmi.Element(sp[1]))
            if len(sp) > 2:
                try: float(sp[2])
                except ValueError:
                    sfacs.extend([gemmi.Element(x) for x in sp[2:]])
        elif ins == "HKLF":
            info["hklf"] = int(sp[1])
        elif not re_kwd.search(ins):
            if not 4 < len(sp) < 13:
                logger.write("cannot parse this line: {}".format(l))
                continue
            site = gemmi.SmallStructure.Site()
            site.label = sp[0]
            try:
                site.element = sfacs[int(sp[1])-1]
            except:
                logger.error("failed to parse: {}".format(l))
                continue

            if site.label.startswith("Q"):
                logger.write("skip Q peak: {}".format(l))
                continue
            
            site.fract.fromlist(list(map(float, sp[2:5])))
            if len(sp) > 5:
                q = abs(float(sp[5]))
                if q > 10: q = q % 10 # FIXME proper handling
                site.occ = q
            if len(sp) > 11:
                u = list(map(float, sp[6:12]))
                site.aniso = gemmi.SMat33d(u[0], u[1], u[2], u[5], u[4], u[3])
                #TODO site.u_iso needs to be set?
            else:
                site.u_iso = float(sp[6])

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

    symms = list(set(symms))
    sg = gemmi.find_spacegroup_by_ops(gemmi.GroupOps(symms))
    # in case of non-regular setting, gemmi.SpaceGroup cannot be constructed anyway.
    if sg is None:
        logger.error("Cannot construct space group from symbols: {}".format([x.triplet() for x in symms]))
    else:
        ss.spacegroup_hm = sg.hm + (" :{}".format(sg.ext) if sg.ext!="\0" else "")

    if sg is not None: # debug
        sgops = set(gemmi.SpaceGroup(ss.spacegroup_hm).operations())
        opdiffs = sgops.symmetric_difference(symms)
        if opdiffs:
            logger.write("ops= {}".format(" ".join([x.triplet() for x in symms])))

    return ss, info
# read_shelx_ins()

def read_shelx_hkl(cell, sg, file_in=None, lines_in=None):
    assert (file_in, lines_in).count(None) == 1
    hkls, vals, sigs = [], [], []
    for l in open(file_in) if file_in else lines_in:
        if l.startswith(";"): continue
        if not l.strip() or len(l) < 25: continue

        hkl = int(l[:4]), int(l[4:8]), int(l[8:12])
        if hkl == (0,0,0): break
        hkls.append(hkl)
        vals.append(float(l[12:20]))
        sigs.append(float(l[20:28]))
        # batch = l[28:32]
        # wavelength = l[32:40]

    ints = gemmi.Intensities()
    ints.set_data(cell, sg, hkls, vals, sigs)
    ints.merge_in_place(gemmi.DataType.Mean) # TODO may want Anomalous (in case of X-ray)
    logger.write(" Multiplicity: max= {} mean= {:.1f} min= {}".format(numpy.max(ints.nobs_array),
                                                                     numpy.mean(ints.nobs_array),
                                                                     numpy.min(ints.nobs_array)))
    i_sigi = numpy.lib.recfunctions.unstructured_to_structured(numpy.vstack((ints.value_array, ints.sigma_array)).T,
                                                               numpy.dtype([("value", numpy.float32),
                                                                            ("sigma", numpy.float32)]))
    asudata = gemmi.ValueSigmaAsuData(cell, sg, ints.miller_array, i_sigi)
    return asudata
# read_shelx_hkl()

def read_smcif_shelx(cif_in):
    logger.write("Reading small molecule cif: {}".format(cif_in))
    b = gemmi.cif.read(cif_in).sole_block()
    res_str = b.find_value("_shelx_res_file")
    hkl_str = b.find_value("_shelx_hkl_file")
    if not res_str: raise RuntimeError("_shelx_res_file not found in {}".format(cif_in))
    if not hkl_str: raise RuntimeError("_shelx_hkl_file not found in {}".format(cif_in))
    
    ss, info = read_shelx_ins(lines_in=res_str.splitlines())
    asudata = read_shelx_hkl(ss.cell, ss.find_spacegroup(), lines_in=hkl_str.splitlines())
    return asudata, ss, info
# read_smcif_shelx()
