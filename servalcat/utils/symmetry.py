"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import subprocess
import numpy
import re
from servalcat.utils import logger
from servalcat.utils import fileio
from servalcat.utils import model
from servalcat.utils import generate_operators

def get_matrices_using_relion(sym):
    ps = subprocess.check_output(["relion_refine", "--sym", sym.strip(), "--print_symmetry_ops"])

    ret = []
    read_flag = -1
    for l in ps.splitlines():
        if b"R(" in l:
            ret.append(numpy.zeros((3,3)))
            read_flag = 0
        elif 0 <= read_flag < 3:
            ret[-1][read_flag,:] = [float(x) for x in l.split()]
            read_flag += 1
        elif read_flag >= 3:
            read_flag = -1

    return ret
# get_matrices_using_relion()

def operators_from_symbol(op):
    r = re.search("^((?P<a>I|T|O)|(?P<b>C|D)(?P<n>[0-9]+))$", op.upper())
    if not r:
        raise RuntimeError("Invalid point group symbol: {}".format(op))
    a, b, n = r.group("a"), r.group("b"), r.group("n")
    if n is not None and int(n) <= 0:
        raise RuntimeError("Non positive order given: {}".format(op))

    # RELION's conventions
    if a == "I":
        order1 = order2 = 5
        axis1 = numpy.array([0, 0.85065, 0.52573])
        axis2 = numpy.array([0.52573, 0, 0.85065])
    elif a == "O":
        order1 = order2 = 4
        axis1 = numpy.array([0,0,1.0])
        axis2 = numpy.array([0,1.0,0])
    elif a == "T":
        order1 = order2 = 3
        axis1 = numpy.array([0.0,0.0,1.0])
        axis2 = numpy.array([0.0,0.94280904,-0.33333333])
    elif b == "D":
        order1 = int(n)
        axis1 = numpy.array([0,0,1.0])
        order2 = 2
        axis2 = numpy.array([1.,0.,0.])
    elif b == "C":
        order1 = int(n)
        axis1 = numpy.array([0,0,1.0])
        order2 = 0
        axis2 = None
    return generate_operators.generate_all_elements(axis1, order1, axis2, order2)
# operators_from_symbol()

def show_operators_axis_angle(ops):
    for i, op in enumerate(ops):
        ax, ang = generate_operators.Rotation2AxisAngle_general(op)
        logger.write(" operator {:3d} angle= {:7.3f} deg axis= {}".format(i+1, numpy.rad2deg(ang), list(ax)))
# show_operators_axis_angle()

def show_ncs_operators_axis_angle(ops):
    # ops: List of gemmi.NcsOp
    for i, op in enumerate(ops):
        op2 = numpy.array(op.tr.mat.tolist())
        ax, ang = generate_operators.Rotation2AxisAngle_general(op2)
        axlab = "[{: .4f}, {: .4f}, {: .4f}]".format(*ax)
        trlab = "[{: 9.4f}, {: 9.4f}, {: 9.4f}]".format(*op.tr.vec.tolist())
        logger.write(" operator {:3s} angle= {:7.3f} deg axis= {} trans= {} {}".format(op.id, numpy.rad2deg(ang),
                                                                                       axlab, trlab,
                                                                                       "given" if op.given else ""))
# show_operators_axis_angle()

def read_helical_parameters_from_mmcif(cif_in):
    doc = fileio.read_cif_safe(cif_in)
    b = list(filter(lambda b: b.find_loop("_atom_site.id"), doc))[0]
    deltaphi = b.find_value("_em_helical_entity.angular_rotation_per_subunit")
    deltaz = b.find_value("_em_helical_entity.axial_rise_per_subunit")
    deltaphi, deltaz = float(deltaphi), float(deltaz)
    axsym = b.find_value("_em_helical_entity.axial_symmetry")
    return axsym, deltaphi, deltaz
# read_helical_parameters_from_mmcif()

def generate_helical_operators(st, start_xyz, center, axsym, deltaphi, deltaz, padding=1):
    if not axsym: axsym = "C1"
    _, _, axtrs = operators_from_symbol(axsym)
    all_z = [cra.atom.pos.z for cra in st[0].all()]
    min_z, max_z = min(all_z), max(all_z)
    min_n, max_n = int((min_z-padding-start_xyz[2])/deltaz), int((st.cell.c+start_xyz[2]-max_z-padding)/deltaz)
    ops = []
    for i in range(-min_n, max_n+1):
        deg = deltaphi*i
        t = numpy.deg2rad(deg)
        m = generate_operators.AngleAxis2rotatin(numpy.array([0,0,1.]), t)
        s = numpy.array([0,0,deltaz*i])
        for a in axtrs:
            mat = numpy.dot(m, a)
            news = s + numpy.dot(mat, -center) + center
            tr = gemmi.Transform(gemmi.Mat33(mat), gemmi.Vec3(*news))
            newop = gemmi.NcsOp(tr, str(len(ops)+1), tr.is_identity())
            ops.append(newop)

    return ops
# generate_helical_operators()

"""
def write_ncsc_for_refmac(file_name, matrices, xyz_in=None, map_in=None):
    if xyz_in:
        st = gemmi.read_structure(xyz_in)
        cell = st.cell
    if map_in:
        ma = gemmi.read_ccp4_map(map_in)
        cell = ma.grid.unit_cell
        
    A = numpy.array(cell.orthogonalization_matrix.tolist())
    center = numpy.sum(A,axis=1) / 2
    
    ofs = open(file_name, "w")
    for m in matrices:
        transl = numpy.dot(m, -center) + center
        m_str = " ".join([str(x) for x in m.flatten()])
        t_str = " ".join([str(x) for x in transl])
        ofs.write("ncsc matrix {} {}\n".format(m_str, t_str))
    ofs.close()
# write_ncsc_for_refmac()
"""

def make_NcsOps_from_matrices(matrices, cell=None, center=None):
    if center is None:
        A = numpy.array(cell.orthogonalization_matrix.tolist())
        center = numpy.sum(A,axis=1) / 2

    center = gemmi.Vec3(*center)
    ops = []
    for i, m in enumerate(matrices):
        m = gemmi.Mat33(m)
        transl = m.multiply(-center) + center
        op = gemmi.NcsOp(gemmi.Transform(m, transl), str(i+1), m.is_identity())
        ops.append(op)
        
    return ops
# make_NcsOps_from_matrices()

def write_NcsOps_for_refmac(ncs_ops, file_name):
    def make_line(tr):
        m = tr.mat.tolist()
        m_str = " ".join([str(m[i][j]) for i in range(3) for j in range(3)])
        t_str = " ".join([str(x) for x in tr.vec.tolist()])
        return "ncsc matrix {} {}\n".format(m_str, t_str)
    
    ofs = open(file_name, "w")

    # REFMAC requires identity op
    if not any([x.tr.is_identity() for x in ncs_ops]):
        ofs.write(make_line(gemmi.Transform()))

    for op in ncs_ops:
        if not op.given: ofs.write(make_line(op.tr))
    ofs.close()
# write_NcsOps_for_refmac()

# TODO def euler2matrix(euler):
# TODO def polar2matrix(polar):

def parse_ncsc_keywords(kwd_str):
    # FIXME handle lines ending with -
    lines = kwd_str.splitlines()
    ret = []
    for l in lines:
        l = l.split()
        if len(l) < 3: continue
        if l[0].lower().startswith("ncsc"):
            if l[1].lower().startswith("matr"):
                vals = [float(x) for x in l[2:]]
                if len(vals) != 12:
                    print("Bad nsc matrix line: {}".format(" ".join(l)))
                    continue
                op = gemmi.NcsOp()
                op.tr.mat.fromlist([vals[3*x:3*x+3] for x in range(3)])
                op.tr.vec.fromlist(vals[9:])
                op.id = str(len(ret)+1)
                op.given = op.tr.is_identity()
                ret.append(op)
            elif l[1].lower().startswith("eule"):
                pass # TODO
            elif l[1].lower().startswith("pola"):
                pass # TODO
    return ret
# parse_ncsc_keywords()

def apply_shift_for_ncsops(ncsops, shift):
    new_ops = []
    s = gemmi.Vec3(*shift)
    for op in ncsops:
        newt = op.tr.vec + s - op.tr.mat.multiply(s)
        newop = gemmi.NcsOp(gemmi.Transform(op.tr.mat, newt), op.id, op.given)
        new_ops.append(newop)

    return new_ops
# apply_shift_for_ncsops()

def write_symmetry_expanded_model(st, prefix, pdb=False, cif=False, cif_ref=None):
    st_new = st.clone()
    model.expand_ncs(st_new)
    fileio.write_model(st_new, prefix=prefix, pdb=pdb, cif=cif, cif_ref=cif_ref)
# write_symmetry_expanded_model()
