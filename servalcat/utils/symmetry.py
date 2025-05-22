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

def add_symmetry_args(parser, require_pg=False):
    parser.add_argument('--pg', required=require_pg, help="Point group symbol")
    parser.add_argument('--twist', type=float, help="Helical twist (degree)")
    parser.add_argument('--rise', type=float, help="Helical rise (Angstrom)")
    parser.add_argument('--center', type=float, nargs=3, help="Origin of symmetry. Default: center of the box")
    parser.add_argument('--axis1', type=float, nargs=3, help="Axis1 (if I: 5-fold, O: 4-fold, T: 3-fold)")
    parser.add_argument('--axis2', type=float, nargs=3, help="Axis2 (if I: 5-fold, O: 4-fold, T: 3-fold, Dn: 2-fold)")
# add_symmetry_args()
    
def update_ncs_from_args(args, st, map_and_start=None, filter_contacting=False,
                         helical_min_n=None, helical_max_n=None):
    is_helical = args.twist is not None
    if not is_helical and not args.pg:
        if len(st.ncs) > 0:
            logger.writeln("Strict NCS detected from model.")
            show_ncs_operators_axis_angle(st.ncs)
        return
    
    if len(st.ncs) > 0:
        logger.writeln(" WARNING: NCS information in model file will be ignored")

    ncsops = ncsops_from_args(args, st.cell, map_and_start=map_and_start, st=st,
                              helical_min_n=helical_min_n, helical_max_n=helical_max_n)

    st.ncs = [x for x in ncsops if not x.tr.is_identity()]
    # To write identity op to the output model
    idop_id = next((x.id for x in ncsops if x.tr.is_identity()), None)
    if idop_id:
        st.info["_struct_ncs_oper.id"] = idop_id

    if filter_contacting:
        model.filter_contacting_ncs(st)
# update_ncs_from_args()

def ncsops_from_args(args, cell, map_and_start=None, st=None, helical_min_n=None, helical_max_n=None):
    is_helical = args.twist is not None
    
    if map_and_start is not None:
        start_xyz = numpy.array(map_and_start[0].get_position(*map_and_start[1]).tolist())
    else:
        start_xyz = numpy.zeros(3)

    if args.center is None:
        A = cell.orth.mat.array
        center = numpy.sum(A, axis=1) / 2 #+ start_xyz
        logger.writeln("Center: {}".format(center))
    else:
        center = numpy.array(args.center)
        
    if is_helical:
        ncsops = generate_helical_operators(start_xyz, center,
                                            args.pg, args.twist, args.rise,
                                            axis1=args.axis1, axis2=args.axis2,
                                            st=st, min_n=helical_min_n, max_n=helical_max_n)
        logger.writeln("{} helical operators found".format(len(ncsops)))
    else:
        _, _, ops = operators_from_symbol(args.pg, axis1=args.axis1, axis2=args.axis2)
        logger.writeln("{} operators found for {}".format(len(ops), args.pg))
        show_operators_axis_angle(ops)
        ncsops = make_NcsOps_from_matrices(ops, cell=cell, center=center)

    return ncsops
# ncsops_from_args()

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

def operators_from_symbol(op, axis1=None, axis2=None):
    r = re.search("^([CDITO])([0-9]*)$", op.upper())
    if not r:
        raise RuntimeError("Invalid point group symbol: {}".format(op))
    a, n = r.groups()
    if n:
        n = int(n)
        if n <= 0:
            raise RuntimeError("Non positive order given: {}".format(op))
        elif a in ("T", "O"):
            raise RuntimeError("You cannot give number after T,O: {}".format(op))
        elif a == "I" and n > 4:
            raise RuntimeError("Only I1-4 are supported for I variants: {}".format(op))

    if axis1 is not None: axis1 = numpy.array(axis1)
    if axis2 is not None: axis2 = numpy.array(axis2)

    # RELION's conventions
    if a == "I":
        order1 = order2 = 5
        if not n or n == 2:
            if axis1 is None: axis1 = numpy.array([0, 0.85065, 0.52573])
            if axis2 is None: axis2 = numpy.array([0.52573, 0, 0.85065])
        elif n == 1:
            if axis1 is None: axis1 = numpy.array([0.85065, 0., 0.52573])
            if axis2 is None: axis2 = numpy.array([0.52573, 0.85065, 0.])
        elif n == 3:
            if axis1 is None: axis1 = numpy.array([0.723606, 0.525732, 0.447214])
            if axis2 is None: axis2 = numpy.array([-0.276391, -0.850645, 0.447216])
        elif n == 4:
            if axis1 is None: axis1 = numpy.array([-0.723606, -0.525732, 0.447214])
            if axis2 is None: axis2 = numpy.array([0., 0., 1.])
    elif a == "O":
        order1 = order2 = 4
        if axis1 is None: axis1 = numpy.array([0,0,1.0])
        if axis2 is None: axis2 = numpy.array([0,1.0,0])
    elif a == "T":
        order1 = order2 = 3
        if axis1 is None: axis1 = numpy.array([0.0,0.0,1.0])
        if axis2 is None: axis2 = numpy.array([0.0,0.94280904,-0.33333333])
    elif a == "D":
        order1 = n
        if axis1 is None: axis1 = numpy.array([0,0,1.0])
        order2 = 2
        if axis2 is None: axis2 = numpy.array([1.,0.,0.])
    elif a == "C":
        order1 = n
        if axis1 is None: axis1 = numpy.array([0,0,1.0])
        order2 = 0
        axis2 = None
    return generate_operators.generate_all_elements(axis1, order1, axis2, order2)
# operators_from_symbol()

def show_operators_axis_angle(ops):
    for i, op in enumerate(ops):
        ax, ang = generate_operators.Rotation2AxisAngle_general(op)
        logger.writeln(" operator {:3d} angle= {:7.3f} deg axis= {}".format(i+1, numpy.rad2deg(ang), numpy.array_str(ax)))
# show_operators_axis_angle()

def show_ncs_operators_axis_angle(ops):
    # ops: List of gemmi.NcsOp
    for i, op in enumerate(ops):
        op2 = op.tr.mat.array
        ax, ang = generate_operators.Rotation2AxisAngle_general(op2)
        axlab = "[{: .4f}, {: .4f}, {: .4f}]".format(*ax)
        trlab = "[{: 9.4f}, {: 9.4f}, {: 9.4f}]".format(*op.tr.vec.tolist())
        logger.writeln(" operator {:3s} angle= {:7.3f} deg axis= {} trans= {} {}".format(op.id, numpy.rad2deg(ang),
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

def generate_helical_operators(start_xyz, center, axsym, deltaphi, deltaz, axis1=None, axis2=None,
                               st=None, min_n=None, max_n=None, padding=2.):
    if not axsym: axsym = "C1"
    if axis1 is None: axis1 = numpy.array([0,0,1.])
    else: axis1 /= numpy.linalg.norm(axis1)
    _, _, axtrs = operators_from_symbol(axsym, axis1, axis2)
    if min_n is None or max_n is None:
        assert st is not None
        all_z = numpy.dot([cra.atom.pos.tolist() for cra in st[0].all()], axis1)
        min_z, max_z = numpy.min(all_z), numpy.max(all_z)
        direc = numpy.argmax([numpy.dot(axis1, v) for v in ((1.,0,0), (0,1.,0), (0,0,1.))]) # assume axis1 along any of a,b,c axis
        if min_n is None:
            min_n = -int((min_z - padding - start_xyz[direc]) / deltaz)
        if max_n is None:
            max_n = int((st.cell.parameters[direc] + start_xyz[direc] - max_z - padding) / deltaz)
    if min_n > max_n:
        min_n, max_n = max_n, min_n
    ops = []
    for i in range(min_n, max_n+1):
        deg = deltaphi*i
        t = numpy.deg2rad(deg)
        m = generate_operators.AngleAxis2rotatin(axis1, t)
        s = numpy.array(axis1 * deltaz*i)
        for a in axtrs:
            mat = numpy.dot(m, a)
            news = s + numpy.dot(mat, -center) + center
            tr = gemmi.Transform(gemmi.Mat33(mat), gemmi.Vec3(*news))
            newop = gemmi.NcsOp(tr, str(len(ops)+1), tr.is_identity())
            ops.append(newop)

    return ops
# generate_helical_operators()

def make_NcsOps_from_matrices(matrices, cell=None, center=None):
    if center is None:
        A = cell.orth.mat.array
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

def find_center_of_origin(mat, vec): # may not be unique.
    tmp = numpy.identity(3) - numpy.array(mat.array)
    ret = numpy.dot(numpy.linalg.pinv(tmp), vec.tolist())
    resid = vec.tolist() - (numpy.dot(mat.array, -ret) + ret)
    return gemmi.Vec3(*ret), gemmi.Vec3(*resid)
# find_center_of_origin()

def ncs_ops_for_refmac(ncs_ops):
    def make_line(tr):
        m = tr.mat.tolist()
        m_str = " ".join([str(m[i][j]) for i in range(3) for j in range(3)])
        t_str = " ".join([str(x) for x in tr.vec.tolist()])
        return "ncsc matrix {} {}".format(m_str, t_str)
    
    ret = []
    # REFMAC requires identity op
    if not any(x.tr.is_identity() for x in ncs_ops):
        ret.append(make_line(gemmi.Transform()))

    for op in ncs_ops:
        if not op.given: ret.append(make_line(op.tr))
    return ret
# ncs_ops_for_refmac()

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
