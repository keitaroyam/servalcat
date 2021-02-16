"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
import gemmi
import subprocess
import numpy
from servalcat.utils import fileio

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

    for op in ncs_ops: ofs.write(make_line(op.tr))
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
    st_new.expand_ncs(gemmi.HowToNameCopiedChain.Short)
    fileio.write_model(st_new, prefix=prefix, pdb=pdb, cif=cif, cif_ref=cif_ref)
# write_symmetry_expanded_model()
