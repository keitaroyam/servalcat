"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import numpy
import copy
eps_l = 1.0e-5

def generate_all_elements(axis1, order1, axis2=None, order2=0, toler=1.0e-6, toler2=1.e-3):
    # 
    # Generate all group elements. Output will be as a list of cyclic groups.
    #
    if axis2 is None: axis2 = numpy.array([0.0,0.0,1.0])
    grp_out = []
    order_out = []
    axes_out = []
    if order2 == 0:
        grp_out = generate_cyclic(axis1, order1)
        order_out.append(order1)
        axes_out.append(axis1)
        return order_out, axes_out, grp_out
    
    grp_out = generate_cyclic(axis1,order1)
    order_out.append(order1)
    axes_out.append(axis1)
    grp1 = generate_cyclic(axis2,order2)
    grp_out = add_groups_together(grp_out,grp1,toler2)
    order_out.append(order2)
    axes_out.append(axis2)
    
    grp_out_new = copy.copy(grp_out)
    things_to_do = True
    while things_to_do:
        things_to_do = False
        for i, ri in enumerate(grp_out):
            for j, rj in enumerate(grp_out):
                if i !=0 and j!=0 and i != j:
                    r3 = numpy.dot(ri,rj)
                    if not is_in_the_list_rotation(r3, grp_out, toler2):
                        things_to_do = True
                        order3, axis3 = find_order(r3, toler2)
                        grp3 = generate_cyclic(axis3, order3)
                        grp_out_new = add_groups_together(grp_out_new, grp3,toler2)
                        order_out.append(order3)
                        axes_out.append(axis3)
                    r3 = numpy.dot(rj, ri)
                    if not is_in_the_list_rotation(r3, grp_out, toler2):
                        things_to_do = True
                        order3, axis3 = find_order(r3, toler2)
                        grp3 = generate_cyclic(axis3,order3)
                        grp_out_new = add_groups_together(grp_out_new, grp3,toler2)
                        order_out.append(order3)
                        axes_out.append(axis3)
        grp_out = copy.copy(grp_out_new)
    #
    # Filter out axes (if they are parallel to each other then select one, for the order we should take
    # the highest order). In our case it should happen only for the group O. We may have 2 and four fold symmetries
    # with the same axis
    axes_out_new = []
    order_out_new = []
    for i, axisi in enumerate(axes_out):
        order_cp = order_out[i]
        for j, axisj in enumerate(axes_out):
            if i < j:
                cangle = numpy.dot(axisi,axisj)/(numpy.linalg.norm(axisi)*numpy.linalg.norm(axisj))
                if numpy.abs(cangle-1.0) < toler or numpy.abs(cangle+1) < toler:
                    # same axis
                    if order_cp <= order_out[j]:
                        order_cp= 0
                        break
        if order_cp > 0:
            axes_out_new.append(axisi)
            order_out_new.append(order_cp)
            
    return order_out_new, axes_out_new, grp_out
# generate_all_elements()

def find_order(r, toler=1.0e-3):
    order = 1
    r_id = numpy.identity(3)
    r3 = numpy.copy(r_id)
    things_to_do = True
    A = r_id
    while things_to_do and order < 100:
        things_to_do = False
        r3 = numpy.dot(r3, r)
        if numpy.sum(numpy.abs(r3-r_id)) > toler:
            A = A + r3
            things_to_do = True
            order += 1
            
    if order >= 100:
        raise RuntimeError("The order of the group is too high: order > 100")
    A = A/order
    axis_l = find_axis(A)
        
    return order, axis_l
# find_order()
            
def add_groups_together(grp_in, grp_add, toler=1.e-3):
    grp_out = copy.copy(grp_in)
    grp_out.extend(filter(lambda r: not is_in_the_list_rotation(r, grp_out, toler), grp_add))
    #for r in grp_add:
    #    if not is_in_the_list_rotation(r, grp_out):
    #        grp_out.append(r)
    return grp_out
# add_groups_together()

def generate_cyclic(axis, order):
    #
    #  This function generates all cyclic group elements using axis and order of the group
    if order <=0 or numpy.sum(numpy.abs(axis)) < eps_l:
        raise RuntimeError("Either order or axis is zero. order= {} axis= {}".format(order, axis))
    gout = []
    id_matr = numpy.identity(3)
    gout.append(id_matr)
    angle = 2.0*numpy.pi/order
    axis = axis/numpy.linalg.norm(axis)
    exp_matr = numpy.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
    axis_outer = numpy.outer(axis, axis)
    m_int = id_matr - axis_outer
    for i in range(order-1):
        angle_l = angle*(i+1)
        stheta = numpy.sin(angle_l)
        ctheta = numpy.cos(angle_l)
        m_l = exp_matr*stheta + m_int*ctheta +axis_outer
        m_l = numpy.where(numpy.abs(m_l) < 1e-9, 0, m_l)
        gout.append(m_l)

    return gout
# generate_cyclic()

def AngleAxis2rotatin(axis, angle):
    #
    #  Convert axis and ange to a rotation matrix. Here we use a mtrix form of the relatiionship
    # IT may not be the moost efficient algorithm, but it should work (it is more elegant)
    if numpy.sum(numpy.abs(axis)) < eps_l:
        raise RuntimeError("Axis is zero. axis= {} angle= {}".format(axis, angle))
    id_matr = numpy.identity(3)
    axis = axis/numpy.sqrt(numpy.dot(axis,axis))
    exp_matr = numpy.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
    axis_outer = numpy.outer(axis, axis)
    m_int = id_matr - axis_outer
    stheta = numpy.sin(angle)
    ctheta = numpy.cos(angle)
    m_l = exp_matr*stheta + m_int*ctheta +axis_outer
    m_l = numpy.where(numpy.abs(m_l) < 1e-9, 0, m_l)
    return m_l
# AngleAxis2rotatin()

def Rotation2AxisAngle_cyclic(m_in, eps_l=1.0e-5):
    #
    # Here we assume that rotation matrix is an element of a cyclic group
    # This routine gives the smallest angle for this cyclic group. 
    # To find axis of the rotation we use the fact that if we define 
    # A = 1/n sum_i-0^(n-1) (R^i) then this operator is a projector to the axis of rotation
    # i.e. for Ax will be on the axis for any x. IT could be equal 0, in this case we select another x
    A = m_in
    m1 = m_in
    id_matr = numpy.identity(3)
    cycle_number = 1
    ended = False
    while not ended and cycle_number < 200:
        if numpy.sum(numpy.abs(m1-id_matr)) < eps_l:
            ended = True
            break
        m1 = numpy.dot(m1,m_in)
        A = A + m1
        cycle_number = cycle_number + 1
    # take a ranom vector
    if cycle_number >= 150 :
        print("matrix ",m_in)
        print("Try to change the tolerance: eps_l = XXX")
        raise RuntimeError("The matrix does not seem to be producing a finite cyclic group")
    A = A/cycle_number
    axis = numpy.zeros(3)
    for xin in ((0,0,1.), (0,1.,0), (1.,0,0,)):
        axis = numpy.dot(A,xin)
        if numpy.dot(axis,axis) > eps_l:
            axis = axis/numpy.sqrt(numpy.dot(axis,axis))
        if numpy.dot(axis,axis) >= eps_l:
            break

    if axis[2] < 0.0:
        axis = -axis
    elif axis[2] == 0.0 and axis[1] < 0.0:
        axis = -axis
    angle = 2.0*numpy.pi/cycle_number
    axis[axis==0.]=0.
    return axis,angle,cycle_number
# Rotation2AxisAngle_cyclic()

def Rotation2AxisAngle_general(m_in, eps_l=1.0e-5):
    #
    #  This routine should work for any rotation matrix
    axis = numpy.array([1, 0.0, 0.0])
    angle = numpy.arccos(max(-1.0, numpy.min((numpy.trace(m_in)-1)/2.0)))
    if numpy.sum(numpy.abs(m_in-numpy.transpose(m_in))) < eps_l:
        # 
        # It is a symmetric matrix. so I and m_in form a cyclic group
        A = (numpy.identity(3) + m_in)/2.0
        axis = numpy.zeros(3)
        for a in ((0,0,1.), (0,1.,0), (1.,0,0,)):
            axis = numpy.dot(A, a)
            if numpy.linalg.norm(axis) >= eps_l: break
    else:
        axis[0] = m_in[1,2] - m_in[2,1]
        axis[1] = m_in[0,2] - m_in[2,0]
        axis[2] = m_in[0,1] - m_in[1,0]
    if axis[2] < 0.0:
        axis = -axis
        angle = 2.0*numpy.pi - angle
    elif axis[2] < eps_l and axis[1] < 0.0:
        axis = -axis
        angle = 2.0*numpy.pi - angle
    axis = axis/numpy.linalg.norm(axis)
    axis[axis==0.]=0.
    return axis, angle
# Rotation2AxisAngle_general()

def is_in_the_list_rotation(m_in, m_list, toler = 1.0e-3):
    id_matr = numpy.identity(3)
    return numpy.any(numpy.abs(numpy.trace(numpy.dot(numpy.transpose(m_in), m_list)-id_matr[:,None], axis1=0,axis2=2)) < toler)
# is_in_the_list_rotation()

def closest_rotation(m_in, m_list):
    id_matr = numpy.identity(3)
    return min(numpy.abs(numpy.trace(numpy.dot(numpy.transpose(m_in), m_list)-id_matr[:,None], axis1=0,axis2=2)))
# closest_rotation()

def find_axis(amatr):
    #
    #  We assume that amatr is a projector. I.e. y = amatr x is on the the symmetry axis. 
    # To avoid problem of 0 vector we try several times to make sure that 0 vector is not generated
    axis1 = numpy.zeros(3)
    for a in ((0,0,1.), (0,1.,0), (1.,0,0,)):
        axis1 = numpy.dot(amatr, a)
        if numpy.linalg.norm(axis1) >= 0.001:
            break

    axis1 /= numpy.linalg.norm(axis1)
    axis1 = numpy.around(axis1, 10)
    axis1 /= numpy.linalg.norm(axis1)
    # Remove annoying negative signs
    axis1[axis1==0.]=0.
    
    if axis1[2] < 0.0:
        axis1 *= -1
    elif axis1[2] == 0.0 and axis1[1] < 0.0:
        axis1 *= -1

    return axis1
# find_axis()

def rotate_group_elements(Rg, matrices):
    #
    #   assume an input list and return a list of matrices
    #
    mm_out = []
    Rg_t = numpy.transpose(Rg)
    for i, mm in enumerate(matrices):
        m1 = numpy.dot(numpy.dot(Rg_t, mm), Rg)
        mm_out.append(m1)
    return(mm_out)
# rotate_group_elements()
    
if __name__ == "__main__":
    import sys
    from servalcat.utils import symmetry
    symbol = sys.argv[1]
    order, axes, grp = symmetry.operators_from_symbol(symbol)
    #print(order)
    #print(axes)
    #print(grp)
    #quit()
    rgs = symmetry.get_matrices_using_relion(symbol)
    all_ok = True
    max_diff = 0
    for i, m in enumerate(grp):
        #print("Op", i)
        #print(m)
        #ok = is_in_the_list_rotation(m, rgs, toler=1e-4)
        diff = closest_rotation(m, rgs)
        ok = diff < 1e-4
        #print("match? {} {:.1e}".format(ok, diff))
        if not ok: all_ok = False
        if diff > max_diff: max_diff = diff
    print("Final=", all_ok, max_diff)
