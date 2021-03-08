"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
import numpy
import copy
eps_l = 1.0e-5

def generate_all_elements(axis1, order1, axis2=None, order2=0, toler=1.0e-6):
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
    grp_out = add_groups_together(grp_out,grp1)
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
                    if not is_in_the_list_rotation(r3, grp_out):
                        things_to_do = True
                        order3, axis3 = find_order(r3)
                        grp3 = generate_cyclic(axis3, order3)
                        grp_out_new = add_groups_together(grp_out_new, grp3)
                        order_out.append(order3)
                        axes_out.append(axis3)
                    r3 = numpy.dot(rj, ri)
                    if not is_in_the_list_rotation(r3, grp_out):
                        things_to_do = True
                        order3, axis3 = find_order(r3)
                        grp3 = generate_cyclic(axis3,order3)
                        grp_out_new = add_groups_together(grp_out_new, grp3)
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
            
def add_groups_together(grp_in, grp_add):
    grp_out = copy.copy(grp_in)
    grp_out.extend(filter(lambda r: not is_in_the_list_rotation(r, grp_out), grp_add))
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
    return m_l
# AngleAxis2rotatin()

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
