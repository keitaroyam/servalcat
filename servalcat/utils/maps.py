"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
from servalcat.utils import logger

def mask_from_model():
    pass

def half2full(map_h1, map_h2):
    assert map_h1.shape == map_h2.shape
    assert map_h1.unit_cell == map_h2.unit_cell
    tmp = (numpy.array(map_h1)+numpy.array(map_h2))/2.
    gr = gemmi.FloatGrid(tmp, map_h1.unit_cell, map_h1.spacegroup)
    return gr
# half2full()

def write_ccp4_map(filename, array, cell=None, sg=None, mask_for_extent=None, grid_start=None):
    if type(array) == numpy.ndarray:
        # TODO check dtype
        if sg is None: sg = gemmi.SpaceGroup(1)
        grid = gemmi.FloatGrid(array, cell, sg)
    else:
        # TODO check type
        grid = array
        
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True) # float, update stats

    if mask_for_extent is not None:
        tmp = numpy.where(numpy.array(mask_for_extent)>0)
        if grid_start is not None:
            grid_start = numpy.array(grid_start)[:,None]
            grid_shape = numpy.array(grid.shape)[:,None]
            tmp -= grid_start
            tmp += (grid_shape*numpy.floor(1-tmp/grid_shape)).astype(int) + grid_start

        l = [(min(x), max(x)) for x in tmp]
        box = gemmi.FractionalBox()
        for i in (0, 1):
            fm = grid.get_fractional(l[0][i], l[1][i], l[2][i])
            box.extend(fm)

        logger.write(" setting extent: {} {}".format(box.minimum, box.maximum))
        ccp4.set_extent(box)
    elif grid_start is not None:
        logger.write(" setting starting grid: {} {} {}".format(*grid_start))
        new_grid = gemmi.FloatGrid(grid.get_subarray(*(list(grid_start)+list(grid.shape))),
                                   cell,
                                   sg)
        ccp4.grid = new_grid
        for i in range(3):
            ccp4.set_header_i32(5+i, grid_start[i])

    ccp4.write_ccp4_map(filename)
# write_ccp4_map()
