# Care starting grid!!
"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import os
import gemmi
import numpy
import argparse
import json
from servalcat.utils import logger
from servalcat import utils
from servalcat.spa.sfcalc import write_shifts_json

def add_arguments(parser):
    parser.description = 'Trim maps and shift models into a small new box.'
    parser.epilog = 'If --mask is provided, a boundary is decided using the mask and --padding. Otherwise the model is used.'
    parser.add_argument('--maps',
                        required=True,
                        nargs="+",
                        help='Input map file(s)')
    parser.add_argument('--mask',
                        help='Mask file')    
    parser.add_argument('--model',
                        nargs="+",
                        help='Input atomic model file(s)')
    parser.add_argument('--padding',
                        type=float,
                        default=10.0,
                        help='in angstrom unit')
    parser.add_argument('--mask_cutoff',
                        type=float,
                        default=1e-5,
                        help='Mask value cutoff to define boundary')
    parser.add_argument('--noncubic',
                        action='store_true')
    parser.add_argument('--noncentered',
                        action='store_true',
                        help='If specified non-centered trimming is performed. Not recommended if having some symmetry')
    parser.add_argument('--force_cell',
                        type=float,
                        nargs=6,
                        help='Force cell')
    parser.add_argument('--disable_cell_check',
                        action='store_true')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def check_maps(map_files, disable_cell_check=False):
    logger.write("Input map files:")
    params = []
    for f in map_files:
        m = utils.fileio.read_ccp4_map(f)
        g = m.grid
        params.append((g.unit_cell.parameters, g.shape, g.spacing))

    shapes = set([x[1] for x in params])
    if len(shapes) > 1:
        raise RuntimeError("Error: different grid size included")

    cells = set([x[0] for x in params])
    if len(cells) > 1:
        if disable_cell_check:
            logger.write("WARNING: Cells are different. Using the first one.")
        else:
            raise RuntimeError("Error: different cell parameters included")

    return params[0]
        
def main(args):
    cell, grid_shape, spacing = check_maps(args.maps, args.disable_cell_check)
    if args.force_cell:
        cell = args.force_cell

    cell = gemmi.UnitCell(*cell)
    sts, cif_refs = [], []
    if args.model:
        for m in args.model:
            st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(m)
            st.spacegroup_hm = "P1"
            st.cell = cell
            sts.append(st)
            cif_refs.append(cif_ref)
        
    if args.mask:
        logger.write("Using mask to decide border: {}".format(args.mask))
        mask = utils.fileio.read_ccp4_map(args.mask).grid
        mask.set_unit_cell(cell)
        mask.spacegroup = gemmi.SpaceGroup(1)
    elif args.model:
        logger.write("Using model to decide border: {}".format(args.model))
        mask = gemmi.FloatGrid(*grid_shape)
        mask.set_unit_cell(cell)
        mask.spacegroup = gemmi.SpaceGroup(1)
        for st in sts:
            mask.mask_points_in_constant_radius(st[0], args.padding, 1.)
    else:
        raise RuntimeError("Give mask or model")

    tmp = numpy.where(numpy.array(mask)>args.mask_cutoff)
    limits = [(min(x), max(x)) for x in tmp]
    # Option to keep cubic?
    spacing = numpy.array(spacing)
    p = args.padding / spacing
    p = p.astype(int)
    logger.write("Limits: {}".format(limits))
    logger.write("Padding: {}".format(p))
    if args.noncentered:
        logger.write("Non-centered trimming will be performed.")
        slices = [slice(max(0, limits[i][0]-p[i]),
                        min(limits[i][1]+p[i]+1, grid_shape[i])) for i in range(3)]
    else:
        logger.write("Centered trimming will be performed.")
        slices = [0, 0, 0]
        for i in range(3):
            ctr = (grid_shape[i]-1)/2 
            rad = max(ctr-(limits[i][0]-p[i]), (limits[i][1]+p[i])-ctr)
            logger.write("Rad{}= {}".format(i, rad))
            if rad < grid_shape[i]/2:
                slices[i] = slice(int(ctr-rad), int(ctr+rad)+1, None)
            else:
                slices[i] = slice(0, grid_shape[i], None)
            
    logger.write("Slices: {}".format(slices))
    if not args.noncubic:
        min_s = min([slices[i].start for i in range(3)])
        max_s = max([slices[i].stop for i in range(3)])
        slices = [slice(min_s, max_s, None) for i in range(3)]
        logger.write("Cubic Slices: {}".format(slices))

    if args.model:
        shifts = -mask.get_position(slices[0].start, slices[1].start, slices[2].start)
        logger.write("Shift for model: {}".format(shifts))
        for st in sts:
            for mol in st:
                for cra in mol.all():
                    cra.atom.pos += shifts

    new_shape = [slices[0].stop-slices[0].start,
                 slices[1].stop-slices[1].start,
                 slices[2].stop-slices[2].start]
    tmp = mask.get_position(*new_shape)
    new_cell = gemmi.UnitCell(tmp[0], tmp[1], tmp[2], cell.alpha, cell.beta, cell.gamma)

    if args.model:
        for i, st in enumerate(sts):
            spext = utils.fileio.splitext(os.path.basename(args.model[i]))
            st.cell = new_cell
            if len(st.ncs) > 0:
                new_ops = utils.symmetry.apply_shift_for_ncsops(st.ncs, shifts)
                st.ncs.clear()
                st.ncs.extend(new_ops)
                logger.write(" Writing symmetry expanded model for shifted model")
                utils.symmetry.write_symmetry_expanded_model(st, spext[0]+"_trimmed_local_expanded",
                                                             pdb=True, cif=True)

            utils.fileio.write_model(st, file_name=spext[0]+"_trimmed"+spext[1], cif_ref=cif_refs[i])
    
    for f in args.maps:
        logger.write("Slicing {}".format(f))
        g = numpy.array(utils.fileio.read_ccp4_map(f).grid)
        newg = g[slices[0], slices[1], slices[2]]
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = gemmi.FloatGrid(newg, new_cell, mask.spacegroup)
        ccp4.update_ccp4_header(2, True) # float, update stats
        ccp4.write_ccp4_map(os.path.basename(utils.fileio.splitext(f)[0])+"_trimmed.mrc")

    write_shifts_json("trim_shifts.json",
                      cell=cell.parameters, shape=grid_shape,
                      new_cell=new_cell.parameters, new_shape=list(map(int, new_shape)),
                      shifts=shifts.tolist())
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
