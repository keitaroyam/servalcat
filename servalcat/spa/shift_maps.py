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

def add_arguments(parser):
    parser.description = 'Trim maps and shift models into a small new box.'
    parser.epilog = 'If --mask is provided, a boundary is decided using the mask and --padding. Otherwise the model is used.'
    parser.add_argument('--maps', nargs="+", action="append",
                        help='Input map file(s)')
    parser.add_argument('--mask',
                        help='Mask file')    
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--model', nargs="+", action="append",
                        help='Input atomic model file(s)')
    parser.add_argument('--padding', type=float, default=10.0,
                        help='in angstrom unit')
    parser.add_argument('--mask_cutoff', type=float, default=0.5,
                        help='Mask value cutoff to define boundary')
    parser.add_argument('--noncubic',
                        action='store_true')
    parser.add_argument('--noncentered',
                        action='store_true',
                        help='If specified non-centered trimming is performed. Not recommended if having some symmetry')
    parser.add_argument('--force_cell', type=float, nargs=6,
                        help='Force cell')
    parser.add_argument('--disable_cell_check',
                        action='store_true')
    parser.add_argument("--shifts", help="Specify shifts.json to use precalculated parameters")
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def check_maps(map_files, pixel_size=None, disable_cell_check=False):
    logger.write("Input map files:")
    params = []
    for f in map_files:
        g, gs = utils.fileio.read_ccp4_map(f, pixel_size=pixel_size)
        params.append((g.unit_cell.parameters, g.shape, g.spacing, gs))

    shapes = set([x[1] for x in params])
    if len(shapes) > 1:
        raise RuntimeError("Error: different grid size included")

    starts = set([tuple(x[3]) for x in params])
    if len(starts) > 1:
        raise RuntimeError("Error: different origins included")
    
    cells = set([x[0] for x in params])
    if len(cells) > 1:
        if disable_cell_check:
            logger.write("WARNING: Cells are different. Using the first one.")
        else:
            raise RuntimeError("Error: different cell parameters included")

    return params[0]
# check_maps()

def write_shifts_json(filename, cell, shape, new_cell, new_shape, starts, shifts):
    json.dump(dict(cell=cell,
                   grid=shape,
                   new_cell=new_cell,
                   new_grid=new_shape,
                   starts=starts,
                   shifts=shifts),
              open(filename, "w"), indent=2)
# write_shifts_json()

def determine_shape_and_shift(mask, grid_start, padding, mask_cutoff=1e-5, noncentered=False, noncubic=False,
                              json_out="trim_shifts.json"):
    grid_shape = numpy.array(mask.shape)
    spacing = numpy.array(mask.spacing)
    grid_start = numpy.array(grid_start)
    grid_end = grid_start + grid_shape # for indexing grid_end-1 is the end
    cell = mask.unit_cell
    logger.write("Original grid start: {:4d} {:4d} {:4d}".format(*grid_start))
    logger.write("         grid   end: {:4d} {:4d} {:4d}".format(*(grid_end-1)))
    tmp = numpy.where(numpy.array(mask)>mask_cutoff) - grid_start[:,None]
    tmp += (grid_shape[:,None]*numpy.floor(1-tmp/grid_shape[:,None])).astype(int) + grid_start[:,None]
    limits = [(min(x), max(x)) for x in tmp]
    p = numpy.ceil(padding / spacing).astype(int)
    logger.write("Limits: {} {} {}".format(*limits))
    logger.write("Padding: {} {} {}".format(*p))
    slices = [0, 0, 0]
    if noncentered:
        logger.write("Non-centered trimming will be performed.")
        for i in range(3):
            start = max(grid_start[i], limits[i][0]-p[i])
            stop = min(limits[i][1]+p[i]+1, grid_end[i])
            if (stop-start)%2 == 1:
                if start > 0: start -= 1
                elif stop < grid_end[i]: stop += 1
            slices[i] = slice(start, stop)
    else:
        logger.write("Centered trimming will be performed.")
        for i in range(3):
            ctr = (grid_shape[i]-1)/2 + grid_start[i]
            rad = max(ctr-(limits[i][0]-p[i]), (limits[i][1]+p[i])-ctr)
            logger.write("Rad{}= {}".format(i, rad))
            if rad < grid_shape[i]/2:
                slices[i] = slice(int(ctr-rad), int(ctr+rad)+1, None)
            else:
                slices[i] = slice(grid_start[i], grid_end[i], None)
            
    logger.write("Slices: {}".format(slices))
    if not noncubic:
        # only works when input grid is cubic; otherwise need to expand
        if len(set(mask.shape)) > 1:
            raise RuntimeError("Input grid is not cubic. Try --noncubic")
        min_s = min([slices[i].start for i in range(3)])
        max_s = max([slices[i].stop for i in range(3)])
        slices = [slice(min_s, max_s, None) for i in range(3)]
        logger.write("Cubic Slices: {}".format(slices))

    starts = [slices[i].start for i in range(3)]
        
    # Shifts for model
    shifts = -mask.get_position(slices[0].start, slices[1].start, slices[2].start)
    logger.write("Shift for model: {} {} {}".format(*shifts.tolist()))

    new_shape = [slices[0].stop-slices[0].start,
                 slices[1].stop-slices[1].start,
                 slices[2].stop-slices[2].start]
    tmp = mask.get_position(*new_shape)
    new_cell = gemmi.UnitCell(tmp[0], tmp[1], tmp[2], cell.alpha, cell.beta, cell.gamma)
    
    logger.write("New Cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*new_cell.parameters))
    logger.write("New grid: {} {} {}".format(*new_shape))

    if json_out:
        write_shifts_json(json_out,
                          cell=mask.unit_cell.parameters, shape=mask.shape,
                          new_cell=new_cell.parameters, new_shape=list(map(int, new_shape)),
                          starts=list(map(int, starts)),
                          shifts=shifts.tolist())

    return new_cell, new_shape, starts, shifts
# determine_shape_and_shift()    
        
def main(args):
    if not args.maps and not args.shifts:
        raise RuntimeError("Give --maps or --shifts")

    if args.maps:
        args.maps = sum(args.maps, [])
    else:
        args.maps = []
    
    if args.shifts:
        if args.noncubic or args.noncentered or args.mask:
            raise RuntimeError("You cannot specify --noncubic/--noncentered/--mask if --shifts given")
        if not args.maps and not args.model:
            raise RuntimeError("Give --maps or --model")
        info = json.load(open(args.shifts))
        cell = info["cell"]
    elif args.maps:
        cell, grid_shape, spacing, grid_start = check_maps(args.maps, args.pixel_size, args.disable_cell_check)
      
    if args.force_cell:
        cell = args.force_cell

    cell = gemmi.UnitCell(*cell)
    sts, cif_refs = [], []
    if args.model:
        args.model = sum(args.model, [])
        for m in args.model:
            st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(m)
            st.spacegroup_hm = "P1"
            st.cell = cell
            sts.append(st)
            cif_refs.append(cif_ref)
        
    if args.mask:
        logger.write("Using mask to decide border: {}".format(args.mask))
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
        assert mask.shape == grid_shape
        mask.set_unit_cell(cell)
        mask.spacegroup = gemmi.SpaceGroup(1)
        if args.mask not in args.maps: # need to check with normalized paths?
            args.maps.append(args.mask)
    elif args.model and not args.shifts:
        logger.write("Using model to decide border: {}".format(args.model))
        mask = gemmi.FloatGrid(*grid_shape)
        mask.set_unit_cell(cell)
        mask.spacegroup = gemmi.SpaceGroup(1)
        for st in sts:
            mask.mask_points_in_constant_radius(st[0], args.padding, 1.)
    elif not args.shifts:
        raise RuntimeError("Give mask or model")

    if not args.shifts:
        new_cell, new_shape, starts, shifts = determine_shape_and_shift(mask=mask, grid_start=grid_start,
                                                                        padding=args.padding,
                                                                        mask_cutoff=args.mask_cutoff,
                                                                        noncentered=args.noncentered,
                                                                        noncubic=args.noncubic)
    else:
        new_cell = gemmi.UnitCell(*info["new_cell"])
        new_shape, starts = info["new_grid"], info["starts"]
        shifts = gemmi.Position(*info["shifts"])
        
    if args.model:
        for i, st in enumerate(sts):
            st.cell = new_cell
            # apply shifts
            for mol in st:
                for cra in mol.all():
                    cra.atom.pos += shifts

            # save files
            spext = utils.fileio.splitext(os.path.basename(args.model[i]))
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
        g = utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size)[0]
        newg = g.get_subarray(*(list(starts)+list(new_shape)))
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = gemmi.FloatGrid(newg, new_cell, g.spacegroup)
        ccp4.update_ccp4_header(2, True) # float, update stats
        ccp4.write_ccp4_map(os.path.basename(utils.fileio.splitext(f)[0])+"_trimmed.mrc")

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
