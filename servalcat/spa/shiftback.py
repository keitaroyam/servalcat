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
    parser.description = 'Shift back for Refmac local refinement results'
    parser.add_argument('--model',
                        required=True,
                        help='Input atomic model file')
    parser.add_argument('--refine_mtz',
                        required=True,
                        help='Refmac mtz file for shift_back mode')
    parser.add_argument('--shifts',
                        required=True,
                        default="shifts.json",
                        help='Shift information file')
    parser.add_argument('--ncsc_file',
                        help='')
    parser.add_argument('--output_prefix',
                        default="refined",
                        help='output file prefix')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def refmac_mtz_in_original_cell(org_cell, org_grid_size, new_grid_size, shifts, mtz_in, mtz_out):
    targets = (("FWT", "PHWT"), ("DELFWT", "PHDELWT"))
    
    shifts_frac = org_cell.fractionalize(gemmi.Position(*shifts)).tolist()

    # Output mtz
    mtz = gemmi.Mtz()
    mtz.spacegroup = gemmi.SpaceGroup("P1")
    mtz.cell = org_cell
    mtz.add_dataset('HKL_base')
    for l in ["H", "K", "L"]: mtz.add_column(l, "H")

    data = None
    for i in range(len(targets)):
        d_min, m = utils.fileio.read_map_from_mtz(mtz_in, targets[i], new_grid_size)
        F = numpy.fft.fftn(m, org_grid_size).conj()
        grid = gemmi.ReciprocalComplexGrid(F.astype(numpy.complex64), mtz.cell, mtz.spacegroup)
        asu = grid.prepare_asu_data(dmin=d_min)
        if data is None:
            data = numpy.empty((len(asu), 3+2*len(targets)))
            data[:,:3] = asu.miller_array

        shift_factor = numpy.exp(-2j*numpy.pi*numpy.dot(asu.miller_array, shifts_frac))
        F_shift = asu.value_array * shift_factor
        data[:,3+2*i] = numpy.absolute(F_shift)
        data[:,3+2*i+1] = numpy.angle(F_shift, deg=True)

        mtz.add_column(targets[i][0], "F")
        mtz.add_column(targets[i][1], "P")
        
    mtz.set_data(data)
    mtz.write_to_file(mtz_out)
# refmac_mtz_in_original_cell()

def shift_back(xyz_in, refine_mtz, shifts_json, ncsc_in, out_prefix):
    logger.write("Reading shifts info from {}".format(shifts_json))
    info = json.load(open(shifts_json))
    for k in info:
        logger.write(" {}= {}".format(k, info[k]))

    org_cell = gemmi.UnitCell(*info["cell"])
    shifts = gemmi.Position(*info["shifts"])

    if refine_mtz:
        logger.write("Transforming MTZ: {}".format(refine_mtz))
        refmac_mtz_in_original_cell(org_cell,
                                    info["grid"],
                                    info["new_grid"],
                                    info["shifts"],
                                    refine_mtz,
                                    out_prefix+".mtz")

    if xyz_in:
        logger.write("Shifting back model: {}".format(xyz_in))
        st, cif_ref = utils.fileio.read_structure_from_pdb_and_mmcif(xyz_in)

        st.cell = org_cell
        for model in st:
            for cra in model.all():
                cra.atom.pos -= shifts

        if ncsc_in:
            logger.write("Original NCSC: {}".format(ncsc_in))
            ncsops = utils.keywords.parse_ncsc_keywords(open(ncsc_in).read())
            st.ncs.clear()
            st.ncs.extend([x for x in ncsops if not x.tr.is_identity()])

        utils.fileio.write_model(st, out_prefix,
                                 pdb=True, cif=True, cif_ref=cif_ref)
        

# shift_back()


def main(args):
    shift_back(args.model, args.refine_mtz, args.shifts,
               args.ncsc_file, args.output_prefix)

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
    
