"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import os
import subprocess
import pipes
from servalcat import utils
from servalcat import spa

def add_arguments(parser):
    parser.description = 'FSC calculation'

    parser.add_argument('--model',
                        required=True,
                        help="")
    parser.add_argument('--maps',
                        required=True,
                        nargs="+",
                        help='Input map file(s)')
    parser.add_argument('--mask_radius',
                        default=3,
                        type=float,
                        help='')
    parser.add_argument('--resolution',
                        type=float,
                        required=True,
                        help='')
    parser.add_argument('--fsc_out',
                        default="fsc.dat",
                        help='')
# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def read_and_fft_maps(filenames, d_min, mask, check_consistency=True):
    hkldata = None

    last_cell, last_shape, last_sg = None, None, None
    
    for idx, mapin in enumerate(filenames):
        m = gemmi.read_ccp4_map(mapin).grid
        if check_consistency:
            if last_cell is not None:
                assert m.unit_cell == last_cell
                assert m.shape == last_shape
                assert m.spacegroup == last_sg
            last_cell, last_shape, last_sg = m.unit_cell, m.shape, m.spacegroup


        m = gemmi.FloatGrid(numpy.array(m) * mask, m.unit_cell, m.spacegroup)
        asu = gemmi.transform_map_to_f_phi(m).prepare_asu_data(dmin=d_min)
        label = "Fmap{}".format(idx+1)
        if hkldata is None:
            df = utils.hkl.df_from_asu_data(asu, label)
            hkldata = utils.hkl.HklData(m.unit_cell, m.spacegroup, False, df)
        else:
            hkldata.merge_asu_data(asu, label)
            
    return hkldata
# read_and_fft_maps()


def main(args):
    st = gemmi.read_structure(args.model)

    ref_grid = gemmi.read_ccp4_map(args.maps[0]).grid
    mask = gemmi.FloatGrid(*ref_grid.shape)
    mask.set_unit_cell(ref_grid.unit_cell)
    mask.spacegroup = ref_grid.spacegroup
    st.cell = ref_grid.unit_cell
    st.spacegroup_hm = "P1"
    mask.mask_points_in_constant_radius(st[0], args.mask_radius, 1.)

    
    fc = utils.model.calc_fc_em(st, args.resolution)

    hkldata = read_and_fft_maps(args.maps, args.resolution, mask)
    

    hkldata.merge_asu_data(fc, "FC")
    hkldata.setup_relion_binning()
    #print(hkldata.df)


    ofs = open(args.fsc_out, "w")
    ofs.write("bin n d_max d_min {}\n".format(" ".join(["Fmap{}".format(i+1) for i in range(len(args.maps))])))
    fsc_list, n_list = [], []
    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        fc = hkldata.df["FC"][sel]
        ofs.write("{:3d} {:7d} {:7.3f} {:7.3f}".format(i_bin, fc.size, bin_d_max, bin_d_min))
        for i_map in range(len(args.maps)):
            fo = hkldata.df["Fmap{}".format(i_map+1)][sel]
            fsc = numpy.real(numpy.corrcoef(fo, fc)[1,0])
            ofs.write(" {:.4f}".format(fsc))
        ofs.write("\n")
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
