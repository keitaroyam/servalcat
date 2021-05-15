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
from servalcat.utils import logger
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
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('-r', '--mask_radius',
                        type=float,
                        help='')
    parser.add_argument('-d', '--resolution',
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

def read_and_fft_maps(filenames, d_min, mask=None, pixel_size=None, check_consistency=True):
    hkldata = None

    last_cell, last_shape, last_sg = None, None, None
    
    for idx, mapin in enumerate(filenames):
        m = utils.fileio.read_ccp4_map(mapin, pixel_size=pixel_size)[0]
        if check_consistency:
            if last_cell is not None:
                assert m.unit_cell == last_cell
                assert m.shape == last_shape
                assert m.spacegroup == last_sg
            last_cell, last_shape, last_sg = m.unit_cell, m.shape, m.spacegroup

        if mask is not None:
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
    st = utils.fileio.read_structure(args.model)
    ref_grid = utils.fileio.read_ccp4_map(args.maps[0], pixel_size=args.pixel_size)[0]
    st.cell = ref_grid.unit_cell
    st.spacegroup_hm = "P1"

    if len(st.ncs) > 0:
        utils.model.expand_ncs(st)
    
    if args.mask_radius is not None:
        mask = gemmi.FloatGrid(*ref_grid.shape)
        mask.set_unit_cell(ref_grid.unit_cell)
        mask.spacegroup = ref_grid.spacegroup
        mask.mask_points_in_constant_radius(st[0], args.mask_radius, 1.)    
    else:
        mask = None
    
    fc = utils.model.calc_fc_fft(st, args.resolution, source="electron")
    hkldata = read_and_fft_maps(args.maps, args.resolution, mask, pixel_size=args.pixel_size)
    
    hkldata.merge_asu_data(fc, "FC")
    hkldata.setup_relion_binning()
    #print(hkldata.df)


    ofs = open(args.fsc_out, "w")
    ofs.write("# Mask_radius= {}\n".format(args.mask_radius))
    for i, f in enumerate(args.maps): ofs.write("# Map{} = {}\n".format(i+1, f))
    ofs.write("bin n d_max d_min {}\n".format(" ".join(["Fmap{}".format(i+1) for i in range(len(args.maps))])))
    fsc_list = [[] for i in range(len(args.maps))]
    n_list = [[] for i in range(len(args.maps))]
    for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
        sel = i_bin == hkldata.df.bin
        fc = hkldata.df["FC"][sel]
        ofs.write("{:3d} {:7d} {:7.3f} {:7.3f}".format(i_bin, fc.size, bin_d_max, bin_d_min))
        for i_map in range(len(args.maps)):
            fo = hkldata.df["Fmap{}".format(i_map+1)][sel]
            fsc = numpy.real(numpy.corrcoef(fo, fc)[1,0])
            ofs.write(" {:.4f}".format(fsc))
            fsc_list[i_map].append(fsc)
            n_list[i_map].append(fc.size)
        ofs.write("\n")
    ofs.write("\n")
    ofs.write("#              FSCaverage =")
    for i in range(len(args.maps)):
        fsc = numpy.array(fsc_list[i])
        n = numpy.array(n_list[i])
        fsca = sum(fsc*n)/sum(n)
        ofs.write(" {:.4f}".format(fsca))
    ofs.write("\n")
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
