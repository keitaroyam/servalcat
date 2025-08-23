"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import pandas
import os
import json
import argparse
from servalcat.utils import logger
from servalcat import utils

def add_arguments(parser):
    parser.description = 'Calculate real space local correlation map from half maps and model'
    parser.add_argument("--halfmaps", required=True, nargs=2,
                       help="Input half map files")
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kernel", type=int, 
                       help="Kernel radius in pixel")
    group.add_argument("--kernel_ang", type=float,
                       help="Kernel radius in Angstrom (hard sphere)")
    parser.add_argument('--mask',
                        help="mask file")
    parser.add_argument('--model', 
                        help='Input atomic model file')
    parser.add_argument('--resolution', type=float,
                        help='default: nyquist resolution')
    parser.add_argument("-s", "--source", choices=["electron", "xray", "neutron", "custom"], default="electron")
    parser.add_argument("--trim", action='store_true', help="Write trimmed map")
    parser.add_argument('-o', '--output_prefix', default="ccmap",
                        help="default: %(default)s")
# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def setup_coeffs_for_halfmap_cc(maps, d_min, mask=None, st=None):
    hkldata = utils.maps.mask_and_fft_maps(maps, d_min, mask)
    hkldata.setup_relion_binning("ml")
    utils.maps.calc_noise_var_from_halfmaps(hkldata)

    nref = len(hkldata.df.index)
    F1w = numpy.zeros(nref, dtype=complex)
    F2w = numpy.zeros(nref, dtype=complex)
    F1 = hkldata.df.F_map1.to_numpy()
    F2 = hkldata.df.F_map2.to_numpy()

    logger.writeln("Calculating weights for half map correlation.")
    logger.writeln(" weight = sqrt(FSChalf / (2*var_noise + var_signal))")
    hkldata.binned_df["ml"]["w2_half_varsignal"] = 0.
    for i_bin, idxes in hkldata.binned("ml"):
        fscfull = hkldata.binned_df["ml"].FSCfull[i_bin]
        if fscfull < 0:
            break # stop here so that higher resolution are all zero
        fsc = fscfull / (2 - fscfull)
        var_fo = 2 * hkldata.binned_df["ml"].var_noise[i_bin] + hkldata.binned_df["ml"].var_signal[i_bin]
        w = numpy.sqrt(fsc / var_fo)
        hkldata.binned_df["ml"].loc[i_bin, "w2_half_varsignal"] = fsc / var_fo * hkldata.binned_df["ml"].var_signal[i_bin]
        F1w[idxes] = F1[idxes] * w
        F2w[idxes] = F2[idxes] * w

    hkldata.df["F_map1w"] = F1w
    hkldata.df["F_map2w"] = F2w

    return hkldata
# setup_coeffs_for_halfmap_cc()

def add_coeffs_for_model_cc(hkldata, st, source="electron"):
    hkldata.df["FC"] = utils.model.calc_fc_fft(st, d_min=hkldata.d_min_max()[0]-1e-6,
                                               source=source, miller_array=hkldata.miller_array())
    nref = len(hkldata.df.index)
    FCw = numpy.zeros(nref, dtype=complex)
    FPw = numpy.zeros(nref, dtype=complex)
    FP = hkldata.df.FP.to_numpy()
    FC = hkldata.df.FC.to_numpy()

    logger.writeln("Calculating weights for map-model correlation.")
    logger.writeln(" weight for Fo = sqrt(FSCfull / var(Fo))")
    logger.writeln(" weight for Fc = sqrt(FSCfull / var(Fc))")
    hkldata.binned_df["ml"]["w_mapmodel_c"] = 0.
    hkldata.binned_df["ml"]["w_mapmodel_o"] = 0.
    hkldata.binned_df["ml"]["var_fc"] = 0.
    for i_bin, idxes in hkldata.binned("ml"):
        fscfull = hkldata.binned_df["ml"].FSCfull[i_bin]
        if fscfull < 0: break
        var_fc = numpy.var(FC[idxes])
        wc = numpy.sqrt(fscfull / var_fc)
        wo = numpy.sqrt(fscfull / numpy.var(FP[idxes]))
        FCw[idxes] = FC[idxes] * wc
        FPw[idxes] = FP[idxes] * wo
        hkldata.binned_df["ml"].loc[i_bin, "w_mapmodel_c"] = wc
        hkldata.binned_df["ml"].loc[i_bin, "w_mapmodel_o"] = wo
        hkldata.binned_df["ml"].loc[i_bin, "var_fc"] = var_fc
    
    hkldata.df["FPw"] = FPw
    hkldata.df["FCw"] = FCw
# add_coeffs_for_model_cc()

def model_stats(st, modelcc_map, halfcc_map, loggraph_out=None, json_out=None):
    tmp = dict(chain=[], seqid=[], resn=[], CC_mapmodel=[], CC_halfmap=[])
    for chain in st[0]:
        for res in chain:
            mm = numpy.mean([modelcc_map.interpolate_value(atom.pos) for atom in res])
            hc = numpy.mean([halfcc_map.interpolate_value(atom.pos) for atom in res])
            tmp["chain"].append(chain.name)
            tmp["seqid"].append(str(res.seqid))
            tmp["resn"].append(res.name)
            tmp["CC_mapmodel"].append(mm)
            tmp["CC_halfmap"].append(hc)

    df = pandas.DataFrame(tmp)
    df["sqrt_CC_full"] = numpy.sqrt(2 * df.CC_halfmap / (1 + df.CC_halfmap))
    if loggraph_out is not None:
        with open(loggraph_out, "w") as ofs:
            for c, g in df.groupby("chain", sort=False):
                ofs.write("$TABLE: Chain {} :".format(c))
                ofs.write("""
$GRAPHS
: average correlations :A:2,4,5,6:
$$
chain seqid resn CC(map,model) CC_half sqrt(CC_full)
$$
$$
""")
                ofs.write(g.to_string(header=False, index=False))
                ofs.write("\n\n")
    if json_out is not None:
        df.to_json(json_out, orient="records", indent=2)
    return df
# model_stats()

def main(args):
    maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
    grid_shape = maps[0][0].shape
    if args.mask:
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None

    if args.resolution is None:
        d_min = utils.maps.nyquist_resolution(maps[0][0])
    else:
        d_min = args.resolution

    hkldata = setup_coeffs_for_halfmap_cc(maps, d_min, mask)
    if args.kernel is None:
        prefix = "{}_r{}A".format(args.output_prefix, args.kernel_ang)
        knl = hkldata.hard_sphere_kernel(r_ang=args.kernel_ang, grid_size=grid_shape)
    else:
        prefix = "{}_r{}px".format(args.output_prefix, args.kernel)
        knl = utils.maps.raised_cosine_kernel(args.kernel)
        
    halfcc_map = utils.maps.local_cc(hkldata.fft_map("F_map1w", grid_size=grid_shape),
                                     hkldata.fft_map("F_map2w", grid_size=grid_shape),
                                     knl, method="simple" if args.kernel is None else "scipy")

    halfcc_map_in_mask = halfcc_map.array[mask.array>0.5] if mask is not None else halfcc_map
    logger.writeln("Half map CC: min/max= {:.4f} {:.4f}".format(numpy.min(halfcc_map_in_mask), numpy.max(halfcc_map_in_mask)))
    utils.maps.write_ccp4_map(prefix+"_half.mrc", halfcc_map, hkldata.cell, hkldata.sg,
                              mask_for_extent=mask if args.trim else None)

    if args.model:
        st = utils.fileio.read_structure(args.model)
        utils.model.remove_charge([st])
        ccu = utils.model.CustomCoefUtil()
        if args.source == "custom":
            ccu.read_from_cif(st, args.model)
            ccu.show_info()
            ccu.set_coeffs(st)
        utils.model.expand_ncs(st)
        st.cell = hkldata.cell
        st.spacegroup_hm = hkldata.sg.xhm()
        add_coeffs_for_model_cc(hkldata, st, args.source)
        modelcc_map = utils.maps.local_cc(hkldata.fft_map("FPw", grid_size=grid_shape),
                                          hkldata.fft_map("FCw", grid_size=grid_shape),
                                          knl, method="simple" if args.kernel is None else "scipy")
        modelcc_map_in_mask = modelcc_map.array[mask.array>0.5] if mask is not None else modelcc_map
        logger.writeln("Model-map CC: min/max= {:.4f} {:.4f}".format(numpy.min(modelcc_map_in_mask), numpy.max(modelcc_map_in_mask)))
        utils.maps.write_ccp4_map(prefix+"_model.mrc", modelcc_map, hkldata.cell, hkldata.sg,
                                  mask_for_extent=mask if args.trim else None)
        model_stats(st, modelcc_map, halfcc_map, loggraph_out=prefix+"_byresidue.log", json_out=prefix+"_byresidue.json")
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
