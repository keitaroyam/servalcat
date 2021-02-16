"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import argparse
import json
from servalcat.utils import logger
from servalcat import utils

def add_arguments(parser):
    parser.description = 'Structure factor calculation for Refmac'
    parser.add_argument('--map',
                        required=False,
                        help='Input map file')
    parser.add_argument('--halfmaps',
                        required=False,
                        nargs=2,
                        help='Input half map files')
    parser.add_argument('--mapref',
                        required=False,
                        help='Reference map file')
    parser.add_argument('--mask',
                        required=False,
                        help='Mask file')
    parser.add_argument('--model',
                        required=False,
                        help='Input atomic model file')
    parser.add_argument('--output_masked_prefix',
                        required=False,
                        default="masked_fs",
                        help='output MTZ file name for masked F')
    parser.add_argument('--output_mtz_prefix',
                        required=False,
                        default="starting_map",
                        help='output MTZ file name for original F')
    parser.add_argument('--output_model_prefix',
                        required=False,
                        default="shifted_local",
                        help='output model file name')
    parser.add_argument('--mask_radius',
                        required=False,
                        type=float,
                        help='')
    parser.add_argument('--resolution',
                        required=True,
                        type=float,
                        help='')
    parser.add_argument('--shift',
                        action='store_true',
                        help='')
    parser.add_argument('--blur',
                        required=False,
                        nargs="+",
                        type=float,
                        help='Sharpening or blurring B')
    parser.add_argument('--relion_pg',
                        required=False,
                        help='RELION point group symbol for strict symmetry')
    parser.add_argument('--ignore_symmetry',
                        required=False,
                        help='Ignore symmetry information in the model file')
    parser.add_argument('--remove_multiple_models',
                        help='Keep 1st model only')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def write_map_mtz(asu_data, mtz_out, blurs=None):
    if not blurs: blurs = []
    if 0 not in blurs:
        blurs.insert(0, 0)
    
    mtz = gemmi.Mtz()
    mtz.spacegroup = asu_data.spacegroup
    mtz.cell = asu_data.unit_cell
    mtz.add_dataset('HKL_base')
    for label in ['H', 'K', 'L']: mtz.add_column(label, 'H')
    for b in blurs:
        if b == 0:
            mtz.add_column("Fout0", "F")
        else:
            mtz.add_column("FoutBlur_{:.2f}".format(b), "F")

    mtz.add_column("Pout0", "P")

    data = numpy.empty((len(asu_data), 3+len(blurs)+1))
    data[:,:3] = asu_data.miller_array
    s2 = asu_data.make_1_d2_array()
    
    for i, b in enumerate(blurs):
        if b == 0:
            data[:,3+i] = numpy.absolute(asu_data.value_array)
        else:
            tempfac = numpy.exp(-b*s2/4.)
            data[:,3+i] = numpy.absolute(asu_data.value_array * tempfac)
            
    data[:,-1] = numpy.angle(asu_data.value_array, deg=True)

    mtz.set_data(data)
    mtz.write_to_file(mtz_out)
# write_map_mtz()

def scale_maps(maps_in, map_ref, d_min):
    fs = []
    for m in maps_in+[map_ref]:
        asu = gemmi.transform_map_to_f_phi(m).prepare_asu_data(dmin=d_min)
        fs.append(asu)

    binner = utils.hkl.Binner(fs[0], style="relion")
    #fs, binner = fft_and_binning([map_ref]+maps_in, d_min)
    f_ref, f_inps = fs[0], fs[1:]
    d_array = f_ref.make_d_array()
    f_inps_scaled = [f_obs.copy() for f_obs in f_inps]
    logger.write(" Saving sf stats as f_stats.dat")
    ofs = open("f_stats.dat", "w")
    lab_fos = " ".join(["F_obs.{n} F_scaled.{n}".format(n=i+1) for i in range(len(f_inps))])
    ofs.write("bin count d_min F_ref {}\n".format(lab_fos))
    for i_bin, c_bin in zip(binner.bins, binner.bin_counts):
        sel = binner.bin_array == i_bin
        fr = f_ref.value_array[sel]
        s_fr = numpy.std(fr)
        d = min(d_array[sel])
        avg_strs = []
        for f_obs, f_obs_scaled in zip(f_inps, f_inps_scaled):
            fo = f_obs.value_array[sel]
            s_fo = numpy.std(fo)
            fo_scaled = fo*s_fr/s_fo
            f_obs_scaled.value_array[sel] = fo_scaled
            avg_strs.append("{:e} {:e}".format(numpy.average(numpy.abs(fo)), numpy.average(numpy.abs(fo_scaled))))

        ofs.write("{:3d} {:6d} {:7.3f} {:e} {}\n".format(i_bin, c_bin, d,
                                                         numpy.average(numpy.abs(fr)), " ".join(avg_strs)))
                                                                

    maps_scaled = [gemmi.transform_f_phi_grid_to_map(f.get_f_phi_on_grid(size=map_ref.shape))
                   for f in f_inps_scaled]
    return maps_scaled
# scale_maps()

def main(args):
    if args.mask and args.mask_radius:
        logger.write("You cannot give mask and mask_radius")
        return
    if args.mask_radius and not args.model:
        logger.write("You need model when you give mask_radius")
        return
    
    input_maps = []
    input_map_labels = []
    args.resolution -= 1e-6

    if args.map:
        logger.write("Input map: {}".format(args.map))
        map_obs = gemmi.read_ccp4_map(args.map).grid
        input_maps.append(map_obs)
        input_map_labels.append("obs")
        logger.write("Grid spacings: {} {} {}".format(*map_obs.spacing))
    else:
        map_obs = None
        
    if args.halfmaps:
        logger.write("Half map 1: {}".format(args.halfmaps[0]))
        logger.write("Half map 2: {}".format(args.halfmaps[1]))
        map_h1 = gemmi.read_ccp4_map(args.halfmaps[0]).grid
        map_h2 = gemmi.read_ccp4_map(args.halfmaps[1]).grid
        assert map_h1.shape == map_h2.shape
        assert map_h1.unit_cell == map_h2.unit_cell

        if map_obs is None:
            map_obs = utils.maps.half2full(map_h1, map_h2)
            input_maps.append(map_obs)
            input_map_labels.append("obs")
            logger.write("Grid spacings: {} {} {}".format(*map_obs.spacing))
        else:
            assert map_obs.shape == map_h1.shape
            assert map_obs.unit_cell == map_h1.unit_cell
            
        input_maps.extend([map_h1, map_h2])
        input_map_labels.extend(["half1", "half2"])
        
    unit_cell = map_obs.unit_cell
    spacegroup = gemmi.SpaceGroup(1)

    #if not args.map:
    #    map_obs = gemmi.FloatGrid((numpy.array(map_h1) + map_h2)/2, unit_cell, spacegroup)

    logger.write("Extent: {}".format(map_obs.shape))
    logger.write("Cell: {}".format(unit_cell.parameters))

    if args.mapref:
        if args.mapref == args.map:
            map_ref = map_obs
            logger.write("Using input map as a reference")
        else:
            map_ref = gemmi.read_ccp4_map(args.mapref).grid
            logger.write("Reference map: {}".format(args.mapref))
            assert unit_cell == map_ref.unit_cell
            assert map_obs.shape == map_ref.shape

        # Overwrite input_maps
        logger.write("Scaling maps..")
        input_maps = scale_maps(input_maps, map_ref, args.resolution)

    # Create mask
    mask = None
    if args.mask:
        logger.write("Input mask file: {}".format(args.mask))
        mask = numpy.array(gemmi.read_ccp4_map(args.mask).grid)
    
    st_new = None
    if args.model: # and 
        logger.write("Input model: {}".format(args.model))
        st = gemmi.read_structure(args.model)
        if args.remove_multiple_models and len(st) > 1:
            logger.write(" Removing models 2-{}".format(len(st)))
            for i in reversed(range(1, len(st))):
                del st[i]

        if args.ignore_symmetry and len(st.ncs) > 0:
            logger.write("Removing symmetry information from model.")
            st.ncs.clear()

        if args.relion_pg:
            logger.write("Point group symmetry given as RELION symbol: {}".format(args.relion_pg))
            if len(st.ncs) > 0:
                logger.write(" WARNING: NCS information in model file will be ignored")

            ops = utils.symmetry.get_matrices_using_relion(args.relion_pg)
            if ops:
                logger.write(" {} operators found".format(len(ops)))
                ops = utils.symmetry.make_NcsOps_from_matrices(ops, cell=unit_cell)
                st.ncs.clear()
                st.ncs.extend([x for x in ops if not x.tr.is_identity()])

        elif len(st.ncs) > 0:
            logger.write("Strict NCS detected from model.")

        if len(st.ncs) > 0:
            logger.write(" Writing NCS file")
            utils.symmetry.write_NcsOps_for_refmac(st.ncs, "ncsc_global.txt")
        
        st_new = st.clone()
        if len(st.ncs) > 0 and args.shift:
            logger.write("Expanding symmetry.")
            st.expand_ncs(gemmi.HowToNameCopiedChain.Short)
            logger.write(" Saving expanded model: input_model_expanded.*")
            utils.fileio.write_model(st, "input_model_expanded", pdb=True, cif=True)
    
        if not mask and args.mask_radius:
            logger.write("Creating mask..")
            mask = gemmi.FloatGrid(*map_obs.shape)
            mask.set_unit_cell(unit_cell)
            mask.spacegroup = spacegroup
            mask.mask_points_in_constant_radius(st[0], args.mask_radius, 1.)
            ccp4 = gemmi.Ccp4Map()
            ccp4.grid = mask
            ccp4.update_ccp4_header(2, True) # float, update stats
            ccp4.write_ccp4_map("mask_from_model.ccp4")
            logger.write("Mask file written: mask_from_model.ccp4")

    # TODO Apply sharpening/blurring here?
    # 
    
    logger.write("Saving original maps as mtz files..")
    for ma, lab in zip(input_maps, input_map_labels):
        asu_obs = gemmi.transform_map_to_f_phi(ma).prepare_asu_data(dmin=args.resolution)
        write_map_mtz(asu_obs, args.output_mtz_prefix+"_"+lab+".mtz", blurs=args.blur)

    if mask:
        # Mask maps
        logger.write("Applying mask..")
        input_maps = [gemmi.FloatGrid(numpy.array(ma)*mask, unit_cell, spacegroup)
                      for ma in input_maps]

        if args.shift:
            logger.write(" Shifting maps and/or model..")
            tmp = numpy.where(numpy.array(mask)>0)
            limits = [(min(x), max(x)) for x in tmp]
            logger.write("  New limits: {} {} {}".format(*limits))
            min_points = [x[0] for x in limits]
            lu, lv, lw = limits
            # padding
            spacing = numpy.array(map_obs.spacing)
            p = args.mask_radius/spacing if args.mask_radius else 3/spacing
            p = p.astype(int)
            logger.write("  Spacing: {} {} {}".format(*map_obs.spacing))
            logger.write("  Padding: {} {} {}".format(*p))
            ll = [slice(max(0, limits[i][0]-p[i]),
                        min(limits[i][1]+p[i]+1, map_obs.shape[i])) for i in range(3)]
            #ll = [slice(28,120), slice(32,116), slice(36,112)]
            
            logger.write("  Slice {}".format(ll))
            
            # FIXME does not work for non-orthogonal cell
            new_shape = [ll[0].stop-ll[0].start,
                         ll[1].stop-ll[1].start,
                         ll[2].stop-ll[2].start]
            tmp = map_obs.point_to_position(*new_shape)
            new_cell = gemmi.UnitCell(tmp[0], tmp[1], tmp[2], 90, 90, 90)
            logger.write("  New Cell: {:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f}".format(*new_cell.parameters))
            logger.write("  New grid: {} {} {}".format(*new_shape))

            if st_new:
                shifts = -mask.point_to_position(ll[0].start, ll[1].start, ll[2].start)
                logger.write("  Shifts: {:.4f} {:.4f} {:.4f}".format(*shifts.tolist()))
                for cra in st_new[0].all():
                    cra.atom.pos += shifts
                
                st_new.cell = new_cell
                st_new.spacegroup_hm = "P 1"
                if len(st_new.ncs) > 0:
                    new_ops = utils.symmetry.apply_shift_for_ncsops(st_new.ncs, shifts)
                    st_new.ncs.clear()
                    st_new.ncs.extend(new_ops)
                    logger.write(" Writing NCS file for shifted model")
                    utils.symmetry.write_NcsOps_for_refmac(st_new.ncs, "ncsc_local.txt")
                    logger.write(" Writing symmetry expanded model for shifted model")
                    utils.symmetry.write_symmetry_expanded_model(st_new, "shifted_local_expanded",
                                                                 pdb=True, cif=True)

                logger.write(" Saving shifted model..")
                utils.fileio.write_model(st_new, "shifted_local", pdb=True, cif=True)


            logger.write(" Saving masked and shifted maps as mtz files..")
            for ma, lab in zip(input_maps, input_map_labels):
                logger.write("  Processing {} map".format(lab))
                org_grid = numpy.array(ma)
                new_grid = org_grid[ll[0], ll[1], ll[2]]

                new_grid = gemmi.FloatGrid(new_grid, new_cell, spacegroup)
                asu_new = gemmi.transform_map_to_f_phi(new_grid).prepare_asu_data(dmin=args.resolution)
                write_map_mtz(asu_new, args.output_mtz_prefix+"_"+lab+".mtz", blurs=args.blur)

            json.dump(dict(cell=unit_cell.parameters,
                           grid=map_obs.shape,
                           new_cell=new_cell.parameters,
                           new_grid=list(map(int, new_shape)),
                           shifts=shifts.tolist()),
                      open("shifts.json", "w"), indent=2)
        else:
            logger.write(" Saving input model with unit cell information")
            st.cell = unit_cell
            st.spacegroup_hm = "P 1"
            utils.fileio.write_model(st, "starting_model", pdb=True, cif=True)
            
            logger.write(" Saving masked maps as mtz files..")
            for ma, lab in zip(input_maps, input_map_labels):
                logger.write("  Processing {} map".format(lab))
                asu_new = gemmi.transform_map_to_f_phi(ma).prepare_asu_data(dmin=args.resolution)
                write_map_mtz(asu_new, args.output_mtz_prefix+"_"+lab+".mtz", blurs=args.blur)

    # TODO write NCS file for REFMAC5
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
