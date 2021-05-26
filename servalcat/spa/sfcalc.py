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
from servalcat.utils import logger
from servalcat import utils
from servalcat.spa import shift_maps

def add_sfcalc_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--halfmaps", nargs=2, help="Input half map files")
    group.add_argument("--map", help="Use only if you really do not have half maps.")
    parser.add_argument('--mapref',
                        help='Reference map file')
    parser.add_argument('--mask',
                        help='Mask file')
    parser.add_argument('--model',
                        help='Input atomic model file')
    parser.add_argument('--mask_radius',
                        type=float, default=3,
                        help='')
    parser.add_argument('--padding',
                        type=float, 
                        help='Default: 2*mask_radius')
    parser.add_argument('--no_mask',
                        action='store_true')
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--resolution',
                        type=float,
                        help='')
    parser.add_argument('--no_shift',
                        action='store_true',
                        help='')
    parser.add_argument('--blur',
                        nargs="+", # XXX probably no need to be multiple
                        type=float,
                        help='Sharpening or blurring B')
    parser.add_argument('--pg',
                        help="Point group symbol for strict symmetry. The coordinate system is consitent with RELION.")
    parser.add_argument('--twist', type=float, help="Helical twist (degree)")
    parser.add_argument('--rise', type=float, help="Helical rise (Angstrom)")
    parser.add_argument('--ignore_symmetry',
                        help='Ignore symmetry information in the model file')
    parser.add_argument('--remove_multiple_models', action='store_true', 
                        help='Keep 1st model only')
    parser.add_argument("--b_before_mask", type=float)
    parser.add_argument('--no_sharpen_before_mask', action='store_true',
                        help='By default half maps are sharpened before masking by std of signal and unsharpened after masking. This option disables it.')
    parser.add_argument('--no_fix_microheterogeneity', action='store_true', 
                        help='By default it will fix microheterogeneity for Refmac')

# add_sfcalc_args()

def add_arguments(parser):
    parser.description = 'Structure factor calculation for Refmac'
    add_sfcalc_args(parser)
    parser.add_argument('--output_masked_prefix',
                        default="masked_fs",
                        help='output MTZ file name for masked F')
    parser.add_argument('--output_mtz_prefix',
                        default="starting_map",
                        help='output MTZ file name for original F')
    parser.add_argument('--output_model_prefix',
                        default="shifted_local",
                        help='output model file name')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def write_map_mtz(hkldata, mtz_out, map_labs, sig_lab=None, blurs=None):
    if not blurs: blurs = []
    if 0 not in blurs: blurs = [0.] + blurs
    
    nblur = len(blurs)
    ncol = 3+len(map_labs)*(1+nblur)
    if sig_lab: ncol += nblur
    data = numpy.empty((len(hkldata.df.index), ncol))
    data[:,:3] = hkldata.df[["H","K","L"]]
    s2 = 1./hkldata.d_spacings()**2
    for i, lab in enumerate(map_labs):
        for j, b in enumerate(blurs):
            f = numpy.abs(hkldata.df[lab])
            if b != 0: f *= numpy.exp(-b*s2/4.)
            data[:,3+i*(1+nblur)+j] = f
        data[:,3+i*(1+nblur)+nblur] = numpy.angle(hkldata.df[lab], deg=True)
        
    if sig_lab:
        for j, b in enumerate(blurs):
            sigf = hkldata.df[sig_lab]
            if b != 0: sigf *= numpy.exp(-b*s2/4.)
            data[:,3+len(map_labs)*(1+nblur)+j] = sigf

    mtz = gemmi.Mtz()
    mtz.spacegroup = hkldata.sg
    mtz.cell = hkldata.cell
    mtz.add_dataset('HKL_base')
    for label in ['H', 'K', 'L']: mtz.add_column(label, 'H')

    for lab in map_labs:
        lab_root = lab[1:] if lab[0]=="F" else lab
        for b in blurs:
            labf = lab
            if b != 0: labf = "{}Blur_{:.2f}".format(labf, b).replace("Fout0", "Fout")
            mtz.add_column(labf, "F")
        mtz.add_column("P"+lab_root, "P")
    if sig_lab:
        for b in blurs:
            labsigf = sig_lab
            if b != 0: labsigf = "{}Blur_{:.2f}".format(labsigf, b).replace("Fout0", "Fout")
            mtz.add_column(labsigf, "Q")
        
    mtz.set_data(data)
    mtz.write_to_file(mtz_out)
# write_map_mtz()

def scale_maps(maps_in, map_ref, d_min):
    fs = []
    for m in [[map_ref,None]]+maps_in:
        asu = gemmi.transform_map_to_f_phi(m[0]).prepare_asu_data(dmin=d_min)
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
    return [[x]+y[1:] for x,y in zip(maps_scaled, maps_in)]
# scale_maps()

def main(args, monlib=None):
    ret = {} # instructions for refinement
    
    if (args.twist, args.rise).count(None) == 1:
        raise RuntimeError("ERROR: give both helical paramters --twist and --rise")

    is_helical = args.twist is not None

    if args.no_mask:
        args.mask_radius = None
        if not args.no_shift:
            logger.write("WARNING: setting --no_shift because --no_mask is given")
            args.no_shift = True
        if args.mask:
            logger.write("WARNING: Your --mask is ignored because --no_mask is given")
            args.mask = None

    if args.resolution is None and args.model and utils.fileio.splitext(args.model)[1].endswith("cif"):
        doc = gemmi.cif.read(args.model)
        block = doc.sole_block()
        reso_str = block.find_value("_em_3d_reconstruction.resolution")
        logger.write("WARNING: --resolution not given. Using _em_3d_reconstruction.resolution = {}".format(reso_str))
        args.resolution = float(reso_str)

    if args.resolution is None:
        raise RuntimeError("ERROR: --resolution is needed.")
        
    resolution = args.resolution - 1e-6

    if args.halfmaps:
        maps = [utils.fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
        assert maps[0][0].shape == maps[1][0].shape
        assert maps[0][0].unit_cell == maps[1][0].unit_cell
        assert maps[0][1] == maps[1][1]
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]

    grid_start = maps[0][1]
    unit_cell = maps[0][0].unit_cell
    spacegroup = gemmi.SpaceGroup(1)
    start_xyz = numpy.array(maps[0][0].get_position(*grid_start).tolist())
    A = numpy.array(unit_cell.orthogonalization_matrix.tolist())
    center = numpy.sum(A, axis=1) / 2 + start_xyz

    if args.mapref:
        logger.write("Reference map: {}".format(args.mapref))
        map_ref, ref_start = utils.fileio.read_ccp4_map(args.mapref, pixel_size=args.pixel_size)
        assert unit_cell == map_ref.unit_cell
        assert maps[0][0].shape == map_ref.shape
        assert maps[0][1] == ref_start

        # Overwrite input_maps
        logger.write("Scaling maps..")
        maps = scale_maps(maps, map_ref, resolution)

    # Create mask
    mask = None
    if args.mask:
        logger.write("Input mask file: {}".format(args.mask))
        mask = numpy.array(utils.fileio.read_ccp4_map(args.mask)[0])
    
    st_new = None
    if args.model: # and 
        logger.write("Input model: {}".format(args.model))
        st = utils.fileio.read_structure(args.model)
        st.cell = unit_cell
        st.spacegroup_hm = "P 1"

        if not args.no_fix_microheterogeneity:
            # TODO need to check external restraints

            if monlib is None:
                # FIXME should use user provided libraries
                monlib = utils.restraints.load_monomer_library(st[0].get_all_residue_names())
            mhtr_mods = utils.model.microheterogeneity_for_refmac(st, monlib)
            ret["inscode_mods"] = mhtr_mods
            
        model_format = utils.fileio.check_model_format(args.model)
        chain_id_len_max = max([len(x) for x in utils.model.all_chain_ids(st)])
        if chain_id_len_max > 1 and model_format == ".pdb":
            logger.write("Long chain ID (length: {}) detected. Will use mmcif format".format(chain_id_len_max))
            model_format = ".mmcif"

        ret["model_format"] = model_format

        if args.remove_multiple_models and len(st) > 1:
            logger.write(" Removing models 2-{}".format(len(st)))
            for i in reversed(range(1, len(st))):
                del st[i]

        if args.ignore_symmetry and len(st.ncs) > 0:
            logger.write("Removing symmetry information from model.")
            st.ncs.clear()

        if is_helical:
            ops = utils.symmetry.generate_helical_operators(st, start_xyz, center,
                                                            args.pg, args.twist, args.rise)
            logger.write("{} helical operators found".format(len(ops)))
            st.ncs.clear()
            st.ncs.extend([x for x in ops if not x.tr.is_identity()])
            #ret["helical"] = ops
            utils.model.filter_helical_contacting(st)
        elif args.pg:
            logger.write("Point group symmetry: {}".format(args.pg))
            if len(st.ncs) > 0:
                logger.write(" WARNING: NCS information in model file will be ignored")

            _, _, ops = utils.symmetry.operators_from_symbol(args.pg)
            if ops:
                logger.write(" {} operators found".format(len(ops)))
                ops = utils.symmetry.make_NcsOps_from_matrices(ops, cell=unit_cell, center=center)
                st.ncs.clear()
                st.ncs.extend([x for x in ops if not x.tr.is_identity()])

        elif len(st.ncs) > 0:
            logger.write("Strict NCS detected from model.")

        if len(st.ncs) > 0:
            logger.write(" Writing NCS file")
            utils.symmetry.write_NcsOps_for_refmac(st.ncs, "ncsc_global.txt")
            ret["ncsc_file"] = "ncsc_global.txt"
        
        st_new = st.clone()
        if len(st.ncs) > 0 and not args.no_shift:
            utils.model.expand_ncs(st)
            logger.write(" Saving expanded model: input_model_expanded.*")
            utils.fileio.write_model(st, "input_model_expanded", pdb=True, cif=True)
    
        if not mask and args.mask_radius:
            logger.write("Creating mask..")
            mask = gemmi.FloatGrid(*maps[0][0].shape)
            mask.set_unit_cell(unit_cell)
            mask.spacegroup = spacegroup
            mask.mask_points_in_constant_radius(st[0], args.mask_radius, 1.)
            ccp4 = gemmi.Ccp4Map()
            ccp4.grid = mask
            ccp4.update_ccp4_header(2, True) # float, update stats
            ccp4.write_ccp4_map("mask_from_model.ccp4")
            logger.write("Mask file written: mask_from_model.ccp4")
    else:
        model_format = None
        
    if args.no_shift:
        logger.write(" Saving input model with unit cell information")
        utils.fileio.write_model(st, "starting_model", pdb=True, cif=True)
        ret["model_file"] = "starting_model" + model_format

    if mask:
        # Mask maps
        if args.no_sharpen_before_mask or len(maps) < 2:
            logger.write("Applying mask..")
            maps = [[gemmi.FloatGrid(numpy.array(ma[0])*mask, unit_cell, spacegroup)]+ma[1:]
                    for ma in maps]
        else:
            logger.write("Sharpen-mask-unsharpen..")
            maps = utils.maps.sharpen_mask_unsharpen(maps, mask, resolution, b=args.b_before_mask)

        if not args.no_shift:
            logger.write(" Shifting maps and/or model..")
            if args.padding is None: args.padding = args.mask_radius * 2
            new_cell, new_shape, starts, shifts = shift_maps.determine_shape_and_shift(mask=mask,
                                                                                       grid_start=grid_start,
                                                                                       padding=args.padding,
                                                                                       mask_cutoff=0.1,
                                                                                       noncentered=True,
                                                                                       noncubic=True,
                                                                                       json_out="shifts.json")
            ret["shifts"] = shifts
            vol_mask = numpy.count_nonzero(numpy.array(mask)>0.5)
            vol_map = new_shape[0] * new_shape[1] * new_shape[2]
            ret["vol_ratio"] = vol_mask/vol_map
            logger.write(" Vol_mask/Vol_map= {:.2e}".format(ret["vol_ratio"]))
            
            if st_new:
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
                    ret["ncsc_file"] = "ncsc_local.txt"
                    logger.write(" Writing symmetry expanded model for shifted model")
                    utils.symmetry.write_symmetry_expanded_model(st_new, "shifted_local_expanded",
                                                                 pdb=True, cif=True)

                logger.write(" Saving shifted model..")
                utils.fileio.write_model(st_new, "shifted_local", pdb=True, cif=True)
                ret["model_file"] = "shifted_local" + model_format

            logger.write(" Trimming maps..")
            for i in range(len(maps)): # Update maps
                suba = maps[i][0].get_subarray(*(list(starts)+list(new_shape)))
                new_grid = gemmi.FloatGrid(suba, new_cell, spacegroup)
                maps[i][0] = new_grid

        
    hkldata = utils.maps.mask_and_fft_maps(maps, resolution, None)
    hkldata.setup_relion_binning()
    if len(maps) == 2:
        logger.write(" Calculating noise variances..")
        map_labs = ["Fmap1", "Fmap2", "Fout0"]
        sig_lab = "SIGFout0"
        ret["lab_sigf"] = sig_lab
        ret["lab_f_half1"] = "Fmap1"
        # TODO Add SIGF in case of half maps, when refmac is ready
        ret["lab_phi_half1"] = "Pmap1"
        ret["lab_f_half2"] = "Fmap2"
        ret["lab_phi_half2"] = "Pmap2"
        utils.maps.calc_noise_var_from_halfmaps(hkldata)
        hkldata.df[sig_lab] = 0.
        for i_bin, bin_d_max, bin_d_min in hkldata.bin_and_limits():
            sel = i_bin == hkldata.df.bin
            hkldata.df.loc[sel, sig_lab] = numpy.sqrt(hkldata.binned_df["var_noise"][i_bin])

        d_eff_full = hkldata.d_eff("FSCfull")
        logger.write("Effective resolution from FSCfull= {:.2f}".format(d_eff_full))
        ret["d_eff"] = d_eff_full
    else:
        map_labs = ["Fout0"]
        sig_lab = None

    if args.no_shift:
        logger.write("Saving original maps as mtz files..")
        mtzout = args.output_mtz_prefix+".mtz"
    else:
        logger.write(" Saving masked maps as mtz files..")
        mtzout = args.output_masked_prefix+"_obs.mtz"

    hkldata.df.rename(columns=dict(F_map1="Fmap1", F_map2="Fmap2", FP="Fout0"), inplace=True)
    write_map_mtz(hkldata, mtzout,
                  map_labs=map_labs, sig_lab=sig_lab, blurs=args.blur)
    ret["mtz_file"] = mtzout
    ret["lab_f"] = "Fout" + ("Blur_{:.2f}".format(args.blur[0]) if args.blur else "0")
    ret["lab_phi"] = "Pout0"
    return ret
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
