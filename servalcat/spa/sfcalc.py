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

# TODO change the name; "sfcalc" is misleading actually - this prepares input files for Refmac

def add_sfcalc_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--halfmaps", nargs=2, help="Input half map files")
    group.add_argument("--map", help="Use this only if you really do not have half maps.")
    parser.add_argument('--mask',
                        help='Mask file')
    parser.add_argument('--model',
                        help='Input atomic model file')
    parser.add_argument('--mask_radius',
                        type=float, default=3,
                        help='')
    parser.add_argument('--mask_soft_edge',
                        type=float, default=0,
                        help='Add soft edge to model mask. Should use with --no_sharpen_before_mask?')
    parser.add_argument('--padding',
                        type=float, 
                        help='Default: 2*mask_radius')
    parser.add_argument('--no_mask', action='store_true')
    parser.add_argument('--invert_mask', action='store_true', help='not for refinement.')
    parser.add_argument('--pixel_size', type=float,
                        help='Override pixel size (A)')
    parser.add_argument('--resolution',
                        type=float,
                        help='')
    parser.add_argument('--no_trim',
                        action='store_true',
                        help='Keep original box (not recommended)')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--blur',
                        type=float, default=0,
                        help='Sharpening or blurring B')
    utils.symmetry.add_symmetry_args(parser) # add --pg etc
    parser.add_argument('--contacting_only', action="store_true", help="Filter out non-contacting NCS")
    parser.add_argument('--ignore_symmetry',
                        help='Ignore symmetry information (MTRIX/_struct_ncs_oper) in the model file')
    parser.add_argument('--keep_multiple_models', action='store_true', 
                        help='Multi-models will be kept; by default only 1st model is kept because REFMAC5 does not support it')
    parser.add_argument('--no_link_check', action='store_true', 
                        help='Do not find and fix link records in input model.')
    parser.add_argument("--b_before_mask", type=float,
                        help="sharpening B value for sharpen-mask-unsharpen procedure. By default it is determined automatically.")
    parser.add_argument('--no_sharpen_before_mask', action='store_true',
                        help='By default half maps are sharpened before masking by std of signal and unsharpened after masking. This option disables it.')
    parser.add_argument('--no_fix_microheterogeneity', action='store_true', 
                        help='By default it will fix microheterogeneity for Refmac')
    parser.add_argument('--no_fix_resi9999', action='store_true', 
                        help='By default it will split chain if max residue number > 9999 which is not supported by Refmac')
    parser.add_argument('--no_check_ncs_overlaps', action='store_true', 
                        help='Disable model overlap (e.g. expanded model is used with --pg) test')
    parser.add_argument('--no_check_mask_with_model', action='store_true', 
                        help='Disable mask test using model')
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
    parser.add_argument('--shifted_model_prefix',
                        default="shifted",
                        help='output (shifted) model file name')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def lab_f_suffix(blur):
    if blur is None or blur == 0.:
        return ""
    elif blur > 0:
        return "Blur_{:.2f}".format(blur)
    else:
        return "Sharp_{:.2f}".format(-blur)
# lab_f_suffix()

def write_map_mtz(hkldata, mtz_out, map_labs, sig_lab=None, blur=0):
    nblur = 2 if blur != 0 else 1
    mean_f = hkldata.df[map_labs].abs().mean().min()
    data_labs = map_labs + ([sig_lab] if sig_lab else [])

    if mean_f < 1:
        scale = 10. / mean_f
        logger.writeln("Mean(|F|)= {:.2e} may be too small for Refmac. Applying scale= {:.1f}".format(mean_f, scale))
        for lab in data_labs:
            hkldata.df[lab] *= scale

    mtz_labs = data_labs + []
    mtz_types = {}
    if sig_lab: mtz_types[sig_lab] = "Q"

    if blur != 0:
        temp = hkldata.debye_waller_factors(b_iso=blur)
        for lab in data_labs:
            data = hkldata.df[lab]
            newlab = lab + lab_f_suffix(blur)
            if numpy.iscomplexobj(data): data = numpy.abs(data)
            hkldata.df[newlab] = data * temp
            mtz_labs.append(newlab)
            mtz_types[newlab] = "F" if lab != sig_lab else "Q"

    hkldata.write_mtz(mtz_out, labs=mtz_labs, types=mtz_types,
                      phase_label_decorator=lambda x: "P"+x[1:])
# write_map_mtz()

def determine_b_before_mask(st, maps, grid_start, mask, resolution):
    logger.writeln("Determining b_before_mask..")
    # work in masked map for the speed
    new_cell, new_shape, starts, shifts = shift_maps.determine_shape_and_shift(mask=mask,
                                                                               grid_start=grid_start,
                                                                               padding=5,
                                                                               mask_cutoff=0.5,
                                                                               noncentered=True,
                                                                               noncubic=True,
                                                                               json_out=None)
    st = st.clone()
    st.cell = new_cell
    st.spacegroup_hm = "P 1"
    for cra in st[0].all():
        cra.atom.pos += shifts
        cra.atom.b_iso = 0
        cra.atom.aniso = gemmi.SMat33f(0,0,0,0,0,0)

    newmaps = []
    for i in range(len(maps)): # Update maps
        g = gemmi.FloatGrid(maps[i][0].array*mask,
                            maps[i][0].unit_cell, maps[i][0].spacegroup)

        suba = g.get_subarray(starts, new_shape)
        new_grid = gemmi.FloatGrid(suba, new_cell, st.find_spacegroup())
        newmaps.append([new_grid]+maps[i][1:])

    hkldata = utils.maps.mask_and_fft_maps(newmaps, resolution)
    hkldata.df["FC"] = utils.model.calc_fc_fft(st, resolution - 1e-6, source="electron",
                                               miller_array=hkldata.miller_array())
    k, b = hkldata.scale_k_and_b("FC", "FP")
    return -b
# determine_b_before_mask()

def main(args, monlib=None):
    ret = {} # instructions for refinement
    
    if (args.twist, args.rise).count(None) == 1:
        raise SystemExit("ERROR: give both helical paramters --twist and --rise")
    if args.twist is not None:
        logger.writeln("INFO: setting --contacting_only because helical symmetry is given")
        args.contacting_only = True
    if args.no_mask:
        args.mask_radius = None
        if not args.no_trim:
            logger.writeln("WARNING: setting --no_trim because --no_mask is given")
            args.no_trim = True
        if args.mask:
            logger.writeln("WARNING: Your --mask is ignored because --no_mask is given")
            args.mask = None

    #if args.mask_soft_edge > 0:
    #    logger.writeln("INFO: --mask_soft_edge={} is given. Turning off sharpen_before_mask.".format(args.mask_soft_edge))
    #    args.no_sharpen_before_mask = True

    if args.resolution is None and args.model and utils.fileio.splitext(args.model)[1].endswith("cif"):
        doc = gemmi.cif.read(args.model)
        if len(doc) != 1:
            raise SystemExit("cannot find resolution from cif. Give --resolution")
        block = doc.sole_block()
        reso_str = block.find_value("_em_3d_reconstruction.resolution")
        try:
            args.resolution = float(reso_str)
        except:
            raise SystemExit("ERROR: _em_3d_reconstruction.resolution is invalid. Give --resolution")
        logger.writeln("WARNING: --resolution not given. Using _em_3d_reconstruction.resolution = {}".format(reso_str))

    if args.resolution is None:
        raise SystemExit("ERROR: --resolution is needed.")
        
    resolution = args.resolution - 1e-6

    if args.halfmaps:
        maps = utils.fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
    else:
        maps = [utils.fileio.read_ccp4_map(args.map, pixel_size=args.pixel_size)]

    grid_start = maps[0][1]
    unit_cell = maps[0][0].unit_cell
    spacegroup = gemmi.SpaceGroup(1)
    start_xyz = numpy.array(maps[0][0].get_position(*grid_start).tolist())
    A = numpy.array(unit_cell.orthogonalization_matrix.tolist())
    center = numpy.sum(A, axis=1) / 2 #+ start_xyz

    # Create mask
    mask = None
    if args.mask:
        logger.writeln("Input mask file: {}".format(args.mask))
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    
    st_new = None
    if args.model: # and 
        logger.writeln("Input model: {}".format(args.model))
        st = utils.fileio.read_structure(args.model)
        st.cell = unit_cell
        st.spacegroup_hm = "P 1"
        model_format = utils.fileio.check_model_format(args.model)

        if monlib is None:
            # FIXME should use user provided libraries
            monlib = utils.restraints.load_monomer_library(st)

        if not args.no_link_check:
            utils.restraints.find_and_fix_links(st, monlib)

        max_seq_num = max([max(res.seqid.num for res in chain) for model in st for chain in model])
        if max_seq_num > 9999 and model_format == ".pdb":
            logger.writeln("Max residue number ({}) exceeds 9999. Will use mmcif format".format(max_seq_num))
            model_format = ".mmcif"

        if 1: # workaround for Refmac
            # TODO need to check external restraints
            st.entities.clear()
            st.setup_entities()
            topo = gemmi.prepare_topology(st, monlib, ignore_unknown_links=True)
            ret["refmac_fixes"] = utils.refmac.FixForRefmac(st, topo, 
                                                            fix_microheterogeneity=not args.no_fix_microheterogeneity,
                                                            fix_resimax=not args.no_fix_resi9999,
                                                            fix_nonpolymer=False)
            chain_id_len_max = max([len(x) for x in utils.model.all_chain_ids(st)])
            if chain_id_len_max > 1 and model_format == ".pdb":
                logger.writeln("Long chain ID (length: {}) detected. Will use mmcif format".format(chain_id_len_max))
                model_format = ".mmcif"
            if model_format == ".mmcif": ret["refmac_fixes"].fix_nonpolymer(st)
            st.entities.clear()

        ret["model_format"] = model_format

        if not args.keep_multiple_models and len(st) > 1:
            logger.writeln(" Removing models 2-{}".format(len(st)))
            for i in reversed(range(1, len(st))):
                del st[i]

        if len(st.ncs) > 0 and args.ignore_symmetry:
            logger.writeln("Removing symmetry information from model.")
            st.ncs.clear()
        utils.symmetry.update_ncs_from_args(args, st, map_and_start=maps[0], filter_contacting=args.contacting_only)
        st_new = st.clone()
        if len(st.ncs) > 0:
            if not args.no_check_ncs_overlaps and utils.model.check_symmetry_related_model_duplication(st):
                raise SystemExit("\nError: Too many symmetery-related contacts detected.\n"
                                 "It is very likely you gave symmetry-expanded model along with symmetry operators.")
            
            ret["ncsc"] = utils.symmetry.ncs_ops_for_refmac(st.ncs)
            utils.model.expand_ncs(st)
            logger.writeln(" Saving expanded model: input_model_expanded.*")
            utils.fileio.write_model(st, "input_model_expanded", pdb=True, cif=True)

        if mask is not None and not args.no_check_mask_with_model:
            if not utils.maps.test_mask_with_model(mask, st):
                raise SystemExit("\nError: Model is out of mask.\n"
                                 "Please check your --model and --mask. You can disable this test with --no_check_mask_with_model.")
            
        if mask is None and args.mask_radius:
            logger.writeln("Creating mask..")
            mask = utils.maps.mask_from_model(st, args.mask_radius, soft_edge=args.mask_soft_edge, grid=maps[0][0])
            utils.maps.write_ccp4_map("mask_from_model.ccp4", mask)
    else:
        model_format = None
        
    if st_new:
        logger.writeln(" Saving input model with unit cell information")
        utils.fileio.write_model(st_new, "starting_model", pdb=True, cif=True)
        ret["model_file"] = "starting_model" + model_format

    if mask is not None:
        if args.invert_mask:
            logger.writeln("Inverting mask..")
            mask_max, mask_min = numpy.max(mask), numpy.min(mask)
            logger.writeln("  mask_max, mask_min= {}, {}".format(mask_max, mask_min))
            mask = mask_max + mask_min - mask
        
        # Mask maps
        if args.no_sharpen_before_mask or len(maps) < 2:
            logger.writeln("Applying mask..")
            for ma in maps: ma[0].array[:] *= mask
        else:
            logger.writeln("Sharpen-mask-unsharpen..")
            b_before_mask = args.b_before_mask
            if b_before_mask is None: b_before_mask = determine_b_before_mask(st, maps, grid_start, mask, resolution)                
            maps = utils.maps.sharpen_mask_unsharpen(maps, mask, resolution, b=b_before_mask)

        if not args.no_trim:
            logger.writeln(" Shifting maps and/or model..")
            if args.padding is None: args.padding = args.mask_radius * 2
            new_cell, new_shape, starts, shifts = shift_maps.determine_shape_and_shift(mask=mask,
                                                                                       grid_start=grid_start,
                                                                                       padding=args.padding,
                                                                                       mask_cutoff=0.5,
                                                                                       noncentered=True,
                                                                                       noncubic=True,
                                                                                       json_out=None)
            ret["shifts"] = shifts
            vol_mask = numpy.count_nonzero(mask.array>0.5)
            vol_map = new_shape[0] * new_shape[1] * new_shape[2] # XXX assuming all orthogonal
            ret["vol_ratio"] = vol_mask/vol_map
            logger.writeln(" Vol_mask/Vol_map= {:.2e}".format(ret["vol_ratio"]))
            
            if st_new:
                st_new.cell = new_cell
                st_new.spacegroup_hm = "P 1"

                logger.writeln(" Saving model in trimmed map..")
                utils.fileio.write_model(st_new, args.shifted_model_prefix, pdb=True, cif=True)
                ret["model_file"] = args.shifted_model_prefix + model_format

            logger.writeln(" Trimming maps..")
            for i in range(len(maps)): # Update maps
                suba = maps[i][0].get_subarray(starts, new_shape)
                new_grid = gemmi.FloatGrid(suba, new_cell, spacegroup)
                maps[i][0] = new_grid

    hkldata = utils.maps.mask_and_fft_maps(maps, resolution, None)
    hkldata.setup_relion_binning()
    if len(maps) == 2:
        logger.writeln(" Calculating noise variances..")
        map_labs = ["Fmap1", "Fmap2", "Fout"]
        sig_lab = "SIGFout"
        ret["lab_sigf"] = sig_lab + lab_f_suffix(args.blur)
        ret["lab_f_half1"] = "Fmap1" + lab_f_suffix(args.blur)
        # TODO Add SIGF in case of half maps, when refmac is ready
        ret["lab_phi_half1"] = "Pmap1"
        ret["lab_f_half2"] = "Fmap2" + lab_f_suffix(args.blur)
        ret["lab_phi_half2"] = "Pmap2"
        utils.maps.calc_noise_var_from_halfmaps(hkldata)
        hkldata.df[sig_lab] = 0.
        for i_bin, idxes in hkldata.binned():
            hkldata.df.loc[idxes, sig_lab] = numpy.sqrt(hkldata.binned_df["var_noise"][i_bin])

        d_eff_full = hkldata.d_eff("FSCfull")
        logger.writeln("Effective resolution from FSCfull= {:.2f}".format(d_eff_full))
        ret["d_eff"] = d_eff_full
    else:
        map_labs = ["Fout"]
        sig_lab = None

    if args.no_mask:
        logger.writeln("Saving unmasked maps as mtz file..")
        mtzout = args.output_mtz_prefix+".mtz"
    else:
        logger.writeln(" Saving masked maps as mtz file..")
        mtzout = args.output_masked_prefix+"_obs.mtz"

    hkldata.df.rename(columns=dict(F_map1="Fmap1", F_map2="Fmap2", FP="Fout"), inplace=True)
    if "shifts" in ret:
        for lab in map_labs: # apply phase shift
            logger.writeln("  applying phase shift for {} with translation {}".format(lab, -ret["shifts"]))
            hkldata.translate(lab, -ret["shifts"])
        
    write_map_mtz(hkldata, mtzout,
                  map_labs=map_labs, sig_lab=sig_lab, blur=args.blur)
    ret["mtz_file"] = mtzout
    ret["lab_f"] = "Fout" + lab_f_suffix(args.blur)
    ret["lab_phi"] = "Pout"
    return ret
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
