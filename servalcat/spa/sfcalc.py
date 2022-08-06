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
    group.add_argument('--no_shift',
                       action='store_true',
                       help='Keep map origin so that output maps overlap with the input maps. '
                            'Now this is on by default. Use --shift_if_trim if you want the previous default behavior.')
    group.add_argument('--shift_if_trim',
                       action='store_true',
                       help='Shift model and map origin if map is trimmed. This option is to emulate previous default behavior.')
    parser.add_argument('--blur',
                        nargs="+", # XXX probably no need to be multiple
                        type=float,
                        help='Sharpening or blurring B')
    utils.symmetry.add_symmetry_args(parser) # add --pg etc
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

"""
Possible choices:
 - no mask => no shift AND no trim
 - mask AND trim => shift OR no shift
 - mask AND no trim => no shift
"""

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

def write_map_mtz(hkldata, mtz_out, map_labs, sig_lab=None, blurs=None): # TODO use hkldata.write_mtz
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
            if b != 0: labf = "{}{}".format(labf, lab_f_suffix(b))
            mtz.add_column(labf, "F")
        mtz.add_column("P"+lab_root, "P")
    if sig_lab:
        for b in blurs:
            labsigf = sig_lab
            if b != 0: labsigf = "{}{}".format(labsigf, lab_f_suffix(b))
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

def determine_b_before_mask(st, maps, grid_start, mask, resolution):
    logger.write("Determining b_before_mask..")
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

    if args.no_shift:
        logger.write("DeprecationWarning: --no_shift is now on by default, and this option will be removed in the future.")
    if not args.shift_if_trim:
        args.no_shift = True
    
    if (args.twist, args.rise).count(None) == 1:
        raise SystemExit("ERROR: give both helical paramters --twist and --rise")

    if args.no_mask:
        args.mask_radius = None
        if not args.no_shift:
            logger.write("WARNING: setting --no_shift because --no_mask is given")
            args.no_shift = True
        if not args.no_trim:
            logger.write("WARNING: setting --no_trim because --no_mask is given")
            args.no_trim = True
        if args.mask:
            logger.write("WARNING: Your --mask is ignored because --no_mask is given")
            args.mask = None
    elif args.no_trim and not args.no_shift:
        logger.write("WARNING: setting --no_shift because --no_trim is given (and --no_mask not given)")
        args.no_shift = True
        

    if args.resolution is None and args.model and utils.fileio.splitext(args.model)[1].endswith("cif"):
        doc = gemmi.cif.read(args.model)
        if len(doc) != 1:
            raise SystemExit("cannot find resolution from cif. Give --resolution")
        block = doc.sole_block()
        reso_str = block.find_value("_em_3d_reconstruction.resolution")
        try:
            float(reso_str)
        except:
            raise SystemExit("ERROR: _em_3d_reconstruction.resolution is invalid. Give --resolution")
        logger.write("WARNING: --resolution not given. Using _em_3d_reconstruction.resolution = {}".format(reso_str))
        args.resolution = float(reso_str)

    if args.resolution is None:
        raise SystemExit("ERROR: --resolution is needed.")
        
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
    center = numpy.sum(A, axis=1) / 2 #+ start_xyz

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
        mask = utils.fileio.read_ccp4_map(args.mask)[0]
    
    st_new = None
    if args.model: # and 
        logger.write("Input model: {}".format(args.model))
        st = utils.fileio.read_structure(args.model)
        st.cell = unit_cell
        st.spacegroup_hm = "P 1"

        if monlib is None:
            # FIXME should use user provided libraries
            monlib = utils.restraints.load_monomer_library(st)

        if not args.no_link_check:
            utils.restraints.find_and_fix_links(st, monlib)

        if not args.no_fix_microheterogeneity or not args.no_fix_resi9999:
            # TODO need to check external restraints
            st.entities.clear()
            st.setup_entities()
            topo = gemmi.prepare_topology(st, monlib, ignore_unknown_links=True)
            ret["refmac_fixes"] = utils.refmac.FixForRefmac(st, topo, 
                                                            fix_microheterogeneity=not args.no_fix_microheterogeneity,
                                                            fix_resimax=not args.no_fix_resi9999)
            st.entities.clear()
        model_format = utils.fileio.check_model_format(args.model)
        chain_id_len_max = max([len(x) for x in utils.model.all_chain_ids(st)])
        if chain_id_len_max > 1 and model_format == ".pdb":
            logger.write("Long chain ID (length: {}) detected. Will use mmcif format".format(chain_id_len_max))
            model_format = ".mmcif"

        max_seq_num = max([max(res.seqid.num for res in chain) for model in st for chain in model])
        if max_seq_num > 9999 and model_format == ".pdb":
            logger.write("Max residue number ({}) exceeds 9999. Will use mmcif format".format(max_seq_num))
            model_format = ".mmcif"

        ret["model_format"] = model_format

        if not args.keep_multiple_models and len(st) > 1:
            logger.write(" Removing models 2-{}".format(len(st)))
            for i in reversed(range(1, len(st))):
                del st[i]

        if len(st.ncs) > 0:
            if args.ignore_symmetry:
                logger.write("Removing symmetry information from model.")
                st.ncs.clear()
            else:
                # remove already-applied symmetries, which can confuse refmac
                for i in reversed(range(len(st.ncs))):
                    if st.ncs[i].given: del st.ncs[i]

        utils.symmetry.update_ncs_from_args(args, st, map_and_start=maps[0], filter_model_helical_contacting=True)
        st_new = st.clone()
        if len(st.ncs) > 0:
            if not args.no_check_ncs_overlaps and utils.model.check_symmetry_related_model_duplication(st):
                raise SystemExit("\nError: Too many symmetery-related contacts detected.\n"
                                 "It is very likely you gave symmetry-expanded model along with symmetry operators.")
            
            logger.write(" Writing NCS file")
            utils.symmetry.write_NcsOps_for_refmac(st.ncs, "ncsc.txt")
            ret["ncsc_file"] = "ncsc.txt"
        
            utils.model.expand_ncs(st)
            logger.write(" Saving expanded model: input_model_expanded.*")
            utils.fileio.write_model(st, "input_model_expanded", pdb=True, cif=True)

        if mask is not None and not args.no_check_mask_with_model:
            if not utils.maps.test_mask_with_model(mask, st):
                raise SystemExit("\nError: Model is out of mask.\n"
                                 "Please check your --model and --mask. You can disable this test with --no_check_mask_with_model.")
            
        if mask is None and args.mask_radius:
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
        
    if args.no_shift and st_new:
        logger.write(" Saving input model with unit cell information")
        utils.fileio.write_model(st_new, "starting_model", pdb=True, cif=True)
        ret["model_file"] = "starting_model" + model_format

    if mask is not None:
        if args.invert_mask:
            logger.write("Inverting mask..")
            mask_max, mask_min = numpy.max(mask), numpy.min(mask)
            logger.write("  mask_max, mask_min= {}, {}".format(mask_max, mask_min))
            mask = mask_max + mask_min - mask
        
        # Mask maps
        if args.no_sharpen_before_mask or len(maps) < 2:
            logger.write("Applying mask..")
            for ma in maps: ma[0].array[:] *= mask
        else:
            logger.write("Sharpen-mask-unsharpen..")
            b_before_mask = args.b_before_mask
            if b_before_mask is None: b_before_mask = determine_b_before_mask(st, maps, grid_start, mask, resolution)                
            maps = utils.maps.sharpen_mask_unsharpen(maps, mask, resolution, b=b_before_mask)

        if not args.no_trim:
            logger.write(" Shifting maps and/or model..")
            if args.padding is None: args.padding = args.mask_radius * 2
            new_cell, new_shape, starts, shifts = shift_maps.determine_shape_and_shift(mask=mask,
                                                                                       grid_start=grid_start,
                                                                                       padding=args.padding,
                                                                                       mask_cutoff=0.5,
                                                                                       noncentered=True,
                                                                                       noncubic=True,
                                                                                       json_out="shifts.json")
            ret["shifts"] = shifts
            vol_mask = numpy.count_nonzero(mask.array>0.5)
            vol_map = new_shape[0] * new_shape[1] * new_shape[2] # XXX assuming all orthogonal
            ret["vol_ratio"] = vol_mask/vol_map
            logger.write(" Vol_mask/Vol_map= {:.2e}".format(ret["vol_ratio"]))
            
            if st_new:
                st_new.cell = new_cell
                st_new.spacegroup_hm = "P 1"

                if not args.no_shift:
                    for cra in st_new[0].all():
                        cra.atom.pos += shifts
                
                if not args.no_shift and len(st_new.ncs) > 0:
                    new_ops = utils.symmetry.apply_shift_for_ncsops(st_new.ncs, shifts)
                    st_new.ncs.clear()
                    st_new.ncs.extend(new_ops)
                    logger.write(" Writing NCS file for shifted model")
                    utils.symmetry.write_NcsOps_for_refmac(st_new.ncs, "ncsc_{}.txt".format(args.shifted_model_prefix))
                    ret["ncsc_file"] = "ncsc_{}.txt".format(args.shifted_model_prefix)
                    logger.write(" Writing symmetry expanded model for shifted model")
                    utils.symmetry.write_symmetry_expanded_model(st_new, "{}_expanded".format(args.shifted_model_prefix),
                                                                 pdb=True, cif=True)

                logger.write(" Saving model in trimmed map..")
                utils.fileio.write_model(st_new, args.shifted_model_prefix, pdb=True, cif=True)
                ret["model_file"] = args.shifted_model_prefix + model_format # the same name whether --no_shift given or not

            logger.write(" Trimming maps..")
            for i in range(len(maps)): # Update maps
                suba = maps[i][0].get_subarray(starts, new_shape)
                new_grid = gemmi.FloatGrid(suba, new_cell, spacegroup)
                maps[i][0] = new_grid

    blur0 = args.blur[0] if args.blur else None
    hkldata = utils.maps.mask_and_fft_maps(maps, resolution, None)
    hkldata.setup_relion_binning()
    if len(maps) == 2:
        logger.write(" Calculating noise variances..")
        map_labs = ["Fmap1", "Fmap2", "Fout"]
        sig_lab = "SIGFout"
        ret["lab_sigf"] = sig_lab + lab_f_suffix(blur0)
        ret["lab_f_half1"] = "Fmap1" + lab_f_suffix(blur0)
        # TODO Add SIGF in case of half maps, when refmac is ready
        ret["lab_phi_half1"] = "Pmap1"
        ret["lab_f_half2"] = "Fmap2" + lab_f_suffix(blur0)
        ret["lab_phi_half2"] = "Pmap2"
        utils.maps.calc_noise_var_from_halfmaps(hkldata)
        hkldata.df[sig_lab] = 0.
        for i_bin, idxes in hkldata.binned():
            hkldata.df.loc[idxes, sig_lab] = numpy.sqrt(hkldata.binned_df["var_noise"][i_bin])

        d_eff_full = hkldata.d_eff("FSCfull")
        logger.write("Effective resolution from FSCfull= {:.2f}".format(d_eff_full))
        ret["d_eff"] = d_eff_full
    else:
        map_labs = ["Fout"]
        sig_lab = None

    if args.no_mask:
        logger.write("Saving unmasked maps as mtz file..")
        mtzout = args.output_mtz_prefix+".mtz"
    else:
        logger.write(" Saving masked maps as mtz file..")
        mtzout = args.output_masked_prefix+"_obs.mtz"

    hkldata.df.rename(columns=dict(F_map1="Fmap1", F_map2="Fmap2", FP="Fout"), inplace=True)
    if "shifts" in ret and args.no_shift:
        for lab in map_labs: # apply phase shift
            logger.write("  applying phase shift for {} with translation {}".format(lab, -ret["shifts"]))
            hkldata.translate(lab, -ret["shifts"])
        
    write_map_mtz(hkldata, mtzout,
                  map_labs=map_labs, sig_lab=sig_lab, blurs=args.blur)
    ret["mtz_file"] = mtzout
    ret["lab_f"] = "Fout" + lab_f_suffix(blur0)
    ret["lab_phi"] = "Pout"
    return ret
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
