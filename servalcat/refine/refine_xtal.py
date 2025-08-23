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
import shutil
import argparse
from servalcat.utils import logger
from servalcat import utils
from servalcat.xtal.sigmaa import decide_mtz_labels, process_input, calculate_maps, calculate_maps_int, calculate_maps_twin
from servalcat.refine.xtal import LL_Xtal
from servalcat.refine.refine import Geom, Refine, RefineParams, update_meta, print_h_options, load_config
from servalcat.refmac import refmac_keywords
from servalcat import ext
b_to_u = utils.model.b_to_u

def add_arguments(parser):
    parser.description = "program to refine crystallographic structures"
    parser.add_argument("--hklin", required=True)
    parser.add_argument('--hklin_free',
                        help='Input MTZ file for test flags')
    parser.add_argument("-d", '--d_min', type=float)
    parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int,
                        help="Number of bins for statistics (default: auto)")
    parser.add_argument('--nbins_ml', type=int,
                        help="Number of bins for ML parameters (default: auto)")
    parser.add_argument("--labin", help="F,SIGF,FREE input")
    parser.add_argument('--labin_free',
                        help='MTZ column of --hklin_free')
    parser.add_argument('--labin_llweight',
                        help=argparse.SUPPRESS) # testing
    parser.add_argument('--free', type=int,
                        help='flag number for test set')
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--ligand', nargs="*", action="append",
                        help="restraint dictionary cif file(s)")
    parser.add_argument('--newligand_continue', action='store_true',
                        help="Make ad-hoc restraints for unknown ligands (not recommended)")
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no"],
                        help="all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input. "
                        "Default: %(default)s")
    parser.add_argument('--hout', action='store_true', help="write hydrogen atoms in the output model")
    parser.add_argument('--jellybody', action='store_true',
                        help="Use jelly body restraints")
    parser.add_argument('--jellybody_params', nargs=2, type=float,
                        metavar=("sigma", "dmax"), default=[0.01, 4.2],
                        help="Jelly body sigma and dmax (default: %(default)s)")
    parser.add_argument('--jellyonly', action='store_true',
                        help="Jelly body only (experimental, may not be useful)")
    parser.add_argument('--find_links', action='store_true', 
                        help='Automatically add links')
    parser.add_argument('--keywords', nargs='+', action="append",
                        help="refmac keyword(s)")
    parser.add_argument('--keyword_file', nargs='+', action="append",
                        help="refmac keyword file(s)")
    parser.add_argument('--randomize', type=float, default=0,
                        help='Shake coordinates with specified rmsd')
    parser.add_argument('--ncycle', type=int, default=10,
                        help="number of CG cycles (default: %(default)d)")
    parser.add_argument('--weight', type=float,
                        help="refinement weight (default: auto)")
    parser.add_argument('--no_weight_adjust', action='store_true', 
                        help='Do not adjust weight during refinement')
    parser.add_argument('--target_bond_rmsz_range', nargs=2, type=float, default=[0.5, 1.],
                        help='Bond rmsz range for weight adjustment (default: %(default)s)')
    parser.add_argument('--ncsr', action='store_true', 
                        help='Use local NCS restraints')
    parser.add_argument('--adpr_weight', type=float, default=1.,
                        help="ADP restraint weight (default: %(default)f)")
    parser.add_argument('--occr_weight', type=float, default=0.,
                        help="Occupancy restraint weight (default: %(default)f)")
    parser.add_argument('--bfactor', type=float,
                        help="reset all atomic B values to specified value")
    parser.add_argument('--fix_xyz', action="store_true")
    parser.add_argument('--adp',  choices=["fix", "iso", "aniso"], default="iso")
    parser.add_argument('--refine_all_occ', action="store_true")
    parser.add_argument('--max_dist_for_adp_restraint', type=float, default=4.)
    parser.add_argument('--adp_restraint_power', type=float)
    parser.add_argument('--adp_restraint_exp_fac', type=float)
    parser.add_argument('--adp_restraint_no_long_range', action='store_true')
    parser.add_argument('--adp_restraint_mode', choices=["diff", "kldiv"], default="kldiv")
    parser.add_argument('--unrestrained',  action='store_true', help="No positional restraints")
    parser.add_argument('--refine_h', action="store_true", help="Refine hydrogen (default: restraints only)")
    parser.add_argument('--refine_dfrac', action="store_true", help="Refine deuterium fraction (neutron only)")
    parser.add_argument('--twin', action="store_true", help="Turn on twin refinement")
    parser.add_argument("-s", "--source", choices=["electron", "xray", "neutron"], required=True)
    parser.add_argument('--no_solvent',  action='store_true',
                        help="Do not consider bulk solvent contribution")
    parser.add_argument('--use_work_in_est',  action='store_true',
                        help="Use work reflections in ML parameter estimates")
    parser.add_argument('--keep_charges',  action='store_true',
                        help="Use scattering factor for charged atoms. Use it with care.")
    parser.add_argument("--keep_entities", action='store_true',
                        help="Do not override entities")
    parser.add_argument('--allow_unusual_occupancies', action="store_true", help="Allow negative or more than one occupancies")
    parser.add_argument('-o','--output_prefix')
    parser.add_argument("--write_trajectory", action='store_true',
                        help="Write all output from cycles")
    parser.add_argument("--vonmises", action='store_true',
                        help="Experimental: von Mises type restraint for angles")
    parser.add_argument("--prefer_intensity", action='store_true')
    parser.add_argument("--config",
                        help="Config file (.yaml)")
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def main(args):
    if args.refine_dfrac and args.source != "neutron": # TODO should check params.is_refined()
        raise SystemExit("--refine_dfrac can only be used for the neutron source")
    if args.labin_llweight and args.twin:
        raise SystemExit("--labin_llweight not supported for twin refinement")
    if args.ligand: args.ligand = sum(args.ligand, [])
    if not args.output_prefix:
        args.output_prefix = utils.fileio.splitext(os.path.basename(args.model))[0] + "_refined"

    keywords = []
    if args.keywords or args.keyword_file:
        if args.keywords: keywords = sum(args.keywords, [])
        if args.keyword_file: keywords.extend(l for f in sum(args.keyword_file, []) for l in open(f))
    params = refmac_keywords.parse_keywords(keywords)
    refine_cfg = load_config(args.config, args, params)
    hklin = args.hklin
    labin = args.labin
    if labin is not None:
        labin = labin.split(",")
    elif utils.fileio.is_mmhkl_file(hklin):
        hklin = utils.fileio.read_mmhkl(hklin)
        labin = decide_mtz_labels(hklin, prefer_intensity=args.prefer_intensity)
    software_items = utils.fileio.software_items_from_mtz(hklin)
    try:
        hkldata, sts, fc_labs, args.free, use_in_est = process_input(
            hklin=hklin,
            labin=labin,
            n_bins_ml=args.nbins_ml,
            n_bins_stat=args.nbins,
            free=args.free,
            xyzins=[args.model],
            source=args.source,
            d_max=args.d_max,
            d_min=args.d_min,
            use="work" if args.use_work_in_est else "test",
            max_mlbins=30,
            keep_charges=args.keep_charges,
            allow_unusual_occupancies=args.allow_unusual_occupancies,
            hklin_free=args.hklin_free,
            labin_free=args.labin_free,
            labin_llweight=args.labin_llweight)

    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if "FREE" in hkldata.df:
        use_in_target = "work"
    else:
        use_in_target = "all"

    is_int = "I" in hkldata.df
    st = sts[0]
    utils.model.fix_deuterium_residues(st)
    if args.unrestrained:
        monlib = gemmi.MonLib()
        topo = None
        if args.hydrogen == "yes":
            h_change = gemmi.HydrogenChange.NoChange
        else:
            h_change = gemmi.HydrogenChange.Remove
            st.remove_hydrogens()
            if args.hydrogen == "all":
                logger.writeln("\nWARNING: in unrestrained refinement hydrogen atoms are not generated.\n")
        for i, cra in enumerate(st[0].all()):
            cra.atom.serial = i + 1
    else:
        try:
            monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                           stop_for_unknowns=not args.newligand_continue,
                                                           params=params)
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))
        if not args.keep_entities:
            utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
        utils.restraints.find_and_fix_links(st, monlib, find_metal_links=args.find_links,
                                            add_found=args.find_links)
        h_change = {"all":gemmi.HydrogenChange.ReAddKnown,
                    "yes":gemmi.HydrogenChange.NoChange,
                    "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
        try:
            topo, _ = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                        check_hydrogen=(args.hydrogen=="yes"),
                                                        params=params)
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))

        if h_change == gemmi.HydrogenChange.ReAddKnown and args.source == "neutron":
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus,
                                           default_scale=utils.restraints.default_proton_scale)

    print_h_options(h_change, st[0].has_hydrogen(), args.refine_h, args.hout, geom_only=False)
    
    # initialize values
    utils.model.reset_adp(st[0], args.bfactor, args.adp)
    utils.model.initialize_values(st[0], refine_cfg.initialisation)
        
    # auto weight
    if args.weight is None:
        logger.writeln("Estimating weight using resolution")
        reso = hkldata.d_min_max()[0]
        args.weight = numpy.exp(reso * 0.9104 + 0.2162)
        logger.writeln(" Will use weight= {:.2f}".format(args.weight))
        
    if args.ncsr:
        ncslist = utils.restraints.prepare_ncs_restraints(st)
    else:
        ncslist = False
    if args.refine_dfrac:
        # needed to make Fc calculation expand H/D (ugh)
        st.has_d_fraction = True
    refine_params = RefineParams(st, refine_xyz=not args.fix_xyz,
                                 adp_mode=dict(fix=0, iso=1, aniso=2)[args.adp],
                                 refine_occ=args.refine_all_occ,
                                 refine_dfrac=args.refine_dfrac, cfg=refine_cfg,
                                 exclude_h_ll=not args.refine_h)
    geom = Geom(st, topo, monlib, refine_params,
                shake_rms=args.randomize, adpr_w=args.adpr_weight, occr_w=args.occr_weight, params=params,
                unrestrained=args.unrestrained or args.jellyonly, use_nucleus=(args.source=="neutron"),
                ncslist=ncslist)
    geom.geom.angle_von_mises = args.vonmises
    geom.geom.adpr_max_dist = args.max_dist_for_adp_restraint
    if args.adp_restraint_power is not None: geom.geom.adpr_d_power = args.adp_restraint_power
    if args.adp_restraint_exp_fac is not None: geom.geom.adpr_exp_fac = args.adp_restraint_exp_fac
    if args.adp_restraint_no_long_range: geom.geom.adpr_long_range = False
    geom.geom.adpr_mode = args.adp_restraint_mode
    if args.jellybody or args.jellyonly:
        geom.geom.ridge_sigma, geom.geom.ridge_dmax = args.jellybody_params
    if args.jellyonly: geom.geom.ridge_exclude_short_dist = False

    ll = LL_Xtal(hkldata, args.free, st, monlib, source=args.source,
                 use_solvent=not args.no_solvent, use_in_est=use_in_est, use_in_target=use_in_target,
                 twin=args.twin)
    refiner = Refine(st, geom, refine_cfg, refine_params, ll=ll,
                     unrestrained=args.unrestrained)

    stats = refiner.run_cycles(args.ncycle, weight=args.weight,
                               weight_adjust=not args.no_weight_adjust,
                               weight_adjust_bond_rmsz_range=args.target_bond_rmsz_range,
                               stats_json_out=args.output_prefix + "_stats.json")
    update_meta(st, stats[-1], ll)
    st.meta.software = software_items + st.meta.software
    refiner.st.name = args.output_prefix
    utils.fileio.write_model(refiner.st, args.output_prefix, pdb=True, cif=True, hout=args.hout)
    if st.has_d_fraction: # XXX make hout when neutron
        st_hd_expand = refiner.st.clone()
        st_hd_expand.store_deuterium_as_fraction(False)
        utils.fileio.write_model(st_hd_expand, args.output_prefix + "_hd_expand", pdb=True, cif=True, hout=True)
    if refine_cfg.write_trajectory:
        utils.fileio.write_model(refiner.st_traj, args.output_prefix + "_traj", cif=True)

    if ll.twin_data:
        # replace hkldata
        hkldata = calculate_maps_twin(ll.hkldata, ll.b_aniso, ll.fc_labs, ll.D_labs, ll.twin_data,
                                      use=use_in_target)
    elif is_int:
        calculate_maps_int(ll.hkldata, ll.b_aniso, ll.fc_labs, ll.D_labs, use=use_in_target)
    else:
        calculate_maps(ll.hkldata, ll.b_aniso, ll.fc_labs, ll.D_labs, args.output_prefix + "_stats.log",
                       use=use_in_target)

    # Write mtz file
    if ll.twin_data:
        labs = ["F_est", "F_exp"]
    elif is_int:
        labs = ["I", "SIGI", "F_est"]
    else:
        labs = ["FP", "SIGFP"]
    labs.extend(["FOM", "FWT", "DELFWT", "FC"])
    if "FAN" in hkldata.df:
        labs.append("FAN")
    if not args.no_solvent:
        labs.append("FCbulk")
    if "FREE" in hkldata.df:
        labs.append("FREE")
    if args.labin_llweight:
        labs.append("llweight")
    labs += ll.D_labs + ["S"] # for debugging, for now
    mtz_out = args.output_prefix+".mtz"
    hkldata.write_mtz(mtz_out, labs=labs, types={"FOM": "W", "FP":"F", "SIGFP":"Q", "I":"J", "SIGI":"Q", "F_est": "F", "F_exp": "F", "llweight": "R"})

# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
