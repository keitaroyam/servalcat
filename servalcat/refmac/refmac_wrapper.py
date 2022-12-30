"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import json
import os
import sys
import tempfile
import subprocess
import argparse
from collections import OrderedDict
from servalcat.utils import logger
from servalcat.refmac import refmac_keywords
from servalcat import utils

def add_arguments(parser):
    parser.description = 'Run REFMAC5 with gemmi-prepared restraints'
    parser.add_argument('--exe', default="refmac5", help='refmac5 binary')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("opts", nargs="+",
                        help="HKLIN hklin XYZIN xyzin...")
    parser.add_argument('--auto_box_with_padding', type=float, help="Determine box size from model with specified padding")
    parser.add_argument('--no_adjust_hydrogen_distances', action='store_true', help="By default it adjusts hydrogen distances using ideal values. This option is to disable it.")
    parser.add_argument('--keep_original_output', action='store_true', help="with .org extension")
    parser.add_argument('--prefix', help="output prefix")

    # TODO --cell to override unit cell?

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def parse_keywords(inputs):
    # these make keywords will be ignored (just passed to refmac): hout,ribo,valu,spec,form,sdmi,segi
    ret = refmac_keywords.parse_keywords(inputs)
    def sorry(s): raise SystemExit("Sorry, '{}' is not supported".format(s))
    if ret["make"].get("hydr") == "f":
        sorry("make hydr full")
    if ret["make"].get("buil") == "y":
        sorry("make build yes")
    return ret
# parse_keywords()

def prepare_crd(xyzin, crdout, ligand, make, monlib_path=None, h_pos="elec", auto_box_with_padding=None,
                no_adjust_hydrogen_distances=False):
    assert h_pos in ("elec", "nucl")
    h_change = dict(a=gemmi.HydrogenChange.ReAddButWater,
                    y=gemmi.HydrogenChange.NoChange,
                    n=gemmi.HydrogenChange.Remove)[make.get("hydr", "a")]
    st = utils.fileio.read_structure(xyzin)
    if not st.cell.is_crystal():
        if auto_box_with_padding is not None:
            st.cell = utils.model.box_from_model(st[0], auto_box_with_padding)
            st.spacegroup_hm = "P 1"
            logger.writeln("Box size from the model with padding of {}: {}".format(auto_box_with_padding, st.cell.parameters))
        else:
            raise SystemExit("Error: unit cell is not defined in the model.")


    # we do not have DOD. will not change ND4->NH4 and SPW->SPK, as hydrogen atom names are different
    for chain in st[0]:
        for res in chain:
            if res.name == "DOD":
                logger.writeln("Warning: changing DOD to HOH (chain {} residue {})".format(chain.name, res.seqid))
                res.name = "HOH"

    utils.model.setup_entities(st, clear=True, force_subchain_names=True)

    # TODO read dictionary from xyzin (priority: user cif -> monlib -> xyzin
    try:
        monlib = utils.restraints.load_monomer_library(st,
                                                       monomer_dir=monlib_path,
                                                       cif_files=ligand,
                                                       stop_for_unknowns=not make.get("newligand"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if make.get("cispept", "y") == "y": st.assign_cis_flags()

    utils.restraints.find_and_fix_links(st, monlib, add_found=(make.get("link", "n")=="y"))
    for con in st.connections:
        if con.link_id not in ("?", "", "gap") and con.link_id not in monlib.links:
            logger.writeln(" removing unknown link id ({}). Ad-hoc link will be generated.".format(con.link_id))
            con.link_id = ""

    refmac_fixes = None
    max_seq_num = max([max(res.seqid.num for res in chain) for model in st for chain in model])
    if max_seq_num > 9999:
        logger.writeln("Max residue number ({}) exceeds 9999. Needs workaround.".format(max_seq_num))
        topo = gemmi.prepare_topology(st, monlib, ignore_unknown_links=True)
        refmac_fixes = utils.refmac.FixForRefmac(st, topo, 
                                                 fix_microheterogeneity=False,
                                                 fix_resimax=True,
                                                 fix_nonpolymer=False)

    if make.get("hydr") == "a": logger.writeln("(re)generating hydrogen atoms")
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=logger, reorder=True, ignore_unknown_links=False) # we should remove logger here??
    if make.get("hydr") != "n" and st[0].has_hydrogen():
        if h_pos == "nucl" and (make.get("hydr") == "a" or not no_adjust_hydrogen_distances):
            resnames = st[0].get_all_residue_names()
            utils.restraints.check_monlib_support_nucleus_distances(monlib, resnames)
            logger.writeln("adjusting hydrogen position to nucleus")
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=1.1)
        elif h_pos == "elec" and make.get("hydr") == "y" and not no_adjust_hydrogen_distances:
            logger.writeln("adjusting hydrogen position to electron cloud")
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)

    doc = gemmi.prepare_refmac_crd(st, topo, monlib, h_change)
    doc.write_file(crdout, style=gemmi.cif.Style.NoBlankLines)
    logger.writeln("crd file written: {}".format(crdout))
    return refmac_fixes
# prepare_crd()

def get_output_model_names(xyzout):
    # ref: WRITE_ATOMS_REFMAC in oppro_allocate.f
    if xyzout is None: xyzout = "XYZOUT"
    pdb, mmcif = "", ""
    if len(xyzout) > 3:
        if xyzout.lower().endswith("pdb"):
            mmcif = xyzout[:-4] + ".mmcif"
            pdb = xyzout
        else:
            if xyzout.lower().endswith("cif") and len(xyzout) > 5:
                if xyzout.lower().endswith("mmcif"):
                    mmcif = xyzout
                    pdb = xyzout[:-6] + ".pdb"
                else:
                    mmcif = xyzout
                    pdb = xyzout[:-4] + ".pdb"
            else:
                mmcif = xyzout + ".mmcif"
                pdb = xyzout
    else:
        mmcif = xyzout + ".mmcif"
        pdb = xyzout
        
    return pdb, mmcif
# get_output_model_names()

def modify_output(pdbout, cifout, fixes, keep_original_output=False):
    st = utils.fileio.read_structure(cifout)
    if os.path.exists(pdbout):
        st.raw_remarks = gemmi.read_pdb(pdbout).raw_remarks
    if fixes is not None:
        fixes.modify_back(st)

    suffix = ".org"
    os.rename(cifout, cifout + suffix)
    utils.fileio.write_mmcif(st, cifout, cifout + suffix)
    
    chain_id_len_max = max([len(x) for x in utils.model.all_chain_ids(st)])
    seqnums = [res.seqid.num for chain in st[0] for res in chain]
    if chain_id_len_max > 1 or min(seqnums) <= -1000 or max(seqnums) >= 10000:
        logger.writeln("This structure cannot be saved as an official PDB format. Using hybrid-36. Header part may be inaccurate.")
    st.expand_hd_mixture()
    os.rename(pdbout, pdbout + suffix)
    utils.fileio.write_pdb(st, pdbout)
    if not keep_original_output:
        os.remove(pdbout + suffix)
        os.remove(cifout + suffix)
# modify_output()

def main(args):
    if len(args.opts) % 2 != 0: raise SystemExit("Invalid number of args")
    args.ligand = sum(args.ligand, []) if args.ligand else []
    try:
        refmac_ver = utils.refmac.check_version(args.exe)
    except OSError as e:
        raise SystemExit("Error: Cannot execute {}. Check Refmac instllation or use --exe to give the location.\n{}".format(args.exe, e))
    if not refmac_ver:
        raise SystemExit("Error: Cannot get Refmac version.")
    logger.writeln("Refmac version: {}".format(".".join(str(x) for x in refmac_ver)))
    if refmac_ver < (5, 8, 404):
        raise SystemExit("Error: this version of Refmac is not supported. Update to 5.8.404 or newer")
    print("Waiting for input..")
    inputs = []
    for l in sys.stdin:
        inputs.append(l)
        if l.strip().lower().startswith("end"):
            break

    opts = OrderedDict((args.opts[2*i].lower(), args.opts[2*i+1]) for i in range(len(args.opts)//2))
    xyzin = opts.get("xyzin")
    xyzout = opts.get("xyzout")
    libin = opts.pop("libin", None)
    if libin: args.ligand.append(libin)
    if not args.monlib:
        # if --monlib is given, it has priority.
        args.monlib = opts.pop("clibd_mon", None)
    for k in ("temp1", "scrref"): # scrref has priority
        if k in opts:
            logger.writeln("updating CCP4_SCR from {}={}".format(k, opts[k]))
            os.environ["CCP4_SCR"] = os.path.dirname(opts[k]) # XXX "." may be given, which causes problem (os.path.isdir("") is False)
    utils.refmac.ensure_ccp4scr()
    
    keywords = parse_keywords(inputs) # TODO read psrestin also?

    # TODO what if restin is given or make cr prepared is given?
    # TODO check make pept/link/suga/ss/conn/symm/chain

    # Process model
    crdout = None
    refmac_fixes = None
    if xyzin is not None:
        #tmpfd, crdout = tempfile.mkstemp(prefix="gemmi_", suffix=".crd") # TODO use dir=CCP4_SCR
        #os.close(tmpfd)
        crdout = "gemmi_{}_{}.crd".format(utils.fileio.splitext(os.path.basename(xyzin))[0], os.getpid())
        refmac_fixes = prepare_crd(xyzin, crdout, args.ligand, make=keywords["make"], monlib_path=args.monlib,
                                   h_pos="nucl" if keywords.get("source")=="ne" else "elec",
                                   auto_box_with_padding=args.auto_box_with_padding,
                                   no_adjust_hydrogen_distances=args.no_adjust_hydrogen_distances)
        opts["xyzin"] = crdout

    if keywords["make"].get("exit"):
        return

    if args.prefix:
        if "xyzin" in opts and "xyzout" not in opts: opts["xyzout"] = args.prefix + ".pdb"
        if "hklin" in opts and "hklout" not in opts: opts["hklout"] = args.prefix + ".mtz"
        if "tlsin" in opts and "tlsout" not in opts: opts["tlsout"] = args.prefix + ".tls"
        
    # Run Refmac
    cmd = [args.exe] + list(sum(tuple(opts.items()), ()))
    env = os.environ
    logger.writeln("Running REFMAC5..")
    if args.monlib:
        logger.writeln("CLIBD_MON={}".format(args.monlib))
        env["CLIBD_MON"] = os.path.join(args.monlib, "") # should end with /
    logger.writeln(" ".join(cmd))
    p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True, env=env)
    if crdout: p.stdin.write("make cr prepared\n")
    p.stdin.write("".join(inputs))
    p.stdin.close()
    for l in iter(p.stdout.readline, ""):
        logger.write(l)
    retcode = p.wait()
    logger.writeln("\nRefmac finished with exit code= {}".format(retcode))

    # Modify output
    if xyzin is not None:
        pdbout, cifout = get_output_model_names(xyzout)
        if os.path.exists(cifout):
            modify_output(pdbout, cifout, refmac_fixes, args.keep_original_output)
# main()

def command_line():
    import sys
    args = parse_args(sys.argv[1:])
    if args.prefix:
        logger.set_file(args.prefix + ".log")
    logger.write_header(command="refmac")
    main(args)

if __name__ == "__main__":
    command_line()
