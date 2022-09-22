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
from servalcat.utils import logger
from servalcat import utils

def add_arguments(parser):
    parser.description = 'Run REFMAC5 with gemmi-prepared restraints'
    parser.add_argument('--exe', default="refmac5", help='refmac5 binary')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("opts", nargs="+",
                        help="HKLIN hklin XYZIN xyzin...")

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def get_opt(opts, kwd):
    for i in range(len(opts)-1):
        if opts[i].lower() == kwd.lower():
            return opts[i+1], opts[:i] + opts[i+2:]
    return None, opts
# get_opt()

def parse_keywords(inputs):
    # these make keywords will be ignored (just passed to refmac): hout,ribo,valu,spec,form,sdmi,segi
    ret = {"make":{}}
    for l in inputs:
        if "!" in l: l = l[:l.index("!")]
        if "#" in l: l = l[:l.index("#")]
        s = l.split()
        ntok = len(s)
        if s[0].lower().startswith("make"):
            itk = 1
            r = ret["make"]
            while itk < ntok:
                if s[itk].lower().startswith("hydr"): #default a
                    r["hydr"] = s[itk+1][0].lower()
                    if r["hydr"] not in "yanf":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("chec"): # default n?
                    tmp = s[itk+1].lower()
                    if tmp.startswith(("none", "0")):
                        r["check"] = "0"
                    elif tmp.startswith(("liga", "n")):
                        r["check"] = "n"
                    elif tmp.startswith(("all", "y")):
                        r["check"] = "y"
                    else:
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("newl"): #default e
                    tmp = s[itk+1].lower()
                    if tmp.startswith("e"): # exit
                        r["newligand"] = "h"
                    elif tmp.startswith(("c", "y", "noex")): # noexit
                        r["newligand"] = "y"
                    else:
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("buil"): #default n
                    r["build"] = s[itk+1][0].lower()
                    if r["build"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("pept"): # default n
                    r["pept"] = s[itk+1][0].lower()
                    if r["pept"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("link"):
                    r["link"] = s[itk+1][0].lower()
                    if r["link"] not in "ynd0": # what is 0?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("suga"):
                    r["sugar"] = s[itk+1][0].lower()
                    if r["sugar"] not in "ynds": # what is s?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("conn"): # TODO read conn_tolerance? (make conn tole val)
                    r["conn"] = s[itk+1][0].lower()
                    if r["conn"] not in "ynd0": # what is 0?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("symm"):
                    r["symm"] = s[itk+1][0].lower()
                    if r["symm"] not in "ync": # what is 0?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("chai"):
                    r["chain"] = s[itk+1][0].lower()
                    if r["chain"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("cisp"):
                    r["cispept"] = s[itk+1][0].lower()
                    if r["cispept"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("ss"):
                    r["ss"] = s[itk+1][0].lower()
                    if r["ss"] not in "ydn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("exit"):
                    r["exit"] = s[itk+1][0].lower() == "y" # TODO refmac works just with "make exit"
                    itk += 2
                else:
                    itk += 1
        elif s[0].lower().startswith(("sour", "scat")):
            k = s[1].lower()
            if k.startswith("em"):
                ret["source"] = "em"
            elif k.startswith("e"):
                ret["source"] = "ec"
            elif k.startswith("n"):
                ret["source"] = "ne"
            else:
                ret["source"] = "xr"
            # TODO check mb, lamb

    def sorry(s): raise SystemExit("Sorry, {} is not supported".format(s))
    if ret["make"].get("hydr") == "f":
        sorry("make hydr full")
    if ret["make"].get("newligand") == "y":
        sorry("make newligand continue")
    if ret["make"].get("buil") == "y":
        sorry("make build yes")

    return ret
# parse_keywords()

def prepare_crd(xyzin, crdout, ligand, make, monlib_path=None, h_pos="elec"):
    assert h_pos in ("elec", "nucl")
    h_change = dict(a=gemmi.HydrogenChange.ReAddButWater,
                    y=gemmi.HydrogenChange.NoChange,
                    n=gemmi.HydrogenChange.Remove)[make.get("hydr", "a")]
    st = utils.fileio.read_structure(xyzin)
    if st.input_format in (gemmi.CoorFormat.Pdb, gemmi.CoorFormat.ChemComp):
        st.entities.clear()
        st.setup_entities()
    try:
        monlib = utils.restraints.load_monomer_library(st,
                                                       monomer_dir=monlib_path,
                                                       cif_files=ligand,
                                                       stop_for_unknowns=True) # until we have make newligand option
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if make.get("cispept", "y") == "y": st.assign_cis_flags()
    if make.get("link", "n") == "y": # TODO support it correctly, and also make link define. what is 0?
        utils.restraints.find_and_fix_links(st, monlib) # TODO fix for unknown links

    if make.get("hydr") == "a":
        logger.write("generating hydrogen atoms")
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=logger, reorder=True, ignore_unknown_links=True)
    if make.get("hydr") != "n" and st[0].has_hydrogen():
        if h_pos == "nucl":
            resnames = st[0].get_all_residue_names()
            utils.restraints.check_monlib_support_nucleus_distances(monlib, resnames)
            logger.write("adjusting hydrogen position to nucleus")
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=1.1)
        elif make.get("hydr") != "a":
            logger.write("adjusting hydrogen position to electron cloud")
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)

    doc = gemmi.cif.Document()
    crd = gemmi.prepare_crd(st, topo, h_change)
    rst = gemmi.prepare_rst(topo, monlib, st.cell)
    doc.add_copied_block(crd)
    doc.add_copied_block(rst)
    doc.write_file(crdout, style=gemmi.cif.Style.NoBlankLines)
    logger.write("crd file written: {}".format(crdout))
# prepare_crd()

def main(args):
    args.ligand = sum(args.ligand, []) if args.ligand else []
    inputs = []
    for l in sys.stdin:
        inputs.append(l)
        if l.strip().lower() == "end":
            break

    xyzin, opts = get_opt(args.opts, "xyzin")
    libin, _ = get_opt(opts, "libin")
    keywords = parse_keywords(inputs) # TODO expand @
    if libin: args.ligand.append(libin)

    # TODO what if restin is given or make cr prepared is given?
    # TODO check make pept/link/suga/ss/conn/symm/chain

    # Process model
    crdout = None
    if xyzin is not None:
        #tmpfd, crdout = tempfile.mkstemp(prefix="gemmi_", suffix=".crd") # TODO use dir=CCP4_SCR
        #os.close(tmpfd)
        crdout = "gemmi_{}.crd".format(utils.fileio.splitext(os.path.basename(xyzin))[0])
        prepare_crd(xyzin, crdout, args.ligand, make=keywords["make"], monlib_path=args.monlib,
                    h_pos="nucl" if keywords.get("source")=="ne" else "elec")
        opts.extend(["xyzin", crdout])

    if keywords["make"].get("exit"):
        return
        
    # Run Refmac
    cmd = [args.exe] + opts
    env = os.environ
    logger.write("Running REFMAC5..")
    if args.monlib:
        logger.write("CLIBD_MON={}".format(args.monlib))
        env["CLIBD_MON"] = os.path.join(args.monlib, "") # should end with /
    logger.write(" ".join(cmd))
    p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True, env=env)
    if crdout: p.stdin.write("make cr prepared\n")
    p.stdin.write("".join(inputs))
    p.stdin.close()
    for l in iter(p.stdout.readline, ""):
        logger.write(l, end="")
    p.wait()

    # TODO append monomer description to mmcif file.
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
    
