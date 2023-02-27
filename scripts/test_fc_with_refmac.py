"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import time
import os
import argparse
import subprocess
from servalcat.utils import logger
from servalcat import utils

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--refmacmtz')
    parser.add_argument('--auto_box_with_padding', type=float, help="Determine box size from model with specified padding")
    parser.add_argument('-d', '--resolution', type=float, required=True)
    parser.add_argument("--source", choices=["electron", "xray"], default="electron")
    return parser.parse_args(arg_list)
# parse_args()

def run_refmac(pdbin, d_min, source):
    cmd = ["refmac5", "xyzin", pdbin]
    ofs = open("refmac_sfcalc.log", "w")
    p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=ofs, stderr=ofs,
                         universal_newlines=True)
    p.stdin.write("MODE SFCALC\n")
    p.stdin.write("make hydr yes\n")
    if source == "electron": p.stdin.write("source em mb\n")
    p.stdin.write("Sfcalc cr2f\n")
    p.stdin.write("Reso {}\n".format(d_min))
    p.stdin.close()
    print("exit with", p.wait())
    return "sfcalc_from_crd.mtz"

def main(args):
    logger.set_file("servalcat_test_fc.log")
    st = utils.fileio.read_structure(args.model)
    st.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
    if not st.cell.is_crystal() and args.auto_box_with_padding is not None:
        st.cell = utils.model.box_from_model(st[0], args.auto_box_with_padding)
    if not st.cell.is_crystal():
        logger.error("ERROR: No unit cell information. Give --cell.")
        return

    if not args.refmacmtz:
        xyzin = "for_refmac.pdb"
        st.write_pdb(xyzin)
        args.refmacmtz = run_refmac(xyzin, args.resolution, args.source)
        fc_refmac = utils.fileio.read_asu_data_from_mtz(args.refmacmtz, ["Fout0", "Pout0"])
    else:
        fc_refmac = utils.fileio.read_asu_data_from_mtz(args.refmacmtz, ["FC", "PHIC"])
    hkldata = utils.hkl.hkldata_from_asu_data(fc_refmac, "FC_refmac")
    
    monlib = utils.restraints.load_monomer_library(st)

    ofs = open("fc_refmac_gemmi.dat", "w")
    ofs.write("blur cutoff rate d_max d_min fsc\n")
    for blur in (None,):#(0, 20, 40, 60):
        for cutoff in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        #for cutoff in (1e-3,1e-4,):
            for rate in (1.5,):
                t0 = time.time()
                fc_asu = utils.model.calc_fc_fft(st, args.resolution, cutoff=cutoff, rate=rate,
                                                 mott_bethe=args.source == "electron",
                                                 monlib=monlib, source=args.source, blur=blur)
                tt = time.time() - t0
                if "FC" in hkldata.df: del hkldata.df["FC"]
                hkldata.merge_asu_data(fc_asu, "FC")
                print("SIZE==", len(hkldata.df.index))
                hkldata.setup_relion_binning()
                for i_bin, idxes in hkldata.binned():
                    bin_d_min = hkldata.binned_df.d_min[i_bin]
                    bin_d_max = hkldata.binned_df.d_max[i_bin]
                    Fc = hkldata.df.FC.to_numpy()[idxes]
                    Fcr = hkldata.df.FC_refmac.to_numpy()[idxes]
                    fsc = numpy.real(numpy.corrcoef(Fc, Fcr)[1,0])
                    ofs.write("{} {:.1e} {:.1f} {:7.3f} {:7.3f} {:.4f}\n".format(blur, cutoff, rate,
                                                                                 bin_d_max, bin_d_min,
                                                                                 fsc))

                cc = numpy.real(numpy.corrcoef(Fc, Fcr)[1,0])
                logger.writeln("blur= {} cutoff= {:.1e} rate={:.1f} CC={:.4f} time={:.2f} s".format(blur, cutoff, rate, cc, tt))

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
