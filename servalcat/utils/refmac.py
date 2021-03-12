"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import subprocess
import pipes
import json
import copy
import re
import os
from servalcat.utils import logger

re_version = re.compile("#.* Refmac *version ([^ ]+) ")
re_error = re.compile('(warn|error *[:]|error *==|^error)', re.IGNORECASE)

def check_version(exe="refmac5"):
    p = subprocess.Popen([exe], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    p.stdin.write("end\n")
    p.stdin.close()
    ver = ()
    for l in iter(p.stdout.readline, ""):
        r_ver = re_version.search(l)
        if r_ver:
            ver = tuple(map(int, r_ver.group(1).split(".")))
    p.wait()
    return ver
# check_version()

def external_restraints_json_to_keywords(json_in):
    ret = []
    exte_list = json.load(open(json_in))
    for e in exte_list:
        if "use" in e:
            ret.append("EXTERNAL USE {}".format(e["use"]))
        if "dmax" in e:
            ret.append("EXTERNAL DMAX {0}".format(e["dmax"]))
        if "weight_scale" in e:
            ret.append("EXTERNAL WEIGHT SCALE {0}".format(e["weight_scale"]))
        if "weight_gmwt" in e:
            ret.append("EXTERNAL WEIGHT GMWT {0}".format(e["weight_gmwt"]))
        if "file" in e:
            ret.append("@"+e["file"])

    return "\n".join(ret) + "\n"
# external_restraints_json_to_keywords()

class Refmac:
    def __init__(self, **kwargs):
        self.prefix = "refmac"
        self.hklin = self.xyzin = ""
        self.lab_f = None
        self.lab_sigf = None
        self.lab_phi = None
        self.hydrogen = "all"
        self.hout = "no"
        self.ncycle = 10
        self.resolution = None
        self.weight_fixed = None
        self.weight_auto_scale = None
        self.bfactor = None
        self.jellybody = None
        self.jellybody_sigma, self.jellybody_dmax = 0.01, 4.2
        self.ncsr = None
        self.keyword_files = []
        self.keywords = []
        self.external_restraints_json = None
        self.exe = "refmac5"
        self.show_log = False # summary only if false
        self.global_mode = kwargs.get("global_mode")
        
        if self.global_mode == "spa":
            self.source = "em mb"
            self.solvent = "no"
        else:
            self.source = "xray"
            self.solvent = "yes"
            
        for k in kwargs:
            if k == "args":
                self.init_from_args(kwargs["args"])
            else:
                setattr(self, k, kwargs[k])
    # __init__()

    def init_from_args(self, args):
        self.hklin = args.mtz
        self.xyzin = args.model
        self.libin = args.ligand
        self.ncycle = args.ncycle
        self.lab_f = args.lab_f
        self.lab_phi = args.lab_phi
        self.lab_sigf = args.lab_sigf
        self.hydrogen = args.hydrogen
        self.hout = args.hout
        self.ncsr = args.ncsr
        self.bfactor = args.bfactor
        self.jellybody = args.jellybody
        self.jellybody_sigma, self.jellybody_dmax = args.jellybody_params
        self.resolution = args.resolution
        self.weight_auto_scale = args.weight_auto_scale
        self.weight_fixed = args.weight
        self.keyword_files = args.keyword_file
        self.keywords = args.keywords
        self.external_restraints_json = args.external_restraints_json
        self.exe = args.exe
        self.show_log = args.show_refmac_log
    # init_from_args()

    def copy(self, **kwargs):
        ret = copy.copy(self)
        for k in kwargs:
            setattr(ret, k, kwargs[k])
            
        return ret
    # copy()
    
    def make_keywords(self):
        ret = ""
        labin = []
        if self.lab_f: labin.append("FP={}".format(self.lab_f))
        if self.lab_sigf: labin.append("SIGFP={}".format(self.lab_sigf))
        if self.lab_phi: labin.append("PHIB={}".format(self.lab_phi))
        ret += "labin {}\n".format(" ".join(labin))


        ret += "make hydr {}\n".format(self.hydrogen)
        ret += "make hout {}\n".format("yes" if self.hout else "no")
        ret += "solvent {}\n".format(self.solvent)
        ret += "source {}\n".format(self.source)
        ret += "ncycle {}\n".format(self.ncycle)
        if self.resolution is not None:
            ret += "reso {}\n".format(self.resolution)
        if self.weight_fixed is not None:
            ret += "weight matrix {}\n".format(self.weight_fixed)
        elif self.weight_auto_scale is not None:
            ret += "weight auto {}\n".format(self.weight_auto_scale)
        else:
            ret += "weight auto\n"

        if self.bfactor is not None:
            ret += "bfactor set {}\n".format(self.bfactor)
        if self.jellybody:
            ret += "ridge dist sigma {:.3f}\n".format(self.jellybody_sigma)
            ret += "ridge dist dmax {:.2f}\n".format(self.jellybody_dmax)
        if self.ncsr:
            ret += "ncsr {}\n".format(self.ncsr)

        if self.external_restraints_json:
            ret += external_restraints_json_to_keywords(self.external_restraints_json)
            
        if self.keyword_files:
            for f in self.keyword_files:
                ret += "@{}\n".format(f)

        if self.keywords:
            ret += "\n".join(self.keywords).strip() + "\n"

        return ret
    # make_keywords()

    def xyzout(self): return self.prefix + ".pdb"
    def hklout(self): return self.prefix + ".mtz"

    def make_cmd(self):
        cmd = [self.exe]
        cmd.extend(["hklin", pipes.quote(self.hklin)])
        cmd.extend(["hklout", pipes.quote(self.hklout())])
        cmd.extend(["xyzin", pipes.quote(self.xyzin)])
        cmd.extend(["xyzout", pipes.quote(self.xyzout())])
        if self.libin:
            cmd.extend(["libin", self.libin])

        return cmd
    # make_cmd()

    def run_refmac(self):
        cmd = self.make_cmd()
        stdin = self.make_keywords()
        open(self.prefix+".inp", "w").write(stdin)

        logger.write("Running REFMAC5..")
        logger.write("{} <<__eof__ > {}".format(" ".join(cmd), self.prefix+".log"))
        logger.write(stdin, end="")
        logger.write("__eof__")

        p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        p.stdin.write(stdin)
        p.stdin.close()

        log = open(self.prefix+".log", "w")
        cycle = 0
        re_lastcycle = re.compile("Cycle *{}. Rfactor analysis".format(self.ncycle+1))
        rmsbond = ""
        rmsangle = ""
        
        for l in iter(p.stdout.readline, ""):
            log.write(l)

            if self.show_log:
                print(l, end="")
                continue
            
            r_ver = re_version.search(l)
            if r_ver:
                logger.write("Starting Refmac {}".format(r_ver.group(1)))

            r_err = re_error.search(l)
            if r_err:
                if self.global_mode == "spa":
                    if "Figure of merit of phases has not been assigned" in l:
                        continue
                    elif "They will be assumed to be equal to 1.0" in l:
                        continue
                logger.write(l, end="")
            
            if self.global_mode == "spa":
                if "CGMAT cycle number =" in l:
                    cycle = int(l[l.index("=")+1:])
                    if cycle == 1:
                        logger.write("cycle FSCaverage")
                elif re_lastcycle.search(l):
                    cycle = self.ncycle + 1
                elif "Average Fourier shell correlation    =" in l and cycle > 0:
                    fsc = l[l.index("=")+1:].strip()
                    if cycle == 1:
                        note = "(initial)"
                    elif cycle > self.ncycle:
                        note = "(final)"
                    else:
                        note = ""
                        
                    logger.write("{:5d} {} {}".format(cycle-1, fsc, note))
                    cycle = 0
                elif "Rms BondLength" in l:
                    rmsbond = l
                elif "Rms BondAngle" in l:
                    rmsangle = l
        
        ret = p.wait()
        if rmsbond:
            logger.write("                      Initial    Final")
            logger.write(rmsbond.rstrip())
            logger.write(rmsangle.rstrip())
        logger.write("REFMAC5 finished with exit code= {}".format(ret))

        if not os.path.isfile(self.xyzout()) or not os.path.isfile(self.hklout()):
            raise RuntimeError("REFMAC5 did not produce output files.")
        
    # run_refmac()
# class Refmac
