"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import subprocess
import pipes
import json
import copy
import re
import os
import tempfile
from servalcat.utils import logger
from servalcat.utils import fileio

re_version = re.compile("#.* Refmac *version ([^ ]+) ")
re_error = re.compile('(warn|error *[:]|error *==|^error)', re.IGNORECASE)
re_outlier_start = re.compile("\*\*\*\*.*outliers")

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

def read_tls_file(tlsin):
    # TODO sort out L/S units - currently use Refmac tlsin/out as is
    # TODO change to gemmi::TlsGroup?
    
    groups = [] 
    for l in open(tlsin):
        l = l.strip()
        if l.startswith("TLS"):
            title = l[4:]
            groups.append(dict(title=title, ranges=[], origin=None, T=None, L=None, S=None))
        elif l.startswith("RANG"):
            r = l[l.index(" "):].strip()
            groups[-1]["ranges"].append(r)
        elif l.startswith("ORIG"):
            try:
                groups[-1]["origin"] = gemmi.Position(*(float(x) for x in l.split()[1:]))
            except:
                raise ValueError("Prolem with TLS file: {}".format(l))
        elif l.startswith("T   "):
            try:
                groups[-1]["T"] = [float(x) for x in l.split()[1:7]]
            except:
                raise ValueError("Prolem with TLS file: {}".format(l))
        elif l.startswith("L   "):
            try:
                groups[-1]["L"] = [float(x) for x in l.split()[1:7]]
            except:
                raise ValueError("Prolem with TLS file: {}".format(l))
        elif l.startswith("S   "):
            try:
                groups[-1]["S"] = [float(x) for x in l.split()[1:10]]
            except:
                raise ValueError("Prolem with TLS file: {}".format(l))

    return groups
# read_tls_file()

def write_tls_file(groups, tlsout):
    with open(tlsout, "w") as f:
        for g in groups:
            f.write("TLS {}\n".format(g["title"]))
            for r in g["ranges"]:
                f.write("RANGE {}\n".format(r))
            if g["origin"] is not None:
                f.write("ORIGIN ")
                f.write(" ".join("{:8.4f}".format(x) for x in g["origin"].tolist()))
                f.write("\n")
            for k in "TLS":
                if g[k] is not None:
                    f.write("{:4s}".format(k))
                    f.write(" ".join("{:8.4f}".format(x) for x in g[k]))
                    f.write("\n")
# write_tls_file()

class Refmac:
    def __init__(self, **kwargs):
        self.prefix = "refmac"
        self.hklin = self.xyzin = ""
        self.source = "electron"
        self.lab_f = None
        self.lab_sigf = None
        self.lab_phi = None
        self.libin = None
        self.tlsin = None
        self.hydrogen = "all"
        self.hout = False
        self.ncycle = 10
        self.tlscycle = 0
        self.resolution = None
        self.weight_matrix = None
        self.weight_auto_scale = None
        self.bfactor = None
        self.jellybody = None
        self.jellybody_sigma, self.jellybody_dmax = 0.01, 4.2
        self.ncsr = None
        self.shake = None
        self.keyword_files = []
        self.keywords = []
        self.external_restraints_json = None
        self.exe = "refmac5"
        self.monlib_path = None
        self.keep_chain_ids = False
        self.show_log = False # summary only if false
        self.tmpdir = None # set path if CCP4_SCR does not work
        self.global_mode = kwargs.get("global_mode")
        
        for k in kwargs:
            if k == "args":
                self.init_from_args(kwargs["args"])
            else:
                setattr(self, k, kwargs[k])

        self.check_tmpdir()
    # __init__()

    def init_from_args(self, args):
        self.hklin = args.mtz
        self.xyzin = args.model
        self.libin = args.ligand
        self.tlsin = args.tlsin
        self.ncycle = args.ncycle
        self.tlscycle = args.tlscycle
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
        self.weight_matrix = args.weight_matrix
        self.keyword_files = args.keyword_file
        self.keywords = args.keywords
        self.external_restraints_json = args.external_restraints_json
        self.exe = args.exe
        self.show_log = args.show_refmac_log
        self.monlib_path = args.monlib
    # init_from_args()

    def copy(self, **kwargs):
        ret = copy.deepcopy(self)
        for k in kwargs:
            setattr(ret, k, kwargs[k])
            
        return ret
    # copy()

    def check_tmpdir(self):
        tmpdir = os.environ.get("CCP4_SCR")
        if tmpdir:
            if os.path.isdir(tmpdir): # TODO check writability
                return
            try:
                os.makedirs(tmpdir)
                return
            except:
                logger.write("Warning: cannot create CCP4_SCR= {}".format(tmpdir))

        self.tmpdir = tempfile.mkdtemp(prefix="ccp4tmp")
    # check_tmpdir()

    def set_libin(self, ligands):
        if not ligands: return
        if len(ligands) > 1:
            mcif = "merged_ligands.cif" # XXX directory!
            logger.write("Merging ligand cif files: {}".format(ligands))
            fileio.merge_ligand_cif(ligands, mcif)
            self.libin = mcif
        else:
            self.libin = ligands[0]
    # set_libin()
            
    
    def make_keywords(self):
        ret = ""
        labin = []
        if self.lab_f: labin.append("FP={}".format(self.lab_f))
        if self.lab_sigf: labin.append("SIGFP={}".format(self.lab_sigf))
        if self.lab_phi: labin.append("PHIB={}".format(self.lab_phi))
        if labin:
            ret += "labin {}\n".format(" ".join(labin))


        ret += "make hydr {}\n".format(self.hydrogen)
        ret += "make hout {}\n".format("yes" if self.hout else "no")
        
        if self.global_mode == "spa":
            ret += "solvent no\n"
            ret += "scale lssc isot\n"
            ret += "source em mb\n"
        elif self.source == "electron":
            ret += "source ec mb\n"
        elif self.source == "neutron":
            ret += "source n\n"

        ret += "ncycle {}\n".format(self.ncycle)
        if self.resolution is not None:
            ret += "reso {}\n".format(self.resolution)
        if self.weight_matrix is not None:
            ret += "weight matrix {}\n".format(self.weight_matrix)
        elif self.weight_auto_scale is not None:
            ret += "weight auto {:.2e}\n".format(self.weight_auto_scale)
        else:
            ret += "weight auto\n"

        if self.bfactor is not None:
            ret += "bfactor set {}\n".format(self.bfactor)
        if self.jellybody:
            ret += "ridge dist sigma {:.3e}\n".format(self.jellybody_sigma)
            ret += "ridge dist dmax {:.2e}\n".format(self.jellybody_dmax)
        if self.ncsr:
            ret += "ncsr {}\n".format(self.ncsr)
        if self.shake:
            ret += "rand {}\n".format(self.shake)
        if self.tlscycle > 0:
            ret += "refi tlsc {}\n".format(self.tlscycle)
            ret += "tlsout addu\n"
        if self.keep_chain_ids:
            ret += "pdbo keep auth\n"

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
    def tlsout(self): return self.prefix + ".tls"

    def make_cmd(self):
        cmd = [self.exe]
        cmd.extend(["hklin", pipes.quote(self.hklin)])
        cmd.extend(["hklout", pipes.quote(self.hklout())])
        cmd.extend(["xyzin", pipes.quote(self.xyzin)])
        cmd.extend(["xyzout", pipes.quote(self.xyzout())])
        if self.libin:
            cmd.extend(["libin", pipes.quote(self.libin)])
        if self.tlsin:
            cmd.extend(["tlsin", pipes.quote(self.tlsin)])
        if self.tlscycle > 0:
            cmd.extend(["tlsout", pipes.quote(self.tlsout())])
        if self.source == "neutron":
            cmd.extend(["atomsf", pipes.quote(os.path.join(os.environ["CLIBD"], "atomsf_neutron.lib"))])
        if self.tmpdir:
            cmd.extend(["scrref", pipes.quote(os.path.join(self.tmpdir, "refmac5_temp1"))])
            
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

        env = os.environ
        if self.monlib_path: env["CLIBD_MON"] = os.path.join(self.monlib_path, "") # should end with /
        
        p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, env=env)
        p.stdin.write(stdin)
        p.stdin.close()

        log = open(self.prefix+".log", "w")
        cycle = 0
        re_lastcycle = re.compile("Cycle *{}. Rfactor analysis".format(self.ncycle+self.tlscycle+1))
        re_actual_weight = re.compile("Actual weight *([^ ]+) *is applied to the X-ray term")
        rmsbond = ""
        rmsangle = ""
        log_delay = []
        summary_write = (lambda x: log_delay.append(x)) if self.show_log else logger.write
        outlier_flag = False
        last_table_flag = False
        last_table_keys = []
        ret = {"version":None,
               "cycles": [{"cycle":i} for i in range(self.ncycle+self.tlscycle+1)],
               } # metadata
        
        for l in iter(p.stdout.readline, ""):
            log.write(l)

            if self.show_log:
                print(l, end="")
            
            r_ver = re_version.search(l)
            if r_ver:
                ret["version"] = r_ver.group(1)
                summary_write("Starting Refmac {} (PID: {})".format(r_ver.group(1), p.pid))

            # print error/warning
            r_err = re_error.search(l)
            if r_err:
                if self.global_mode == "spa":
                    if "Figure of merit of phases has not been assigned" in l:
                        continue
                    elif "They will be assumed to be equal to 1.0" in l:
                        continue
                summary_write(l.rstrip())

            # print outliers
            r_outl = re_outlier_start.search(l)
            if r_outl:
                outlier_flag = True
                summary_write(l.rstrip())
            elif outlier_flag:
                if l.strip() == "" or "monitored" in l or "dev=" in l or "sigma=" in l.lower() or "sigma.=" in l:
                    summary_write(l.rstrip())
                else:
                    outlier_flag = False

            if "TLS refinement cycle" in l:
                cycle = int(l.split()[-1])
            elif "CGMAT cycle number =" in l:
                cycle = int(l[l.index("=")+1:]) + self.tlscycle
            elif re_lastcycle.search(l):
                cycle = self.ncycle + self.tlscycle + 1
            elif "Overall R factor                     =" in l and cycle > 0:
                rfac = l[l.index("=")+1:].strip()
                if self.global_mode != "spa":
                    summary_write(" cycle= {:3d} R= {}".format(cycle-1, rfac))
            elif "Average Fourier shell correlation    =" in l and cycle > 0:
                fsc = l[l.index("=")+1:].strip()
                if cycle == 1:
                    note = "(initial)"
                elif cycle <= self.tlscycle+1:
                    note = "(TLS)"
                elif cycle > self.ncycle + self.tlscycle:
                    note = "(final)"
                else:
                    note = ""

                ret["cycles"][cycle-1]["fsc_average"] = fsc
                if self.global_mode == "spa":
                    summary_write(" cycle= {:3d} FSCaverage= {} {}".format(cycle-1, fsc, note))
            elif "Rms BondLength" in l:
                rmsbond = l
            elif "Rms BondAngle" in l:
                rmsangle = l

            r_actual_weight = re_actual_weight.search(l)
            if r_actual_weight:
                ret["cycles"][cycle-1]["actual_weight"] = r_actual_weight.group(1)

            # Final table
            if "    Ncyc    Rfact    Rfree     FOM" in l:
                last_table_flag = True
                last_table_keys = l.split()
                if last_table_keys[-1] == "$$": del last_table_keys[-1]
            elif last_table_flag:
                if "$$ Final results $$" in l:
                    last_table_flag = False
                    continue
                sp = l.split()
                if len(sp) == len(last_table_keys) and sp[0] != "$$":
                    cyc = int(sp[last_table_keys.index("Ncyc")])
                    key_name = dict(rmsBOND="rms_bond", zBOND="rmsz_bond",
                                    rmsANGL="rms_angle", zANGL="rmsz_angle",
                                    rmsCHIRAL="rms_chiral")
                    for k in key_name:
                        if k in last_table_keys:
                            ret["cycles"][cyc][key_name[k]] = sp[last_table_keys.index(k)]
                        else:
                            logger.error("table does not have key {}?".format(k))
                            
        retcode = p.wait()
        if log_delay:
            logger.write("== Summary of Refmac ==")
            logger.write("\n".join(log_delay))
                         
        if rmsbond:
            logger.write("                      Initial    Final")
            logger.write(rmsbond.rstrip())
            logger.write(rmsangle.rstrip())
            
        logger.write("REFMAC5 finished with exit code= {}".format(retcode))

        # TODO check timestamp
        if not os.path.isfile(self.xyzout()) or not os.path.isfile(self.hklout()):
            raise RuntimeError("REFMAC5 did not produce output files.")

        return ret
    # run_refmac()
# class Refmac
