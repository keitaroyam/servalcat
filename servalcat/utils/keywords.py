"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
def parse_ncsc_keywords(kwd_str):
    # FIXME handle lines ending with -
    lines = kwd_str.splitlines()
    ret = []
    for l in lines:
        l = l.split()
        if len(l) < 3: continue
        if l[0].lower().startswith("ncsc"):
            if l[1].lower().startswith("matr"):
                vals = [float(x) for x in l[2:]]
                if len(vals) != 12:
                    print("Bad nsc matrix line: {}".format(" ".join(l)))
                    continue
                op = gemmi.NcsOp()
                op.tr.mat.fromlist([vals[3*x:3*x+3] for x in range(3)])
                op.tr.vec.fromlist(vals[9:])
                op.id = str(len(ret)+1)
                op.given = op.tr.is_identity()
                ret.append(op)
            elif l[1].lower().startswith("eule"):
                pass # TODO
            elif l[1].lower().startswith("pola"):
                pass # TODO
    return ret
# parse_ncsc_keywords()
