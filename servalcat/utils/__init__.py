"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from . import logger
from . import symmetry
from . import fileio
from . import hkl
from . import model
from . import maps
from . import refmac
from . import restraints
from . import commands

def make_loggraph_str(df, main_title, title_labs, s2=None, float_format=None):
    if s2 is not None:
        df = df.copy()
        df.insert(0, "1/resol^2", s2)
    ret = "$TABLE: {} :\n".format(main_title)
    ret += "$GRAPHS\n"
    all_labs = list(df.columns)
    for t, labs in title_labs:
        if s2 is not None: labs = ["1/resol^2"] + labs
        ret += ": {} :A:{}:\n".format(t, ",".join(str(all_labs.index(l)+1) for l in labs))
    ret += "$$\n"
    lines = df.to_string(index=False, index_names=False, header=True, float_format=float_format).splitlines()
    ret += lines[0] + "\n$$\n$$\n"
    ret += "\n".join(lines[1:]) + "\n$$\n"
    return ret
# make_loggraph_str()

