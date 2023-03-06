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

def make_loggraph_str(df, main_title, title_labs, float_format=None):
    ret = "$TABLE: {} :\n".format(main_title)
    ret += "$GRAPHS\n"
    #all_labs = []
    all_labs = list(df.columns)
    for t, labs in title_labs:
        ret += ": {} :A:{}:\n".format(t, ",".join(str(all_labs.index(l)+1) for l in labs))
        #all_labs.extend(l for l in labs if l not in all_labs)
    ret += "$$\n"
    ret += " ".join(all_labs) + "\n"
    ret += "$$\n$$\n"
    #ret += df.to_string(columns=all_labs, index=False, index_names=False, header=False) + "\n"
    ret += df.to_string(index=False, index_names=False, header=False, float_format=float_format) + "\n$$\n"
    return ret
# make_loggraph_str()

