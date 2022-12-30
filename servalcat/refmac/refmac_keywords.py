"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger

def parse_line(l, ret):
    s = l.split()
    ntok = len(s)
    if ntok == 0: return
    if s[0].lower().startswith("make"): # TODO: hout,ribo,valu,spec,form,sdmi,segi
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
                    r["newligand"] = False
                elif tmp.startswith(("c", "y", "noex")): # noexit
                    r["newligand"] = True
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
                if itk + 1 < len(s):
                    r["exit"] = s[itk+1][0].lower() == "y"
                    itk += 2
                else:
                    r["exit"] = True
                    itk += 1
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
    elif s[0].lower().startswith("refi"): # TODO support this. Note that only valid with hklin
        itk = 1
        while itk < ntok:
            if s[itk].startswith("type"):
                if itk+1 < ntok and s[itk+1].startswith("unre"):
                    ret["refi"] = {"type": "unre"}
                itk += 2
            else:
                itk += 1
# parse_line()

def parse_lines(lines, ret):
    cont = ""
    for l in lines:
        if "!" in l: l = l[:l.index("!")]
        if "#" in l: l = l[:l.index("#")]
        l = l.strip()
        if not l: continue
        if l[0] == "@":
            f = l[1:]
            parse_lines(open(f).readlines(), ret)
            continue
        if l.split()[-1] == "-":
            cont += l[:l.rfind("-")] + " "
            continue
        if cont:
            l = cont + l
            cont = ""
        parse_line(l, ret)
# parse_lines()
            
def parse_keywords(inputs):
    ret = {"make":{}}
    parse_lines(inputs, ret)
    return ret
# parse_keywords()

