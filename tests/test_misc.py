"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import unittest
import numpy
import pandas
import json
import gemmi
import os
import shutil
import sys
import tempfile
import hashlib
from servalcat import utils

root = os.path.abspath(os.path.dirname(__file__))

class RestrTests(unittest.TestCase):
    def test_merge_dict(self):
        cifs = [os.path.join(root, "dict", f) for f in ("acedrg_link_4D4-MS6.cif", "acedrg_link_MS6-GLY.cif")]
        tmpfd, tmpf = tempfile.mkstemp(prefix="servalcat_merged_", suffix=".cif")
        os.close(tmpfd)
        utils.fileio.merge_ligand_cif(cifs, tmpf)

        doc = gemmi.cif.read(tmpf)
        names = set(["comp_list", "link_list", "mod_list", "comp_4D4", "comp_MS6",
                     "mod_4D4m1", "mod_MS6m1", "mod_MS6m1-0", "mod_GLYm1",
                     "link_MS6-GLY", "link_4D4-MS6"])
        self.assertTrue(names.issubset([x.name for x in doc]))
        # TODO need to test hydrogen generation etc?
        os.remove(tmpf)

    # test_merge_dict()
        
# class RestrTests

if __name__ == '__main__':
    unittest.main()

