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
import pipes
import hashlib
from servalcat import utils
from servalcat.xtal import sigmaa
from servalcat import command_line

root = os.path.abspath(os.path.dirname(__file__))

class XtalTests(unittest.TestCase):
    def test_scale(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        st = utils.fileio.read_structure(pdbin)
        mtz = gemmi.read_mtz_file(mtzin)
        hkldata = utils.hkl.hkldata_from_asu_data(mtz.get_float("I"), "I")

        hkldata.df["FC"] = utils.model.calc_fc_fft(st, mtz.resolution_high(), "xray", mott_bethe=False,
                                                   miller_array=hkldata.miller_array())
        hkldata.df["IC"] = numpy.abs(hkldata.df["FC"].to_numpy())**2
        k, b = hkldata.scale_k_and_b("I", "IC")
        self.assertAlmostEqual(k, 0.00667, places=5)
        self.assertAlmostEqual(b, -8.48374, places=4)
    # test_scale()

    def test_sigmaa(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        args = sigmaa.parse_args(["--hklin", mtzin, "--model", pdbin, "--D_as_exp", "--S_as_exp",
                                  "--labin", "FP,SIGFP", "--nbins", "10"])
        hkldata = sigmaa.main(args)
        os.remove("sigmaa.log")
        os.remove("sigmaa.mtz")

        numpy.testing.assert_array_almost_equal(hkldata.binned_df.d_min,
                                                [5.0834,3.6631,3.0103,2.6155,2.3440,2.1426,1.9855,1.8586,1.7532,1.6640],
                                                decimal=4)
        numpy.testing.assert_array_almost_equal(hkldata.binned_df.D0,
                                                [0.844600,0.983733,1.022023,0.965324,0.996962,1.009091,0.997227,0.989490,0.937605,0.928479],
                                                decimal=4)
        numpy.testing.assert_array_almost_equal(hkldata.binned_df.D1,
                                                [0.297493,0.000004,0.000081,0.000075,0.996606,0.999243,1.000051,0.999999,1.000000,1.000001],
                                                decimal=4)
        numpy.testing.assert_array_almost_equal(hkldata.binned_df.S,
                                                [ 84.857641, 95.794294, 68.853409, 96.124315, 74.231890,110.065386,116.446683, 95.913027, 81.069189, 43.255406],
                                                decimal=4)

    # test_sigmaa()
        
# class XtalTests

if __name__ == '__main__':
    unittest.main()

