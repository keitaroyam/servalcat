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
                                                [0.911309,1.058726,1.089188,1.015436,1.039349,1.036047,1.017594,0.986123,0.939639,0.888208],
                                                decimal=4)
        numpy.testing.assert_array_almost_equal(hkldata.binned_df.D1,
                                                [1.049256,0.000159,0.986066,0.999980,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000],
                                                decimal=4)
        numpy.testing.assert_array_almost_equal(hkldata.binned_df.S,
                                                [101.284865,108.528004,87.098586,104.437671,100.574846,137.784752,131.598254,110.229041,105.206137,64.503932],
                                                decimal=4)

    # test_sigmaa()
        
# class XtalTests

if __name__ == '__main__':
    unittest.main()

