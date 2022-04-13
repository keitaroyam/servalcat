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
                                                [0.9136,1.0628,1.0807,1.0233,1.0223,1.0350,1.0177,0.9951,0.9148,0.8549],
                                                decimal=4)
        numpy.testing.assert_array_almost_equal(hkldata.binned_df.D1,
                                                [1.0601,0.002109,0.999769,0.999991,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000],
                                                decimal=4)
        numpy.testing.assert_array_almost_equal(hkldata.binned_df.S,
                                                [58.016553,63.778586,48.472681,55.611894,60.992111,72.562953,77.010838,61.598969,48.960756,42.537076],
                                                decimal=4)

    # test_sigmaa()
        
# class XtalTests

if __name__ == '__main__':
    unittest.main()

