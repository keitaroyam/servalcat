"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import numpy
import pandas
import gemmi

class Binner:
    def __init__(self, asu, style="relion"):
        if style == "relion":
            cell = asu.unit_cell
            max_cell_edge = numpy.max([cell.a, cell.b, cell.c])
            self.d_array = asu.make_d_array()
            self.bin_array = (max_cell_edge/self.d_array).astype(int)
            self.bins, self.bin_counts = numpy.unique(self.bin_array, return_counts=True)
        else:
            raise Exception("Non-supported binning type")
    # __init__()
# class Binner    

def df_from_asu_data(asu_data, label):
    df = pandas.DataFrame(data=asu_data.miller_array,
                          columns=["H","K","L"])
    df[label] = asu_data.value_array
    return df

def df_from_raw(miller_array, value_array, label):
    df = pandas.DataFrame(data=miller_array,
                          columns=["H","K","L"])
    df[label] = value_array
    return df

class HklData:
    def __init__(self, cell, sg, anomalous, df, binned_df=None):
        self.cell = cell
        self.sg = sg
        self.anomalous = anomalous
        self.df = df
        self.binned_df = binned_df
    # __init__()

    def update_cell(self, cell):
        # update d
        pass

    def switch_to_asu(self):
        # Need to care phases
        pass

    def copy(self):
        return HklData(self.cell, self.sg, self.anomalous,
                       self.df.copy(),
                       self.binned_df.copy() if self.binned_df is not None else None)

    def merge_asu_data(self, asu_data, label, common_only=True):
        if label in self.df:
            raise Exception("Duplicated label")
        
        df_tmp = df_from_asu_data(asu_data, label)
        if common_only:
            self.df = self.df.merge(df_tmp)
        else:
            self.df = self.df.merge(other.df, how="outer")
    # merge_asu_data()

    def miller_array(self):
        return self.df[["H","K","L"]]

    def calc_d(self):
        self.df["d"] = self.cell.calculate_d_array(self.miller_array())
    # calc_d()

    def d_spacings(self):
        if "d" not in self.df or self.df.d.isnull().values.any():
            self.calc_d()
        return self.df.d
    # calc_d()

    def d_min_max(self):
        d = self.d_spacings()
        return min(d), max(d)
    # d_min_max()

    def setup_relion_binning(self):
        max_edge = max(self.cell.parameters[:3])
        if "d" not in self.df or self.df.d.isnull().values.any():
            self.calc_d()
            
        self.df["bin"] = (max_edge/self.df.d).astype(numpy.int)
        self.binned_df = pandas.DataFrame(data=list(range(max(self.df.bin)+1)),
                                          columns=["bin"])
    # setup_relion_binning()

    def bin_and_limits(self):
        bins = set(self.df.bin)
        ret = []
        for i_bin in bins:
            sel = self.df.bin == i_bin
            d_sel = self.df.d[sel]
            ret.append((i_bin, max(d_sel), min(d_sel)))

        return ret
    # bin_and_limits()
    
    def merge(self, other, common_only=True):
        self.merge_df(other, common_only)
    # merge()

    def merge_df(self, other, common_only=True):
        # TODO check space group, cell, anomalous
        # TODO transform to asu with phase shifts
        # TODO check column labels. same names other than HKL?
        # 
        if common_only:
            self.df = self.df.merge(other)
        else:
            df = self.df.merge(other, indicator=True, how="outer")
            df_left = df[df._merge=="left_only"]
            df_right = df[df._merge=="right_only"]
            df_both = df[df._merge=="both"]
    # merge()

    def fft_map(self, label, grid_size=None, sample_rate=3):
        asu = gemmi.ComplexAsuData(self.cell,
                                   self.sg,
                                   self.miller_array(),
                                   self.df[label])
        if grid_size is None:
            grid_size = (0, 0, 0)

        ma = asu.transform_f_phi_to_map(sample_rate=sample_rate, exact_size=grid_size)
        return ma
    # fft_map()
