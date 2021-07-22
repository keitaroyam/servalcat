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
from servalcat.utils import logger

dtypes64 = dict(i=numpy.int64, u=numpy.uint64, f=numpy.float64, c=numpy.complex128)
to64 = lambda x: x.astype(dtypes64.get(x.dtype.kind, x.dtype))

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

def r_factor(fo, fc):
    return numpy.sum(numpy.abs(fo-fc)) / numpy.sum(fo)

def df_from_asu_data(asu_data, label):
    df = pandas.DataFrame(data=asu_data.miller_array,
                          columns=["H","K","L"])
    if asu_data.value_array.dtype.names == ('value', 'sigma'):
        df[label] = to64(asu_data.value_array["value"])
        df["SIG"+label] = to64(asu_data.value_array["sigma"])
    else:
        df[label] = to64(asu_data.value_array)
    return df

def df_from_raw(miller_array, value_array, label):
    df = pandas.DataFrame(data=miller_array,
                          columns=["H","K","L"])
    df[label] = to64(value_array)
    return df

def hkldata_from_asu_data(asu_data, label):
    df = df_from_asu_data(asu_data, label)
    return HklData(asu_data.unit_cell, asu_data.spacegroup, None, df)
# hkldata_from_asu_data()

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

    def calc_epsilon(self):
        self.df["epsilon"] = self.sg.operations().epsilon_factor_without_centering_array(self.miller_array())
    # calc_epsilon()

    def calc_centric(self):
        self.df["centric"] = self.sg.operations().centric_flag_array(self.miller_array()).astype(int)
    # calc_centric()
        
    def d_spacings(self):
        if "d" not in self.df or self.df.d.isnull().values.any():
            self.calc_d()
        return self.df.d
    # calc_d()

    def d_min_max(self):
        d = self.d_spacings()
        return min(d), max(d)
    # d_min_max()

    def setup_binning(self, n_bins, s_power=2):
        sp = self.d_spacings()**(-s_power)
        spmin, spmax = min(sp), max(sp)
        spstep = (spmax-spmin)/n_bins
        bin_limit_ds = numpy.arange(spmin, spmax, spstep)
        
        #bin_limit_ds = [sprange...] # left ends, and right end.
        if len(bin_limit_ds)==n_bins:
            bin_limit_ds = numpy.append(bin_limit_ds, bin_limit_ds[-1]+spstep)
        if bin_limit_ds[-1] < spmax:
            bin_limit_ds[-1] == spmax # difference should be very small..
        bin_limit_ds = bin_limit_ds**(-1/s_power)
        assert len(bin_limit_ds) == n_bins + 1

        bin_centers = [(bin_limit_ds[i]**(-s_power)+bin_limit_ds[i+1]**(-s_power))/2 for i in range(n_bins)]
        sprange = bin_limit_ds**(-s_power)
        sprange[-1] += 1.e-6

        self._bin_and_limits = []
        bin_number = numpy.zeros(len(sp), dtype=numpy.int)
        
        for i in range(1, len(sprange)):
            sel = numpy.where(numpy.logical_and(sprange[i-1]<=sp, sp <sprange[i]))[0]
            bin_number[sel] = i
            self._bin_and_limits.append((i, (bin_limit_ds[i-1], bin_limit_ds[i])))

        self.df["bin"] = bin_number
        self.binned_df = pandas.DataFrame(data=list(range(max(self.df.bin)+1)),
                                          columns=["bin"])
    # setup_binning()
        
    def setup_relion_binning(self):
        max_edge = max(self.cell.parameters[:3])
        if "d" not in self.df or self.df.d.isnull().values.any():
            self.calc_d()
            
        self.df["bin"] = (max_edge/self.df.d).astype(numpy.int)
        self.binned_df = pandas.DataFrame(data=list(range(max(self.df.bin)+1)),
                                          columns=["bin"])
        self._bin_and_limits = []
        bin_numbers = set(self.df.bin)

        # Merge inner/outer shells if too few # TODO smarter way
        bin_counts = []
        modify_table = {}
        for i_bin, g in self.df.groupby("bin", sort=True):
            bin_counts.append([i_bin, len(g.index)])

        for i in range(len(bin_counts)):
            if bin_counts[i][1] < 10 and i < len(bin_counts)-1:
                bin_counts[i+1][1] += bin_counts[i][1]
                modify_table[bin_counts[i][0]] = bin_counts[i+1][0]
                logger.write("Bin {} only has {} data. Merging with next bin.".format(bin_counts[i][0],
                                                                                      bin_counts[i][1]))
            else: break

        for i in reversed(range(len(bin_counts))):
            if bin_counts[i][1] < 10 and i > 0:
                bin_counts[i-1][1] += bin_counts[i][1]
                modify_table[bin_counts[i][0]] = bin_counts[i-1][0]
                logger.write("Bin {} only has {} data. Merging with previous bin.".format(bin_counts[i][0],
                                                                                          bin_counts[i][1]))
            else: break

        while True:
            flag = True
            for i_bin in modify_table:
                if modify_table[i_bin] in modify_table:
                    modify_table[i_bin] = modify_table[modify_table[i_bin]]
                    flag = False
            if flag: break

        if modify_table:
            for i_bin, g in self.df.groupby("bin", sort=True):
                if i_bin in modify_table:
                    self.df.loc[g.index, "bin"] = modify_table[i_bin]

        # set bin_and_limits
        for i_bin, g in self.df.groupby("bin", sort=True):
            self._bin_and_limits.append((i_bin, (max(g.d), min(g.d))))

    # setup_relion_binning()

    def bin_and_limits(self):
        return self._bin_and_limits
    # bin_and_limits()

    def binned_data_as_array(self, lab):
        vals = numpy.zeros(len(self.df.index), dtype=self.binned_df[lab].dtype)
        grouped = self.df.groupby("bin", sort=False)
        for b, g in grouped:
            vals[g.index] = self.binned_df[lab][b]
        return vals
    # binned_data_as_array()

    def binned(self, sort=True):
        return self.df.groupby("bin", sort=sort)
    
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

    def d_eff(self, label):
        # Effective resolution definied using FSC
        fsc = self.binned_df[label]
        bin_counts = self.df.bin.value_counts()
        a = 0.
        for i_bin, (bin_d_max, bin_d_min) in self.bin_and_limits():
            a += bin_counts[i_bin] * fsc[i_bin]

        fac = (a/sum(bin_counts))**(1/3.)
        d_min = self.d_min_max()[0]
        ret = d_min/fac
        return ret
    # d_eff()

    def scale_k_and_b(self, lab_ref, lab_scaled):
        logger.write("Determining k, B scales between {} and {}".format(lab_ref, lab_scaled))
        s2 = 1/self.d_spacings().to_numpy()**2
        # determine scales that minimize (|f1|-|f2|*k*e^(-b*s2/4))^2
        f1 = self.df[lab_ref].to_numpy()
        f2 = self.df[lab_scaled].to_numpy()
        if numpy.iscomplexobj(f1): f1 = numpy.abs(f1)
        if numpy.iscomplexobj(f2): f2 = numpy.abs(f2)

        sel_pos = numpy.logical_and(f1 > 0, f2 > 0)
        f1p, f2p, s2p = f1[sel_pos], f2[sel_pos], s2[sel_pos]

        # 1st step: minimize (log(|f1|)-log(|f2|*e^k*e^(-b*s2/4)))^2 starting with k=1, b=0.
        tmp = numpy.log(f2p) - numpy.log(f1p)
        # g = [dT/dk, dT/db]
        g = numpy.array([2 * numpy.sum(tmp), -numpy.sum(tmp*s2p)/2])
        H = numpy.zeros((2,2))
        H[0,0] = 2*len(f1p)
        H[1,1] = numpy.sum(s2**2/8)
        H[0,1] = H[1,0] = -numpy.sum(s2)/2
        x = -numpy.dot(numpy.linalg.inv(H), g)
        k1 = numpy.exp(x[0])
        B1 = x[1]
        logger.write(" initial estimate using log: k= {:.2e} B= {:.2e}".format(k1, B1))
        f2tmp = f2 * k1 * numpy.exp(-B1*s2/4)
        logger.write(" R= {:.4f} (was: {:.4f})".format(r_factor(f1, f2tmp), r_factor(f1, f2)))

        """
        self.setup_binning(40)        
        bin_limits = dict(self.bin_and_limits())
        x = []
        y0,y1,y2=[],[],[]
        for i_bin, g in self.binned():
            bin_d_max, bin_d_min = bin_limits[i_bin]
            x.append(1/bin_d_min**2)
            y0.append(numpy.average(f1[g.index]))
            y1.append(numpy.average(f2[g.index]))
            y2.append(numpy.average(f2tmp[g.index]))

        import matplotlib.pyplot as plt
        plt.plot(x, y0, label="FC")
        plt.plot(x, y1, label="FP")
        plt.plot(x, y2, label="FP,scaled")
        plt.legend()
        plt.show()
        """
        
        # 2nd step: - minimize (|f1|-|f2|*k*e^(-b*s2/4))^2 with regularisation
        # not implemented
        
        return k1, B1
    # scale_k_and_b()
