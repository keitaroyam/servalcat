"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import numpy
import numpy.lib.recfunctions
import scipy.optimize
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
    return HklData(asu_data.unit_cell, asu_data.spacegroup, df)
# hkldata_from_asu_data()

def hkldata_from_mtz(mtz, labels, newlabels=None):
    if not set(labels).issubset(mtz.column_labels()):
        raise RuntimeError("All specified coulumns were not found from mtz.")
    
    df = pandas.DataFrame(data=numpy.array(mtz, copy=False), columns=mtz.column_labels())
    df = df.astype({name: 'int32' for name in ['H', 'K', 'L']})
    for lab in set(mtz.column_labels()).difference(labels+["H","K","L"]):
        del df[lab]
        
    if newlabels is not None:
        assert len(newlabels) == len(labels)
        col_types = {x.label:x.type for x in mtz.columns}
        for i in range(1, len(newlabels)):
            if newlabels[i] == "": # means this is phase and should be transferred to previous column
                assert col_types.get(labels[i]) == "P"
                assert col_types.get(labels[i-1]) == "F"
                ph = numpy.deg2rad(df[labels[i]])
                df[labels[i-1]] = df[labels[i-1]] * (numpy.cos(ph) + 1j * numpy.sin(ph))
                del df[labels[i]]
        
        df.rename(columns={x:y for x,y in zip(labels, newlabels) if y != ""}, inplace=True)

    return HklData(mtz.cell, mtz.spacegroup, df)
# hkldata_from_mtz()

def blur_mtz(mtz, B):
    # modify given mtz object
    
    s2 = mtz.make_1_d2_array()
    k2 = numpy.exp(-B*s2/2)
    k = numpy.exp(-B*s2/4)
    i_labs = [c.label for c in mtz.columns if c.type in "JK"]
    f_labs = [c.label for c in mtz.columns if c.type in "FDG"]
    for labs in i_labs, f_labs:
        for l in labs:
            sl = "SIG"+l
            if sl in mtz.column_labels(): labs.append(sl)

    if i_labs:
        logger.write("Intensities: {}".format(" ".join(i_labs)))
        logger.write("  exp(-B*s^2/2) will be multiplied (B= {:.2f})".format(B))
    if f_labs:
        logger.write("Amplitudes:  {}".format(" ".join(f_labs)))
        logger.write("  exp(-B*s^2/4) will be multiplied (B= {:.2f})".format(B))

    for l in i_labs:
        c = mtz.column_with_label(l)
        c.array[:] *= k2
    for l in f_labs:
        c = mtz.column_with_label(l)
        c.array[:] *= k
# blur_mtz()

def mtz_selected(mtz, columns):
    """
    creates a new mtz object having specified `columns` of `mtz`
    """
    columns = ["H", "K", "L"] + columns # TODO make sure no duplicates
    col_dict = {x.label:x for x in mtz.columns}
    col_idxes = {x.label:i for i, x in enumerate(mtz.columns)}

    notfound = list(set(columns) - set(col_idxes))
    if notfound:
        raise RuntimeError("specified columns not found: {}".format(str(notfound)))

    # copy metadata
    mtz2 = gemmi.Mtz()
    for k in ("spacegroup", "cell", "history", "title"):
        setattr(mtz2, k, getattr(mtz, k))

    for ds in mtz.datasets:
        ds2 = mtz2.add_dataset("")
        for k in ("cell", "id", "crystal_name", "dataset_name", "project_name", "wavelength"):
            setattr(ds2, k, getattr(ds, k))

    # copy selected columns
    for col in columns:
        mtz2.add_column(col, col_dict[col].type,
                        dataset_id=col_dict[col].dataset_id, expand_data=False)

    idxes = [col_idxes[col] for col in columns]
    data = numpy.array(mtz, copy=False)[:, idxes]
    mtz2.set_data(data)
    return mtz2
# mtz_selected()
    
class HklData:
    def __init__(self, cell, sg, df=None, binned_df=None):
        self.cell = cell
        self.sg = sg
        self.df = df
        self.binned_df = binned_df
        self._bin_and_indices = []
    # __init__()

    def update_cell(self, cell):
        # update d
        pass

    def switch_to_asu(self):
        # Need to care phases
        pass

    def copy(self, d_min=None, d_max=None):
        # FIXME we should reset_index here? after resolution truncation, max(df.index) will be larger than size.
        if (d_min, d_max).count(None) == 2:
            df = self.df.copy()
            binned_df = self.binned_df.copy() if self.binned_df is not None else None
        else:
            if d_min is None: d_min = 0
            if d_max is None: d_max = float("inf")
            sel = self.d_spacings().between(d_min, d_max, inclusive=True)
            df = self.df[sel].copy()
            binned_df = None # no way to keep it
        
        return HklData(self.cell, self.sg,  df, binned_df)
    # copy()

    def merge_asu_data(self, asu_data, label, common_only=True):
        if self.df is not None and label in self.df:
            raise Exception("Duplicated label")
        
        df_tmp = df_from_asu_data(asu_data, label)

        if self.df is None:
            self.df = df_tmp
        elif common_only:
            self.df = self.df.merge(df_tmp)
        else:
            self.df = self.df.merge(other.df, how="outer")
    # merge_asu_data()

    def miller_array(self):
        return self.df[["H","K","L"]]

    def s_array(self):
        hkl = self.miller_array()
        return numpy.dot(hkl, self.cell.fractionalization_matrix)

    def debye_waller_factors(self, b_cart=None, b_iso=None):
        if b_iso is not None:
            s2 = 1 / self.d_spacings()**2
            return numpy.exp(-b_iso / 4 * s2)
        if b_cart is not None:
            b_star = b_cart.transformed_by(self.cell.fractionalization_matrix)
            return numpy.exp(-b_star.r_u_r(self.miller_array().to_numpy()) / 4)
    
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

    def sort_by_resolution(self, ascending=False):
        self.d_spacings()
        self.df.sort_values("d", ascending=ascending, inplace=True)
    # sort_by_resolution()

    def d_min_max(self):
        d = self.d_spacings()
        return numpy.min(d), numpy.max(d)
    # d_min_max()

    def complete(self):
        # make complete set
        all_hkl = gemmi.make_miller_array(self.cell, self.sg, self.d_min_max()[0])
        match = gemmi.HklMatch(self.miller_array(), all_hkl)
        missing_hkl_df = pandas.DataFrame(all_hkl[numpy.asarray(match.pos) < 0], columns=["H","K","L"])
        self.df = pandas.concat([self.df, missing_hkl_df])
        logger.write("Completing hkldata: {} reflections were missing".format(len(missing_hkl_df.index)))
        self.calc_d()
    # complete()

    def completeness(self, label=None):
        if label is None:
            n_missing = numpy.sum(self.df.isna().any(axis=1))
        else:
            n_missing = numpy.sum(self.df[label].isna())
        n_all = len(self.df.index)
        return (n_all-n_missing)/n_all
    # completeness()

    def setup_binning(self, n_bins, method=gemmi.Binner.Method.Dstar2):
        self.df.reset_index(drop=True, inplace=True)
        s2 = 1/self.d_spacings().to_numpy()**2
        binner = gemmi.Binner()
        binner.setup_from_1_d2(n_bins, method, s2, self.cell)
        self._bin_and_indices = []
        d_limits = 1 / numpy.sqrt(binner.limits)
        bin_number = binner.get_bins_from_1_d2(s2)
        d_max_all = []
        d_min_all = []
        for i in range(binner.size):
            left = numpy.max(self.d_spacings()) if i == 0 else d_limits[i-1]
            right = numpy.min(self.d_spacings()) if i == binner.size -1 else d_limits[i]
            sel = numpy.where(bin_number==i)[0] # slow?
            d_max_all.append(left)
            d_min_all.append(right)
            self._bin_and_indices.append((i, sel))

        self.df["bin"] = bin_number
        self.binned_df = pandas.DataFrame(dict(d_max=d_max_all, d_min=d_min_all), index=list(range(binner.size)))
    # setup_binning()

    def setup_relion_binning(self, sort=False):
        max_edge = max(self.cell.parameters[:3])
        if sort:
            self.sort_by_resolution()
        self.df.reset_index(drop=True, inplace=True) # to allow numpy.array indexing
            
        self.df["bin"] = (max_edge/self.d_spacings()+0.5).astype(numpy.int)
        # Merge inner/outer shells if too few # TODO smarter way
        bin_counts = []
        bin_ranges = {}
        modify_table = {}
        for i_bin, g in self.df.groupby("bin", sort=True):
            bin_counts.append([i_bin, g.index])
            bin_ranges[i_bin] = (numpy.max(g.d), numpy.min(g.d))

        for i in range(len(bin_counts)):
            if len(bin_counts[i][1]) < 10 and i < len(bin_counts)-1:
                bin_counts[i+1][1] = bin_counts[i+1][1].union(bin_counts[i][1])
                modify_table[bin_counts[i][0]] = bin_counts[i+1][0]
                logger.write("Bin {} only has {} data. Merging with next bin.".format(bin_counts[i][0],
                                                                                      len(bin_counts[i][1])))
            else: break

        for i in reversed(range(len(bin_counts))):
            if i > 0 and len(bin_counts[i][1])/len(bin_counts[i-1][1]) < 0.5:
                bin_counts[i-1][1] = bin_counts[i-1][1].union(bin_counts[i][1])
                modify_table[bin_counts[i][0]] = bin_counts[i-1][0]
                logger.write("Bin {} only has {} data. Merging with previous bin.".format(bin_counts[i][0],
                                                                                          len(bin_counts[i][1])))
            else: break

        while True:
            flag = True
            for i_bin in modify_table:
                if modify_table[i_bin] in modify_table:
                    modify_table[i_bin] = modify_table[modify_table[i_bin]]
                    flag = False
            if flag: break

        for i_bin in modify_table:
            new_bin = modify_table[i_bin]
            self.df["bin"] = numpy.where(self.df["bin"].to_numpy() == i_bin, new_bin, self.df["bin"].to_numpy())
            bin_ranges[new_bin] = (max(bin_ranges[i_bin][0], bin_ranges[new_bin][0]),
                                   min(bin_ranges[i_bin][1], bin_ranges[new_bin][1]))

        self._bin_and_indices = []
        bin_all = []
        d_max_all = []
        d_min_all = []
        for i_bin, indices in bin_counts:
            if i_bin in modify_table: continue
            #if sort: # want this, but we cannot take len() for slice. we can add ncoeffs to binned_df
            #    self._bin_and_indices.append((i_bin, slice(numpy.min(indices), numpy.max(indices))))
            #else:
            self._bin_and_indices.append((i_bin, indices))
                
            bin_all.append(i_bin)
            d_max_all.append(bin_ranges[i_bin][0])
            d_min_all.append(bin_ranges[i_bin][1])
        self.binned_df = pandas.DataFrame(dict(d_max=d_max_all, d_min=d_min_all), index=bin_all)
    # setup_relion_binning()

    def binned_data_as_array(self, lab):
        vals = numpy.zeros(len(self.df.index), dtype=self.binned_df[lab].dtype)
        for i_bin, idxes in self.binned():
            vals[idxes] = self.binned_df[lab][i_bin]
        return vals
    # binned_data_as_array()

    def binned(self):
        return self._bin_and_indices
    
    def merge(self, other, common_only=True):
        self.merge_df(other, common_only)
    # merge()

    def merge_df(self, other, common_only=True):
        # TODO check space group, cell
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

    def as_numpy_arrays(self, labels, omit_nan=True):
        tmp = self.df[labels]
        if omit_nan: tmp = tmp[~tmp.isna().any(axis=1)]
        return [tmp[lab].to_numpy() for lab in labels]
    # as_numpy_arrays()

    def as_asu_data(self, label=None, data=None, label_sigma=None):
        if label is None: assert data is not None
        else: assert data is None

        if label_sigma is not None:
            assert data is None
            assert not numpy.iscomplexobj(self.df[label])
            sigma = self.df[label_sigma]
            data = numpy.lib.recfunctions.unstructured_to_structured(self.df[[label,label_sigma]].to_numpy(),
                                                                     numpy.dtype([("value", numpy.float32), ("sigma", numpy.float32)]))
        elif data is None:
            data = self.df[label]
            
        if numpy.iscomplexobj(data):
            asutype = gemmi.ComplexAsuData
        elif issubclass(data.dtype.type, numpy.integer):
            asutype = gemmi.IntAsuData
        elif label_sigma is not None:
            asutype = gemmi.ValueSigmaAsuData
        else:
            asutype = gemmi.FloatAsuData
        
        return asutype(self.cell, self.sg,
                       self.miller_array(), data)
    # as_asu_data()

    def fft_map(self, label=None, data=None, grid_size=None, sample_rate=3):
        asu = self.as_asu_data(label=label, data=data)
        if grid_size is None:
            ma = asu.transform_f_phi_to_map(sample_rate=sample_rate, exact_size=(0, 0, 0)) # half_l=True
        else:
            ma = gemmi.transform_f_phi_grid_to_map(asu.get_f_phi_on_grid(grid_size)) # half_l=False
            
        return ma
    # fft_map()

    def d_eff(self, label):
        # Effective resolution definied using FSC
        fsc = self.binned_df[label]
        a = 0.
        for i_bin, idxes in self.binned():
            a += len(idxes) * fsc[i_bin]

        fac = (a/len(self.df.index))**(1/3.)
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
        H[1,1] = numpy.sum(s2p**2/8)
        H[0,1] = H[1,0] = -numpy.sum(s2p)/2
        x = -numpy.dot(numpy.linalg.inv(H), g)
        k1 = numpy.exp(x[0])
        B1 = x[1]
        logger.write(" initial estimate using log: k= {:.2e} B= {:.2e}".format(k1, B1))
        f2tmp = f2 * k1 * numpy.exp(-B1*s2/4)
        r_step0 = r_factor(f1, f2)
        r_step1 = r_factor(f1, f2tmp)
        logger.write(" R= {:.4f} (was: {:.4f})".format(r_step1, r_step0))

        # 2nd step: - minimize (|f1|-|f2|*k*e^(-b*s2/4))^2 iteratively (TODO with regularisation)

        def grad2(x):
            t = numpy.exp(-x[1]*s2/4)
            tmp = (f1-f2*x[0]*t)*f2*t
            return numpy.array([-2.*numpy.sum(tmp),
                                0.5*x[0]*numpy.sum(tmp*s2)])

        def hess2(x):
            h = numpy.zeros((2, 2))
            t = numpy.exp(-x[1]*s2/4)
            t2 = t**2
            h[0,0] = numpy.sum(f2**2 * t2) * 2
            h[1,1] = numpy.sum(f2 * s2**2/4 * (-f1/2*t + f2*x[0]*t2)) * x[0]
            h[1,0] = numpy.sum(f2 * s2 * (f1/2*t - f2*x[0]*t2))
            h[0,1] = h[1,0]
            return h

        res = scipy.optimize.minimize(fun=lambda x: sum((f1-f2*x[0]*numpy.exp(-x[1]*s2/4))**2),
                                      jac=grad2,
                                      hess=hess2,
                                      method="Newton-CG",
                                      x0=numpy.array([k1, B1]),
                                      )
        logger.write(str(res))
        k2, B2 = res.x
        f2tmp2 = f2 * k2 * numpy.exp(-B2*s2/4)
        r_step2 = r_factor(f1, f2tmp2)
        logger.write(" Least-square estimate: k= {:.2e} B= {:.2e}".format(k2, B2))
        logger.write(" R= {:.4f}".format(r_step2))

        if 0:
            self.setup_binning(40)        
            x = []
            y0,y1,y2,y3=[],[],[],[]
            for i_bin, idxes in self.binned():
                bin_d_min = hkldata.binned_df.d_min[i_bin]
                bin_d_max = hkldata.binned_df.d_max[i_bin]
                x.append(1/bin_d_min**2)
                y0.append(numpy.average(f1[idxes]))
                y1.append(numpy.average(f2[idxes]))
                y2.append(numpy.average(f2tmp[idxes]))
                y3.append(numpy.average(f2tmp2[idxes]))

            import matplotlib.pyplot as plt
            plt.plot(x, y0, label="FC")
            plt.plot(x, y1, label="FP")
            plt.plot(x, y2, label="FP,scaled")
            plt.plot(x, y3, label="FP,scaled2")
            plt.legend()
            plt.show()

        if r_step2 < r_step1:
            return k2, B2
        else:
            return k1, B1
    # scale_k_and_b()

    def translate(self, lab, shift):
        # apply phase shift
        assert numpy.iscomplexobj(self.df[lab])
        
        if type(shift) != gemmi.Position:
            shift = gemmi.Position(*shift)
            
        self.df[lab] *= numpy.exp(2.j*numpy.pi*numpy.dot(self.miller_array(),
                                                         self.cell.fractionalize(shift).tolist()))
    # translate()

    def write_mtz(self, mtz_out, labs, types=None):
        if types is None: types = {}
        ndata = sum(2 if numpy.iscomplexobj(self.df[lab]) else 1 for lab in labs)

        data = numpy.empty((len(self.df.index), ndata + 3), dtype=numpy.float32)
        data[:,:3] = self.df[["H","K","L"]]
        idx = 3
        for lab in labs:
            if numpy.iscomplexobj(self.df[lab]):
                data[:,idx] = numpy.abs(self.df[lab])
                data[:,idx+1] = numpy.angle(self.df[lab], deg=True)
                idx += 2
            else:
                data[:,idx] = self.df[lab]
                idx += 1

        mtz = gemmi.Mtz()
        mtz.spacegroup = self.sg
        mtz.cell = self.cell
        mtz.add_dataset('HKL_base')
        for label in ['H', 'K', 'L']: mtz.add_column(label, 'H')

        for lab in labs:
            if numpy.iscomplexobj(self.df[lab]):
                mtz.add_column(lab, "F")
                mtz.add_column(("PH"+lab).replace("FWT", "WT"), "P")
            else:
                typ = types.get(lab)
                if typ is None:
                    if issubclass(self.df[lab].dtype.type, numpy.integer):
                        typ = "I"
                    else:
                        typ = "R"
                mtz.add_column(lab, typ)
    
        mtz.set_data(data)
        mtz.write_to_file(mtz_out)
    # write_mtz()
