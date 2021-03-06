"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import scipy.optimize
from servalcat.utils import logger
from servalcat.utils import hkl

def mask_from_model():
    pass

def half2full(map_h1, map_h2):
    assert map_h1.shape == map_h2.shape
    assert map_h1.unit_cell == map_h2.unit_cell
    tmp = (numpy.array(map_h1)+numpy.array(map_h2))/2.
    gr = gemmi.FloatGrid(tmp, map_h1.unit_cell, map_h1.spacegroup)
    return gr
# half2full()

def sharpen_mask_unsharpen(maps, mask, d_min, b=None):
    assert len(maps) < 3
    if b is None and len(maps) != 2:
        raise RuntimeError("Cannot determine sharpening")

    hkldata = mask_and_fft_maps(maps, d_min)
    normalizer = numpy.ones(len(hkldata.df.index))
    if len(maps) == 2:
        labs = ["F_map1", "F_map2"]
    else:
        labs = ["FP"]
        
    # 1. Sharpen
    if b is None:
        hkldata.setup_relion_binning()
        calc_noise_var_from_halfmaps(hkldata)
        logger.write("""$TABLE: Normalizing before masking:
$GRAPHS: ln(Mn(|F|)) :A:1,2:
: Normalizer :A:1,3:
: FSC(full) :A:1,4:
$$ 1/resol^2 ln(Mn(|F|)) normalizer FSC $$
$$""")
        bin_limits = dict(hkldata.bin_and_limits())
        for i_bin, g in hkldata.binned():
            bin_d_max, bin_d_min = bin_limits[i_bin]
            Fo = g.FP.to_numpy()
            FSCfull = hkldata.binned_df.FSCfull[i_bin]
            sig_fo = numpy.std(Fo)
            if FSCfull > 0:
                n_fo = sig_fo * numpy.sqrt(FSCfull)
            else:
                n_fo = sig_fo # XXX not a right way
                
            normalizer[g.index] = n_fo
            for lab in labs: hkldata.df.loc[g.index, lab] /= n_fo
            logger.write("{:.4f} {:.2f} {:.3f} {:.4f}".format(1/bin_d_min**2,
                                                              numpy.log(numpy.average(numpy.abs(Fo))),
                                                              n_fo, FSCfull))

        logger.write("$$")

    else:
        logger.write("Sharpening B before masking= {}".format(b))
        s2 = 1./hkldata.d_spacings()**2
        normalizer[:] = numpy.exp(-b*s2/4.)
        for lab in labs: hkldata.df.loc[:, lab] /= normalizer

    # 2. Mask
    new_maps = []
    for lab in labs:
        m = hkldata.fft_map(lab, grid_size=mask.shape)
        new_maps.append([gemmi.FloatGrid(numpy.array(m)*mask, hkldata.cell, hkldata.sg), None])

    # 3. Unsharpen
    hkldata = mask_and_fft_maps(new_maps, d_min)
    for lab in labs: hkldata.df[lab] *= normalizer
    
    new_maps = []
    for i, lab in enumerate(labs):
        m = hkldata.fft_map(lab, grid_size=mask.shape)
        new_maps.append([m]+maps[i][1:])

    return new_maps
# sharpen_mask_unsharpen()

def mask_and_fft_maps(maps, d_min, mask=None):
    assert len(maps) <= 2
    asus = []
    for m in maps:
        g = m[0]
        if mask is not None:
            g = gemmi.FloatGrid(numpy.array(g)*mask,
                                g.unit_cell, g.spacegroup)

        asus.append(gemmi.transform_map_to_f_phi(g).prepare_asu_data(dmin=d_min))

    if len(maps) == 2:
        df = hkl.df_from_asu_data(asus[0], "F_map1")
        hkldata = hkl.HklData(asus[0].unit_cell, asus[0].spacegroup, False, df)
        hkldata.merge_asu_data(asus[1], "F_map2")
        hkldata.df["FP"] = (hkldata.df.F_map1 + hkldata.df.F_map2)/2.
    else:
        df = hkl.df_from_asu_data(asus[0], "FP")
        hkldata = hkl.HklData(asus[0].unit_cell, asus[0].spacegroup, False, df)
        
    return hkldata
# mask_and_fft_maps()
    
def calc_noise_var_from_halfmaps(hkldata):
    # Scale
    #iniscale = utils.scaling.InitialScaler(fo_asu, fc_asu, aniso=True)
    #iniscale.run()
    #scale_for_fo = iniscale.get_scales()
    #fo_asu.value_array[:] *= scale_for_fo
    #asu1.value_array[:] *= scale_for_fo
    #asu2.value_array[:] *= scale_for_fo

    s_array = 1./hkldata.d_spacings()
    hkldata.binned_df["var_noise"] = 0.
    hkldata.binned_df["var_signal"] = 0.
    hkldata.binned_df["FSCfull"] = 0.
    
    logger.write("Bin Ncoeffs d_max   d_min   FSChalf var.noise   scale")
    bin_limits = dict(hkldata.bin_and_limits())
    for i_bin, g in hkldata.binned():
        bin_d_max, bin_d_min = bin_limits[i_bin]
        # scale
        scale = 1. #numpy.sqrt(var_cmpl(fc)/var_cmpl(fo))
        #hkldata.df.loc[sel, "FP"] *= scale
        #hkldata.df.loc[sel, "F_map1"] *= scale
        #hkldata.df.loc[sel, "F_map2"] *= scale
        
        sel1 = g.F_map1.to_numpy()
        sel2 = g.F_map2.to_numpy()

        if sel1.size < 3:
            logger.write("WARNING: skipping bin {} with size= {}".format(i_bin, sel1.size))
            continue

        fsc = numpy.real(numpy.corrcoef(sel1, sel2)[1,0])
        varn = numpy.var(sel1-sel2)/4
        vart = numpy.var(sel1+sel2)/4
        logger.write("{:3d} {:7d} {:7.3f} {:7.3f} {:.4f} {:e} {}".format(i_bin, sel1.size, bin_d_max, bin_d_min,
                                                                         fsc, varn, scale))
        hkldata.binned_df.loc[i_bin, "var_noise"] = varn
        hkldata.binned_df.loc[i_bin, "var_signal"] = vart-varn
        hkldata.binned_df.loc[i_bin, "FSCfull"] = 2*fsc/(1+fsc)
# calc_noise_var_from_halfmaps()

def write_ccp4_map(filename, array, cell=None, sg=None, mask_for_extent=None, grid_start=None):
    if type(array) == numpy.ndarray:
        # TODO check dtype
        if sg is None: sg = gemmi.SpaceGroup(1)
        grid = gemmi.FloatGrid(array, cell, sg)
    else:
        # TODO check type
        grid = array
        
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True) # float, update stats

    if mask_for_extent is not None:
        tmp = numpy.where(numpy.array(mask_for_extent)>0)
        if grid_start is not None:
            grid_start = numpy.array(grid_start)[:,None]
            grid_shape = numpy.array(grid.shape)[:,None]
            tmp -= grid_start
            tmp += (grid_shape*numpy.floor(1-tmp/grid_shape)).astype(int) + grid_start

        l = [(min(x), max(x)) for x in tmp]
        box = gemmi.FractionalBox()
        for i in (0, 1):
            fm = grid.get_fractional(l[0][i], l[1][i], l[2][i])
            box.extend(fm)

        logger.write(" setting extent: {} {}".format(box.minimum, box.maximum))
        ccp4.set_extent(box)
    elif grid_start is not None:
        logger.write(" setting starting grid: {} {} {}".format(*grid_start))
        new_grid = gemmi.FloatGrid(grid.get_subarray(*(list(grid_start)+list(grid.shape))),
                                   cell,
                                   sg)
        ccp4.grid = new_grid
        for i in range(3):
            ccp4.set_header_i32(5+i, grid_start[i])

    ccp4.write_ccp4_map(filename)
# write_ccp4_map()

def optimize_peak(grid, ini_pos):
    logger.write("Finding peak using interpolation..")
    x = grid.unit_cell.fractionalize(ini_pos)
    logger.write("       x0: [{}, {}, {}]".format(*x.tolist()))
    logger.write("       f0: {}".format(-grid.tricubic_interpolation(x)))

    res = scipy.optimize.minimize(fun=lambda x:-grid.tricubic_interpolation(gemmi.Fractional(*x)),
                                  x0=x.tolist(),
                                  jac=lambda x:-numpy.array(grid.tricubic_interpolation_der(gemmi.Fractional(*x))[1:])
                                  )
    logger.write(str(res))
    final_pos = grid.unit_cell.orthogonalize(gemmi.Fractional(*res.x))
    logger.write(" Move from initial: [{:.3f}, {:.3f}, {:.3f}] A".format(*(final_pos-ini_pos).tolist()))
    return final_pos
# optimize_peak()
