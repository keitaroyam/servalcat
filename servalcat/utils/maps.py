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
import scipy.signal
from servalcat.utils import logger
from servalcat.utils import hkl

def new_grid_like(gr):
    newgr = type(gr)(*gr.shape)
    newgr.set_unit_cell(gr.unit_cell)
    newgr.spacegroup = gr.spacegroup
    return newgr
# new_grid_like()

def mask_from_model():
    pass

def test_mask_with_model(mask, st, mask_threshold=.5, inclusion_cutoff=.8):
    logger.write("Testing mask with model..")
    n_all = 0
    n_out = 0
    for cra in st[0].all():
        v = mask.interpolate_value(cra.atom.pos)
        n_all += 1
        if v < mask_threshold:
            logger.write(" WARNING: mask value at {} = {:.1f}".format(cra, v))
            n_out += 1

    logger.write(" n_atoms= {} n_out_of_mask= {} ({:.2%})".format(n_all, n_out, n_out/n_all if n_all>0 else 0))
    return 1 - n_out/n_all > inclusion_cutoff
# test_mask_with_model()

def half2full(map_h1, map_h2):
    assert map_h1.shape == map_h2.shape
    assert map_h1.unit_cell == map_h2.unit_cell
    tmp = (map_h1.array + map_h2.array)/2.
    gr = gemmi.FloatGrid(tmp, map_h1.unit_cell, map_h1.spacegroup)
    return gr
# half2full()

def nyquist_resolution(map_grid):
    grid_shape = map_grid.shape
    rec_cell = map_grid.unit_cell.reciprocal().parameters
    resolutions = [2./rec_cell[i]/grid_shape[i] for i in (0,1,2)]
    return max(resolutions)
# nyquist_resolution()

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
        for i_bin, idxes in hkldata.binned():
            bin_d_min = hkldata.binned_df.d_min[i_bin]
            Fo = hkldata.df.FP.to_numpy()[idxes]
            FSCfull = hkldata.binned_df.FSCfull[i_bin]
            sig_fo = numpy.std(Fo)
            if FSCfull > 0:
                n_fo = sig_fo * numpy.sqrt(FSCfull)
            else:
                n_fo = sig_fo # XXX not a right way
                
            normalizer[idxes] = n_fo
            for lab in labs: hkldata.df.loc[idxes, lab] /= n_fo
            logger.write("{:.4f} {:.2f} {:.3f} {:.4f}".format(1/bin_d_min**2,
                                                              numpy.log(numpy.average(numpy.abs(Fo))),
                                                              n_fo, FSCfull))

        logger.write("$$")

    else:
        logger.write("Sharpening B before masking= {}".format(b))
        s2 = 1./hkldata.d_spacings()**2
        normalizer[:] = numpy.exp(-b*s2/4.)
        for lab in labs: hkldata.df.loc[:, lab] /= normalizer

    # 2. Mask, FFT, and unsharpen
    for lab in labs:
        m = hkldata.fft_map(lab, grid_size=mask.shape)
        m.array[:] *= mask
        #write_ccp4_map("debug_{}.ccp4".format(lab), new_maps[-1][0])
        rg = gemmi.transform_map_to_f_phi(m)
        hkldata.df[lab] = rg.get_value_by_hkl(hkldata.miller_array()) * normalizer

    # TODO can return here for most use cases?
    
    new_maps = []
    for i, lab in enumerate(labs):
        m = hkldata.fft_map(lab, grid_size=mask.shape)
        new_maps.append([m]+maps[i][1:])

    return new_maps
# sharpen_mask_unsharpen()

def mask_and_fft_maps(maps, d_min, mask=None):
    assert len(maps) <= 2
    hkldata = None
    for i, m in enumerate(maps):
        if len(maps) == 2:
            lab = "F_map{}".format(i+1)
        else:
            lab = "FP"
        g = m[0]
        if mask is not None:
            g.array[:] *= mask
        f_grid = gemmi.transform_map_to_f_phi(g)
        if hkldata is None:
            asudata = f_grid.prepare_asu_data(dmin=d_min)
            hkldata = hkl.hkldata_from_asu_data(asudata, lab)
        else:
            hkldata.df[lab] = f_grid.get_value_by_hkl(hkldata.miller_array())

    if len(maps) == 2:
        hkldata.df["FP"] = (hkldata.df.F_map1 + hkldata.df.F_map2)/2.
        
    return hkldata
# mask_and_fft_maps()
    
def calc_noise_var_from_halfmaps(hkldata):
    hkldata.binned_df["var_noise"] = 0.
    hkldata.binned_df["var_signal"] = 0.
    hkldata.binned_df["FSCfull"] = 0.
    
    logger.write("Bin Ncoeffs d_max   d_min   FSChalf var.noise")
    for i_bin, idxes in hkldata.binned():
        bin_d_min = hkldata.binned_df.d_min[i_bin]
        bin_d_max = hkldata.binned_df.d_max[i_bin]
        
        sel1 = hkldata.df.F_map1.to_numpy()[idxes]
        sel2 = hkldata.df.F_map2.to_numpy()[idxes]

        if sel1.size < 3:
            logger.write("WARNING: skipping bin {} with size= {}".format(i_bin, sel1.size))
            continue

        fsc = numpy.real(numpy.corrcoef(sel1, sel2)[1,0])
        varn = numpy.var(sel1-sel2)/4
        vart = numpy.var(sel1+sel2)/4
        logger.write("{:3d} {:7d} {:7.3f} {:7.3f} {:.4f} {:e}".format(i_bin, sel1.size, bin_d_max, bin_d_min,
                                                                      fsc, varn))
        hkldata.binned_df.loc[i_bin, "var_noise"] = varn
        hkldata.binned_df.loc[i_bin, "var_signal"] = vart-varn
        hkldata.binned_df.loc[i_bin, "FSCfull"] = 2*fsc/(1+fsc)
# calc_noise_var_from_halfmaps()

def write_ccp4_map(filename, array, cell=None, sg=None, mask_for_extent=None, mask_threshold=0.5, mask_padding=5,
                   grid_start=None, grid_shape=None, update_cell=False):
    """
    - If mask_for_extent is set: grid_shape is ignored
    - grid_shape must be specified together with grid_start.
    - mask_padding unit: px
    """
    logger.write("Writing map file: {}".format(filename))
    ccp4 = gemmi.Ccp4Map()
    
    if type(array) == numpy.ndarray:
        # TODO check dtype
        if sg is None: sg = gemmi.SpaceGroup(1)
        ccp4.grid = gemmi.FloatGrid(array, cell, sg)
    else:
        # TODO check type
        ccp4.grid = array
        if cell is not None: ccp4.grid.set_unit_cell(cell)
        if sg is not None: ccp4.grid.spacegroup = sg

    ccp4.update_ccp4_header(2, True) # float, update stats

    if mask_for_extent is not None: # want to crop part of map using mask
        tmp = numpy.where(mask_for_extent.array > mask_threshold)
        if grid_start is not None:
            grid_start = numpy.array(grid_start)[:,None]
            shape = numpy.array(ccp4.grid.shape)[:,None]
            tmp -= grid_start
            tmp += (shape*numpy.floor(1-tmp/shape)).astype(int) + grid_start
            
        l = [(min(x)-mask_padding, max(x)+mask_padding) for x in tmp]
        grid_start = [l[i][0] for i in range(3)]
        grid_shape = [l[i][1]-l[i][0]+1 for i in range(3)]
        
    if grid_start is not None: # want to change origin
        new_shape = ccp4.grid.shape if grid_shape is None else grid_shape
        logger.write(" setting starting grid: {} {} {}".format(*grid_start))
        logger.write(" setting     new shape: {} {} {}".format(*new_shape))
        
        new_cell = ccp4.grid.unit_cell
        if update_cell:
            abc = [new_cell.parameters[i]*new_shape[i]/ccp4.grid.shape[i] for i in range(3)]
            new_cell = gemmi.UnitCell(abc[0], abc[1], abc[2],
                                      new_cell.alpha, new_cell.beta, new_cell.gamma)
            logger.write(" setting      new cell: {:6.2f} {:6.2f} {:6.2f} {:5.1f} {:5.1f} {:5.1f}".format(*new_cell.parameters))
            cell_grid = new_shape
        else:
            cell_grid = ccp4.grid.shape
            
        new_grid = gemmi.FloatGrid(ccp4.grid.get_subarray(grid_start, new_shape),
                                   new_cell,
                                   ccp4.grid.spacegroup)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = new_grid
        ccp4.update_ccp4_header(2, True) # float, update stats
        for i in range(3):
            ccp4.set_header_i32(5+i, grid_start[i])
            ccp4.set_header_i32(1+i, new_shape[i])
            ccp4.set_header_i32(8+i, cell_grid[i])

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

def raised_cosine_kernel(r1, dr=2):
    assert r1 > 2
    assert dr >= 0
    assert r1 > dr
    
    boxsize = 2 * r1 + 1
    x, y, z = numpy.meshgrid(range(boxsize), range(boxsize), range(boxsize))
    cen = boxsize // 2
    r0 = r1 - dr
    d = numpy.sqrt((x-cen)**2+(y-cen)**2+(z-cen)**2)
    kern = 0.5 + 0.5 * numpy.cos(numpy.pi * (d - r0) / (r1 - r0))
    kern[d<=r0] = 1
    kern[d>=r1] = 0
    kern /= numpy.sum(kern)
    return kern
# raised_cosine_kernel()

def local_var(grid, kernel):
    mean_x2 = scipy.signal.fftconvolve(grid.array**2, kernel, "same")
    mean_x = scipy.signal.fftconvolve(grid.array, kernel, "same")
    var_x = new_grid_like(grid)
    var_x.array[:] = mean_x2 - mean_x**2
    var_x.array[var_x.array<0] = 0 # due to loss of significance
    return var_x
# local_var()

def local_cc(map1, map2, kernel):
    localcc = new_grid_like(map1)
    mean_1_sqr = scipy.signal.fftconvolve(map1.array**2, kernel, "same")
    mean_2_sqr = scipy.signal.fftconvolve(map2.array**2, kernel, "same")
    mean_1 = scipy.signal.fftconvolve(map1.array, kernel, "same")
    mean_2 = scipy.signal.fftconvolve(map2.array, kernel, "same")
    mean_12 = scipy.signal.fftconvolve(map1.array * map2.array, kernel, "same")
    var_1 = mean_1_sqr - mean_1**2
    var_1[var_1 < 0] = 0
    var_2 = mean_2_sqr - mean_2**2
    var_2[var_2 < 0] = 0
    covar_12 = mean_12 - mean_1 * mean_2
    localcc.array[:] = covar_12 / numpy.sqrt(var_1 * var_2)
    return localcc
# local_cc()
